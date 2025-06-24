from typing import Dict, Optional
import os
import torch
import numpy as np
import pickle
import zarr
import shutil
import copy
from tqdm import tqdm
from filelock import FileLock
import multiprocessing
import concurrent.futures

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

register_codecs()


def _get_nested(data: Dict, key: str):
    """Utility function to retrieve a value from a nested dictionary given a dotted key string."""
    parts = key.split('.')
    out = data
    for p in parts:
        out = out[p]
    return out


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication.
    Both inputs are (...,4) in (x,y,z,w) convention.
    Returns the product with the same shape.
    """
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([x, y, z, w], axis=-1)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    q_conj = q.copy()
    q_conj[..., :3] *= -1.0
    return q_conj


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to axis-angle representation.
    Input shape (...,4) (x,y,z,w).
    Returns (...,3) axis-angle (axis * angle).
    """
    eps = 1e-8
    qw = q[..., 3:4]
    q_xyz = q[..., :3]
    norm_xyz = np.linalg.norm(q_xyz, axis=-1, keepdims=True) + eps
    angle = 2.0 * np.arctan2(norm_xyz, np.clip(qw, -1.0, 1.0))  # (...,1)
    axis = q_xyz / norm_xyz
    return axis * angle


def _compute_actions(ee_pose: np.ndarray, gripper: np.ndarray, rotation_transformer: RotationTransformer):
    """Compute action as delta pose (pos+rot) + delta gripper.
    ee_pose: (T,7)  pos(3) + quat(4)
    gripper: (T,*)  we use first element as scalar control
    Returns (T,10): delta_pos(3) + rotation_rep(6) + delta_gripper(1)
    The last timestep duplicates the previous action so that len(action)==T.
    """
    pos = ee_pose[:, :3]
    quat = ee_pose[:, 3:]

    # position deltas
    delta_pos = np.diff(pos, axis=0, prepend=pos[0:1])  # (T,3)

    # orientation deltas
    q_next = quat[1:]
    q_curr = quat[:-1]
    q_delta = _quat_mul(q_next, _quat_conj(q_curr))  # (T-1,4)
    aa = _quat_to_axis_angle(q_delta)  # (T-1,3)
    rot_rep = rotation_transformer.forward(aa.astype(np.float32))  # (T-1,6)
    rot_rep = np.concatenate([np.zeros((1, rot_rep.shape[-1]), dtype=np.float32), rot_rep], axis=0)  # prepend zero for first step

    # gripper delta (use first element)
    gripper_scalar = gripper[:, 0]
    delta_gripper = np.diff(gripper_scalar, axis=0, prepend=gripper_scalar[0:1])  # (T,)
    delta_gripper = delta_gripper[:, None]  # (T,1)

    action = np.concatenate([delta_pos, rot_rep, delta_gripper], axis=-1).astype(np.float32)
    return action


def _convert_pkl_to_replay(store, shape_meta, dataset_path, rotation_transformer,
                            n_workers: Optional[int] = None, max_inflight_tasks: Optional[int] = None):
    """Convert a folder of pickle demonstrations into a ReplayBuffer."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys = []
    lowdim_keys = []
    for key, attr in shape_meta['obs'].items():
        if attr.get('type', 'low_dim') == 'rgb':
            rgb_keys.append(key)
        else:
            lowdim_keys.append(key)

    # enumerate demonstration files (assume numeric names)
    demo_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    demo_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    episode_ends = []
    total_steps = 0

    # First pass to gather episode lengths
    demo_lengths = []
    for fname in demo_files:
        with open(os.path.join(dataset_path, fname), 'rb') as f:
            traj = pickle.load(f)
        demo_len = len(traj)
        demo_lengths.append(demo_len)
        total_steps += demo_len
        episode_ends.append(total_steps)

    _ = meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None, overwrite=True)

    # ========== load low-dim data ==========
    lowdim_buffers = {key: np.zeros((total_steps,) + tuple(shape_meta['obs'][key]['shape']), dtype=np.float32)
                      for key in lowdim_keys}
    actions_buffer = np.zeros((total_steps,) + tuple(shape_meta['action']['shape']), dtype=np.float32)

    # ========== prepare image datasets ==========
    img_arrays = {}
    for key in rgb_keys:
        c, h, w = shape_meta['obs'][key]['shape']
        compressor = Jpeg2k(level=50)
        img_arrays[key] = data_group.require_dataset(
            name=key,
            shape=(total_steps, h, w, c),
            chunks=(1, h, w, c),
            compressor=compressor,
            dtype=np.uint8
        )

    # ========== second pass to fill data ==========
    global_idx = 0
    with tqdm(total=total_steps, desc='Parsing pkl demos', mininterval=1.0) as pbar:
        for demo_idx, fname in enumerate(demo_files):
            with open(os.path.join(dataset_path, fname), 'rb') as f:
                traj = pickle.load(f)

            # gather arrays for this episode
            ee_pose_ep = []
            gripper_ep = []
            rgb_ep = {k: [] for k in rgb_keys}

            for step in traj:
                # low-dim obs
                for key in lowdim_keys:
                    value = _get_nested(step, key)  # key is dotted path
                    lowdim_buffers[key][global_idx] = np.asarray(value, dtype=np.float32)

                # store ee_pose and gripper for actions
                ee_pose_ep.append(_get_nested(step, 'observation.ee_pose'))
                gripper_ep.append(_get_nested(step, 'observation.gripper'))

                # images
                for key in rgb_keys:
                    img = _get_nested(step, key)  # expect H,W,C uint8
                    rgb_ep[key].append(np.asarray(img, dtype=np.uint8))

                global_idx += 1
                pbar.update(1)

            # stack episode arrays to numpy
            ee_pose_ep = np.asarray(ee_pose_ep, dtype=np.float32)
            gripper_ep = np.asarray(gripper_ep, dtype=np.float32)
            action_ep = _compute_actions(ee_pose_ep, gripper_ep, rotation_transformer)

            # write actions
            start_idx = global_idx - len(traj)
            actions_buffer[start_idx:global_idx] = action_ep

            # write images with thread pool
            def _img_copy(zarr_arr, z_idx, np_img):
                try:
                    zarr_arr[z_idx] = np_img
                    _ = zarr_arr[z_idx]
                    return True
                except Exception:
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for local_step_idx in range(len(traj)):
                    abs_idx = start_idx + local_step_idx
                    for key in rgb_keys:
                        if len(futures) >= max_inflight_tasks:
                            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in done:
                                assert f.result(), 'Failed to encode image!'
                        futures.add(executor.submit(_img_copy, img_arrays[key], abs_idx, rgb_ep[key][local_step_idx]))
                # wait remaining
                done, _ = concurrent.futures.wait(futures)
                for f in done:
                    assert f.result(), 'Failed to encode image!'

    # save low-dim and action data to zarr
    for key, buf in lowdim_buffers.items():
        _ = data_group.array(name=key, data=buf, shape=buf.shape, chunks=buf.shape, compressor=None, dtype=buf.dtype)
    _ = data_group.array(name='action', data=actions_buffer, shape=actions_buffer.shape,
                         chunks=actions_buffer.shape, compressor=None, dtype=actions_buffer.dtype)

    return ReplayBuffer(root)


class PklReplayImageDataset(BaseImageDataset):
    """Dataset that converts a folder of pickled demonstrations into a ReplayBuffer compatible with Diffusion-Policy."""

    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 n_obs_steps: Optional[int] = None,
                 rotation_rep: str = 'rotation_6d',
                 use_cache: bool = False,
                 seed: int = 42,
                 val_ratio: float = 0.0):

        rotation_transformer = RotationTransformer(from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_path = os.path.abspath(dataset_path.rstrip('/') + '.pkl_zarr.zip')
            lock_path = cache_path + '.lock'
            print('Acquiring lock on PKL cache...')
            with FileLock(lock_path):
                if not os.path.exists(cache_path):
                    try:
                        print('Cache does not exist. Building new one!')
                        replay_buffer = _convert_pkl_to_replay(zarr.MemoryStore(), shape_meta,
                                                               dataset_path, rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_path) as zip_store:
                            replay_buffer.save_to_store(zip_store)
                    except Exception as e:
                        if os.path.exists(cache_path):
                            shutil.rmtree(cache_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from disk.')
                    with zarr.ZipStore(cache_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(zip_store, zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_pkl_to_replay(zarr.MemoryStore(), shape_meta, dataset_path,
                                                   rotation_transformer)

        # determine rgb and low-dim keys from shape_meta
        rgb_keys = [k for k, v in shape_meta['obs'].items() if v.get('type', 'low_dim') == 'rgb']
        lowdim_keys = [k for k, v in shape_meta['obs'].items() if v.get('type', 'low_dim') == 'low_dim']

        key_first_k = {}
        if n_obs_steps is not None:
            for k in rgb_keys + lowdim_keys:
                key_first_k[k] = n_obs_steps

        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        train_mask = ~val_mask

        sampler = SequenceSampler(replay_buffer, sequence_length=horizon, pad_before=pad_before,
                                  pad_after=pad_after, episode_mask=train_mask, key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.train_mask = train_mask

    # ================= dataset utility methods =================
    def get_validation_dataset(self):
        val = copy.copy(self)
        val.sampler = SequenceSampler(self.replay_buffer, sequence_length=self.horizon,
                                      pad_before=self.pad_before, pad_after=self.pad_after,
                                      episode_mask=~self.train_mask)
        val.train_mask = ~self.train_mask
        return val

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # actions – use range normalizer
        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = get_range_normalizer_from_stat(stat)

        # low-dim observations
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos') or key.endswith('qpos'):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                normalizer[key] = get_range_normalizer_from_stat(stat)

        # image observations
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)

        obs_dict = {}
        for key in self.rgb_keys:
            img = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0  # T,C,H,W
            obs_dict[key] = img
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data 
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    pkl_replay_image_dataset = PklReplayImageDataset(
        shape_meta={
            "obs": {
                "dave_image": {
                    "shape": (3, 480, 640),
                    "type": "rgb"
                },
                "wrist_image": {
                    "shape": (3, 480, 640),
                    "type": "rgb"
                },
                "robot0_eef_pos": {
                    "shape": (3,)
                },
                "robot0_eef_quat": {
                    "shape": (4,)
                },
                "robot0_gripper_qpos": {
                    "shape": (2,)
                }
            },
            "action": {"shape": (10,)}
        },
        dataset_path = "/Users/nikunj/Desktop/CIL",
        horizon = 16,
        pad_before = 0,
        pad_after = 0,
        n_obs_steps = None,
        rotation_rep = 'rotation_6d',
        use_cache = False,
        seed = 42,
        val_ratio = 0.0
    )

    train_dataloader = DataLoader(pkl_replay_image_dataset, batch_size=4, shuffle=True)
    batch = next(iter(train_dataloader))
    print(batch)