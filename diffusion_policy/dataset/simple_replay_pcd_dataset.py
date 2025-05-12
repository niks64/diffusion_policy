from typing import Dict, List, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from robomimic.utils.obs_utils import pcd_to_voxel, WS_SIZE, VOXEL_RESO
from robomimic.utils.obs_utils import WORKSPACE as VALID_WORKSPACE

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    get_voxel_identity_normalizer
)
from diffusion_policy.utils.action_utils import (apply_se3_augmentation_to_pcd, globalize_abs_action,
                                                 apply_se3_augmentation_to_abs_action, apply_se3_augmentation_to_lowdim, get_random_se3_transform,
                                                 )
import robomimic.utils.obs_utils as ObsUtils
register_codecs()

from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix

class PCDNormalizer(nn.Module):
    def __init__(self, gripper_centric, bbox_size_m=0.2) -> None:
        super().__init__()
        self.gripper_centric = gripper_centric
        self.bbox_size_m = bbox_size_m

        self.ws_range = torch.from_numpy(VALID_WORKSPACE).float()
    
    def normalize(self, pcd):
        device = pcd.device
        pcd[...,3:] = pcd[...,3:] * 2 - 1
        if self.gripper_centric:
            pcd[...,:3] = pcd[...,:3] / (self.bbox_size_m / 2)
        else:
            ws_range = self.ws_range.to(device)
            ws_min, ws_max = ws_range[:,0].view(1, 1, 1, -1).to(device), ws_range[:,1].view(1, 1, 1, -1).to(device)
            pcd[...,:3] = (pcd[...,:3] - ws_min) / (ws_max - ws_min) * 2 - 1
        
        return pcd

    def unnormalize(self, pcd, previous_ee_pose):
        # no need to unnormalize pcd
        return pcd

class DepthNormalizer(nn.Module):
    def __init__(self, stat) -> None:
        super().__init__()
        self.min = torch.from_numpy(stat['min'])
        self.max = torch.from_numpy(stat['max'])
    
    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        device = x.device
        self.min, self.max = self.min.to(device), self.max.to(device)
        x[x > self.max] = self.max
        x = (x - self.min) / (self.max - self.min)
        return x * 2 - 1

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        device = x.device
        self.min, self.max = self.min.to(device), self.max.to(device)
        return (x + 1) / 2 * (self.max - self.min) + self.min
    
class VoxelNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, voxel):
        voxel = voxel * 2 - 1
        return voxel
    
class SimpleReplayPCDDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            se2_augmentation=False,
            fix_point_num=1024,
            use_voxel=False,
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        assert 'abs' in dataset_path, "this class only supports abs actions"

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)

        pcd_keys = list()
        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        self.depth_range = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'pcd':
                pcd_keys.append(key)
            elif type == 'rgb':
                rgb_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
                # self.depth_range[key] = ObsUtils.DEPTH_MINMAX[key]
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in pcd_keys + rgb_keys + depth_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        assert max_train_episodes <= replay_buffer.n_episodes, f"max_train_episodes={max_train_episodes} is greater than total episodes={replay_buffer.n_episodes}"
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        self.sequence_length = max(n_obs_steps, horizon)
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=self.sequence_length,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
            )
        
        self.se2_augmentation = se2_augmentation
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.pcd_keys = pcd_keys
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        assert abs_action == True, "this class only supports abs action"

        self.fix_point_num = fix_point_num
        self.use_voxel = use_voxel
        self.ws_center = np.array([0, 0], dtype=np.float32)
        self.ws_size = WS_SIZE
        if self.use_voxel:
            self.voxel_size = WS_SIZE / VOXEL_RESO

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.sequence_length,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])

        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                magnitute = max(np.max([stat['max'][:2] - self.ws_center, self.ws_center - stat['min'][:2]]), self.ws_size/2)
                stat['min'][:2] = self.ws_center - magnitute
                stat['max'][:2] = self.ws_center + magnitute
                stat['mean'][:2] = self.ws_center
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
                # this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('eef_pos'):
                magnitute = max(np.max([stat['max'][:2] - self.ws_center, self.ws_center - stat['min'][:2]]), self.ws_size/2)
                stat['min'][:2] = self.ws_center - magnitute
                stat['max'][:2] = self.ws_center + magnitute
                stat['mean'][:2] = self.ws_center
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # pcd 
        for key in self.pcd_keys:
            if self.use_voxel:
                normalizer[key] = get_voxel_identity_normalizer()
            else:
                stat = {'min': torch.tensor(np.array([*VALID_WORKSPACE[:,0], 0., 0., 0.]), dtype=torch.float32),
                        'max': torch.tensor(np.array([*VALID_WORKSPACE[:,1], 1., 1., 1.]), dtype=torch.float32),}
                normalizer[key] = get_range_normalizer_from_stat(stat)

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # depth
        # for key in self.depth_keys:
        #     # depth_min, depth_max = self.depth_range[key]
        #     # stat = {'min': np.array([depth_min], dtype=np.float32), 'max': np.array([depth_max], dtype=np.float32)}
        #     # normalizer[key] = DepthNormalizer(stat) 
        #     depth_min, depth_max = self.depth_range[key]
        #     stat = {'min': np.array([depth_min], dtype=np.float32), 'max': np.array([depth_max], dtype=np.float32)}
        #     # normalizer[key] = DepthNormalizer(stat) 
        #     normalizer[key] = get_range_normalizer_from_stat(stat) 

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def get_data(self, k, obs_dict, data):
        return data[k][:self.n_obs_steps]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        eef_pose = np.concatenate([data['robot0_eef_pos'], data['robot0_eef_quat']], axis=1) # [3] + [4]

        A_slice = slice(self.horizon)
        action = data['action'][A_slice].astype(np.float32)

        if self.se2_augmentation:
            transform, params = get_random_se3_transform(trans_sigma=0.0,theta_x_sigma=0,theta_y_sigma=0,theta_z_sigma=45)
        
        obs_dict = dict()

        for key in self.pcd_keys:
            pcd = self.get_data(key, obs_dict, data)


            current_eef_pose = eef_pose[self.n_obs_steps-1:self.n_obs_steps,...].copy()


            if self.se2_augmentation: # aug on the local pcd WRT gripper frame
                assert self.abs_action
                pcd = apply_se3_augmentation_to_pcd(transform, pcd)
                action = apply_se3_augmentation_to_abs_action(transform, action)

            if self.use_voxel:
                gripper_crop = None
                pcd = pcd_to_voxel(pcd, gripper_crop, voxel_size=self.voxel_size)

            obs_dict[key] = pcd
            # obs_dict['robot0_eef_pos'] = previous_eef_poss[0]

        T_slice = slice(self.n_obs_steps)
        for key in self.rgb_keys:
            rgb = self.get_data(key, obs_dict, data)
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(rgb,-1,1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]

        for key in self.depth_keys:
            depth = self.get_data(key, obs_dict, data)
            # move channel last to channel first 
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(depth,-1,1).astype(np.float32)
            # T,C,H,W
            del data[key]
        
        for key in self.lowdim_keys:
            lowdim = self.get_data(key, obs_dict, data)
            # move channel last to channel first 
            # T,H,W,C
            # convert uint8 image to float32

            if self.se2_augmentation: # aug on the local pcd WRT gripper frame
                lowdim = apply_se3_augmentation_to_lowdim(transform, lowdim, key)

            obs_dict[key] = lowdim.astype(np.float32)
            del data[key]

        debug=False
        if debug:
            global_action = globalize_abs_action(action[None, ...], eef_pose[0:1][None,...], is_emptys[-1:][None,...], local_type=self.local_type)
            from diffusion_policy.utils.visualizer import visualize_pcd_and_action, visualize_pcd_action_and_pose, visualize_pcd_and_pose, visualize_voxel
            visualize_pcd_and_action([pcd[0], pcd[1]], action)
            visualize_pcd_and_action([data['pcd'][0], data['pcd'][-1]], global_action[0])
            visualize_pcd_action_and_pose([pcd[0], pcd[-1]], action, eef_pose)
            visualize_pcd_action_and_pose([pcd[0], pcd[-1]], action, np.concatenate([obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat']], axis=-1))
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()-1
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    obs_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'pcd':
            obs_keys.append(key)
        elif type == 'rgb':
            obs_keys.append(key)
        elif type == 'depth':
            obs_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    # obs_keys = ['spaceview_image', 'spaceview_depth', 'robot0_eye_in_hand_image', 'pcd']
    # lowdim_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
    # shape_meta['obs']['pcd'] = {'shape': [4412, 6], 'type': 'pcd'}
    # shape_meta['obs']['spaceview_image'] = {'shape': [3, 128, 128], 'type': 'rgb'}
    # shape_meta['obs']['spaceview_depth'] = {'shape': [1, 128, 128], 'type': 'depth'}

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        
        # obs_keys = pcd_keys
        with tqdm(total=n_steps*(len(obs_keys)), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in obs_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    
                    if 'pcd' in key:
                        n, c = shape
                        obs_shape = (n_steps,n,c)
                        obs_chunk = (1,n,c)
                        this_compressor = None
                        dtype = np.float32
                    elif 'image' in key:
                        c,h,w = shape
                        obs_shape = (n_steps,h,w,c)
                        obs_chunk = (1,h,w,c)
                        this_compressor = Jpeg2k(level=50)
                        dtype = np.uint8
                    elif 'depth' in key:
                        c,h,w = shape
                        obs_shape = (n_steps,h,w,c)
                        obs_chunk = (1,h,w,c)
                        this_compressor = None
                        dtype = np.float32
                    
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=obs_shape,
                        chunks=obs_chunk,
                        compressor=this_compressor,
                        dtype=dtype
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        
                        hdf5_arr = demo['obs'][key]

                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))
                                

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                            executor.submit(img_copy, 
                                img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
