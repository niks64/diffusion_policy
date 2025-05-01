import collections
import math
import os
import pathlib
import copy
import dill
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import tqdm
import wandb
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env

class RobomimicMultiviewRunnerNoVideo(BaseImageRunner):
    """
    Runner without video recording. Evaluates a policy on an alternate camera view.
    The `alt_render_obs_key` determines the camera view used (e.g. 'sideview_image').
    This image is remapped to the key expected by the policy (e.g. 'agentview_image').
    """
    def __init__(self,
            output_dir,
            dataset_path,
            shape_meta: dict,
            n_train=10,
            n_train_vis=0, # Set to 0 as no visualization/video is recorded
            train_start_idx=0,
            n_test=22,
            n_test_vis=0, # Set to 0 as no visualization/video is recorded
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',  # expected key for policy input
            alt_obs_key='agentview_image',  # alternate view key
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            reward_shaping=True
        ):
        super().__init__(output_dir)

        # We still need the output_dir for logging, but create it if it doesn't exist
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        shape_meta = copy.deepcopy(shape_meta)

        if n_envs is None:
            n_envs = n_train + n_test
        dataset_path = os.path.expanduser(dataset_path)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False
        env_meta['env_kwargs']['reward_shaping'] = reward_shaping
        if alt_obs_key.split("_")[0] not in env_meta['env_kwargs']['camera_names']:
            env_meta['env_kwargs']['camera_names'].append(alt_obs_key.split("_")[0])

        camera_shape = [3, 128, 128]

        new_obs = {}
        for key, value in shape_meta["obs"].items():
            if "image" in key:
                camera_shape = value["shape"]
                break

        new_obs[alt_obs_key] = {
            "shape": camera_shape,
            "type": "rgb"
        }

        new_obs["robot0_eye_in_hand_image"] = {
            "shape": camera_shape,
            "type": "rgb"
        }

        for key, value in shape_meta["obs"].items():
            if "image" not in key:
                new_obs[key] = value

        shape_meta["obs"] = new_obs

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # Enable rendering for observation generation even without video recording
        enable_render = True # Set to True to get image observations

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                enable_render=enable_render
            )
            robomimic_env.env.hard_reset = False
            # Wrap directly with RobomimicImageWrapper and MultiStepWrapper
            return MultiStepWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=alt_obs_key # Use alternate view
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    enable_render=False # No rendering needed for dummy env
                )
            # Wrap directly with RobomimicImageWrapper and MultiStepWrapper
            return MultiStepWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=alt_obs_key # Use alternate view here as well
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                init_state = f[f'data/demo_{train_idx}/states'][0]

                # init_fn no longer needs enable_render or file_path logic
                def init_fn(env, init_state=init_state):
                    # switch to init_state reset
                    assert isinstance(env.env, RobomimicImageWrapper)
                    env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            # init_fn no longer needs enable_render or file_path logic
            def init_fn(env, seed=seed):
                # switch to seed reset
                assert isinstance(env.env, RobomimicImageWrapper)
                env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.default_view_key = render_obs_key
        self.alt_render_obs_key = alt_obs_key
        # Removed fps and crf attributes

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data - remove video paths
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}MultiviewNoVideo {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                np_obs_dict = dict(obs)
                if self.alt_render_obs_key != self.default_view_key:
                    np_obs_dict[self.default_view_key] = np_obs_dict[self.alt_render_obs_key]
                    del np_obs_dict[self.alt_render_obs_key]

                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round - remove video paths
            # env.render() is not called as there's no VideoRecordingWrapper
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # No need to clear video buffer
        _ = env.reset()

        # log - remove video logging
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            # Removed video logging

        for prefix, value in max_rewards.items():
            ms_name = prefix+'mean_score'
            ms_value = np.mean(value)
            log_data[ms_name] = ms_value

            sr_name = prefix+'success_rate'
            sr_value = sum(x == 1.0 for x in value) / len(value) * 100
            log_data[sr_name] = sr_value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction 