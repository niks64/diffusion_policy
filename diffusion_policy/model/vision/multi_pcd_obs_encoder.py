from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.noise_adder import (PointCloudJitter, RandomPointDropout, FarthestPointSampling, NoisePointAdder,
                                                       PointCloudOffsetter, RandomColorDropout)
from diffusion_policy.model.vision.crop_randomizer import VoxelCropRandomizer

def send_cons_to_device(self, constants, device):
    l = []
    for cons in constants:
        cons = cons.to(device)
        l.append(cons)
    return l

# Custom transform for clipping
class RGBDTransform(nn.Module):
    def __init__(self, mean, std, min_val, max_val, device='cuda:0'):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        assert x.shape[1] == 4, "Input should have 4 channels (RGBD)"
        mean, std = send_cons_to_device([self.mean, self.std], x.device)
        # x[3:4] = torch.clamp(x[3:4], min=self.min_val, max=self.max_val)
        # x[3:4] = (x[3:4] - self.min_val) / (self.max_val - self.min_val)
        x = (x - mean[:, None, None]) / std[:, None, None]
        return x
    
    

# Custom center crop for 3d voxel (4, X, Y, Z)
class CenterCrop3D(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.half_size = torch.tensor(size)//2
    
    def forward(self, x):
        assert x.shape[1] == 4, "Input should have 4 channels (RGBD)"
        assert len(x.shape) == 5, "Input should have a shape of 5 (B, C, X, Y, Z)"
        half_size = send_cons_to_device([self.half_size], x.device)
        _, _, X, Y, Z = x.shape
        return x[:, :, X//2-self.half_size[0]:X//2+self.half_size[0], Y//2-self.half_size[1]:Y//2+self.half_size[1], Z//2-self.half_size[2]:Z//2+self.half_size[2]]

class MultiPCDObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            depth_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            sample_point_num: int=1024,
            voxel_model: nn.Module=None,
            pcd_model: nn.Module=None,
            use_voxel: bool=False,
            point_noise_augmentation: bool=False,
            jitter_noise_augmentation: bool=False,
            point_dropout_augmentation: bool=False,
            point_offset_augmentation: bool=False,
            point_colordrop_augmentation: bool=False,
            n_obs_steps: int=1,
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        pcd_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=mean, std=std)
                if imagenet_norm and shape[0] == 4: #rgbd
                    mean.append(0.)
                    std.append(1.)
                    depth_min, depth_max = 0.1, 1.1
                    this_normalizer = RGBDTransform(mean=mean, std=std, min_val=depth_min, max_val=depth_max, device=self.device)
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'depth':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(depth_model, dict):
                        # have provided model for each key
                        this_model = depth_model[key]
                    else:
                        assert isinstance(depth_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(depth_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
               
                this_transform = nn.Sequential(this_resizer, this_randomizer)
                key_transform_map[key] = this_transform
            elif type == 'pcd':
                pcd_keys.append(key)
                if use_voxel:
                    key_shape_map[key] = (4, 64, 64, 64) # override shape
                    this_crop_randomizer = nn.Identity()
                    if random_crop:
                        this_crop_randomizer = VoxelCropRandomizer(crop_depth=58, crop_height=58, crop_width=58)
                    this_transform = nn.Sequential(this_crop_randomizer)
                    key_transform_map[key] = this_transform
                    assert isinstance(voxel_model, nn.Module)     
                    if use_group_norm:
                        pcd_model = replace_submodules(
                            root_module=voxel_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm3d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = voxel_model
                    if pcd_model is not None:
                        del pcd_model
                else:
                    # configure re  size
                    input_shape = shape
                    # configure noiser (data augmentation)
                    n, c = input_shape
                    # this_point_sampler = nn.Identity()
                    this_point_sampler = FarthestPointSampling(sample_point_num)
                    this_dropout_noiser = nn.Identity()
                    if point_dropout_augmentation:
                        this_dropout_noiser = RandomPointDropout(p=0.3)
                    this_point_adder = nn.Identity()
                    if point_noise_augmentation:
                        this_point_adder = NoisePointAdder()
                    this_gaussian_noiser = nn.Identity()
                    if jitter_noise_augmentation:
                        this_gaussian_noiser = PointCloudJitter()
                    this_color_dropper = nn.Identity()
                    if point_colordrop_augmentation:
                        this_color_dropper = RandomColorDropout()
                    this_random_offsetter = nn.Identity()
                    if point_offset_augmentation:
                        this_random_offsetter = PointCloudOffsetter()
                    
                    this_transform = nn.Sequential(this_point_sampler, this_point_adder, this_dropout_noiser, this_gaussian_noiser, this_color_dropper, this_random_offsetter)
                    key_transform_map[key] = this_transform
            
                    self.pcd_key_type = shape_meta['obs'][key]['type']
                    assert isinstance(pcd_model, nn.Module)   
                    if use_group_norm:
                        pcd_model = replace_submodules(
                            root_module=pcd_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16, 
                                num_channels=x.num_features)
                        )
                        pcd_model = replace_submodules(
                            root_module=pcd_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm1d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=max(1, x.num_features // 16), 
                                num_channels=x.num_features)
                        )
                    if next(pcd_model.parameters()).is_cuda:
                        pcd_model = pcd_model.to('cuda') # for pointnext
                    key_model_map[key] = pcd_model
                    if voxel_model is not None:
                        del voxel_model

            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'nonlearn':
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
            # share model for next_gripper obs with gripper obs
            if 'next' in key:
                key_transform_map[key] = key_transform_map[key.replace('_next', '')]

        rgb_keys = sorted(rgb_keys)
        pcd_keys = sorted(pcd_keys)
        low_dim_keys = sorted(low_dim_keys)
        self.obs_keys = rgb_keys + pcd_keys + low_dim_keys

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.pcd_keys = pcd_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.use_voxel = use_voxel
        self.n_obs_steps = n_obs_steps

    def forward(self, obs_dict, batch_size):
        features = list()
        # process rgb input
        if self.share_rgb_model:
            raise NotImplementedError("Sharing RGB model is not supported yet")
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                # if single_modality and (key not in obs_keys):
                #     continue
                img = obs_dict[key]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature.reshape(batch_size, -1))
        
        for key in self.pcd_keys:
            # if single_modality and (key not in obs_keys):
            #     continue
            data = obs_dict[key]
            # assert data.shape[1:] == self.key_shape_map[key]
            data = self.key_transform_map[key](data)
            if self.use_voxel:
                # data = rearrange(data, "b c h w d -> b c d w h")
                # data = torch.flip(data, (2, 3))
                feature = self.key_model_map[key](data)
            else:
                B, num_points, device = data.shape[0], data.shape[1], data.device
                data = rearrange(data, 'b n c -> (b n) c')
                x, pos = data[:,3:], data[:,:3]
                batch = torch.arange(B).repeat_interleave(num_points).to(device)
                feature = self.key_model_map[key](x, pos, batch)
            
            features.append(feature.reshape(batch_size, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            # if single_modality and (key not in obs_keys):
            #         continue
            data = obs_dict[key]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data.reshape(batch_size, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        example_output = list()
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape']) if not (self.use_voxel and key=='pcd') else (4, 64, 64, 64)
            temporal_shape = self.n_obs_steps
            this_obs = torch.zeros(
                (batch_size*temporal_shape,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict, batch_size=1)
        output_shape = example_output.shape[1:]
        return output_shape

