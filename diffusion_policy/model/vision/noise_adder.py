import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
import diffusion_policy.model.common.tensor_util as tu
from torch_geometric.nn import fps
from einops import repeat

def randint(low, high=None, size=None):
    if high is None:
        high = low
        low = 0
    if size is None:
        size = low.shape if isinstance(low, torch.Tensor) else high.shape
    return torch.randint(2**63 - 1, size=size) % (high - low) + low

class GaussianNoiseAdder(nn.Module):
    """
    Randomly add high-frequency gaussian noise on images.
    """
    def __init__(
        self,
        input_shape,
        # start_channel=4, #default setting is add noise to the second RGBD of (B,8,H,W)
        mean = 0.0,
        sigma = 0.1,
        clip=True
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            start_channel (int): start adding noise from this channel
            mean (float): gaussian mean
            sigma (float): gaussian sigma
            clip (bool): if True, clip gaussian noise to -1 to 1
        """
        super().__init__()

        assert len(input_shape) == 3 # (C, H, W)

        self.input_shape = input_shape
        # self.start_channel = start_channel
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        return list(self.input_shape)

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B, C, H, W].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        device = inputs.device
        if self.training:
            # generate random crops
            gaussian = torch.normal(self.mean, self.sigma, size=(inputs.shape)).to(device)
            if self.clip:
                gaussian = torch.clip(gaussian, -1, 1)
            inputs = inputs + gaussian
        out = inputs
        return out
    
    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, start_channel={})".format(
            self.input_shape, self.start_channel)
        return msg
    
class PatchNoiseAdder(nn.Module):
    """
    Randomly add color patches into images.
    """
    def __init__(
        self,
        input_shape,
        consistent_color_per_obs=False
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            consistent_color_per_obs (bool): if true, assign same color for both obs and goal
        """
        super().__init__()

        assert len(input_shape) == 3 # (C, H, W)
        C, H, W = input_shape

        self.input_shape = input_shape
        self.patch_H, self.patch_W = int(np.floor(H/3)), int(np.floor(W/3))
        self.min_cut = 10
        self.consistent_color_per_obs = consistent_color_per_obs

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        return list(self.input_shape)

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random patches for each input in the batch, and then reshapes
        inputs to [B C H W].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        device = inputs.device
        B, C, H, W = inputs.shape
        if self.training:
            # generate random colors
            if self.consistent_color_per_obs:
                colors = torch.rand(size=(B,C//2)).repeat(1,2) * 2 - 1 # rescale to normalized image range
            else:
                colors = torch.rand(size=(B,C)) * 2 - 1 
            colors = colors.to(device)
            # generate random patch shape
            patch_h, patch_w = torch.randint(self.min_cut, self.patch_H, size=(B, 1, 1)), torch.randint(self.min_cut, self.patch_W, size=(B, 1, 1))
            # generate random location on image
            location_h, location_w = randint(H - patch_h, size=(B, 1, 1)), randint(W - patch_w, size=(B, 1, 1))
            # calculate patch mask (B, 1, H, W)
            cutouts = torch.empty((B, C, H, W), dtype=inputs.dtype, device=device)
            for i, (img, h11, w11, loc_h, loc_w) in enumerate(zip(inputs, patch_h, patch_w, location_h, location_w)):
                cut_img = img.clone()
                
                # add random box
                cut_img[:, loc_h:loc_h + h11, loc_w:loc_w + w11] = torch.tile(
                    colors[i].reshape(-1,1,1),                                                
                    (1,) + cut_img[:, loc_h:loc_h + h11, loc_w:loc_w + w11].shape[1:])
                
                cutouts[i] = cut_img
            out = cutouts
        else:
            out = inputs
        return out
    
    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, start_channel={})".format(
            self.input_shape, self.start_channel)
        return msg
    
class PointCloudJitter(nn.Module):
    """
    Randomly jitter points in a point cloud.
    """
    def __init__(self, sigma=0.02, clip=0.05, color_sigma=0.02, color_clip=0.05):
        """
        Args:
            sigma (float): standard deviation of the jitter
            clip (float): maximum absolute value of the jitter
        """
        super().__init__()
        self.sigma = sigma
        self.clip = clip
        self.color_sigma = color_sigma
        self.color_clip = color_clip

    def forward(self, batch_data):
        """
        Apply jittering to the input point cloud.
        
        Args:
            batch_data (torch.Tensor): input point cloud of shape (B, N, C)
        
        Returns:
            torch.Tensor: jittered point cloud of shape (B, N, C)
        """
        B, N, C = batch_data.shape
        assert self.clip > 0
        device = batch_data.device
        if self.training:
            if C == 3:  # Spatial data only
                jittered_data = torch.clamp(
                    self.sigma * torch.randn(B, N, C, device=device), 
                    -self.clip, self.clip
                )
            elif C == 6:  # Spatial data + RGB
                # Jitter for coordinates (x, y, z)
                spatial_jitter = torch.clamp(
                    torch.normal(mean=0.0, std=self.sigma, size=(B, N, 3), device=device),
                    -self.clip, self.clip
                )
                # Jitter for colors (R, G, B)
                color_jitter = torch.clamp(
                    torch.normal(mean=0.0, std=self.color_sigma, size=(B, N, 3), device=device),
                    -self.color_clip, self.color_clip
                )
                jittered_data = torch.cat([spatial_jitter, color_jitter], dim=-1)
                
            batch_data += jittered_data
        return batch_data

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(sigma={}, clip={})".format(self.sigma, self.clip)
        return msg
    
class RandomPointDropout(nn.Module):
    def __init__(self, p=0.3):
        """
        Randomly drops points in a point cloud and replaces them with existing points.

        Args:
            p (float): Probability of dropping each point.
        """
        super().__init__()
        self.p = p

    def forward(self, points):
        """
        Args:
            points (Tensor): Point cloud tensor with shape [N, 3] or [N, C].

        Returns:
            Tensor: Augmented point cloud tensor with the same shape as input.
        """
        if self.training:
            B, N, C = points.shape
            device = points.device

            p = np.random.random() * self.p

            # Create dropout mask
            dropout_mask = torch.rand(B, N, device=device) < p

            # Ensure at least one point remains per cloud
            no_keep = dropout_mask.sum(dim=1) == N
            dropout_mask[no_keep, torch.randint(0, N, (no_keep.sum(),), device=device)] = False

            # Get indices of valid points per batch
            valid_indices = (~dropout_mask).float()

            # Compute probabilities to sample replacement points
            probs = valid_indices / valid_indices.sum(dim=1, keepdim=True)

            # Sample replacement indices
            replacement_indices = torch.multinomial(probs, N, replacement=True)

            # Use batch indexing to replace dropped points
            batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, N)
            points_aug = points.clone()
            points_aug[dropout_mask] = points[batch_idx[dropout_mask], replacement_indices[dropout_mask]]

            return points_aug
        
        else:
            return points

class RandomColorDropout(nn.Module):
    def __init__(self, p=0.2):
        """
        Randomly drops RGB in a point cloud and replaces them with [0,0,0].

        Args:
            p (float): Probability of dropping each point.
        """
        super().__init__()
        self.p = p

    def forward(self, points):
        """
        Args:
            points (Tensor): Point cloud tensor with shape [N, 3] or [N, C].

        Returns:
            Tensor: Augmented point cloud tensor with the same shape as input.
        """
        if self.training:
            if points.dim() == 3:
                B, N, C = points.shape
                mask = torch.rand(B, N, device=points.device) < self.p
                points[mask] = 0
            elif points.dim() == 2:
                N, C = points.shape
                mask = torch.rand(N, device=points.device) < self.p
                points[mask] = 0
        return points
        
class FarthestPointSampling(nn.Module):
    def __init__(self, num_points=512):
        """
        Samples points from a point cloud using farthest point sampling.

        Args:
            num_points (int): Number of points to sample.
        """
        super().__init__()
        self.num_points = num_points

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): Point cloud tensor with shape [B, N, C]

        Returns:
            torch.Tensor: Sampled point cloud with shape [B, num_points, C]
        """
        B, N, C = points.shape
        if N <= self.num_points:
            return points
        device = points.device
        
        # Calculate ratio based on num_points
        ratio = self.num_points / N
        
        # Create batch tensor (indicating which batch each point belongs to)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        
        # Reshape points for fps function
        pos = points.reshape(B*N, C)
        
        # Perform farthest point sampling
        idx = fps(pos, ratio=ratio, batch=batch)
        
        # Gather the sampled points
        result = pos[idx].reshape(B, self.num_points, C)
        
        return result
    
class NoisePointAdder(nn.Module):
    def __init__(self, range=1.):
        """
        Args:
            range (float): The range within which noise points are added in each axis (x, y, z). Default is 1 because assume noralized pcd.
            num_noise_points (int): Number of noise points to add.
        """
        super().__init__()
        self.range = range

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): Input point cloud tensor of shape (B, N, C) where B is batch size, N is the number of points, and C is the number of channels (typically 3 for x, y, z).
        Returns:
            torch.Tensor: Augmented point cloud with added noise points.
        """
        if self.training:
            B, N, C = points.shape
            device = points.device

            num_noise_points = int(N * 0.05 * np.random.random())

            # handle num_noise_points == 0
            num_noise_points = num_noise_points if num_noise_points > 0 else 1
            
            # Generate random noise points within the specified range
            noise_points = (torch.rand(B, num_noise_points, C, device=device) - 0.5) * 2 * self.range
            
            # Concatenate the noise points to the original point cloud
            points = torch.cat([points, noise_points], dim=1)
            
        return points
    
class PointCloudOffsetter(nn.Module):
    def __init__(self, range=0.1):
        """
        Args:
            range (float): The offset range within which noise points are added in each pcd.
        """
        super().__init__()
        self.range = range

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): Input point cloud tensor of shape (B, N, C) where B is batch size, N is the number of points, and C is the number of channels (typically 3 for x, y, z).
        Returns:
            torch.Tensor: Augmented point cloud with offsets.
        """
        if self.training:
            B = points.shape[0]
            device = points.device

            n_offsets = torch.rand(B, device=device) * self.range
            points[:, :, :3] += n_offsets.view(-1, 1, 1)

        return points
    
