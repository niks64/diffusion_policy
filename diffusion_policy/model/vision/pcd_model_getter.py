# pointnet2 implementation from torch_geometry (https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py)
# pointTransformer is from torch_geometry (https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py)
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, global_mean_pool, knn, knn_graph, PointTransformerConv
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

from einops import rearrange
from pointnext import PointNext, pointnext_s

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class PointSpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer for 3D Point Clouds.

    Based on the concept of Spatial Softmax for 2D images, adapted for 3D point clouds.
    Instead of a 2D spatial probability distribution, we create a 1D probability
    distribution along each spatial dimension (x, y, z) based on point features.
    The expected value along each dimension becomes a coordinate of the keypoint.
    """
    def __init__(
        self,
        input_channels,
        num_kp=32,
        temperature=1.0,
        noise_std=0.0,
    ):
        """
        Args:
            input_channels (int): Number of input features per point (C).
            num_kp (int): Number of keypoints to predict.
            temperature (float): Temperature term for the softmax.
            noise_std (float): Add random noise to the predicted keypoints during training.
        """
        super(PointSpatialSoftmax, self).__init__()
        self.input_channels = input_channels
        self.num_kp = num_kp
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
        self.noise_std = noise_std

        # Linear layer to map input features to keypoint weights
        self.fc = torch.nn.Linear(input_channels, num_kp * 3) # Output weights for x, y, z for each keypoint

        self.kps = None


    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(input_channels={}, num_kp={}, temperature={}, noise={})'.format(
            self.input_channels, self.num_kp, self.temperature.item(), self.noise_std)

    def forward(self, points, features):
        """
        Forward pass through the spatial softmax layer for point clouds.

        Args:
            points (torch.Tensor): Input point cloud coordinates of shape [B, N, 3],
                                   where B is batch size and N is the number of points.
            features (torch.Tensor): Input features per point of shape [B, N, C],
                                     where C is the number of input channels.

        Returns:
            torch.Tensor: Predicted keypoints of shape [B, K, 3], where K is the number of keypoints.
        """
        batch_size, num_points, _ = points.shape

        # Predict weights for each keypoint and each dimension (x, y, z)
        weights = self.fc(features)  # [B, N, num_kp * 3]
        weights = weights.view(batch_size, num_points, self.num_kp, 3) # [B, N, K, 3]

        # Apply softmax along the point dimension for each keypoint and each dimension
        attention_x = F.softmax(weights[:, :, :, 0] / self.temperature, dim=1) # [B, N, K]
        attention_y = F.softmax(weights[:, :, :, 1] / self.temperature, dim=1) # [B, N, K]
        attention_z = F.softmax(weights[:, :, :, 2] / self.temperature, dim=1) # [B, N, K]

        # Compute the expected value (weighted sum) of the coordinates
        expected_x = torch.sum(points[:, :, 0].unsqueeze(-1) * attention_x, dim=1) # [B, K]
        expected_y = torch.sum(points[:, :, 1].unsqueeze(-1) * attention_y, dim=1) # [B, K]
        expected_z = torch.sum(points[:, :, 2].unsqueeze(-1) * attention_z, dim=1) # [B, K]

        # Stack the expected coordinates to get the keypoints
        keypoints = torch.stack([expected_x, expected_y, expected_z], dim=-1) # [B, K, 3]

        if self.training:
            noise = torch.randn_like(keypoints) * self.noise_std
            keypoints += noise

        self.kps = keypoints.detach()
        return keypoints
    
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.ratio = npoint / 1024.0
        # Removed previous conv_blocks and bn_blocks and replaced with torch_geometry based convs.
        self.convs = nn.ModuleList()
        for mlp in mlp_list:
            self.convs.append(PointNetConv(MLP(mlp, norm='LayerNorm'), add_self_loops=False))
    
    def forward(self, x, pos, batch=None):
        
        idx = fps(pos, batch, ratio=self.ratio)  # [B, npoint]
        x_dst = None if x is None else x[idx]
        pos_dst = pos[idx]

        out_list = []
        for i, r in enumerate(self.radius_list):
            if batch is not None:
                batch_idx = batch[idx]
            else:
                batch_idx = None
            row, col = radius(pos, pos[idx], r, batch, batch_idx,
                                max_num_neighbors=32)
            edge_index = torch.stack([col, row], dim=0)
            
            out = self.convs[i]((x, x_dst), (pos, pos_dst), edge_index)
            out_list.append(out)

        x = torch.cat(out_list, dim=-1) 
        pos = pos[idx]
        if batch is not None:
            batch = batch[idx]
        return x, pos, batch
    
# Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
class PointNet2MSG(nn.Module):
    def __init__(self, pcd_feature_dim=3, num_classes=128):
        super().__init__()

        sa1_outdim = [64, 128, 128]
        sa2_indim = np.sum(sa1_outdim) + 3
        sa2_outdim = [128, 256, 256]
        sa3_indim = np.sum(sa2_outdim) + 3
        sa3_outdim = 1024

        self.sa1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128], mlp_list=[[pcd_feature_dim+3, 32, 32, sa1_outdim[0]], [pcd_feature_dim+3, 64, sa1_outdim[1]], [pcd_feature_dim+3, 96, sa1_outdim[2]]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128], mlp_list=[[sa2_indim, 64, 64, sa2_outdim[0]], [sa2_indim, 128, sa2_outdim[1]], [sa2_indim, 128, sa2_outdim[2]]])
        self.sa3 = GlobalSAModule(MLP([sa3_indim, 256, 512, sa3_outdim], norm='LayerNorm'))
        self.mlp = MLP([sa3_outdim, 512, 256, num_classes], norm='LayerNorm', dropout=0.5)

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x, pos, batch = self.sa3(*sa2_out)
        return self.mlp(x)

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2SSG(torch.nn.Module):
    def __init__(self, feature_dim, num_classes=128, model_size='small'):
        super().__init__()

        assert model_size in ['small', 'medium']
        if model_size == 'medium':
            dims = [64,128,256,512,1024]
        elif model_size == 'small':
            dims = [32,64,128,256,512]
        else:
            raise ValueError(f"Unknown model size {model_size}")

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.25, 0.05, MLP([feature_dim+3, dims[0], dims[0], dims[1]], norm='LayerNorm'))
        self.sa2_module = SAModule(0.25, 0.1, MLP([dims[1] + 3, dims[1], dims[1], dims[2]], norm='LayerNorm'))
        self.sa3_module = GlobalSAModule(MLP([dims[2] + 3, dims[2], dims[3], dims[4]], norm='LayerNorm'))

        # self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.2, norm=None)
        self.mlp = MLP([dims[4], num_classes], norm=None)

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x)

class PointNet(torch.nn.Module):
    """Encoder for Pointcloud from DP3
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels+3, block_channel[0]),
            torch.nn.GroupNorm(block_channel[0]//16, block_channel[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(block_channel[0], block_channel[1]),
            torch.nn.GroupNorm(block_channel[1]//16, block_channel[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(block_channel[1], block_channel[2]),
            torch.nn.GroupNorm(block_channel[2]//16, block_channel[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        self.final_projection = torch.nn.Linear(block_channel[-1], out_channels)
        

    def forward(self, x, pos, batch=None):
        # idx = fps(pos, batch, ratio=0.8)
        x = torch.cat([pos, x], axis=1)
        x = self.mlp(x)
        if batch is not None:
            x = global_max_pool(x, batch)
        else:
            x = torch.max(x, dim=0, keepdim=True)[0]
        x = self.final_projection(x)
        return x
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, dropout=0.2,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """
    def __init__(self, in_channels, out_channels, ratio=0.5, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        # self.feature_mlp = MLP([in_channels+3, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 128, out_channels], dropout=0.2, norm=None)

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        # x = self.feature_mlp(x) + self.pos_mlp(pos)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return out

class PointNextModel(nn.Module):

    def __init__(self, in_channels=6, out_channels=40, dropout=0., norm_method='layernorm', use_softmax=False):
        super().__init__()

        backbone_outdim = 512
        if use_softmax:
            num_kp = 32
            self.softmax = PointSpatialSoftmax(backbone_outdim, num_kp=num_kp, temperature=1.0, noise_std=0.0)
            mlp_indim = num_kp * 3
        else:
            mlp_indim = backbone_outdim

        encoder = pointnext_s(in_dim=in_channels, strides=[4, 4, 2, 2])
        self.backbone = PointNext(backbone_outdim, encoder=encoder)

        norm_method = norm_method.lower()
        self.norm_method = norm_method
        if norm_method == 'batchnorm':
            self.norm = nn.BatchNorm1d(backbone_outdim)
            mlp_norm = 'BatchNorm'
        elif norm_method == 'layernorm':
            self.norm = nn.LayerNorm(backbone_outdim)
            mlp_norm = 'LayerNorm'
        elif norm_method == 'none':
            self.norm = nn.Identity()
            mlp_norm = None
        else:
            raise ValueError("Unsupported norm_method: choose 'batchnorm', 'layernorm', or 'none'")

        self.relu = nn.ReLU()
        self.mlp = MLP([mlp_indim, out_channels],
                       norm='none',
                       dropout=dropout)
        
        self.use_softmax = use_softmax

    def forward(self, x, pos, batch=None):
        if batch is not None:
            batch_size = batch.max().item() + 1
            x = x.view(batch_size, -1, x.size(-1))
            pos = pos.view(batch_size, -1, pos.size(-1))
        else:
            batch_size, num_points, feat_dim = x.shape()
        
        cpu = False
        if not x.is_cuda: # if input tensor is on cpu, move to cuda due to pointnext's cuda operator
            cpu = True
            x = x.to('cuda')
            pos = pos.to('cuda')

        x = x.transpose(1, 2)
        pos = pos.transpose(1, 2)
        # x = torch.cat([pos, x], dim=1)

        out, pos = self.backbone(x, pos)
        
        if self.use_softmax:
            out = self.softmax(pos.transpose(1, 2), x.transpose(1, 2))
            out = out.view(batch_size, -1)
            out = self.mlp(out)
        else:
            out = out.mean(dim=-1)
            # out = self.relu(out)
            out = self.mlp(out)
        
        if cpu:
            out = out.to('cpu')
            pos = pos.to('cpu')
        return out

class PointNextModelLocal(nn.Module):

    def __init__(self, in_channels=6, out_channels=40, dropout=0., norm_method='batchnorm'):
        super().__init__()

        self.num_points = 2048
        backbone_outdim = 64
        s = 4
        self.num_output_points = self.num_points // np.power(s, 4) 
        encoder = pointnext_s(in_dim=in_channels, strides=[s, s, s, s])

        self.backbone = PointNext(backbone_outdim, encoder=encoder)

        norm_method = norm_method.lower()
        self.norm_method = norm_method
        if norm_method == 'batchnorm':
            self.norm = nn.BatchNorm1d(backbone_outdim)
            mlp_norm = 'BatchNorm'
        elif norm_method == 'layernorm':
            self.norm = nn.LayerNorm(backbone_outdim)
            mlp_norm = 'LayerNorm'
        elif norm_method == 'none':
            self.norm = nn.Identity()
            mlp_norm = None
        else:
            raise ValueError("Unsupported norm_method: choose 'batchnorm', 'layernorm', or 'none'")

        self.relu = nn.ReLU()
        self.mlp = MLP([backbone_outdim * self.num_output_points, out_channels],
                       norm='none',
                       dropout=dropout)

    def forward(self, x, pos, batch=None):
        if batch is not None:
            batch_size = batch.max().item() + 1
            x = x.view(batch_size, -1, x.size(-1))
            pos = pos.view(batch_size, -1, pos.size(-1))
        else:
            batch_size, num_points, feat_dim = x.shape()
        if not x.is_cuda:
            x = x.to('cuda')
            pos = pos.to('cuda')

        x = x.transpose(1, 2)
        pos = pos.transpose(1, 2)
        x = torch.cat([pos, x], dim=1)

        out = self.backbone(x, pos)
        out = self.norm(out)
        out = out.view(batch_size, -1)
        out = self.relu(out)
        out = self.mlp(out)
        return out
    
def get_pcd_net(name, output_dim):
    pcd_feature_dim = 3 # r, g, b, 
    if name == "pointnet":
        model = PointNet(pcd_feature_dim, out_channels=output_dim)
    elif name == "pointnet2-ssg":
        model = PointNet2SSG(pcd_feature_dim, num_classes=output_dim)
    elif name == "pointnet2-msg":
        model = PointNet2MSG(pcd_feature_dim, num_classes=output_dim)
    elif name == "point-transformer":
        model = PointTransformer(pcd_feature_dim, output_dim, dim_model=[32, 128, 256], k=16)
    elif name == "pointnext":
        assert torch.cuda.is_available(), "PointNext requires CUDA"
        model = PointNextModel(pcd_feature_dim, output_dim).to('cuda')
    elif name == "pointnext-local":
        assert torch.cuda.is_available(), "PointNext-local requires CUDA"
        model = PointNextModelLocal(pcd_feature_dim + 3, output_dim).to('cuda')
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    N=1024
    B = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device='cpu'
    # Example colored point cloud with shape (N, 6)
    obs = torch.rand((B, N, 6))
    obs = rearrange(obs, 'b n c -> (b n) c')

    # Split the point cloud into pos and x
    x, pos = obs[:,3:].to(device), obs[:,:3].to(device)
    # Example batch vector
    batch = torch.arange(B).repeat_interleave(N).to(device)

    # Create an instance of SAModule
    net = get_pcd_net('pointnext', 128).to(device)
    print(count_parameters(net))

    # Forward pass
    out = net(x, pos, batch)