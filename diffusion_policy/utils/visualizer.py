import torch
import numpy as np
import open3d as o3d
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix

def visualize_pcd(points: np.array, mode='color'):
    assert mode in ['color', 'xyz'], "Mode must be 'color' or 'xyz'"
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    
    if mode == 'color':
        assert points.shape[1] >= 6
        pcd.colors = o3d.utility.Vector3dVector(points[:,3:6])
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, origin])

def visualize_pcds(points: list, mode='color'):
    assert mode in ['color', 'xyz'], "Mode must be 'color' or 'xyz'"
    
    pcds = []
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
    
        if mode == 'color':
            assert p.shape[1] >= 6
            pcd.colors = o3d.utility.Vector3dVector(p[:,3:6])
        
        pcds.append(pcd)
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, origin])

def visualize_pcd_and_action(points: list, actions: list[np.array], mode='color'):
    assert mode in ['color', 'xyz'], "Mode must be 'color' or 'xyz'"
    assert actions.shape[1] == 10, "support only abs action."
    
    pcds = []
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
    
        if mode == 'color':
            assert p.shape[1] >= 6
            pcd.colors = o3d.utility.Vector3dVector(p[:,3:6])
        
        pcds.append(pcd)
    
    action_frames = list()
    for a in actions:
        xyz = a[:3]
        rot_6d = a[3:9]
        # Visualize the action, pcd, and origin
        a_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=xyz)
        # Apply rotation to the action frame
        R = rotation_6d_to_matrix(torch.from_numpy(rot_6d)).numpy()
        a_frame.rotate(R, center=xyz)
        action_frames.append(a_frame)
    

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, origin, *action_frames])

def visualize_pcd_and_pose(points: list, xyzxyzw):
    xyz, xyzw = xyzxyzw[:,:3], xyzxyzw[:,3:]
    assert xyz.shape[1] == 3 and xyzw.shape[1] == 4, "xyz and quat must be of shape (N, 3) and (N, 4) respectively"
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)

    pcds = []
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(p[:,3:6])
        pcds.append(pcd)
    
    pose_frames = list()
    for pos, rot in zip(xyz, wxyz):
        # Visualize the action, pcd, and origin
        p_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=pos)
        # Apply rotation to the action frame
        R = quaternion_to_matrix(torch.from_numpy(rot)).numpy()
        p_frame.rotate(R, center=pos)
        pose_frames.append(p_frame)
    

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, origin, *pose_frames])

def visualize_pcd_action_and_pose(points: list, actions: list[np.array], xyzxyzw):
    xyz, xyzw = xyzxyzw[:,:3], xyzxyzw[:,3:]
    assert xyz.shape[1] == 3 and xyzw.shape[1] == 4, "xyz and quat must be of shape (N, 3) and (N, 4) respectively"
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)

    pcds = []
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(p[:,3:6])
        pcds.append(pcd)
    
    pose_frames = list()
    for pos, rot in zip(xyz, wxyz):
        # Visualize the action, pcd, and origin
        pose_frame = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        pose_frame.paint_uniform_color([0, 0, 1])
        pose_frame.translate(pos)
        pose_frames.append(pose_frame)
    
    action_frames = list()
    for a in actions:
        xyz = a[:3]
        rot_6d = a[3:9]
        # Visualize the action, pcd, and origin
        a_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=xyz)
        # Apply rotation to the action frame
        R = rotation_6d_to_matrix(torch.from_numpy(rot_6d)).numpy()
        a_frame.rotate(R, center=xyz)
        action_frames.append(a_frame)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, origin, *pose_frames, *action_frames])

def visualize_voxel(np_voxel: np.array):
    """
    Visualize 3D voxels using Open3D.
    Args:
        np_voxels (np.array): A 4D numpy array of shape (C, D, H, W) representing the voxel grid.
                              C is the number of channels (e.g., color channels), and D, H, W are the dimensions of the voxel grid.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    indices = np.argwhere(np_voxel[0] != 0)
    colors = np_voxel[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], color=colors, marker='s')

    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    plt.show(block=False)

def visualize_voxel_and_xyz_goal(np_voxel: np.array, normalized_xyz_goal: np.array):
    """
    Visualize 3D voxels using Open3D.
    Args:
        np_voxels (np.array): A 4D numpy array of shape (C, D, H, W) representing the voxel grid.
                              C is the number of channels (e.g., color channels), and D, H, W are the dimensions of the voxel grid.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    indices = np.argwhere(np_voxel[0] != 0)
    colors = np_voxel[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], color=colors, marker='s')

    # plot the goal
    voxel_reso = np_voxel.shape[-1]
    xyz_goal = np.clip(normalized_xyz_goal, -1.0, 1.0) / 2 + 0.5# Ensure [0, 1] range

    idx_d = np.clip(xyz_goal[0] * voxel_reso, 0, voxel_reso - 1)
    idx_h = np.clip(xyz_goal[1] * voxel_reso, 0, voxel_reso - 1)
    idx_w = np.clip(xyz_goal[2] * voxel_reso, 0, voxel_reso - 1)
    print(f"goal: {xyz_goal}, idx: {idx_d, idx_h, idx_w}")
    
    ax.scatter(idx_d, idx_h, idx_w, color='red', marker='s')


    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, voxel_reso)
    ax.set_ylim(0, voxel_reso)
    ax.set_zlim(0, voxel_reso)
    plt.show(block=False)