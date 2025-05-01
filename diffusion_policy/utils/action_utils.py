import copy
import torch
import numpy as np
import open3d as o3d
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_random_se3_transform(trans_sigma=0.01, theta_x_sigma=10, theta_y_sigma=10, theta_z_sigma=180):
    """Generate a random SE3 transformation."""

    theta_x = np.random.uniform(-theta_x_sigma, theta_x_sigma)
    theta_y = np.random.uniform(-theta_y_sigma, theta_y_sigma)
    theta_z = np.random.uniform(-theta_z_sigma, theta_z_sigma)
    transform = np.eye(4)
    # NOTE: this 'zyx' has noting to do with the robot action order
    rotation = R.from_euler('ZYX', [theta_z, theta_y, theta_x], degrees=True).as_matrix()
    translation = np.clip(np.random.normal(0, trans_sigma, size=(3,)), -trans_sigma, trans_sigma)

    transform[:3, 3] = translation
    transform[:3,:3] = rotation
    return transform, (theta_x, theta_y, theta_z, translation)


def apply_se3_augmentation_to_abs_action(aug_transform, action):
    """
    Apply SE3 transformation to a pose.
    action: a (N, 10) array where 10 contains (x,y,z,6D_rot,gripper) where 6D_rot is the 6d rotation repr [r1,r2,r3,r4,r5,r6]
    """
    action_rot_mat = np.eye(4)
    
    action_transformed = []
    for a in action:
        # transform xy
        action_mat = np.eye(4)
        action_rot_mat = rotation_6d_to_matrix(torch.from_numpy(a[3:9])).numpy()
        action_mat[:3, 3] = a[:3]
        action_mat[:3,:3] = action_rot_mat

        transformed_pose =  aug_transform @ action_mat
        xyz = transformed_pose[:3, 3] 
        rot6d = matrix_to_rotation_6d(torch.from_numpy(transformed_pose[:3, :3])).numpy()

        a_transformed = np.hstack((xyz, rot6d, a[-1]))
        action_transformed.append(a_transformed)

    return np.stack(action_transformed)

def apply_se3_augmentation_to_lowdim(aug_transform, low_dim, key):
    """
    low dim: a (N, 3) array where 3 contains (x,y,z) when key is 'pos'
    low dim: a (N, 4) array where 4 contains (x,y,z,w) when key is 'quat'
    """
    if "eef_quat" in key:
        # low_dim: (N,4) array of quaternions
        # Use batch conversion to rotation matrices
        rot_mats = R.from_quat(low_dim).as_matrix()  # shape (N,3,3)
        # Get rotation part of augmentation transform
        aug_rot = aug_transform[:3, :3]  # shape (3,3)
        # Apply the augmentation: new_rot = aug_rot @ rot_mat for each in batch
        new_rot_mats = np.matmul(aug_rot, rot_mats)  # shape (N,3,3)
        return R.from_matrix(new_rot_mats).as_quat()
    elif "eef_pos" in key:  
        # low_dim: (N,3) array of positions
        # Convert to homogeneous coordinates
        ones = np.ones((low_dim.shape[0], 1))
        low_dim_hom = np.concatenate([low_dim, ones], axis=1)  # shape (N,4)
        # Apply the augmentation transform to each point
        transformed = (aug_transform @ low_dim_hom.T).T  # shape (N,4)
        return transformed[:, :3]
    elif "gripper_qpos" in key:
        return low_dim
    else:
        raise NotImplementedError(f"low dim key {key} not implemented")
    
def apply_se3_pcd_transform(points, transform):
    """
    Apply SE3 transformation to a set of points.
    points: a (N, 6) array where N is number of points and the last dimension is (x,y,z,r,g,b)
    """
    points_homogeneous = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    transformed_points = (transform @ points_homogeneous.T).T
    points[:, :3] = transformed_points[:, :3]
    return points

def localize_abs_action(action_abs, current_ee_xyzqxyzw, is_goal_empty, local_type):
    """
    Transform action_abs in world coordinate to action_abs in local gripper coordinate
    action_abs: a (H, 10) array 
    current_ee_xyzqxyzw: (1, 7) array
    is_goal_empty: (1, 1)
    """
    assert local_type in ['xyz', 'se3'] 
    action_step, dim = action_abs.shape
    current_ee_xyzqxyzw = np.repeat(current_ee_xyzqxyzw, action_step, axis=0)
    is_goal_empty = np.repeat(is_goal_empty, action_step, axis=0)

    local_action_abs = copy.deepcopy(action_abs)
    
    if local_type == 'xyz':
        # localize eef pose
        local_action_abs[:, :3] = action_abs[:, :3] - current_ee_xyzqxyzw[:,:3]
    elif local_type == 'se3':
        action_abs = action_abs.reshape(-1, dim)
        global_action_mat = torch.eye(4)[None, ...].repeat_interleave(action_step, dim=0)
        global_action_mat[:,:3,:3] = rotation_6d_to_matrix(torch.from_numpy(action_abs[:,3:9]))
        global_action_mat[:,:3,3] = torch.from_numpy(action_abs[:,:3])

        eef_pose_mat = torch.eye(4)[None, ...].repeat_interleave(action_step, dim=0)
        eef_pose_mat[:,:3,:3] = quaternion_to_matrix(torch.from_numpy(current_ee_xyzqxyzw[:,3:]))
        eef_pose_mat[:,:3,3] =  torch.from_numpy(current_ee_xyzqxyzw[:, :3])

        # correction_mat = torch.eye(4)[None,...]
        # correction_mat[:,:3,:3] = torch.from_numpy(R.from_euler('XYZ', [90, 0, 0], degrees=True).as_matrix())
        # global_action_mat = torch.matmul(global_action_mat, correction_mat)
        local_eef_pose_mat = torch.matmul(torch.linalg.inv(eef_pose_mat), global_action_mat)

        # local_eef_pose_mat = torch.matmul(correction_mat, local_eef_pose_mat)
        
        local_action_abs[:, :3] = local_eef_pose_mat[:, :3, 3]
        local_action_abs[:, 3:9] = matrix_to_rotation_6d(local_eef_pose_mat[:,:3,:3])
    
    local_action_abs[is_goal_empty] = np.zeros_like(action_abs[0])
    return local_action_abs.reshape(action_step, dim)
    
    # local_a_list = list()
    # for a, ee, is_empt in zip(action_abs, current_ee_xyzqxyzw, is_goal_empty):
    #     if is_empt:
    #         local_a = get_zero_actions(a.shape)
    #     else:
    #         if local_type == 'se3':
    #             ee_mat = np.eye(4)
    #             ee_mat[:3,:3] = R.from_quat(ee[3:7]).as_matrix()
    #             ee_mat[:3,3] = ee[:3]

    #             # correction_mat = np.eye(4)
    #             # correction_mat[:3,:3] = R.from_euler('ZYX', [90, 0, 0], degrees=True).as_matrix()

    #             a_mat = np.eye(4)
    #             a_mat[:3,:3] = rotation_6d_to_matrix(torch.from_numpy(a[3:9])).numpy()
    #             a_mat[:3,3] = a[:3]
                
    #             # local_a_mat = a_mat @ np.linalg.inv(ee_mat)
    #             # local_a_mat = correction_mat @ np.linalg.inv(ee_mat) @ a_mat
    #             local_a_mat = np.linalg.inv(ee_mat) @ a_mat
    #             local_a = np.zeros(10)
    #             local_a[:3] = local_a_mat[:3,3]
    #             local_a[3:9] = matrix_to_rotation_6d(torch.from_numpy(local_a_mat[:3,:3])).numpy()

    #         elif local_type == 'xyz':
    #             local_a = copy.deepcopy(a)
    #             local_a[:3] = a[:3] - ee[:3]
    #     local_a_list.append(local_a)
    # local_action_abs = np.stack(local_a_list).reshape(batch_size, action_step, dim)

    # return local_action_abs

def globalize_abs_action(action_abs, current_ee_xyzqxyzw, is_goal_empty, local_type):
    """
    THis is the reversed function of localize_abs_action
    action_abs: a (N, H, 10) array 
    current_ee_xyzqxyzw: (N, 7) array
    is_goal_empty: (N, 1)
    """
    assert local_type in ['xyz', 'se3'] 
    batch_size, action_step, dim = action_abs.shape
    is_goal_empty = np.repeat(is_goal_empty, action_step, axis=1).reshape(batch_size*action_step)

    global_action_abs = copy.deepcopy(action_abs)
    
    if local_type == 'xyz':
        # localize eef pose
        global_action_abs[:, :, :3] = action_abs[:, :, :3] + current_ee_xyzqxyzw[:,:3]
    elif local_type == 'se3':
        action_abs = action_abs.reshape(-1, dim)
        local_action_mat = torch.eye(4)[None, ...].repeat_interleave(batch_size*action_step, dim=0)
        local_action_mat[:,:3,:3] = rotation_6d_to_matrix(torch.from_numpy(action_abs[:,3:9]))
        local_action_mat[:,:3,3] = torch.from_numpy(action_abs[:,:3])

        eef_pose_mat = torch.eye(4)[None, ...].repeat_interleave(batch_size, dim=0)
        eef_pose_mat[:,:3,:3] = quaternion_to_matrix(torch.from_numpy(current_ee_xyzqxyzw[:,3:]))
        eef_pose_mat[:,:3,3] =  torch.from_numpy(current_ee_xyzqxyzw[:, :3])
        eef_pose_mat = eef_pose_mat.repeat_interleave(action_step, dim=0)

        # local_eef_pose_mat = np.linalg.inv(current_eef_pose_mat) @ eef_pose_mat
        correction_mat = torch.eye(4)[None,...]
        correction_mat[:,:3,:3] = torch.from_numpy(R.from_euler('XYZ', [-180, 0, 0], degrees=True).as_matrix())
        local_action_mat = torch.matmul(correction_mat, local_action_mat)
        global_action_mat = torch.matmul(eef_pose_mat, local_action_mat)
        
        global_action_abs = global_action_abs.reshape(-1, dim)
        global_action_abs[:, :3] = global_action_mat[:, :3, 3]
        global_action_abs[:, 3:9] = matrix_to_rotation_6d(global_action_mat[:,:3,:3])
    
    global_action_abs[is_goal_empty] = np.zeros_like(action_abs[0])
    return global_action_abs.reshape(batch_size, action_step, dim)

    # action_abs = action_abs.reshape(-1, 10)
    # is_goal_empty = is_goal_empty.reshape(-1,1)
    # current_ee_xyzqxyzw = current_ee_xyzqxyzw.reshape(-1, 7)
    # global_a_list = list()
    
    # for a, ee, is_empt in zip(action_abs, current_ee_xyzqxyzw, is_goal_empty):
    #     if is_empt:
    #         global_a = get_zero_actions(a.shape)
    #     else:
    #         if local_type == 'se3':
    #             ee_mat = np.eye(4)
    #             ee_mat[:3,:3] = ee_mat[:3,:3] = R.from_quat(ee[3:7]).as_matrix()
    #             ee_mat[:3,3] = ee[:3]

    #             correction_mat = np.eye(4)
    #             correction_mat[:3,:3] = R.from_euler('ZYX', [-90, 0, 0], degrees=True).as_matrix()
                
    #             a_mat = np.eye(4)
    #             a_mat[:3,:3] = rotation_6d_to_matrix(torch.from_numpy(a[3:9])).numpy()
    #             a_mat[:3,3] = a[:3]

    #             # global_a_mat = a_mat @ ee_mat
    #             global_a_mat = ee_mat @ correction_mat @ a_mat
    #             global_a = np.zeros(10)
    #             global_a[:3] = global_a_mat[:3,3]
    #             global_a[3:9] = matrix_to_rotation_6d(torch.from_numpy(global_a_mat[:3,:3])).numpy()
    #         elif local_type == 'xyz':
    #             global_a = copy.deepcopy(a)
    #             global_a[:3] = a[:3] + ee[:3]
    #     global_a_list.append(global_a)
    # global_action_abs = np.stack(global_a_list).reshape(batch_size, action_step, dim)

    # return global_action_abs

def apply_se3_augmentation_to_pcd(transform, gripper_centered_pcd):
    """Apply SE3 augmentation to point cloud and action."""

    pcd_transformed = list()
    for p in gripper_centered_pcd:
        p_transformed = apply_se3_pcd_transform(p, transform)
        pcd_transformed.append(p_transformed)
    
    return np.stack(pcd_transformed)

def pcd_se3_augmentation(gripper_centered_pcd):
    """Apply SE3 augmentation to point cloud and action."""
    p = copy.copy(gripper_centered_pcd)

    transform, params = get_random_se3_transform(trans_sigma=0.02,theta_x_sigma=5,theta_y_sigma=5,theta_z_sigma=5)
    p_transformed = apply_se3_pcd_transform(p, transform)
    
    return p_transformed

def add_noise_to_quat(quat, noise_sigma=5, noise_clip=10):
    rot_mat = R.from_quat(quat).as_matrix()

    theta_x = np.clip(np.random.normal(0, noise_sigma), -noise_clip, noise_clip)
    theta_y = np.clip(np.random.normal(0, noise_sigma), -noise_clip, noise_clip)
    theta_z = np.clip(np.random.normal(0, noise_sigma), -noise_clip, noise_clip)
    # NOTE: this 'zyx' has noting to do with the robot action order
    noise_mat = R.from_euler('ZYX', [theta_z, theta_y, theta_x], degrees=True).as_matrix()

    return R.from_matrix(rot_mat @ noise_mat).as_quat()

def add_noise_to_goal(goal, goal_type):
    if 'pcd' in goal_type:
        return pcd_se3_augmentation(goal)
    elif 'pos' in goal_type:
        noise_sigma, noise_clip = 0.03, 0.05 # in meter
        return goal + np.clip(np.random.normal(0, noise_sigma, size=goal.shape), -noise_clip, noise_clip)
    elif 'quat' in goal_type:
        noise_sigma, noise_clip = 5, 10 # in degree
        return add_noise_to_quat(goal, noise_sigma, noise_clip)
    else:
        raise NotImplementedError(f"goal type {goal_type} not implemented")
    
def get_zero_actions(actions_shape):
    """
    action_abs: a (10, ) array 
    """
    action_dim = actions_shape
    assert action_dim == (10,), "Support only 10d abs actions"
    zero_actions = np.zeros(actions_shape)

    # define open gripper as zero action
    zero_actions[-1] = -1

    # 0 rotation
    zero_actions[3:9] = np.eye(3)[:2].reshape(-1)

    return zero_actions

def localize_ee_pose(eef_poses, current_eef_pose, local_type):
    assert len(eef_poses.shape) == 3, "eef_poses must be of shape (N, H, 7)"
    assert len(current_eef_pose.shape) == 2, "eef_poses must be of shape (N, 7)"
    batch_size, n_step, ndim = eef_poses.shape
    eef_poses = eef_poses.reshape(batch_size*n_step, -1)
    if local_type == 'xyz':
        # localize eef pose
        eef_poses[:, :, :3] = eef_poses[:, :, :3] - current_eef_pose[:,:3]
    elif local_type == 'se3':
        # localize eef pose
        current_eef_pose_mat = torch.eye(4)[None, ...].repeat_interleave(batch_size, dim=0)
        current_eef_pose_mat[:,:3,:3] = quaternion_to_matrix(torch.from_numpy(current_eef_pose[:,3:7]))
        current_eef_pose_mat[:,:3,3] = torch.from_numpy(current_eef_pose[:,:3])
        current_eef_pose_mat = current_eef_pose_mat.repeat_interleave(n_step, dim=0)

        eef_pose_mat = torch.eye(4)[None, ...].repeat_interleave(batch_size*n_step, dim=0)
        eef_pose_mat[:,:3,:3] = rotation_6d_to_matrix(torch.from_numpy(eef_poses[:,3:]))
        eef_pose_mat[:,:3,3] =  torch.from_numpy(eef_poses[:, :3])

        # local_eef_pose_mat = np.linalg.inv(current_eef_pose_mat) @ eef_pose_mat
        local_eef_pose_mat = torch.matmul(torch.linalg.inv(current_eef_pose_mat), eef_pose_mat)
        eef_poses[:, :3] = local_eef_pose_mat[:, :3, 3]
        eef_poses[:, 3:] = matrix_to_quaternion(local_eef_pose_mat[:,:3,:3])

    return eef_poses.reshape(batch_size, n_step, ndim)

def localize_lowdim(lowdim, current_eef_pose, key, local_type):
    assert len(lowdim.shape) in [3], "eef_poses must be of shape (N, H, n_dim)"
    assert len(current_eef_pose.shape) == 2, "eef_poses must be of shape (N, 7)"
    assert local_type in ['xyz', 'se3'], "local type must be xyz"
    assert key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'], "key must be robot0_eef_pos, robot0_eef_quat or gripper_qpos"


    if local_type == 'se3':
        if 'quat' in key:
            batch_size, n_step, ndim = lowdim.shape[0], lowdim.shape[1], lowdim.shape[2]
            quat = lowdim.reshape(-1, ndim)
            quat = quaternion_to_matrix(torch.from_numpy(quat))
            current_rot_mat = torch.eye(3)[None, ...].repeat_interleave(batch_size*n_step, dim=0)
            current_rot_mat[:,:3,:3] = quaternion_to_matrix(torch.from_numpy(current_eef_pose[:,3:7])).repeat_interleave(n_step, dim=0)
            local_rot_mat = torch.einsum('nmj,ijk->imk', current_rot_mat.transpose(-2, -1), quat)
            quat = matrix_to_quaternion(local_rot_mat).numpy()
            return quat.reshape(batch_size, -1, ndim)
        elif 'gripper_qpos' in key:
            return lowdim
        elif 'pos' in key:
            return lowdim[:,:3] - current_eef_pose[:,:3][:,None,:]
        else:
            raise NotImplementedError(f"low dim key {key} not implemented")
    elif local_type == 'xyz':
        if 'quat' in key:
            return lowdim
        elif 'gripper_qpos' in key:
            return lowdim
        elif 'pos' in key:
            lowdim = lowdim[:,:3] - current_eef_pose[:,:3][:,None,:]
            return lowdim
        else:
            raise NotImplementedError(f"low dim key {key} not implemented")

if __name__ == "__main__":
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)

    import matplotlib.pyplot as plt
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')
    plt.show()