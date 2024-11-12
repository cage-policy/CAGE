import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat

from utils.constants import *
from utils.transforms import (axis_angle_to_quaternion,
                              rot6d_to_rotation_matrix,
                              rotation_matrix_to_rot6d)


def camera_orientation_correction(image, cam_id, cfg, task, user):
    """
    Some cameras are mounted in a non-standard orientation, this function corrects the orientation and crops the image to square.

    Parameters:
        image: PIL Image
        cam_id: camera id
        cfg: platform configuration
        task: task id
        user: task user id

    Returns:
        rectified square PIL Image of correct orientation and range
    """
    H, W = image.size

    if cfg == 1:
        return F.center_crop(image, min(H, W))
    elif cfg == 2 and cam_id == 'cam_f0461559':
        if user in [13, 14, 16] or (user==7 and task not in [1,2,35,37,41,42,44,45,59,60,75,80,82,83,84,85]):
            return image.transpose(Image.ROTATE_270).crop((0,H-W,W,H))  # bottom crop
        else:
            return F.center_crop(image, min(H, W))
    elif cfg in [3, 4] and cam_id == 'cam_f0172289':
        return image.transpose(Image.ROTATE_90).crop((0,H-W,W,H))
    elif cfg == 5 and cam_id == 'cam_f0461559':
        return image.transpose(Image.ROTATE_270).crop((0,H-W,W,H))
    elif cfg in [6, 7] and cam_id == 'cam_f0271510':
        return image.transpose(Image.ROTATE_270).crop((0,H-W,W,H))

def tcp_to_xyz_rot(tcp, config):
    """
    Convert tcp data to xyz and rotation matrix in base coordinate

    Parameters:
        tcp: array of shape (7,) or (6,) with xyz + quat/axis_angle/euler_angle based on config
        config: int, 1-7

    Returns:
        xyz and rotation matrix of shape (3, 3)
    """
    if config in [1, 2]:    # flexiv
        xyz = tcp[:3]
        rot = quat2mat(tcp[3:7])
    elif config in [3, 4]:  # UR5
        xyz = tcp[:3]
        xyz[:2] *= -1
        M = quat2mat(axis_angle_to_quaternion(tcp[3:6]))
        rot = np.stack([
            [-M[0, 1], M[0, 0], -M[0, 2]],
            [-M[1, 1], M[1, 0], -M[1, 2]],
            [M[2, 1], -M[2, 0], M[2, 2]],
        ])
    elif config == 5:       # Franka
        xyz = tcp[:3]
        R = np.array([
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.]
        ])
        M = quat2mat(axis_angle_to_quaternion(tcp[3:6]))
        rot = M @ R
    elif config in [6, 7]:  # Kuka
        xyz = tcp[:3]
        xyz /= 1000
        rot = euler2mat(*tcp[3:6])
    
    return xyz, rot

def encode_gripper_width(width, config):
    """
    Convert real gripper width (in meters) to command w.r.t. different brands

    Parameters
        width: float
            real gripper width in meters
        config: int, 1-7
            task configuration

    Returns
        encoded gripper width
    """
    assert config in range(1, 8)

    if config in [1, 2]:
        # Dahuan AG-95
        width = width / 0.095 * 1000.
        width = max(0, min(1000, width))
    elif config in [3]:
        # WSG-50
        width = width * 100.
    elif config in [4, 6, 7]:
        # Robotiq 2F-85
        width = 255. - width / 0.085 * 255.
        width = max(0, min(255, width))
    elif config in [5]:
        # Panda
        width = width * 100000.

    return width

def decode_gripper_width(width, config):
    """
    Convert encoded gripper width to real width (in meters) w.r.t. different brands

    Parameters
        width: float
            encoded gripper width
        config: int, 1-7
            task configuration
    
    Returns
        real gripper width in meters
    """
    assert config in range(1, 8)

    if config in [1, 2]:
        # Dahuan AG-95
        width = width / 1000. * 0.095
    elif config in [3]:
        # WSG-50
        width = width / 100.
    elif config in [4, 6, 7]:
        # Robotiq 2F-85
        width = (255. - width) / 255. * 0.085
    elif config in [5]:
        # Panda
        width = width / 100000.

    return width

def get_normalized_actions(actions, base_pose, act_repr):
    """
    Normalize actions to [-1, 1] w.r.t. action representation

    Parameters:
        actions: array of shape [..., act_horizon, 13]
        base_pose: array of shape [..., 4, 4]
        act_repr: str, 'abs' | 'rel' | 'delta_xyz'

    Returns:
        normalized actions of shape [..., act_horizon, act_dim=10]
    """
    assert actions.shape[-1] == 13
    assert base_pose.shape[-2:] == (4, 4)
    assert act_repr in ['abs', 'rel', 'delta_xyz']

    prefix_shape = actions.shape[:-1]
    xyz = actions[..., :3]
    rot = actions[..., 3:12].reshape(*prefix_shape, 3, 3)
    width = actions[..., 12:13]

    # first normalize xyz to 0-1
    xyz = (xyz - TRANS_MIN) / (TRANS_MAX - TRANS_MIN)
    if act_repr == 'rel':
        next_pose = np.zeros((*prefix_shape, 4,4))
        next_pose[..., :3, 3] = xyz
        next_pose[..., :3, :3] = rot
        next_pose[..., 3, 3] = 1
        # use relative actions based on the pose of the last obseration timestamp
        pose = np.expand_dims(np.linalg.inv(base_pose), axis=-3) @ next_pose

        # xyz in [-1, 1] (delta)
        xyz = pose[..., :3, 3]
        rot = pose[..., :3, :3]
    elif act_repr == 'delta_xyz':
        base_xyz = base_pose[..., :3, 3]
        xyz = xyz - np.expand_dims(base_xyz, axis=-2)
    else:
        # use absolute actions (tcp as prediction)
        # xyz in [-1, 1]
        xyz = xyz * 2 - 1
    # rot6d
    rot = rotation_matrix_to_rot6d(rot)

    # always predict actual width
    # normalize width
    width = width / WIDTH_MAX * 2 - 1
    
    # dim = 10
    return np.concatenate([xyz, rot, width], axis=-1).astype(np.float32)

def get_actual_actions(actions, base_pose, act_repr):
    """
    Get actual actions from normalized actions

    Parameters:
        actions: array of shape [..., act_horizon, act_dim=10]
        base_pose: array of shape [..., 4, 4]
        act_repr: str, 'abs' | 'rel' | 'delta_xyz'
    
    Returns:
        actual actions of shape [..., act_horizon, act_dim=10]
    """
    assert actions.shape[-1] == 10
    assert base_pose.shape[-2:] == (4, 4)
    assert act_repr in ['abs', 'rel', 'delta_xyz']

    prefix_shape = actions.shape[:-1]
    xyz = actions[..., :3]
    rot = actions[..., 3:9]
    if act_repr == 'rel':
        pose = np.zeros((*prefix_shape, 4, 4))
        pose[..., :3, 3] = xyz
        pose[..., :3, :3] = rot6d_to_rotation_matrix(rot)
        pose[..., 3, 3] = 1
        pose = np.expand_dims(base_pose, axis=-3) @ pose

        xyz = pose[..., :3, 3]
        rot = rotation_matrix_to_rot6d(pose[..., :3, :3])
    elif act_repr == 'delta_xyz':
        base_xyz = base_pose[..., :3, 3]
        xyz = xyz + np.expand_dims(base_xyz, axis=-2)
    else:
        xyz = (xyz + 1) / 2 
    
    xyz = xyz * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    
    width = actions[..., 9:]
    width = (width + 1) / 2 * WIDTH_MAX

    return np.concatenate([xyz, rot, width], axis=-1).astype(np.float32)
