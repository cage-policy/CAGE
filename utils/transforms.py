import numpy as np


def rotation_matrix_to_rot6d(M):
    '''
    :param M: rotation matrix of shape [..., 3, 3]
    :return: array of shape [..., 6], continuous representation of 3D rotation (first 2 ROWs of M)
    '''
    assert M.shape[-2:] == (3, 3)

    return np.concatenate([M[..., 0, :], M[..., 1, :]], axis=-1)


def rot6d_to_rotation_matrix(rot6d):
    '''
    :param rot6d: array of shape [..., 6], continuous representation of 3D rotation, first 2 ROWs of the rotation matrix
    :return: rotation matrix of shape [..., 3, 3]
    '''
    def normalize_vector(v):
        v_mag = np.sqrt((v**2).sum(axis=-1, keepdims=True))
        v_mag = np.maximum(v_mag, 1e-8)
        return v / v_mag
    
    assert rot6d.shape[-1] == 6

    x_raw, y_raw = rot6d[..., :3], rot6d[..., 3:]

    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw, axis=-1)
    z = normalize_vector(z)
    y = np.cross(z, x, axis=-1)

    return np.stack([x,y,z], axis=-2)


def axis_angle_to_quaternion(axis_angle):
    '''
    :param axis_angle: array of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians
    :return: quaternions of shape (..., 4), with real part first
    '''
    angles = np.linalg.norm(axis_angle, ord=2, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions