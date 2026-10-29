"""Rigid transforms and serial-joint forward kinematics (pure numpy)."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(position, quaternion_xyzw):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    matrix[:3, 3] = position
    return matrix


def forward_chain(joint_positions, parent_frames, child_frames, axes):
    """USD joint convention: parent * local0 * rotation(q) * inverse(local1)."""
    result = np.eye(4)
    for q, parent, child, axis in zip(joint_positions, parent_frames, child_frames, axes, strict=True):
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec(np.asarray(axis)*q).as_matrix()
        result = result @ parent @ rotation @ np.linalg.inv(child)
    return result


def tool_point(flange_pose, offset):
    """flange_pose is [x, y, z, qx, qy, qz, qw]; returns tool position and world lever."""
    rotation = Rotation.from_quat(flange_pose[3:]).as_matrix()
    lever = rotation @ np.asarray(offset)
    return np.asarray(flange_pose[:3]) + lever, lever


def rotation_error(target, actual):
    return Rotation.from_matrix(target @ actual.T).as_rotvec()


def array(value):
    """Isaac Lab 3 data buffers (warp-backed) to float numpy."""
    value = value.torch if hasattr(value, 'torch') else value
    return value.detach().cpu().numpy().astype(float)
