from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R


def rodrigues2extrinsic(vec_rotation: np.ndarray, vec_translation: np.ndarray) -> np.ndarray:
    if vec_rotation.shape != (3,):
        vec_rotation = vec_rotation.reshape((3,))
    if vec_translation.shape != (3, 1):
        vec_translation = vec_translation.reshape((3, 1))
    if vec_rotation.dtype != np.float32:
        vec_rotation = vec_rotation.astype(np.float32)
    if vec_translation.dtype != np.float32:
        vec_translation = vec_translation.astype(np.float32)
    mat_rotation = R.from_rotvec(vec_rotation).as_matrix()
    mat_extrinsic = np.r_[np.c_[mat_rotation, vec_translation], np.array([[0, 0, 0, 1]])]
    return mat_extrinsic


def extrinsic2rodrigues(mat_extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mat_extrinsic.dtype != np.float32:
        mat_extrinsic = mat_extrinsic.astype(np.float32)

    mat_rotation = mat_extrinsic[:3, :3]
    vec_rotation = R.from_matrix(mat_rotation).as_rotvec().reshape(3, 1)
    vec_translation = mat_extrinsic[:3, [3]]
    return vec_rotation, vec_translation


def rodrigues2quaternion(vec_rotation: np.ndarray) -> np.ndarray:
    if vec_rotation.shape != (3,):
        vec_rotation = vec_rotation.reshape((3,))
    if vec_rotation.dtype != np.float32:
        vec_rotation = vec_rotation.astype(np.float32)

    quat_rotation = R.from_rotvec(vec_rotation).as_quat()
    return quat_rotation


def quaternion2rodrigues(quat_rotation: np.ndarray) -> np.ndarray:
    if quat_rotation.shape != (4,):
        quat_rotation = quat_rotation.reshape((4,))
    if quat_rotation.dtype != np.float32:
        quat_rotation = quat_rotation.astype(np.float32)

    vec_rotation = R.from_quat(quat_rotation).as_rotvec().reshape(3, 1)
    return vec_rotation
