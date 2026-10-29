"""Small, dependency-free (numpy only) rigid-transform helpers.

Conventions used throughout this package
----------------------------------------
* 4x4 homogeneous transforms use the *column-vector* convention:
  ``p_world = T_world_body @ p_body``. Name transforms ``T_<to>_<from>``.
* Quaternions are ``(w, x, y, z)`` unless a function says otherwise.
* RPY angles are roll (X), pitch (Y), yaw (Z) applied as extrinsic XYZ,
  i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

USD's ``Gf.Matrix4d`` uses the *row-vector* convention (translation in the
last row). Convert with :func:`from_usd_matrix` / :func:`to_usd_rows`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_wxyz_to_matrix",
    "matrix_to_quat_wxyz",
    "quat_xyzw_to_wxyz",
    "rpy_deg_to_matrix",
    "make_transform",
    "invert_transform",
    "transform_points",
    "rotate_vectors",
    "from_usd_matrix",
    "to_usd_rows",
    "split_scale",
    "rotation_angle_between",
]


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(R) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]])


def rpy_deg_to_matrix(rpy_deg) -> np.ndarray:
    """Roll/pitch/yaw in degrees (extrinsic XYZ) -> 3x3 rotation."""
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float).reshape(3))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_transform(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T) -> np.ndarray:
    """Inverse of a rigid transform (rotation + translation, no scale)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T, points) -> np.ndarray:
    """Apply 4x4 transform to (N, 3) points."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    T = np.asarray(T, dtype=float)
    return P @ T[:3, :3].T + T[:3, 3]


def rotate_vectors(R, vectors) -> np.ndarray:
    """Apply 3x3 rotation to (N, 3) free vectors (forces, normals...)."""
    V = np.asarray(vectors, dtype=float).reshape(-1, 3)
    return V @ np.asarray(R, dtype=float).T


def from_usd_matrix(gf_matrix) -> np.ndarray:
    """``Gf.Matrix4d`` (row-vector convention) -> column-convention 4x4."""
    return np.array(gf_matrix, dtype=float).reshape(4, 4).T


def to_usd_rows(T) -> list:
    """Column-convention 4x4 -> nested rows suitable for ``Gf.Matrix4d(rows)``."""
    return np.asarray(T, dtype=float).T.tolist()


def split_scale(T):
    """Split an affine transform into (rigid T, per-axis scale).

    Assumes no shear (true for anything authored by Isaac Sim importers).
    """
    T = np.asarray(T, dtype=float)
    M = T[:3, :3]
    scale = np.linalg.norm(M, axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("degenerate transform (zero scale)")
    R = M / scale
    # Re-orthonormalise (SVD) to remove numerical drift.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # mirrored asset: keep det(R)=+1, push sign to scale
        R[:, 0] *= -1
        scale[0] *= -1
    return make_transform(R, T[:3, 3]), scale


def rotation_angle_between(Ra, Rb) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    Rd = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    c = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))
