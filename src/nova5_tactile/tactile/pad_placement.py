"""Automatic placement of a tactile pad on a fingertip link (pure numpy).

Given the axis-aligned bounding box of the finger link's colliders *in the
link frame*, plus two hints (where the opposing finger is, and where the
gripper base is), choose the pad frame:

* pad +Z (sensing normal) = the AABB face axis that points most towards the
  opposing finger,
* pad +X (taxel rows) = the remaining axis that points most away from the
  gripper base, i.e. towards the fingertip,
* pad +Y = Z x X (right-handed).

The pad is centred on that inner face, pushed towards the tip so that its
distal edge sits ``tip_margin`` from the end of the finger, and offset along
the normal so that it protrudes ``protrusion`` beyond the finger surface.

This is a heuristic: always check the result visually (pads are coloured)
and fall back to manual ``pad_xyz`` / ``pad_rpy_deg`` if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..math3d import make_transform


@dataclass(frozen=True)
class PadPlacement:
    T_link_pad: np.ndarray  # 4x4, column convention
    normal_axis: int  # link-frame axis index used as pad +Z
    tip_axis: int  # link-frame axis index used as pad +X
    fits: bool  # False if the pad is larger than the finger face


def auto_place_pad(
    aabb_min,
    aabb_max,
    toward_other_finger,
    toward_base,
    size_x: float,
    size_y: float,
    thickness: float,
    protrusion: float = 0.0005,
    tip_margin: float = 0.002,
) -> PadPlacement:
    lo = np.asarray(aabb_min, dtype=float).reshape(3)
    hi = np.asarray(aabb_max, dtype=float).reshape(3)
    if np.any(hi <= lo):
        raise ValueError(f"degenerate AABB: min={lo}, max={hi}")
    d_other = np.asarray(toward_other_finger, dtype=float).reshape(3)
    d_tip = -np.asarray(toward_base, dtype=float).reshape(3)
    if np.linalg.norm(d_other) < 1e-9:
        raise ValueError("toward_other_finger is zero; cannot pick the inner face")

    k = int(np.argmax(np.abs(d_other)))
    sk = 1.0 if d_other[k] >= 0 else -1.0

    rest = [i for i in range(3) if i != k]
    if np.linalg.norm(d_tip[rest]) < 1e-9:
        # no usable tip hint: use the longer remaining AABB extent, +direction
        ext = hi - lo
        j = rest[int(np.argmax(ext[rest]))]
        sj = 1.0
    else:
        j = rest[int(np.argmax(np.abs(d_tip[rest])))]
        sj = 1.0 if d_tip[j] >= 0 else -1.0
    m = [i for i in rest if i != j][0]

    ez = np.zeros(3)
    ez[k] = sk
    ex = np.zeros(3)
    ex[j] = sj
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])

    center = np.zeros(3)
    face = hi[k] if sk > 0 else lo[k]
    center[k] = face + sk * (protrusion - 0.5 * thickness)

    extent_j = hi[j] - lo[j]
    fits = extent_j >= size_x + tip_margin and (hi[m] - lo[m]) >= size_y
    if extent_j >= size_x + tip_margin:
        tip_end = hi[j] if sj > 0 else lo[j]
        center[j] = tip_end - sj * (0.5 * size_x + tip_margin)
    else:
        center[j] = 0.5 * (lo[j] + hi[j])
    center[m] = 0.5 * (lo[m] + hi[m])

    return PadPlacement(make_transform(R, center), normal_axis=k, tip_axis=j, fits=bool(fits))
