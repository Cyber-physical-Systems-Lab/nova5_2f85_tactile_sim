"""World pose of rigid bodies during simulation.

Primary source: ``omni.physx`` ``get_rigidbody_transformation(path)`` (reads
the PhysX actor directly, so it is correct even when physics results are
written only to Fabric and not back to USD). Fallback: USD xform cache.

The quaternion layout returned by PhysX bindings has historically been
(x, y, z, w). :meth:`calibrate` compares against USD right after the
simulation starts (both agree at t=0) and picks the matching layout, so a
binding change cannot silently corrupt the tactile frames.
"""

from __future__ import annotations

import logging

import numpy as np

from ..math3d import from_usd_matrix, make_transform, quat_wxyz_to_matrix, rotation_angle_between, split_scale

log = logging.getLogger(__name__)


def _as_list(v):
    try:
        return [float(v[i]) for i in range(len(v))]
    except TypeError:
        return [float(getattr(v, k)) for k in ("x", "y", "z", "w") if hasattr(v, k)]


class RigidBodyPoseSource:
    def __init__(self, stage, prefer_physx: bool = True):
        self._stage = stage
        self._physx = None
        self.quat_order = "xyzw"
        if prefer_physx:
            try:
                from omni.physx import get_physx_interface

                self._physx = get_physx_interface()
            except Exception as exc:  # noqa: BLE001
                log.warning("omni.physx interface unavailable (%s); using USD poses", exc)

    # ------------------------------------------------------------------ sources
    def _usd(self, path: str) -> np.ndarray:
        from pxr import Usd, UsdGeom

        prim = self._stage.GetPrimAtPath(path)
        if not prim:
            raise KeyError(path)
        T = from_usd_matrix(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        return split_scale(T)[0]

    def _physx_raw(self, path: str):
        if self._physx is None:
            return None
        try:
            res = self._physx.get_rigidbody_transformation(path)
        except Exception:  # noqa: BLE001
            return None
        if not res or not res.get("ret_val", False):
            return None
        return np.array(_as_list(res["position"])), np.array(_as_list(res["rotation"]))

    def _compose(self, pos, quat, order) -> np.ndarray:
        q = quat if order == "wxyz" else np.array([quat[3], quat[0], quat[1], quat[2]])
        return make_transform(quat_wxyz_to_matrix(q), pos)

    # ------------------------------------------------------------------ API
    def calibrate(self, path: str) -> str:
        raw = self._physx_raw(path)
        if raw is None:
            log.info("pose source: USD only")
            return "usd"
        T_usd = self._usd(path)
        errs = {o: rotation_angle_between(self._compose(*raw, o)[:3, :3], T_usd[:3, :3]) for o in ("xyzw", "wxyz")}
        self.quat_order = min(errs, key=errs.get)
        log.info("pose source: PhysX (quat %s, rot err %.2e rad vs USD)", self.quat_order, errs[self.quat_order])
        return self.quat_order

    def get(self, path: str) -> np.ndarray:
        raw = self._physx_raw(path)
        if raw is not None:
            return self._compose(*raw, self.quat_order)
        return self._usd(path)
