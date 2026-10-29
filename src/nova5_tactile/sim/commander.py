"""Joint position commands for the assembled robot.

Primary path: ``isaacsim.core.experimental.prims.Articulation`` (the Isaac
Sim 6.x Core API; radians). If it is unavailable or a call fails, fall back
to writing USD drive targets (``drive:<angular|linear>:physics:targetPosition``,
degrees for revolute joints), which PhysX picks up while simulating. The
fallback is slower but independent of Core-API churn.
"""

from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)


def _np(x):
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    if hasattr(x, "cpu"):
        return np.asarray(x.cpu())
    return np.asarray(x)


class JointCommander:
    def __init__(self, stage, articulation_path: str, joint_paths: dict[str, str], joint_types: dict[str, str]):
        self._stage = stage
        self._joint_paths = joint_paths
        self._joint_types = joint_types
        self._art = None
        self._dof_names: list[str] = []
        self.backend = "usd"
        try:
            from isaacsim.core.experimental.prims import Articulation

            self._art = Articulation(articulation_path)
            self._dof_names = [str(n) for n in self._art.dof_names]
            self.backend = "articulation"
        except Exception as exc:  # noqa: BLE001
            log.warning("experimental Articulation unavailable (%s); using USD drive targets", exc)
        log.info("joint commander backend: %s (%d dofs)", self.backend, len(self._dof_names))

    @property
    def dof_names(self) -> list[str]:
        return list(self._dof_names)

    def set_targets(self, targets_rad: dict[str, float]) -> None:
        if self._art is not None:
            try:
                idx = [self._dof_names.index(n) for n in targets_rad]
                vals = np.array([[float(v) for v in targets_rad.values()]])
                self._art.set_dof_position_targets(vals, dof_indices=idx)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("Articulation.set_dof_position_targets failed (%s); switching to USD fallback", exc)
                self._art = None
                self.backend = "usd"
        for name, val in targets_rad.items():
            self._set_usd_target(name, float(val))

    def _set_usd_target(self, name: str, value_rad: float) -> None:
        prim = self._stage.GetPrimAtPath(self._joint_paths[name])
        kind = self._joint_types.get(name, "revolute")
        token = "angular" if kind == "revolute" else "linear"
        value = math.degrees(value_rad) if kind == "revolute" else value_rad
        attr = prim.GetAttribute(f"drive:{token}:physics:targetPosition")
        if not attr:
            from pxr import UsdPhysics

            attr = UsdPhysics.DriveAPI.Apply(prim, token).CreateTargetPositionAttr()
        attr.Set(value)

    def get_positions(self) -> dict[str, float] | None:
        if self._art is None:
            return None
        try:
            q = _np(self._art.get_dof_positions()).reshape(-1)
            return dict(zip(self._dof_names, map(float, q)))
        except Exception:  # noqa: BLE001
            return None
