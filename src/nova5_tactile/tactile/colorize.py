"""Colour taxel boxes in the viewport by their normal load (taxel_colliders mode).

Only writes ``primvars:displayColor`` (not physics-relevant), so it does not
disturb the simulation. Throttle it (every few frames) for large grids.
"""

from __future__ import annotations

import numpy as np


_GREY = np.array([0.22, 0.22, 0.26])
_ORANGE = np.array([1.00, 0.60, 0.05])
_RED = np.array([1.00, 0.05, 0.02])


def _ramp(x: float):
    """0 -> grey (unloaded), 0.5 -> orange, 1 -> red (>= vmax)."""
    x = float(np.clip(x, 0.0, 1.0))
    c = _GREY + (_ORANGE - _GREY) * (2 * x) if x < 0.5 else _ORANGE + (_RED - _ORANGE) * (2 * x - 1)
    return tuple(float(v) for v in c)


class TaxelColorizer:
    def __init__(self, stage, sensors: dict, vmax: float = 5.0):
        from pxr import UsdGeom

        self._attrs = {}
        for name, s in sensors.items():
            for path, rc in s.pad.colliders.items():
                if rc is None:
                    continue
                prim = stage.GetPrimAtPath(path)
                if prim:
                    self._attrs[(name, rc[0], rc[1])] = UsdGeom.Gprim(prim).GetDisplayColorAttr()
        self.vmax = float(vmax)
        self.enabled = bool(self._attrs)

    def update(self, frames: dict) -> None:
        if not self.enabled:
            return
        from pxr import Gf

        for (name, r, c), attr in self._attrs.items():
            fr = frames.get(name)
            if fr is None:
                continue
            attr.Set([Gf.Vec3f(*_ramp(fr.normal[r, c] / self.vmax))])
