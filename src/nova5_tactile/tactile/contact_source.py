"""Pull PhysX contact reports and group them by the tactile colliders we watch.

Requires the PhysX backend and ``PhysxContactReportAPI`` on the finger links
(authored by the assembly step). ``get_contact_report()`` returns the
contacts of the most recent simulation step, so call :meth:`poll` once per
step / app update *after* stepping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairContacts:
    """All contact points of one (pad collider, other collider) pair."""

    pad_collider: str
    other_collider: str
    positions: np.ndarray  # (N, 3) world [m]
    impulses: np.ndarray  # (N, 3) world [N s], PhysX sign convention (see orient_pair_forces)
    separations: np.ndarray  # (N,) [m]


def _vec3(v) -> tuple[float, float, float]:
    try:
        return float(v[0]), float(v[1]), float(v[2])
    except TypeError:
        return float(v.x), float(v.y), float(v.z)


class PhysxContactSource:
    def __init__(self, watched_colliders):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._sim = get_physx_simulation_interface()
        self._int_to_path = PhysicsSchemaTools.intToSdfPath
        self._watched = set(watched_colliders)

    def poll(self) -> dict[str, list[PairContacts]]:
        headers, data = self._sim.get_contact_report()
        out: dict[str, list[PairContacts]] = {}
        for h in headers:
            n = int(h.num_contact_data)
            if n <= 0:
                continue
            c0 = str(self._int_to_path(h.collider0))
            c1 = str(self._int_to_path(h.collider1))
            roles = [(c, o) for c, o in ((c0, c1), (c1, c0)) if c in self._watched]
            if not roles:
                continue
            start = int(h.contact_data_offset)
            pos = np.empty((n, 3))
            imp = np.empty((n, 3))
            sep = np.empty(n)
            for k in range(n):
                d = data[start + k]
                pos[k] = _vec3(d.position)
                imp[k] = _vec3(d.impulse)
                sep[k] = float(d.separation)
            for pad, other in roles:  # pad-vs-pad contacts are reported to both pads
                out.setdefault(pad, []).append(PairContacts(pad, other, pos, imp, sep))
        return out
