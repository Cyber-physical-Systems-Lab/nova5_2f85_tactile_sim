"""Record tactile frames (+ optional scalars) and save them as a compressed .npz.

Keys written for each pad ``<p>``: ``<p>/normal`` (T, R, C), ``<p>/shear``
(T, R, C, 2), ``<p>/total_normal`` (T,), plus ``time`` (T,), any scalar
channels passed to :meth:`add`, and ``spec`` (JSON string).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .taxel_grid import TaxelGridSpec


class TactileRecorder:
    def __init__(self, spec: TaxelGridSpec):
        self.spec = spec
        self._t: list[float] = []
        self._pads: dict[str, dict[str, list]] = {}
        self._scalars: dict[str, list[float]] = {}

    def __len__(self) -> int:
        return len(self._t)

    def add(self, t: float, frames: dict, **scalars: float) -> None:
        self._t.append(float(t))
        for name, fr in frames.items():
            d = self._pads.setdefault(name, {"normal": [], "shear": [], "total_normal": []})
            d["normal"].append(np.asarray(fr.normal, dtype=np.float32))
            d["shear"].append(np.asarray(fr.shear, dtype=np.float32))
            d["total_normal"].append(float(fr.total_normal))
        for k, v in scalars.items():
            self._scalars.setdefault(k, []).append(float(v) if v is not None else np.nan)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"time": np.asarray(self._t), "spec": np.array(json.dumps(asdict(self.spec)))}
        for name, d in self._pads.items():
            arrays[f"{name}/normal"] = np.stack(d["normal"]) if d["normal"] else np.zeros((0,) + self.spec.shape)
            arrays[f"{name}/shear"] = np.stack(d["shear"]) if d["shear"] else np.zeros((0,) + self.spec.shape + (2,))
            arrays[f"{name}/total_normal"] = np.asarray(d["total_normal"])
        for k, v in self._scalars.items():
            arrays[k] = np.asarray(v)
        np.savez_compressed(path, **arrays)
        return path
