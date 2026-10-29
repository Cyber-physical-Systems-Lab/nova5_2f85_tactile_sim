"""Runtime tactile sensors for the two Robotiq fingertips.

Data flow per step::

    PhysX contact report --(PhysxContactSource.poll)--> PairContacts per pad collider
    finger link pose      --(RigidBodyPoseSource.get)--> T_world_pad = T_world_body @ T_body_pad
    world impulses / dt -> pad-frame forces -> orient_pair_forces -> bin_contacts -> sensor model
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..math3d import invert_transform, rotate_vectors, transform_points
from .sensor_model import SensorModelParams, apply_sensor_model
from .taxel_grid import TactileFrame, TaxelGridSpec, bin_contacts, orient_pair_forces


@dataclass
class PadInfo:
    name: str
    body_path: str
    frame_path: str
    colliders: dict  # collider path -> [r, c] | None
    T_body_pad: np.ndarray


def load_meta(meta_path: str | Path) -> dict:
    return json.loads(Path(meta_path).read_text())


def pads_from_meta(meta: dict, robot_prim: str) -> tuple[TaxelGridSpec, list[PadInfo]]:
    """Map the asset-relative paths stored in the meta file under ``robot_prim``."""
    prefix = robot_prim.rstrip("/")

    def full(rel: str) -> str:
        return f"{prefix}/{rel}" if rel else prefix

    spec = TaxelGridSpec(**meta["tactile"]["spec"])
    pads = []
    for name, p in meta["tactile"]["pads"].items():
        pads.append(PadInfo(
            name=name,
            body_path=full(p["body"]),
            frame_path=full(p["frame"]),
            colliders={full(k): v for k, v in p["colliders"].items()},
            T_body_pad=np.asarray(p["T_body_pad"], dtype=float),
        ))
    return spec, pads


class FingertipTactileSensor:
    def __init__(self, pad: PadInfo, spec: TaxelGridSpec, model: SensorModelParams | None = None, seed: int = 0):
        self.pad = pad
        self.spec = spec
        self.model = model or SensorModelParams()
        self._rng = np.random.default_rng(seed)
        self.last: TactileFrame = TactileFrame.empty(spec)

    @property
    def name(self) -> str:
        return self.pad.name

    def compute(self, pairs, T_world_body: np.ndarray, dt: float, timestamp: float) -> TactileFrame:
        T_world_pad = T_world_body @ self.pad.T_body_pad
        T_pad_world = invert_transform(T_world_pad)
        P, F, IDX = [], [], []
        for pc in pairs:
            p_local = transform_points(T_pad_world, pc.positions)
            f_local = orient_pair_forces(rotate_vectors(T_pad_world[:3, :3], pc.impulses / dt))
            rc = self.pad.colliders.get(pc.pad_collider)
            idx = np.tile(rc if rc is not None else [-1, -1], (len(p_local), 1))
            P.append(p_local)
            F.append(f_local)
            IDX.append(idx)
        if P:
            frame = bin_contacts(np.vstack(P), np.vstack(F), self.spec, taxel_index=np.vstack(IDX), timestamp=timestamp)
        else:
            frame = TactileFrame.empty(self.spec, timestamp)
        frame = apply_sensor_model(frame, self.model, self._rng)
        frame.meta["T_world_pad"] = T_world_pad
        self.last = frame
        return frame


class TwoFingerTactileRig:
    """Owns the contact source and both fingertip sensors."""

    def __init__(self, meta: dict, robot_prim: str, pose_source, physics_dt: float,
                 model: SensorModelParams | None = None):
        from .contact_source import PhysxContactSource  # Isaac-only import

        self.spec, pads = pads_from_meta(meta, robot_prim)
        self.sensors = {p.name: FingertipTactileSensor(p, self.spec, model, seed=i) for i, p in enumerate(pads)}
        watched = [c for p in pads for c in p.colliders]
        self._collider_owner = {c: p.name for p in pads for c in p.colliders}
        self._source = PhysxContactSource(watched)
        self._pose = pose_source
        self.dt = float(physics_dt)

    def calibrate_pose_source(self) -> None:
        first = next(iter(self.sensors.values()))
        self._pose.calibrate(first.pad.body_path)

    def update(self, timestamp: float) -> dict[str, TactileFrame]:
        by_collider = self._source.poll()
        grouped: dict[str, list] = {name: [] for name in self.sensors}
        for collider, pairs in by_collider.items():
            grouped[self._collider_owner[collider]].extend(pairs)
        frames = {}
        for name, sensor in self.sensors.items():
            T_wb = self._pose.get(sensor.pad.body_path)
            frames[name] = sensor.compute(grouped[name], T_wb, self.dt, timestamp)
        return frames
