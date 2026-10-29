"""Simple per-taxel sensor non-idealities for sim-to-real work (pure numpy).

Order of operations: ideal force -> additive Gaussian noise -> deadband ->
saturation -> quantisation. Defaults are all "off" (ideal sensor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .taxel_grid import TactileFrame


@dataclass(frozen=True)
class SensorModelParams:
    noise_std: float = 0.0  # [N] per taxel, normal and each shear axis
    deadband: float = 0.0  # [N] normal values below this read as 0
    saturation: float = 0.0  # [N] per taxel; <= 0 disables
    quantization: float = 0.0  # [N] LSB; <= 0 disables

    @property
    def is_ideal(self) -> bool:
        return self.noise_std <= 0 and self.deadband <= 0 and self.saturation <= 0 and self.quantization <= 0


def apply_sensor_model(
    frame: TactileFrame,
    params: SensorModelParams,
    rng: np.random.Generator | None = None,
) -> TactileFrame:
    if params.is_ideal:
        return frame
    rng = rng if rng is not None else np.random.default_rng()
    n = frame.normal.astype(float).copy()
    s = frame.shear.astype(float).copy()

    if params.noise_std > 0:
        n += rng.normal(0.0, params.noise_std, size=n.shape)
        s += rng.normal(0.0, params.noise_std, size=s.shape)
    n = np.maximum(n, 0.0)

    if params.deadband > 0:
        dead = n < params.deadband
        n[dead] = 0.0
        s[dead] = 0.0

    if params.saturation > 0:
        n = np.minimum(n, params.saturation)
        mag = np.linalg.norm(s, axis=-1, keepdims=True)
        s = np.where(mag > params.saturation, s * (params.saturation / np.maximum(mag, 1e-12)), s)

    if params.quantization > 0:
        q = params.quantization
        n = np.round(n / q) * q
        s = np.round(s / q) * q

    return replace(frame, normal=n, shear=s)
