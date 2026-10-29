"""Tactile sensing. Only the pure-numpy API is imported here; the Isaac Sim
runtime pieces live in ``contact_source`` and ``fingertip_sensor``."""

from .pad_placement import PadPlacement, auto_place_pad
from .sensor_model import SensorModelParams, apply_sensor_model
from .taxel_grid import TactileFrame, TaxelGridSpec, bin_contacts, orient_pair_forces

__all__ = [
    "PadPlacement",
    "SensorModelParams",
    "TactileFrame",
    "TaxelGridSpec",
    "apply_sensor_model",
    "auto_place_pad",
    "bin_contacts",
    "orient_pair_forces",
]
