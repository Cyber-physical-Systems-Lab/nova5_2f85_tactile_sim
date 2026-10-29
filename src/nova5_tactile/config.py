"""Project configuration: ``config/default.toml`` -> typed dataclasses.

Only the standard library is used (``tomllib``, Python >= 3.11; Isaac Sim 6.x
ships Python 3.12), so this module imports fine inside and outside Isaac Sim.

Override any value from the command line with ``--set section.key=value``,
where ``value`` is TOML syntax, e.g.::

    --set tactile.rows=16 --set tactile.pad_mode='"single_collider"'
    --set gripper.mount_xyz='[0, 0, 0.01]'
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.toml"


@dataclass
class SimCfg:
    physics_hz: float = 240.0
    gravity: list = field(default_factory=lambda: [0.0, 0.0, -9.81])
    headless: bool = False


@dataclass
class PathsCfg:
    nova5_urdf: str = "assets/urdf/nova5.urdf"
    nova5_usd_dir: str = "assets/usd/nova5"
    assembled_usd: str = "assets/usd/nova5_2f85_tactile.usd"
    gripper_usd: str = ""
    gripper_usd_candidates: list = field(default_factory=list)

    def resolve(self, p: str) -> Path:
        """Repo-relative -> absolute path (absolute paths pass through)."""
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (REPO_ROOT / pp)


@dataclass
class ArmCfg:
    flange_link_candidates: list = field(default_factory=lambda: ["Link6", "link6", "tool0", "flange"])
    joint_names: list = field(default_factory=lambda: [f"joint{i}" for i in range(1, 7)])
    home_deg: list = field(default_factory=lambda: [0.0, 0.0, 90.0, 0.0, -90.0, 0.0])
    drive_stiffness: float = -1.0  # <0 keeps importer values
    drive_damping: float = -1.0


@dataclass
class GripperCfg:
    base_link_candidates: list = field(default_factory=lambda: ["robotiq_base_link", "base_link"])
    drive_joint: str = "finger_joint"
    open_deg: float = 0.0
    close_deg: float = 45.0
    mount_xyz: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mount_rpy_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    remove_world_joints: bool = True


@dataclass
class FingerPadCfg:
    name: str = ""
    link_candidates: list = field(default_factory=list)
    auto_place: bool = True
    pad_xyz: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pad_rpy_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class TactileCfg:
    pad_mode: str = "taxel_colliders"  # or "single_collider"
    rows: int = 8
    cols: int = 4
    size_x: float = 0.030
    size_y: float = 0.016
    thickness: float = 0.002
    protrusion: float = 0.0005
    tip_margin: float = 0.002
    spread_mode: str = "nearest"
    spread_sigma: float = 0.0
    contact_z_tolerance: float = 0.003
    compliant_stiffness: float = 2.0e4
    compliant_damping: float = 50.0
    static_friction: float = 1.0
    dynamic_friction: float = 0.8
    restitution: float = 0.0
    noise_std: float = 0.0
    deadband: float = 0.0
    saturation: float = 0.0
    quantization: float = 0.0
    left: FingerPadCfg = field(default_factory=FingerPadCfg)
    right: FingerPadCfg = field(default_factory=FingerPadCfg)

    def fingers(self) -> list[FingerPadCfg]:
        return [self.left, self.right]


@dataclass
class DemoCfg:
    robot_prim: str = "/World/robot"
    object_size: float = 0.04
    object_mass: float = 0.1
    settle_s: float = 1.5
    close_s: float = 1.0
    hold_s: float = 2.0
    print_every: int = 30
    live_colors: bool = True


@dataclass
class ProjectCfg:
    sim: SimCfg = field(default_factory=SimCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    arm: ArmCfg = field(default_factory=ArmCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)
    tactile: TactileCfg = field(default_factory=TactileCfg)
    demo: DemoCfg = field(default_factory=DemoCfg)


def _build(cls, data: dict, where: str):
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key '{where}{key}'")
        f = known[key]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise TypeError(f"'{where}{key}' must be a table")
            kwargs[key] = _build(type(default), value, f"{where}{key}.")
        else:
            if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            kwargs[key] = value
    return cls(**kwargs)


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def parse_override(expr: str) -> tuple[str, Any]:
    if "=" not in expr:
        raise ValueError(f"override must look like key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    raw = raw.strip()
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # bare word -> string
    return key.strip(), value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ProjectCfg:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    data = copy.deepcopy(data)
    for expr in overrides or []:
        k, v = parse_override(expr)
        _set_dotted(data, k, v)
    cfg = _build(ProjectCfg, data, "")
    validate(cfg)
    return cfg


def validate(cfg: ProjectCfg) -> None:
    t = cfg.tactile
    if t.pad_mode not in ("taxel_colliders", "single_collider"):
        raise ValueError(f"tactile.pad_mode must be taxel_colliders|single_collider, got {t.pad_mode!r}")
    for f in t.fingers():
        if not f.name or not f.link_candidates:
            raise ValueError("tactile.left/right need 'name' and 'link_candidates'")
    if len(cfg.arm.home_deg) != len(cfg.arm.joint_names):
        raise ValueError("arm.home_deg and arm.joint_names must have the same length")
    for key in ("mount_xyz", "mount_rpy_deg"):
        if len(getattr(cfg.gripper, key)) != 3:
            raise ValueError(f"gripper.{key} must have 3 values")
