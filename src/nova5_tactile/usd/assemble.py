"""Build ``nova5_2f85_tactile.usd``: Nova5 arm + Robotiq 2F-85 + fingertip pads.

Layout of the generated asset (defaultPrim = ``/nova5_2f85``)::

    /nova5_2f85                       ArticulationRootAPI (single articulation)
      nova5/          -> reference to the URDF-imported Nova5 USD
      gripper/        -> reference to the Robotiq 2F-85 USD (re-posed on the flange)
      gripper_mount_joint             FixedJoint  flange -> gripper base
      Looks/tactile_pad_material      physics material (compliant contact)
      .../<left finger link>/tactile_pad/taxel_RR_CC   box colliders
      .../<right finger link>/tactile_pad/taxel_RR_CC

A ``<asset>.meta.json`` sidecar records every resolved prim path (relative
to ``/nova5_2f85``) so the runtime never has to search the stage again.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from ..config import ProjectCfg
from ..math3d import invert_transform, make_transform, rpy_deg_to_matrix, split_scale, transform_points
from ..tactile.pad_placement import auto_place_pad
from ..tactile.taxel_grid import TaxelGridSpec
from . import author, query

log = logging.getLogger(__name__)

ROOT = "/nova5_2f85"


def tactile_spec(cfg: ProjectCfg) -> TaxelGridSpec:
    t = cfg.tactile
    return TaxelGridSpec(
        rows=t.rows, cols=t.cols, size_x=t.size_x, size_y=t.size_y, thickness=t.thickness,
        mode=t.spread_mode, spread_sigma=t.spread_sigma, contact_z_tolerance=t.contact_z_tolerance,
    )


def _ref_path(target: str, out_usd: Path) -> str:
    """Relative reference for local files (keeps the repo relocatable)."""
    if "://" in target:
        return target
    return "./" + os.path.relpath(Path(target).resolve(), out_usd.parent.resolve()).replace(os.sep, "/")


def _rigid_world(prim, cache) -> np.ndarray:
    T, s = split_scale(query.world_matrix(prim, cache))
    if not np.allclose(np.abs(s), 1.0, atol=1e-4):
        log.warning("%s has world scale %s; physics bodies should be unscaled", prim.GetPath(), s)
    return T


def _layer_meters_per_unit(url: str) -> float:
    try:
        st = Usd.Stage.Open(url, Usd.Stage.LoadNone)
        return float(UsdGeom.GetStageMetersPerUnit(st)) if st else 1.0
    except Exception:  # noqa: BLE001 - best effort metadata probe
        return 1.0


def _require(prim, what: str, root, candidates):
    if prim is None:
        names = sorted({b.GetName() for b in query.rigid_bodies(root)})
        raise RuntimeError(f"could not find {what}; tried {list(candidates)}. Rigid bodies available: {names}")
    return prim


def _rel(path: str) -> str:
    return path[len(ROOT):].lstrip("/") if path.startswith(ROOT) else path


def assemble(cfg: ProjectCfg, nova5_usd: str, gripper_usd: str, out_usd: Path) -> dict:
    out_usd = Path(out_usd)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    if out_usd.exists():
        out_usd.unlink()

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)

    arm = stage.DefinePrim(f"{ROOT}/nova5", "Xform")
    arm.GetReferences().AddReference(_ref_path(nova5_usd, out_usd))
    grip = stage.DefinePrim(f"{ROOT}/gripper", "Xform")
    grip.GetReferences().AddReference(_ref_path(gripper_usd, out_usd))
    if not query.rigid_bodies(arm):
        raise RuntimeError(f"Nova5 reference has no rigid bodies: {nova5_usd}")
    if not query.rigid_bodies(grip):
        raise RuntimeError(f"gripper reference has no rigid bodies: {gripper_usd}")

    report: dict = {"warnings": []}

    # ---- one articulation for arm + gripper
    report["removed_articulation_roots"] = author.remove_articulation_roots(root)
    author.apply_articulation_root(root, self_collisions=False)
    report["deactivated_world_joints"] = (
        author.deactivate_world_joints(grip) if cfg.gripper.remove_world_joints else []
    )

    flange = _require(query.find_rigid_body(arm, cfg.arm.flange_link_candidates), "arm flange link", arm,
                      cfg.arm.flange_link_candidates)
    gbase = query.find_rigid_body(grip, cfg.gripper.base_link_candidates) or query.root_body(grip)
    gbase = _require(gbase, "gripper base link", grip, cfg.gripper.base_link_candidates)

    # ---- re-pose the gripper so its base sits on flange * mount
    mpu = _layer_meters_per_unit(gripper_usd)
    if abs(mpu - 1.0) > 1e-9:
        report["warnings"].append(f"gripper asset metersPerUnit={mpu}; compensating with a root scale")
    author.set_xform(grip, np.eye(4), scale=mpu if abs(mpu - 1.0) > 1e-9 else None, suffix="mount")
    cache = UsdGeom.XformCache()
    T_w_flange = _rigid_world(flange, cache)
    T_flange_mount = make_transform(rpy_deg_to_matrix(cfg.gripper.mount_rpy_deg), cfg.gripper.mount_xyz)
    T_w_target = T_w_flange @ T_flange_mount
    T_w_gbase_now = _rigid_world(gbase, cache)  # gripper root currently at identity (* scale)
    T_new_root = T_w_target @ invert_transform(T_w_gbase_now)
    author.set_xform(grip, T_new_root, scale=mpu if abs(mpu - 1.0) > 1e-9 else None, suffix="mount")
    author.add_fixed_joint(stage, f"{ROOT}/gripper_mount_joint", str(flange.GetPath()), str(gbase.GetPath()),
                           T_flange_mount)

    # ---- optional arm drive gains
    all_joints = {}
    for j in query.joints(root):
        b0, b1 = query.joint_bodies(j)
        name = j.GetName()
        if name in all_joints:
            report["warnings"].append(f"duplicate joint name {name!r} ({j.GetPath()}); keeping first")
            continue
        all_joints[name] = {"path": _rel(str(j.GetPath())), "type": query.joint_kind(j), "body0": _rel(b0), "body1": _rel(b1)}
    missing = [n for n in cfg.arm.joint_names if n not in all_joints]
    if missing:
        raise RuntimeError(f"arm joints {missing} not found; available: {sorted(all_joints)}")
    if cfg.gripper.drive_joint not in all_joints:
        raise RuntimeError(f"gripper drive joint {cfg.gripper.drive_joint!r} not found; available: {sorted(all_joints)}")
    if cfg.arm.drive_stiffness >= 0 or cfg.arm.drive_damping >= 0:
        for n in cfg.arm.joint_names:
            author.set_drive_gains(stage.GetPrimAtPath(f"{ROOT}/{all_joints[n]['path']}"),
                                   cfg.arm.drive_stiffness, cfg.arm.drive_damping)

    # ---- tactile pads
    t = cfg.tactile
    spec = tactile_spec(cfg)
    material = author.define_pad_material(
        stage, f"{ROOT}/Looks/tactile_pad_material", t.static_friction, t.dynamic_friction, t.restitution,
        t.compliant_stiffness, t.compliant_damping,
    )
    fingers = {f.name: f for f in t.fingers()}
    links = {
        name: _require(query.find_rigid_body(grip, f.link_candidates), f"{name} finger link", grip, f.link_candidates)
        for name, f in fingers.items()
    }
    cache = UsdGeom.XformCache()  # gripper moved: fresh cache
    aabb = {name: query.collider_aabb(link, cache) for name, link in links.items()}
    T_w_link = {name: _rigid_world(link, cache) for name, link in links.items()}
    base_w = _rigid_world(gbase, cache)[:3, 3]
    names = list(fingers)
    pads = {}
    for name, f in fingers.items():
        other = names[1] if name == names[0] else names[0]
        link = links[name]
        if f.auto_place:
            lo, hi = aabb[name]
            c_self = 0.5 * (lo + hi)
            c_other_w = transform_points(T_w_link[other], [0.5 * (aabb[other][0] + aabb[other][1])])[0]
            T_link_w = invert_transform(T_w_link[name])
            toward_other = transform_points(T_link_w, [c_other_w])[0] - c_self
            toward_base = transform_points(T_link_w, [base_w])[0] - c_self
            placement = auto_place_pad(lo, hi, toward_other, toward_base, t.size_x, t.size_y, t.thickness,
                                       t.protrusion, t.tip_margin)
            T_link_pad = placement.T_link_pad
            if not placement.fits:
                report["warnings"].append(
                    f"{name} pad ({t.size_x}x{t.size_y} m) is larger than the finger face "
                    f"(AABB {np.round(hi - lo, 4).tolist()} m); check placement"
                )
        else:
            T_link_pad = make_transform(rpy_deg_to_matrix(f.pad_rpy_deg), f.pad_xyz)
        built = author.build_pad(stage, link, T_link_pad, spec, t.pad_mode, material)
        author.enable_contact_report(link, threshold=0.0)
        pads[name] = {
            "body": _rel(str(link.GetPath())),
            "frame": _rel(built["frame"]),
            "colliders": {_rel(k): v for k, v in built["colliders"].items()},
            "T_body_pad": T_link_pad.tolist(),
            "link_aabb": [aabb[name][0].tolist(), aabb[name][1].tolist()],
        }

    stage.GetRootLayer().Save()

    meta = {
        "asset_root": ROOT,
        "usd": out_usd.name,
        "sources": {"nova5_usd": str(nova5_usd), "gripper_usd": str(gripper_usd)},
        "flange": _rel(str(flange.GetPath())),
        "gripper_base": _rel(str(gbase.GetPath())),
        "arm_joints": cfg.arm.joint_names,
        "gripper_drive_joint": cfg.gripper.drive_joint,
        "joints": all_joints,
        "tactile": {"pad_mode": t.pad_mode, "spec": asdict(spec), "pads": pads},
        "report": report,
    }
    meta_path = out_usd.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    for w in report["warnings"]:
        log.warning(w)
    return meta
