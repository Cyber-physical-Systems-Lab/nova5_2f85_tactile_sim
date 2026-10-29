"""OpenUSD authoring helpers (UsdPhysics / PhysxSchema / UsdShade).

Everything here only writes *opinions in the current edit target*, so the
referenced Nova5 / Robotiq assets are never modified on disk.
"""

from __future__ import annotations

import logging

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from ..math3d import matrix_to_quat_wxyz
from ..tactile.taxel_grid import TaxelGridSpec
from .query import PAD_PRIM_NAME, iter_prims, joint_bodies, joints

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- xforms
def set_xform(prim: Usd.Prim, T: np.ndarray, scale=None, suffix: str = "") -> None:
    """Replace the prim's xform op order with translate/orient[/scale].

    A ``suffix`` creates namespaced ops (``xformOp:translate:<suffix>``) so we
    never clash with differently-typed ops authored in a referenced layer.
    """
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t = T[:3, 3]
    w, x, y, z = matrix_to_quat_wxyz(T[:3, :3])
    prec = UsdGeom.XformOp.PrecisionDouble
    xf.AddTranslateOp(prec, suffix).Set(Gf.Vec3d(*map(float, t)))
    xf.AddOrientOp(prec, suffix).Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
    if scale is not None:
        xf.AddScaleOp(prec, suffix).Set(Gf.Vec3d(*map(float, np.broadcast_to(scale, 3))))


def _quatf(R) -> Gf.Quatf:
    w, x, y, z = matrix_to_quat_wxyz(R)
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


# --------------------------------------------------------------------------- physics
def add_fixed_joint(stage: Usd.Stage, path: str, body0: str, body1: str, T_body0_joint: np.ndarray) -> Usd.Prim:
    """Fixed joint whose frame is ``T_body0_joint`` in body0 and identity in body1."""
    j = UsdPhysics.FixedJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, T_body0_joint[:3, 3])))
    j.CreateLocalRot0Attr().Set(_quatf(T_body0_joint[:3, :3]))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return j.GetPrim()


def remove_articulation_roots(root: Usd.Prim) -> list[str]:
    removed = []
    for p in iter_prims(root, instance_proxies=False):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed.append(str(p.GetPath()))
        if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
            p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    return removed


def apply_articulation_root(prim: Usd.Prim, self_collisions: bool = False, pos_iters: int = 32, vel_iters: int = 1) -> None:
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    api.CreateEnabledSelfCollisionsAttr().Set(bool(self_collisions))
    api.CreateSolverPositionIterationCountAttr().Set(int(pos_iters))
    api.CreateSolverVelocityIterationCountAttr().Set(int(vel_iters))


def deactivate_world_joints(root: Usd.Prim) -> list[str]:
    """Deactivate joints that attach a body to the world (empty body0/body1)."""
    out = []
    for j in joints(root):
        if j.IsInstanceProxy():
            continue
        b0, b1 = joint_bodies(j)
        if not b0 or not b1:
            j.SetActive(False)
            out.append(str(j.GetPath()))
    return out


def set_drive_gains(joint_prim: Usd.Prim, stiffness: float | None, damping: float | None) -> None:
    kind = "angular" if joint_prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, kind)
    if stiffness is not None and stiffness >= 0:
        drive.CreateStiffnessAttr().Set(float(stiffness))
    if damping is not None and damping >= 0:
        drive.CreateDampingAttr().Set(float(damping))


def enable_contact_report(body: Usd.Prim, threshold: float = 0.0) -> None:
    api = PhysxSchema.PhysxContactReportAPI.Apply(body)
    api.CreateThresholdAttr().Set(float(threshold))


# --------------------------------------------------------------------------- materials
def define_pad_material(
    stage: Usd.Stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
    compliant_stiffness: float,
    compliant_damping: float,
) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    prim = mat.GetPrim()
    pm = UsdPhysics.MaterialAPI.Apply(prim)
    pm.CreateStaticFrictionAttr().Set(float(static_friction))
    pm.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    pm.CreateRestitutionAttr().Set(float(restitution))
    if compliant_stiffness > 0:
        px = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        try:
            px.CreateCompliantContactStiffnessAttr().Set(float(compliant_stiffness))
            px.CreateCompliantContactDampingAttr().Set(float(compliant_damping))
        except AttributeError:  # schema without the typed accessors
            prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(float(compliant_stiffness))
            prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(float(compliant_damping))
    return mat


def bind_physics_material(prim: Usd.Prim, mat: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


# --------------------------------------------------------------------------- tactile pads
PAD_COLOR = Gf.Vec3f(0.95, 0.45, 0.10)
PAD_COLOR_ALT = Gf.Vec3f(0.85, 0.35, 0.05)


def _box(stage, path, center, half_sizes, color, material, contact_offset):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(1.0)  # NB: UsdGeom.Cube defaults to size 2
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(2 * h) for h in half_sizes]))
    cube.CreateDisplayColorAttr().Set([color])
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc.CreateContactOffsetAttr().Set(float(contact_offset))
    pc.CreateRestOffsetAttr().Set(0.0)
    if material is not None:
        bind_physics_material(prim, material)
    return prim


def build_pad(
    stage: Usd.Stage,
    link: Usd.Prim,
    T_link_pad: np.ndarray,
    spec: TaxelGridSpec,
    pad_mode: str,
    material: UsdShade.Material | None,
    contact_offset: float = 0.002,
) -> dict:
    """Author ``<link>/tactile_pad`` (+ colliders). Returns collider bookkeeping.

    ``taxel_colliders``: rows x cols boxes named ``taxel_RR_CC``.
    ``single_collider``: one box named ``surface``.
    """
    if link.IsInstanceProxy():
        raise RuntimeError(
            f"{link.GetPath()} is an instance proxy; cannot add a pad under it. "
            "Make the asset non-instanceable or pick a non-instanced link."
        )
    frame_path = link.GetPath().AppendChild(PAD_PRIM_NAME)
    frame = UsdGeom.Xform.Define(stage, frame_path).GetPrim()
    set_xform(frame, T_link_pad)

    t = spec.thickness
    colliders: dict[str, list[int] | None] = {}
    if pad_mode == "single_collider":
        p = _box(stage, frame_path.AppendChild("surface"), (0, 0, 0), (spec.size_x / 2, spec.size_y / 2, t / 2),
                 PAD_COLOR, material, contact_offset)
        colliders[str(p.GetPath())] = None
    elif pad_mode == "taxel_colliders":
        centers = spec.centers()
        hx, hy = spec.pitch_x / 2, spec.pitch_y / 2
        for r in range(spec.rows):
            for c in range(spec.cols):
                color = PAD_COLOR if (r + c) % 2 == 0 else PAD_COLOR_ALT
                p = _box(stage, frame_path.AppendChild(f"taxel_{r:02d}_{c:02d}"),
                         (centers[r, c, 0], centers[r, c, 1], 0.0), (hx, hy, t / 2), color, material, contact_offset)
                colliders[str(p.GetPath())] = [r, c]
    else:
        raise ValueError(f"unknown pad_mode {pad_mode!r}")
    return {"frame": str(frame_path), "colliders": colliders}
