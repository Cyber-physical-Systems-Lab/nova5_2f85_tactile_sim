"""Minimal test cell authored with plain pxr (stable across Isaac Sim releases)."""

from __future__ import annotations

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from ..usd.author import set_xform


def init_stage(stage: Usd.Stage, world: str = "/World") -> Usd.Prim:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    w = UsdGeom.Xform.Define(stage, world).GetPrim()
    stage.SetDefaultPrim(w)
    return w


def physics_scene(stage: Usd.Stage, path: str, gravity, physics_hz: float) -> Usd.Prim:
    scene = UsdPhysics.Scene.Define(stage, path)
    g = np.asarray(gravity, dtype=float)
    mag = float(np.linalg.norm(g))
    if mag > 0:
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*(g / mag)))
    scene.CreateGravityMagnitudeAttr().Set(mag)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateTimeStepsPerSecondAttr().Set(int(round(physics_hz)))
    px.CreateEnableCCDAttr().Set(False)
    return scene.GetPrim()


def ground(stage: Usd.Stage, path: str = "/World/ground", size: float = 4.0) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(size, size, 0.1))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.37, 0.40)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def dome_light(stage: Usd.Stage, path: str = "/World/dome", intensity: float = 1000.0) -> Usd.Prim:
    light = UsdLux.DomeLight.Define(stage, path)
    light.CreateIntensityAttr().Set(float(intensity))
    return light.GetPrim()


def add_reference(stage: Usd.Stage, path: str, usd: str) -> Usd.Prim:
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(usd)
    return prim


def test_cube(stage: Usd.Stage, path: str, size: float, mass: float, T_world, kinematic: bool = True,
              friction: float = 0.8) -> Usd.Prim:
    """Cube rigid body posed by the 4x4 ``T_world`` (its parent must be at identity)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(float(size))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.45, 0.85)])
    prim = cube.GetPrim()
    set_xform(prim, np.asarray(T_world, dtype=float))
    UsdPhysics.CollisionAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(bool(kinematic))
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass))
    # Materials must not live under a gprim: keep them in /World/Looks.
    mat_path = Sdf.Path("/World/Looks").AppendChild(Sdf.Path(path).name + "_physics_material")
    material = UsdShade.Material.Define(stage, mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr().Set(float(friction))
    pm.CreateDynamicFrictionAttr().Set(float(friction))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return prim


def set_kinematic(prim: Usd.Prim, kinematic: bool) -> None:
    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(bool(kinematic))
