"""NOVA5 on a fixed table with the Robotiq 2F-85. Import after AppLauncher.

Ported from the nova5 branch of go2_mycobot_ral build_realization().
"""
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from . import ROOT, NOVA5_USD
from .robotiq import attach_robotiq, triangle_mesh, preview_material
from .pad_geometry import box
from .robot_cfg import ARM_JOINTS, FLANGE, ARM_EFFORT_NM, actuators

def cube(path, size, position, color):
    cfg = sim_utils.CuboidCfg(size=tuple(size),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1., restitution=0.),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
    cfg.func(path, cfg, translation=tuple(position))


def author_nova5_2f85(stage, root, *, base_height=0., tactile=True, pad_stiffness=1e4, pad_damping=20.,
                      pad_colliders='taxels', taxel_grid=(6, 9)):
    """Author arm + gripper under `root` (USD only, no world joint). Returns the spec dict.

    The NOVA5 base_link carries the articulation root. The arm base sits
    `base_height` above `root`.
    """
    UsdGeom.Xform.Define(stage, root)
    arm_prim = UsdGeom.Xform.Define(stage, root+'/Arm').GetPrim()
    arm_prim.GetReferences().AddReference(str(NOVA5_USD))
    xform = UsdGeom.Xformable(arm_prim); xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, base_height))
    bodies = {p.GetName(): p for p in Usd.PrimRange(arm_prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if 'base_link' not in bodies or FLANGE not in bodies:
        raise RuntimeError(f'Unexpected NOVA5 body names: {list(bodies)}')
    for prim in reversed(list(Usd.PrimRange(arm_prim))):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        # The converter's Newton root schema includes PhysicsArticulationRootAPI.
        if 'NewtonArticulationRootAPI' in prim.GetAppliedSchemas():
            prim.RemoveAPI('NewtonArticulationRootAPI')
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            endpoints = [j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()]
            if any(not t or not stage.GetPrimAtPath(t[0]).HasAPI(UsdPhysics.RigidBodyAPI) for t in endpoints):
                prim.SetActive(False)
    base_path = str(bodies['base_link'].GetPath())
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(base_path))
    gripper = attach_robotiq(stage, root, str(bodies[FLANGE].GetPath()), tactile=tactile,
                             pad_stiffness=pad_stiffness, pad_damping=pad_damping,
                             pad_colliders=pad_colliders, taxel_grid=taxel_grid)
    props = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(base_path))
    props.CreateEnabledSelfCollisionsAttr(False)
    props.CreateSolverPositionIterationCountAttr(16); props.CreateSolverVelocityIterationCountAttr(4)
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
    return {'arm_joints': ARM_JOINTS, 'flange': FLANGE, 'tool_offset': gripper['tool_offset_m'],
            'base_path': base_path, 'asset': str(NOVA5_USD.relative_to(ROOT)),
            'arm_effort_limits_nm': ARM_EFFORT_NM, 'gripper': gripper}


def build_nova5_2f85(stage, root='/World/Robot', *, table_height=.70, table_size=(.3, .3), **kw):
    """Spawn table + arm + gripper (single scene, no cloning); returns (Articulation, spec dict)."""
    cube(root+'_Table', (*table_size, .10), (0, 0, table_height-.05), (.45, .38, .27))
    cube(root+'_TablePedestal', (.18, .18, table_height-.10), (0, 0, (table_height-.10)/2), (.2, .2, .2))
    spec = author_nova5_2f85(stage, root, base_height=table_height, **kw)
    fix = UsdPhysics.FixedJoint.Define(stage, root+'/table_mount')
    fix.CreateBody1Rel().SetTargets([spec['base_path']])
    fix.CreateLocalPos0Attr(Gf.Vec3f(0, 0, table_height))
    joint_pos = dict(zip(ARM_JOINTS, [0., 0., 1., 0., .6, 0.]))
    joint_pos.update(dict.fromkeys(spec['gripper']['joint_names'], 0.))
    robot = Articulation(ArticulationCfg(prim_path=root, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., table_height), joint_pos=joint_pos,
                                                   joint_vel={'.*': 0.}),
        actuators=actuators(spec['gripper']['joint_names'])))
    spec['base_height_m'] = table_height
    return robot, spec


def spawn_sdf_box(stage, path, size, position, *, mass=.05, color=(.9, .025, .015), sdf_resolution=128, approximation='sdf'):
    """Dynamic box with an SDF triangle-mesh collider (TacSL force field needs an SDF)."""
    pts, tris = box(size)
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    body.AddOrientOp().Set(Gf.Quatf(1.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    rb.CreateSolverPositionIterationCountAttr(16); rb.CreateSolverVelocityIterationCountAttr(4); rb.CreateSleepThresholdAttr(0.)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(mass))
    PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
    mesh = triangle_mesh(stage, path+'/geometry', pts, tris)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(preview_material(stage, path+'/Look', color))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
    if approximation == 'sdf':
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim()).CreateSdfResolutionAttr(int(sdf_resolution))
    api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    api.CreateContactOffsetAttr(.001); api.CreateRestOffsetAttr(0.)
    material = UsdShade.Material.Define(stage, path+'/PhysicsMaterial')
    phys = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys.CreateStaticFrictionAttr(1.); phys.CreateDynamicFrictionAttr(.8); phys.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
    return path
