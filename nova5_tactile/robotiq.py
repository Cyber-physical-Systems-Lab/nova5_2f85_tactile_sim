"""Robotiq 2F-85 CAD, rigid links and URDF mimic constraints, plus tactile pads.

Ported from go2_mycobot_ral/scripts/open_team/isaac_robotiq.py. No object
attachment. Contact, drive and pad compliance settings are simulation
assumptions; source URDF mimic-dependent limit signs are repaired.
"""
import xml.etree.ElementTree as ET
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Vt
from . import ROOT, ASSETS, GRIPPER_URDF
from .kinematics import pose_matrix
from . import pad_geometry

FINGER_TIPS = {'left': 'left_inner_finger', 'right': 'right_inner_finger'}


def stl_triangles(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84+50*count:
        raise ValueError('Expected binary STL: '+str(path))
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.frombuffer(raw, dtype=dtype, offset=84)['vertices'].reshape(-1, 3).copy()


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3])))


def origin(element):
    e = element.find('origin')
    xyz = np.fromstring(e.get('xyz', '0 0 0'), sep=' ') if e is not None else np.zeros(3)
    rpy = np.fromstring(e.get('rpy', '0 0 0'), sep=' ') if e is not None else np.zeros(3)
    return pose_matrix(xyz, Rotation.from_euler('xyz', rpy).as_quat())


def preview_material(stage, path, color, roughness=.4):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def triangle_mesh(stage, path, points, triangles):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(triangles).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    return mesh


def pad_materials(stage, root, pad_stiffness, pad_damping):
    """(physics material, look) shared by all pads under root."""
    material = UsdShade.Material.Define(stage, root+'/PadMaterial')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.2); physics.CreateDynamicFrictionAttr(1.0); physics.CreateRestitutionAttr(0.)
    if pad_stiffness:
        # Compliant contact lets the object penetrate the pad, which the SDF force field measures.
        api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        api.CreateCompliantContactStiffnessAttr(float(pad_stiffness))
        api.CreateCompliantContactDampingAttr(float(pad_damping))
    return material, preview_material(stage, root+'/PadLook', (.85, .55, .10), .8)


def make_pad(stage, path, pose, material, look, colliders='taxels', grid=(6, 9), kinematic=False):
    """Elastomer rigid body at `pose` (4x4, parent frame). Returns {collider path: [r, c] | None}.

    The visual mesh is the first non-collision mesh under the body, which is
    what TacSL ray-casts to place its taxels. With colliders='taxels' the
    contact face is tiled by rows x cols box colliders so every PhysX contact
    is attributed to one taxel; 'single' uses one convex hull of the pad.
    """
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTransformOp().Set(Gf.Matrix4d(np.asarray(pose).T.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(bool(kinematic))
    PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(.002)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0., 0., -pad_geometry.THICKNESS/2))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2e-7, 1e-7, 2e-7))
    points, triangles = pad_geometry.frustum()
    visual = triangle_mesh(stage, path+'/elastomer_visual', points, triangles)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(look)
    shapes = {}
    if colliders == 'taxels':
        for (r, c), (pts, tris) in pad_geometry.taxel_boxes(grid):
            shapes[f'{path}/taxel_{r:02d}_{c:02d}'] = ([r, c], pts, tris)
    elif colliders == 'single':
        shapes[path+'/elastomer_collision'] = (None, points, triangles)
    else:
        raise ValueError(f'colliders must be taxels or single, got {colliders!r}')
    for cpath, (rc, pts, tris) in shapes.items():
        mesh = triangle_mesh(stage, cpath, pts, tris)
        UsdGeom.Imageable(mesh).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        api.CreateContactOffsetAttr(.001); api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
    UsdGeom.Xform.Define(stage, path+'/tactile_sensor')  # TacSL sensor prim (parent = elastomer body)
    return {cpath: rc for cpath, (rc, _, _) in shapes.items()}


def attach_tactile_pads(stage, grip_root, poses, pad_stiffness, pad_damping, colliders='taxels', grid=(6, 9)):
    """One elastomer body per fingertip, fixed to its inner finger."""
    material, look = pad_materials(stage, grip_root, pad_stiffness, pad_damping)
    local = pad_geometry.finger_to_pad()
    pads = {}
    for side, finger in FINGER_TIPS.items():
        path = f'{grip_root}/{side}_tactile_pad'
        shapes = make_pad(stage, path, poses[finger]@local, material, look, colliders, grid)
        fix = UsdPhysics.FixedJoint.Define(stage, path+'_mount')
        fix.CreateBody0Rel().SetTargets([f'{grip_root}/{finger}']); fix.CreateBody1Rel().SetTargets([path])
        fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, local[:3, 3]))); fix.CreateLocalRot0Attr(quat(local[:3, :3]))
        pads[side] = {'body': path, 'colliders': shapes}
    return pads


def attach_robotiq(stage, root, flange_path, *, tactile=True, pad_stiffness=1e4, pad_damping=20.,
                   pad_colliders='taxels', taxel_grid=(6, 9)):
    doc = ET.parse(GRIPPER_URDF).getroot()
    links = {e.get('name'): e for e in doc.findall('link')}
    joints = doc.findall('joint')
    base = 'robotiq_85_base_link'; selected = {base}; ordered = []
    for _ in range(10):
        for j in joints:
            if j.find('parent').get('link') in selected and j.find('child').get('link') not in selected:
                selected.add(j.find('child').get('link')); ordered.append(j)
    cache = UsdGeom.XformCache()
    flange = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(flange_path))).T
    parent = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T
    poses = {base: np.linalg.inv(parent)@flange}
    for j in ordered:
        poses[j.find('child').get('link')] = poses[j.find('parent').get('link')]@origin(j)
    grip_root = root+'/Robotiq2F85'
    UsdGeom.Xform.Define(stage, grip_root)
    material = UsdShade.Material.Define(stage, grip_root+'/ContactMaterial')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.0); physics.CreateDynamicFrictionAttr(.8); physics.CreateRestitutionAttr(0.)
    hashes = {str(GRIPPER_URDF.relative_to(ROOT)): hashlib.sha256(GRIPPER_URDF.read_bytes()).hexdigest()}
    for name in sorted(selected):
        link = links[name]; path = grip_root+'/'+name
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTransformOp().Set(Gf.Matrix4d(poses[name].T.tolist()))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
        inertial = link.find('inertial'); mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(float(inertial.find('mass').get('value')))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, origin(inertial)[:3, 3])))
        vals = inertial.find('inertia').attrib
        I = np.array([[float(vals['i'+a+b if a <= b else 'i'+b+a]) for b in 'xyz'] for a in 'xyz'])
        eig, axes = np.linalg.eigh(I)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, eig)))
        mass.CreatePrincipalAxesAttr(quat(axes))
        for kind in ['visual', 'collision']:
            e = link.find(kind); filename = e.find('geometry/mesh').get('filename')
            meshfile = ASSETS/filename.removeprefix('package://')
            hashes[str(meshfile.relative_to(ROOT))] = hashlib.sha256(meshfile.read_bytes()).hexdigest()
            points = stl_triangles(meshfile)
            mesh = triangle_mesh(stage, path+'/'+kind, points, np.arange(len(points)).reshape(-1, 3))
            UsdGeom.Xformable(mesh).AddTransformOp().Set(Gf.Matrix4d(origin(e).T.tolist()))
            if kind == 'visual':
                # Explicit render material avoids the white fallback in imported CAD.
                color = (.055, .065, .075) if name == base else (.22, .24, .27)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(preview_material(stage, path+'/VisualMaterial', color))
            else:
                UsdGeom.Imageable(mesh).MakeInvisible()
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexDecomposition')
                api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                api.CreateContactOffsetAttr(.0005); api.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
    mount = UsdPhysics.FixedJoint.Define(stage, grip_root+'/mount')
    mount.CreateBody0Rel().SetTargets([flange_path]); mount.CreateBody1Rel().SetTargets([grip_root+'/'+base])
    names = []; multipliers = {}; repairs = []
    for j in ordered:
        name = j.get('name'); path = grip_root+'/'+name
        joint = (UsdPhysics.FixedJoint if j.get('type') == 'fixed' else UsdPhysics.RevoluteJoint).Define(stage, path)
        joint.CreateBody0Rel().SetTargets([grip_root+'/'+j.find('parent').get('link')])
        joint.CreateBody1Rel().SetTargets([grip_root+'/'+j.find('child').get('link')])
        o = origin(j); align = np.eye(3)
        if j.get('type') != 'fixed':
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
            align = Rotation.align_vectors([axis/np.linalg.norm(axis)], [[0, 0, 1]])[0].as_matrix()
            joint.CreateAxisAttr('Z')
            mimic = j.find('mimic'); mult = float(mimic.get('multiplier', '1')) if mimic is not None else 1.
            low, high = sorted([0., .725*mult])
            joint.CreateLowerLimitAttr(float(np.rad2deg(low))); joint.CreateUpperLimitAttr(float(np.rad2deg(high)))
            names.append(name); multipliers[name] = mult
            repairs.append({'joint': name, 'source_limits': j.find('limit').attrib, 'sim_limits_rad': [low, high]})
            if mimic is not None:
                api = PhysxSchema.PhysxMimicJointAPI.Apply(joint.GetPrim(), 'rotZ')
                api.CreateReferenceJointRel().SetTargets([grip_root+'/finger_joint'])
                api.CreateGearingAttr(-mult); api.CreateOffsetAttr(0.)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, o[:3, 3])))
        joint.CreateLocalRot0Attr(quat(o[:3, :3]@align)); joint.CreateLocalRot1Attr(quat(align))
    pads = attach_tactile_pads(stage, grip_root, poses, pad_stiffness, pad_damping, pad_colliders, taxel_grid) if tactile else {}
    return {'model': 'robotiq_2f85', 'joint_names': names, 'multipliers': multipliers,
            'contact_bodies': [p['body'] for p in pads.values()] or [grip_root+'/'+f for f in FINGER_TIPS.values()],
            'tactile_pads': pads, 'pad_colliders': pad_colliders if tactile else None,
            'pad_compliance': {'stiffness_n_per_m': pad_stiffness, 'damping': pad_damping} if tactile else None,
            # Tool point between the pad centres when closed (flange +z).
            'tool_offset_m': [0, 0, .145], 'source_sha256': hashes, 'limit_repairs': repairs,
            'mimic': 'PhysxMimicJointAPI; q_slave=multiplier*q_driver',
            'drive_assumptions': 'single explicit finger_joint drive; 2 Nm cap; not calibrated hardware'}
