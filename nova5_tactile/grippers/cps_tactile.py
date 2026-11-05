"""CPS-lab tactile gripper (master-thesis-tactile-gripper) as a two-finger USD articulation. Import after AppLauncher.

Source CAD: github.com/Cyber-physical-Systems-Lab/master-thesis-tactile-gripper (commit 783f026),
converted by scripts/convert_gripper_cad.py into assets/generated/tactile_gripper/:
  assembly.usd   placements of every part (Solid Edge assembly; part geometry not readable on Linux)
  parts/*.usd    geometry of the parts that have STEP exports (base, cylinder, mounting disk,
                 pinion, brick-and-rack carriages, level plates) and of the Daimon DM-Tac WL
                 sensor (vendor STEP 'DM-Tac WL.STEP', supplied separately into third_party/daimon/)

Mechanism: a central pinion drives two racks, each on a carriage ("brick") that slides on two
8 mm rods along the gripper x axis, so the jaws move symmetrically. Each carriage carries a
level plate, a sensor attachment and a Daimon DM-Tac vision-based tactile sensor, which is the
finger. Simulated as two prismatic joints coupled by a PhysX mimic joint (positive = closing).

Gripper frame (CAD): z up towards the mounting disk, fingers hang along -z, jaws close along x.
Sensor frames (CAD): y down the finger, z towards the opposite finger, x across the finger;
origin at the inner bottom corner of the level plate and on one side edge of the sensor, so the
sensor width is twice the origin's offset from the carriage centre line (34.1 mm).

Sensor: Daimon DM-Tac WL from the vendor STEP (39 x 65.3 x 33.9 mm; its frame: y from the cable
end to the tip, -z the gel's outward normal, gel 33 x 40 mm near the tip). The thesis CAD has its
own sensor model whose geometry is not readable here, so the vendor model is mounted by rule:
cable end against the level plate's underside, centred on the carriage, gel face 3.5 mm inboard
of the CAD sensor origin (FACE_INBOARD; matches the thesis front render). The collision shape is
the sensor's bounding box without the cable connector; its inner face is the gel plane. Without
the vendor file the sensor falls back to a 34 x 34.1 x 73 mm box (dimensions measured from the
thesis renders). Outward travel beyond the CAD pose is assumed to be 10 mm.
"""
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from .. import ASSETS

SRC = ASSETS/'generated/tactile_gripper'
MM = .001
# Fallback sensor box when the vendor STEP is missing (see module docstring), metres.
SENSOR_DEPTH = .034       # along x (closing direction)
SENSOR_LENGTH = .073      # below the level plate
GEL_SIZE = (.027, .036)   # (y, z)
GEL_ABOVE_TIP = .020
FACE_INBOARD = .0035      # contact face ahead of the CAD sensor origin, towards the other finger
SENSOR_USD = 'DM_Tac_WL'  # parts/<name>.usd converted from the vendor STEP
OPEN_TRAVEL = .010        # extra opening per finger beyond the CAD pose
MOUNT_TOP_Z = .068        # top of the mounting disk (bolts to the robot flange)
MASS_BASE, MASS_FINGER = .8, .25   # kg, rough (aluminium/printed parts + 131 g sensor)

# STEP part -> assembly instance name (prefix match) and which body carries it.
PARTS = {'Base': ('Basepar1', 'base'), 'Cylinder': ('Cylinderpar1', 'base'), 'Mounting_disk': ('Mounting_diskpar1', 'base'),
         'Pinion': ('pinionpar1', 'base')}
FINGER_ASM = {'left': 'ASM_Adjustable_brick', 'right': 'ASM_Reference_brick'}
FINGER_PARTS = {'Brickpar2': 'Brick_and_rack', 'level_z1par1': 'Level_z1', 'level_referencepar1': 'Level_reference'}


def _key(prim):
    return prim.GetName().removeprefix('tn__').rsplit('_', 1)[0]


def cad_frames():
    """Placements (4x4, gripper frame, metres, column-vector convention) read from the converted assembly."""
    asm = Usd.Stage.Open(str(SRC/'assembly.usd'))
    master = next(p for p in asm.Traverse() if p.GetName() == 'Master')
    cache = UsdGeom.XformCache(); inv = cache.GetLocalToWorldTransform(master).GetInverse()
    out = {'base': {}, 'left': {}, 'right': {}, 'sensor': {}}
    for p in Usd.PrimRange(master):
        m = np.array(cache.GetLocalToWorldTransform(p)*inv).T
        key = _key(p)
        side = next((s for s, a in FINGER_ASM.items() if p.GetParent().GetName().removeprefix('tn__').startswith(a)), None)
        if side and key in FINGER_PARTS:
            out[side][FINGER_PARTS[key]] = m
        elif side and key.startswith('daimon_sensor'):
            out['sensor'][side] = m
        else:
            for part, (inst, body) in PARTS.items():
                if key == inst:
                    out['base'][part] = m
    return out


def vendor_sensor():
    """Bounding boxes of the vendor sensor model in its own frame (metres), or None if not converted.

    Returns lo/hi of the whole sensor, of the gel (the lowest part in z) and of the body without the
    cable connector (parts whose box reaches above the gel-side housing are left out).
    """
    path = SRC/'parts'/f'{SENSOR_USD}.usd'
    if not path.exists():
        return None
    st = Usd.Stage.Open(str(path)); bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    def box(p):
        r = bc.ComputeWorldBound(p).ComputeAlignedRange()
        return np.array(r.GetMin())*MM, np.array(r.GetMax())*MM
    top = st.GetDefaultPrim(); lo, hi = box(top)
    parts = [box(c) for c in Usd.PrimRange(top) if c.IsA(UsdGeom.Xform) and c.GetParent() and c.GetParent().GetParent() == top]
    parts = [b for b in parts if np.all(b[1] > b[0])]
    gel = min(parts, key=lambda b: (b[0][2], b[1][2]-b[0][2]))
    housing_top = max(b[1][2] for b in parts if (b[1]-b[0])[1] > .5*(hi-lo)[1])   # parts spanning most of the length
    return {'lo': lo, 'hi': hi, 'gel_lo': gel[0], 'gel_hi': gel[1], 'body_hi_z': housing_top}


def geometry():
    """Finger geometry in the gripper frame (metres), from the CAD placements and the sensor model."""
    f = cad_frames(); v = vendor_sensor(); g = {'vendor_sensor': v is not None}
    for side, S in f['sensor'].items():
        origin, across, down, inward = S[:3, 3], S[:3, 0], S[:3, 1], S[:3, 2]
        top_z = origin[2]
        face_x = (origin+FACE_INBOARD*inward)[0]
        if v is None:
            width = 2*abs(origin[1])                      # CAD origin sits on the sensor's side edge
            tip_z = top_z+SENSOR_LENGTH*down[2]
            contact = np.array([face_x, 0., tip_z+GEL_ABOVE_TIP])
            g[side] = {'inward': inward, 'width': width, 'top_z': top_z, 'tip_z': tip_z, 'contact': contact}
            continue
        # Vendor frame -> gripper frame: y (cable end -> tip) down, -z (gel normal) towards the other finger.
        yv, zv = np.array([0., 0., -1.]), -inward
        R = np.column_stack([np.cross(yv, zv), yv, zv])
        anchor_v = np.array([(v['lo'][0]+v['hi'][0])/2, v['lo'][1], v['lo'][2]])   # cable end, width centre, gel plane
        t = np.array([face_x, 0., top_z])-R@anchor_v
        V = np.eye(4); V[:3, :3] = R; V[:3, 3] = t
        gel_c = (v['gel_lo']+v['gel_hi'])/2; gel_c[2] = v['lo'][2]
        contact = R@gel_c+t
        tip_z = (R@np.array([0., v['hi'][1], 0.])+t)[2]
        g[side] = {'inward': inward, 'width': float(v['hi'][0]-v['lo'][0]), 'top_z': top_z, 'tip_z': float(tip_z),
                   'contact': contact, 'vendor_T': V,
                   'gel_size': (float(v['gel_hi'][0]-v['gel_lo'][0]), float(v['gel_hi'][1]-v['gel_lo'][1])),
                   'depth': float(v['body_hi_z']-v['lo'][2])}
    g['gap0'] = float(abs(g['right']['contact'][0]-g['left']['contact'][0]))
    mid = (g['left']['contact']+g['right']['contact'])/2
    g['tool_offset'] = float(MOUNT_TOP_Z-mid[2])                     # flange z of the contact midpoint
    g['tip_below_contact'] = float(mid[2]-min(g['left']['tip_z'], g['right']['tip_z']))
    return g


def _ref(stage, path, part, rel):
    """Reference a STEP-derived part (millimetres) with a placement relative to its body."""
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTransformOp().Set(Gf.Matrix4d(rel.T.tolist()))
    mm = UsdGeom.Xform.Define(stage, path+'/mm'); mm.AddScaleOp().Set(Gf.Vec3f(MM, MM, MM))
    mm.GetPrim().GetReferences().AddReference(str(SRC/'parts'/f'{part}.usd'))


def _box(stage, path, lo, hi, color=None, collide=False, material=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cube = UsdGeom.Cube.Define(stage, path); cube.CreateSizeAttr(1.)
    x = UsdGeom.Xformable(cube); x.AddTranslateOp().Set(Gf.Vec3d(*((lo+hi)/2))); x.AddScaleOp().Set(Gf.Vec3f(*(hi-lo)))
    if color is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()); api.CreateContactOffsetAttr(.001); api.CreateRestOffsetAttr(0.)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
    return cube


def _body(stage, path, pose, mass, inertia):
    b = UsdGeom.Xform.Define(stage, path); b.AddTransformOp().Set(Gf.Matrix4d(np.asarray(pose).T.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(b.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(b.GetPrim()).CreateThresholdAttr(0.)
    m = UsdPhysics.MassAPI.Apply(b.GetPrim()); m.CreateMassAttr(float(mass)); m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    return b


def _quat(R):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    return Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3])))


def attach_cps_tactile(stage, root, flange_path):
    """Author the gripper under root/CPSTactile, fixed to flange_path. Returns a spec dict."""
    if not (SRC/'assembly.usd').exists():
        raise FileNotFoundError(f'{SRC}/assembly.usd missing: run scripts/convert_gripper_cad.py first')
    f = cad_frames(); g = geometry()
    cache = UsdGeom.XformCache()
    flange = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(flange_path))).T
    parent = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T
    # Gripper frame in the flange frame: flip z (fingers point out of the flange), disk top on the flange face.
    flange_T_grip = np.diag([1., -1., -1., 1.]); flange_T_grip[2, 3] = MOUNT_TOP_Z
    root_T_grip = np.linalg.inv(parent)@flange@flange_T_grip
    grip = root+'/CPSTactile'
    UsdGeom.Xform.Define(stage, grip)
    mat = UsdShade.Material.Define(stage, grip+'/GelMaterial')
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(1.2); phys.CreateDynamicFrictionAttr(1.0); phys.CreateRestitutionAttr(0.)
    base = _body(stage, grip+'/gripper_base', root_T_grip, MASS_BASE, (2e-3, 2e-3, 2e-3))
    for part, rel in f['base'].items():
        _ref(stage, f'{base.GetPath()}/{part}', part, rel)
    for y in (-.0315, .0315):  # guide rods (8 mm shafts; no STEP), along x
        rod = UsdGeom.Cylinder.Define(stage, f'{base.GetPath()}/rod_{"n" if y < 0 else "p"}')
        rod.CreateRadiusAttr(.004); rod.CreateHeightAttr(.15); rod.CreateAxisAttr('X')
        rod.AddTranslateOp().Set(Gf.Vec3d(0., y, -.016)); rod.CreateDisplayColorAttr([Gf.Vec3f(.7, .72, .75)])
    mount = UsdPhysics.FixedJoint.Define(stage, grip+'/mount')
    mount.CreateBody0Rel().SetTargets([flange_path]); mount.CreateBody1Rel().SetTargets([str(base.GetPath())])
    mount.CreateLocalPos0Attr(Gf.Vec3f(*map(float, flange_T_grip[:3, 3]))); mount.CreateLocalRot0Attr(_quat(flange_T_grip[:3, :3]))
    joints = {}
    for side in ('left', 'right'):
        s = g[side]
        T = np.eye(4); T[:3, 3] = s['contact']          # finger body: gripper orientation, origin at the contact-face centre
        body = _body(stage, f'{grip}/{side}_finger', root_T_grip@T, MASS_FINGER, (3e-4, 3e-4, 2e-4))
        Tinv = np.linalg.inv(T)
        for part, rel in f[side].items():
            _ref(stage, f'{body.GetPath()}/{part}', part, Tinv@rel)
        # Sensor: vendor mesh (visual) + bounding box whose inner face is the gel plane (collision), in the finger frame.
        out = -np.sign(s['inward'][0])                    # +1 for the right finger: its body lies at +x of the face
        z_top, z_tip = s['top_z']-s['contact'][2], s['tip_z']-s['contact'][2]
        if 'vendor_T' in s:
            _ref(stage, f'{body.GetPath()}/DM_Tac_WL', SENSOR_USD, Tinv@s['vendor_T'])
            x0, x1 = sorted([0., out*s['depth']])
            box = _box(stage, f'{body.GetPath()}/sensor_collision', (x0, -s['width']/2, z_tip), (x1, s['width']/2, z_top),
                       collide=True, material=mat)
            UsdGeom.Imageable(box).MakeInvisible()
        else:
            x0, x1 = sorted([0., out*SENSOR_DEPTH])
            _box(stage, f'{body.GetPath()}/sensor', (x0, -s['width']/2, z_tip), (x1, s['width']/2, z_top),
                 color=(.12, .13, .15), collide=True, material=mat)
            gx0, gx1 = sorted([0., -out*.0004])            # gel stands 0.4 mm proud of the face (visual only)
            _box(stage, f'{body.GetPath()}/gel', (gx0, -GEL_SIZE[0]/2, -GEL_SIZE[1]/2), (gx1, GEL_SIZE[0]/2, GEL_SIZE[1]/2),
                 color=(.95, .55, .12))
        j = UsdPhysics.PrismaticJoint.Define(stage, f'{grip}/{side}_finger_joint')
        j.CreateBody0Rel().SetTargets([str(base.GetPath())]); j.CreateBody1Rel().SetTargets([str(body.GetPath())])
        j.CreateAxisAttr('X'); j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, s['contact'])))
        flip = np.diag([-1., -1., 1.]) if side == 'right' else np.eye(3)   # joint +x = closing for both fingers
        j.CreateLocalRot0Attr(_quat(flip)); j.CreateLocalRot1Attr(_quat(flip))
        j.CreateLowerLimitAttr(-OPEN_TRAVEL); j.CreateUpperLimitAttr(g['gap0']/2+.001)
        joints[side] = j
    # Instance name must be rotX/Y/Z even for prismatic joints (single-DOF joints use their own axis).
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(joints['right'].GetPrim(), 'rotX')
    mimic.CreateReferenceJointRel().SetTargets([str(joints['left'].GetPath())]); mimic.CreateGearingAttr(-1.); mimic.CreateOffsetAttr(0.)
    return {'model': 'cps_tactile', 'joint_names': ['left_finger_joint', 'right_finger_joint'], 'driver': 'left_finger_joint',
            'fingers': {'left': f'{grip}/left_finger', 'right': f'{grip}/right_finger'},
            'gap0_m': g['gap0'], 'tool_offset_m': [0., 0., g['tool_offset']], 'tip_below_contact_m': g['tip_below_contact'],
            'q_range_m': [-OPEN_TRAVEL, g['gap0']/2+.001],
            'sensor': ('DM-Tac WL vendor STEP' if g['vendor_sensor'] else 'fallback box'),
            'sensor_m': {k: g['left'].get(k) for k in ('width', 'depth', 'gel_size')} | {'length': g['left']['top_z']-g['left']['tip_z']},
            'assumptions': {'face_inboard_m': FACE_INBOARD, 'open_travel_m': OPEN_TRAVEL}}


def pad_gap_table(gap0, q_range, n=60):
    """(q, gap) with q per finger: both fingers move q towards each other."""
    qs = np.linspace(*q_range, n)
    return [[float(q), float(gap0-2*q)] for q in qs]
