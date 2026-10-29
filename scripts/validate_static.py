"""Static tactile checks on standalone face-up pads (no arm), both backends.

Pads are copies of the fingertip pad (same mesh, colliders and compliant
material), held by a fixed joint to the world like the fingertip pads are held
by the finger. (Kinematic pads make PhysX over-report contact force about 3x
under footprints spanning many taxel colliders; --pad-mount kinematic shows it.)
Boxes with SDF colliders are set on them:
  weight:      200 g, 16x16 mm footprint, centred -> total normal = m g
  tip:         50 g, 6x6 mm, near the fingertip   -> load in the highest cols
  side:        50 g, 6x6 mm, towards +x           -> load in the highest rows
  empty:       nothing on it                      -> every taxel reads 0
  heavy_small / light_big: mass vs footprint controls for the total
Results (averaged over the last 0.5 s) go to <out>/static.json.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True)
p.add_argument('--seconds', type=float, default=2.)
p.add_argument('--pad-stiffness', type=float, default=1e4)
p.add_argument('--pad-damping', type=float, default=20.)
p.add_argument('--pad-colliders', choices=['taxels', 'single'], default='taxels')
p.add_argument('--tactile-stiffness', type=float, default=1.)
p.add_argument('--pad-mount', choices=['joint', 'kinematic'], default='joint', help='dynamic pad on a world fixed joint, or kinematic pad')
p.add_argument('--object-collider', choices=['sdf', 'convexHull'], default='sdf', help='convexHull disables TacSL')
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_physx.physics import PhysxCfg
from pxr import Gf, UsdGeom, UsdPhysics
from nova5_tactile import pad_geometry, tactile
from nova5_tactile.robotiq import pad_materials, make_pad, quat
from nova5_tactile.robot import spawn_sdf_box
from nova5_tactile.contact_taxels import ContactTaxelSensors
from nova5_tactile.taxels import TactileFrame

G = 9.81
# name: (pad-frame (x, y) of the box centre, size, mass)
CASES = {'weight': ((0., 0.), (.016, .016, .012), .2),
         'tip': ((0., .0125), (.006, .006, .006), .05),
         'side': ((.0075, 0.), (.006, .006, .006), .05),
         'empty': (None, None, 0.),
         'heavy_small': ((0., 0.), (.006, .006, .006), .2),
         'light_big': ((0., 0.), (.016, .016, .012), .05)}


def main():
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f'{out} exists; use a fresh output directory')
    out.mkdir(parents=True)
    dt = .0025
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
        physics=PhysxCfg(gpu_collision_stack_size=2**28)))
    sim.set_camera_view(eye=(.15, -.35, .5), target=(.15, 0., .3))
    cfg = sim_utils.GroundPlaneCfg(); cfg.func('/World/Ground', cfg)
    cfg = sim_utils.DomeLightCfg(intensity=1800.); cfg.func('/World/Light', cfg)
    UsdGeom.Xform.Define(sim.stage, '/World/Pads')
    material, look = pad_materials(sim.stage, '/World/Pads', args.pad_stiffness, args.pad_damping)
    flip = np.eye(4); flip[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()  # pad z down: contact face up
    pads = {}; poses = {}; objects = {}; expected = {}
    for k, (name, (xy, size, mass)) in enumerate(CASES.items()):
        pose = flip.copy(); pose[:3, 3] = [.1*k, 0., .3]
        path = f'/World/Pads/{name}'
        pads[name] = {'body': path, 'colliders': make_pad(sim.stage, path, pose, material, look, args.pad_colliders,
                                                          kinematic=args.pad_mount == 'kinematic')}
        if args.pad_mount == 'joint':
            fix = UsdPhysics.FixedJoint.Define(sim.stage, path+'_world')
            fix.CreateBody1Rel().SetTargets([path])
            fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pose[:3, 3]))); fix.CreateLocalRot0Attr(quat(pose[:3, :3]))
        poses[name] = pose
        if xy is not None:
            centre = pose@np.r_[xy, -pad_geometry.THICKNESS-size[2]/2-.001, 1]
            spawn_sdf_box(sim.stage, f'/World/Box_{name}', size, centre[:3], mass=mass, approximation=args.object_collider)
            objects[name] = RigidObject(RigidObjectCfg(prim_path=f'/World/Box_{name}', spawn=None,
                                        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(centre[:3]))))
            expected[name] = {'xy': list(xy), 'normal_n': mass*G}
    targets = {n: f'/World/Box_{n}' if n in objects else '/World/Box_weight' for n in pads}
    use_tacsl = args.object_collider == 'sdf'
    tacsl = tactile.make_tactile_sensors(pads, targets, normal_stiffness=args.tactile_stiffness) if use_tacsl else {}
    contact = ContactTaxelSensors(pads, targets)
    sim.reset()
    backends = ['tacsl', 'contact'] if use_tacsl else ['contact']
    acc = {n: {b: [] for b in backends} for n in pads}
    steps = round(args.seconds/dt); keep = round(.5/dt)
    for step in range(steps):
        sim.step(render=not args.headless)
        for o in objects.values():
            o.update(dt)
        for s in tacsl.values():
            s.update(dt)
        frames = contact.update(poses, dt)
        if step >= steps-keep:
            for n in pads:
                if use_tacsl:
                    r = tactile.read(tacsl[n]); acc[n]['tacsl'].append((r['normal'], r['shear']))
                acc[n]['contact'].append((frames[n].normal, frames[n].shear))
    centres = {'contact': contact.spec.centers()}
    if use_tacsl:
        centres['tacsl'] = tactile.taxel_positions(tacsl['weight'])[..., :2]
    result = {'args': vars(args), 'expected': expected, 'cases': {}}
    for n in pads:
        result['cases'][n] = {}
        for backend in backends:
            normal = np.mean([a for a, _ in acc[n][backend]], 0); shear = np.mean([b for _, b in acc[n][backend]], 0)
            f = TactileFrame(normal, shear)
            peak = np.unravel_index(np.argmax(normal), normal.shape) if f.total_normal > 0 else None
            result['cases'][n][backend] = {'total_normal': f.total_normal, 'total_shear': f.total_shear.tolist(),
                'cop_xy': f.center_of_pressure(centres[backend]).tolist(), 'peak_rc': [int(i) for i in peak] if peak else None,
                'active': int((normal > 0).sum()), 'normal': normal.tolist()}
            print('CASE', n, backend, 'total', round(f.total_normal, 5), 'cop_mm', np.round(f.center_of_pressure(centres[backend])*1e3, 2),
                  'peak', peak, 'active', int((normal > 0).sum()), flush=True)
        if n in objects:
            result['cases'][n]['object_z'] = float(objects[n].data.root_pos_w.torch[0, 2])
    (out/'static.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0


code = 1
try:
    code = main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close(exit_code=code)
raise SystemExit(code)
