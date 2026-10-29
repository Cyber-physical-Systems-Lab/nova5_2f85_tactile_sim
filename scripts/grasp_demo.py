"""NOVA5 + Robotiq 2F-85 + fingertip tactile pads: approach, close, lift a free box.

Logs both pads with two tactile backends side by side: TacSL force field
(SDF penetration) and contact-report taxels (PhysX impulses on per-taxel
colliders), next to each pad's net contact force. Both are in the pad frame
(taxels.py). The close/lift schedule is fixed open-loop; tactile data is
recorded, not used.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True, help='fresh output directory under runs/')
p.add_argument('--seconds', type=float, default=14.)
p.add_argument('--keep-open', action='store_true', help='negative control: never close the gripper')
p.add_argument('--pad-stiffness', type=float, default=1e4, help='compliant pad contact stiffness [N/m], 0 = rigid')
p.add_argument('--pad-damping', type=float, default=20.)
p.add_argument('--tactile-stiffness', type=float, default=1., help='TacSL per-taxel normal stiffness')
p.add_argument('--pad-colliders', choices=['taxels', 'single'], default='taxels',
               help='taxels: one box collider per taxel (contact backend direct); single: one hull, contacts binned by position')
p.add_argument('--noise-std', type=float, default=0., help='contact backend per-taxel noise [N]')
p.add_argument('--deadband', type=float, default=0.)
p.add_argument('--saturation', type=float, default=0.)
p.add_argument('--quantization', type=float, default=0.)
p.add_argument('--debug-taxels', action='store_true', help='draw taxel markers (GUI)')
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app

import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from nova5_tactile import ROOT
from nova5_tactile.robot import build_nova5_2f85, spawn_sdf_box
from nova5_tactile.kinematics import pose_matrix, forward_chain, tool_point, rotation_error, array
from nova5_tactile import tactile
from nova5_tactile.contact_taxels import ContactTaxelSensors
from nova5_tactile.taxels import SensorModelParams

# Top-down grasp seed from go2_mycobot_ral configs/nova5_loaded_grasp.json.
IK_SEED = [0.26833561, 0.00559351, 1.52478671, 0.04041244, -1.57079610, 1.83913481]


def main():
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f'{out} exists; use a fresh output directory')
    out.mkdir(parents=True); (out/'tactile').mkdir()
    dt = .0025
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
        physics=PhysxCfg(enable_external_forces_every_iteration=True, gpu_collision_stack_size=2**28)))
    sim.set_camera_view(eye=(.35, .45, 1.15), target=(-.4, -.25, .8))
    cfg = sim_utils.GroundPlaneCfg(); cfg.func('/World/Ground', cfg)
    cfg = sim_utils.DomeLightCfg(intensity=1800.); cfg.func('/World/Light', cfg)
    robot, spec = build_nova5_2f85(sim.stage, pad_stiffness=args.pad_stiffness or None, pad_damping=args.pad_damping,
                                   pad_colliders=args.pad_colliders)
    robot.cfg.actuators['arm'].stiffness = 250.
    robot.cfg.actuators['arm'].damping = 30.
    center = np.array([-.4, -.25, .825]); offset = np.array(spec['tool_offset'])
    surface = RigidObject(RigidObjectCfg(prim_path='/World/Surface', spawn=sim_utils.CuboidCfg(size=(.28, .28, .04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.), collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18, .30, .40))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-.4, -.25, .78))))
    PhysxSchema.PhysxContactReportAPI.Apply(sim.stage.GetPrimAtPath('/World/Surface')).CreateThresholdAttr(0.)
    spawn_sdf_box(sim.stage, '/World/Object', (.04, .04, .05), center)
    obj = RigidObject(RigidObjectCfg(prim_path='/World/Object', spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(center))))
    pads = spec['gripper']['tactile_pads']
    taxels = tactile.make_tactile_sensors(pads, '/World/Object', normal_stiffness=args.tactile_stiffness,
                                          debug_vis=args.debug_taxels)
    contacts = {side: ContactSensor(ContactSensorCfg(prim_path=pad['body'], filter_prim_paths_expr=['/World/Object'], update_period=0.))
                for side, pad in pads.items()}
    model = SensorModelParams(args.noise_std, args.deadband, args.saturation, args.quantization)
    taxels_c = ContactTaxelSensors(pads, '/World/Object', model=model)
    support = ContactSensor(ContactSensorCfg(prim_path='/World/Surface', filter_prim_paths_expr=['/World/Object'], update_period=0.))
    sim.reset()

    arm = [robot.joint_names.index(n) for n in spec['arm_joints']]; flange = robot.body_names.index(spec['flange'])
    gi = robot.joint_names.index('finger_joint')
    pad_ids = {side: robot.body_names.index(pad['body'].rsplit('/', 1)[-1]) for side, pad in pads.items()}
    def pad_poses():
        poses = array(robot.data.body_link_pose_w)[0]
        return {side: pose_matrix(poses[i, :3], poses[i, 3:]) for side, i in pad_ids.items()}
    joints = {prim.GetName(): UsdPhysics.Joint(prim) for prim in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/Robot'))
              if prim.IsA(UsdPhysics.Joint)}
    parents = []; children = []; axes = []
    def frame(pos, q): return pose_matrix(pos, [*q.GetImaginary(), q.GetReal()])
    for name in spec['arm_joints']:
        j = joints[name]
        parents.append(frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()))
        children.append(frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))
        axes.append(np.eye(3)['XYZ'.index(j.GetPrim().GetAttribute('physics:axis').Get())])
    base = pose_matrix([0, 0, spec['base_height_m']], [0, 0, 0, 1]); orientation = Rotation.from_euler('x', np.pi).as_matrix()
    limits = array(robot.data.joint_pos_limits)[0, arm]
    def ik(target, q0):
        def residual(q):
            f = base@forward_chain(q, parents, children, axes)
            return np.r_[(f@np.r_[offset, 1])[:3]-target, .2*rotation_error(orientation, f[:3, :3])]
        sol = least_squares(residual, q0, bounds=(limits[:, 0]+.03, limits[:, 1]-.03), max_nfev=400)
        if np.linalg.norm(residual(sol.x)) > .002:
            raise ValueError('Grasp IK failed')
        return sol.x
    q0 = ik(center+[0, 0, .16], np.array(IK_SEED)); qclose = ik(center, q0); qlift = ik(center+[0, 0, .14], qclose)
    pos = robot.data.default_joint_pos.torch.clone(); pos[0, arm] = torch.as_tensor(q0, device=sim.device, dtype=pos.dtype)
    robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=pos)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset(); robot.update(dt); obj.update(dt); surface.update(dt)
    for s in taxels.values():
        print('TAXELS', s.cfg.prim_path, s.num_tactile_points, flush=True)
    meta = {'plant': spec, 'dt': dt, 'args': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
            'tactile': {'model': 'TacSL force field (isaaclab_contrib)', 'array_size': tactile.ARRAY_SIZE,
                        'margin_m': tactile.MARGIN, 'taxel_pos_pad_m': {k: tactile.taxel_positions(s).tolist() for k, s in taxels.items()},
                        'contact_backend': {'colliders': args.pad_colliders, 'grid': list(taxels_c.spec.shape),
                                            'taxel_centres_pad_m': taxels_c.spec.centers().tolist(), 'sensor_model': vars(model)},
                        'frame': 'pad frame: z into finger, rows along x (width), cols along y (to fingertip); force on pad'},
            'object_pose_writes_after_initialization': 0, 'controller_uses_tactile': False}
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    sim.stage.Export(str(out/'scene.usda'))

    def blend(x):
        x = np.clip(x, 0, 1); return x*x*(3-2*x)
    records = []; since = None; success = False; success_t = None
    for step in range(round(args.seconds/dt)):
        t = step*dt
        if t < 6: qref = q0+(qclose-q0)*blend((t-1)/5); phase = 'approach'
        elif t < 8: qref = qclose; phase = 'closing'
        else: qref = qclose+(qlift-qclose)*blend((t-8)/4); phase = 'lift_hold'
        targets = pos.clone(); targets[0, arm] = torch.as_tensor(qref, device=sim.device, dtype=targets.dtype)
        targets[0, gi] = 0. if args.keep_open else .725*blend((t-6)/1.5)
        robot.set_joint_position_target_index(target=targets)
        # Arm gravity feedforward; bounded explicit PD on the gripper driver joint.
        effort = torch.zeros_like(targets)
        effort[0, arm] = torch.as_tensor(array(robot.data.gravity_compensation_forces)[0, robot.num_base_dofs:][arm],
                                         device=sim.device, dtype=targets.dtype)
        gq = float(array(robot.data.joint_pos)[0, gi]); gv = float(array(robot.data.joint_vel)[0, gi])
        effort[0, gi] = np.clip(2.*(float(targets[0, gi])-gq)-.04*gv, -2., 2.)
        robot.set_joint_effort_target_index(target=effort); robot.write_data_to_sim()
        sim.step(render=not args.headless); robot.update(dt); obj.update(dt); surface.update(dt)
        for s in [*taxels.values(), *contacts.values(), support]:
            s.update(dt)
        cframes = taxels_c.update(pad_poses(), dt)
        forces = {k: float(np.linalg.norm(array(c.data.force_matrix_w))) for k, c in contacts.items()}
        support_force = float(np.linalg.norm(array(support.data.force_matrix_w)))
        op = array(obj.data.root_pos_w)[0]; tool, _ = tool_point(array(robot.data.body_link_pose_w)[0, flange], offset)
        retained = bool(t > 8 and op[2]-center[2] > .06 and min(forces.values()) > .05 and support_force < .02
                        and np.linalg.norm(op-tool) < .04)
        since = (t if since is None else since) if retained else None
        if since is not None and t-since >= 2 and not success:
            success = True; success_t = t
        if step % 20 == 0:
            snap = {k: tactile.read(s) for k, s in taxels.items()}
            rec = {'t': t, 'phase': phase, 'object': op.tolist(), 'tool': tool.tolist(), 'pad_contact_force_n': forces,
                   'support_force_n': support_force, 'gripper_q': gq, 'retained': retained,
                   'tacsl': {k: {'normal_sum': float(v['normal'].sum()), 'shear_sum': v['shear'].sum((0, 1)).tolist(),
                                 'max_depth_m': float(v['depth'].max()), 'active_taxels': int((v['depth'] > 0).sum())}
                             for k, v in snap.items()},
                   'contact_taxels': {k: {'normal_sum': f.total_normal, 'shear_sum': f.total_shear.tolist(),
                                          'active_taxels': int((f.normal > 0).sum()), 'num_contacts': f.num_contacts,
                                          **f.meta}
                                      for k, f in cframes.items()}}
            records.append(rec)
            np.savez_compressed(out/'tactile'/f'{step//20:06d}.npz', t=t,
                                **{f'tacsl_{k}_{f}': v[f] for k, v in snap.items() for f in v},
                                **{f'contact_{k}_{f}': getattr(v, f) for k, v in cframes.items() for f in ('normal', 'shear')})
            if step % 400 == 0:
                print('STEP', f'{t:.2f}', phase, 'z', round(op[2]-center[2], 4), 'pad_N', {k: round(v, 3) for k, v in forces.items()},
                      'tacsl', {k: (r['active_taxels'], round(r['normal_sum'], 4)) for k, r in rec['tacsl'].items()},
                      'contact', {k: (r['active_taxels'], round(r['normal_sum'], 3)) for k, r in rec['contact_taxels'].items()}, flush=True)
    result = {'success': success, 'success_t': success_t, 'keep_open': args.keep_open, 'records': records,
              'gate': '2 continuous seconds: lift >6cm, both pad contacts >.05N, no support >.02N, tool-object distance <4cm'}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
    return 0 if success else 3


code = 1
try:
    code = main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close(exit_code=code)
raise SystemExit(code)
