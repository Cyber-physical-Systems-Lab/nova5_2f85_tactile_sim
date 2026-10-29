"""ManipulationNet Peg-in-Hole (NIST ATB M1) as an Isaac Lab DirectRLEnv.

One benchmark sub-task per env config (shape A-E, clearance level 1-4). The
episode starts with the matching steel peg already grasped above its hole
(as in Isaac Lab's Factory peg insert); the policy moves the fingertip with
small Cartesian + yaw steps, solved to joint targets by damped least squares.
The benchmark's grasp-from-feeder phase and its 20-task sequence are not
part of this RL env.

All environments reset together (episodes end only on timeout), which lets
the reset step the simulation to settle the grasp, like Factory does.
"""
import json
import carb
import numpy as np
import torch
from pxr import Usd, UsdPhysics
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as tm
from ..kinematics import pose_matrix, forward_chain, rotation_error
from ..gripper_kinematics import mimic_multipliers
from .peg_in_hole_env_cfg import PegInHoleEnvCfg, GENERATED, TABLE_TOP, HOLE_XY, ROBOT_BASE_HEIGHT
from . import peg_in_hole as pih

DOWN_ROLL = np.pi


def squash(x, a, b):
    return 1/(torch.exp(a*x)+b+torch.exp(-a*x))


class PegInHoleEnv(DirectRLEnv):
    cfg: PegInHoleEnvCfg

    def __init__(self, cfg: PegInHoleEnvCfg, render_mode=None, **kwargs):
        self.meta = json.loads((GENERATED/'peg_in_hole.json').read_text())
        super().__init__(cfg, render_mode, **kwargs)
        r = self.robot
        self.arm_ids = [r.joint_names.index(n) for n in self.meta['robot']['arm_joints']]
        mult = mimic_multipliers()
        self.grip_ids = [r.joint_names.index(n) for n in mult]
        self.grip_mult = torch.tensor(list(mult.values()), device=self.device)
        self.driver = r.joint_names.index('finger_joint')
        self.pad_ids = [r.body_names.index(n) for n in self.meta['robot']['pads'].values()]
        self.flange_id = r.body_names.index(self.meta['robot']['flange'])
        jac = r.data.body_link_jacobian_w.torch
        self.jac_offset = 1 if jac.shape[1] == len(r.body_names)-1 else 0
        peg = self.meta['pegs'][cfg.task[0]]; hole = self.meta['holes'][cfg.task]
        self.peg_len, self.grip_yaw, self.grip_width = peg['length_m'], peg['grip_yaw'], peg['grip_width_m']
        self.hole_yaw = hole['yaw']
        gap = np.array(self.meta['robot']['pad_gap_table'])
        self.q_open = float(np.interp(self.grip_width+.006, gap[::-1, 1], gap[::-1, 0]))
        # Pads pre-shaped 0.3 mm inside the peg width; the closing effort then holds it.
        self.q_grasp = float(np.interp(self.grip_width-.0003, gap[::-1, 1], gap[::-1, 0]))
        # Hole top centre (env frame) and the fingertip yaw that aligns the peg with the hole.
        self.hole_top = torch.tensor([HOLE_XY[0], HOLE_XY[1], TABLE_TOP+pih.BOARD_THICKNESS], device=self.device)
        self.goal_yaw = float(np.arctan2(np.sin(self.hole_yaw+self.grip_yaw), np.cos(self.hole_yaw+self.grip_yaw)))
        self.peg_top_above_pads = .005  # peg top sticks out this far above the pad centres
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_success = torch.zeros_like(self.success)
        self.sync_step = 0
        self.q_home = self._home_joints()

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.peg = RigidObject(self.cfg.peg)
        hole_c = np.array(self.meta['holes'][self.cfg.task]['center_m'])
        board_pos = (HOLE_XY[0]-hole_c[0], HOLE_XY[1]-hole_c[1], TABLE_TOP)
        cfg = sim_utils.UsdFileCfg(usd_path=str(GENERATED/f'board_{self.cfg.task}.usd'))
        cfg.func('/World/envs/env_.*/Board', cfg, translation=board_pos)
        table = sim_utils.CuboidCfg(size=(.6, .6, .05), collision_props=sim_utils.CollisionPropertiesCfg(),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.30, .22, .14)))
        table.func('/World/envs/env_.*/Table', table, translation=(HOLE_XY[0], HOLE_XY[1], TABLE_TOP-.025))
        legs = sim_utils.CuboidCfg(size=(.5, .5, TABLE_TOP-.05),
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .25, .27)))
        legs.func('/World/envs/env_.*/TableBase', legs, translation=(HOLE_XY[0], HOLE_XY[1], (TABLE_TOP-.05)/2))
        ped = sim_utils.CuboidCfg(size=(.18, .18, ROBOT_BASE_HEIGHT),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.2, .2, .2)))
        ped.func('/World/envs/env_.*/Pedestal', ped, translation=(0., 0., ROBOT_BASE_HEIGHT/2))
        spawn_ground_plane('/World/ground', GroundPlaneCfg())
        if self.cfg.screenshot_camera:
            from isaaclab.sensors.camera import Camera, CameraCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            w, h = self.cfg.viewer.resolution
            self.shot_cam = Camera(CameraCfg(prim_path='/World/ShotCamera', height=h, width=w, data_types=['rgb'],
                                             update_period=0., renderer_cfg=IsaacRtxRendererCfg(),
                                             spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955,
                                                                              clipping_range=(.01, 100.))))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot'] = self.robot
        self.scene.rigid_objects['peg'] = self.peg
        light = sim_utils.DomeLightCfg(intensity=2000., color=(.8, .8, .8))
        light.func('/World/Light', light)

    def _home_joints(self):
        """Joint angles with the fingertip 6 cm above the hole, pointing down, aligned with the hole."""
        stage = self.sim.stage
        root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
        joints = {p.GetName(): UsdPhysics.Joint(p) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)}
        def frame(pos, q): return pose_matrix(pos, [*q.GetImaginary(), q.GetReal()])
        parents, children, axes = [], [], []
        for n in self.meta['robot']['arm_joints']:
            j = joints[n]
            parents.append(frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()))
            children.append(frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))
            axes.append(np.eye(3)['XYZ'.index(j.GetPrim().GetAttribute('physics:axis').Get())])
        base = pose_matrix([0, 0, ROBOT_BASE_HEIGHT], [0, 0, 0, 1])
        target = self.hole_top.cpu().numpy()+[0, 0, .06+self.peg_len]
        R = Rotation.from_euler('xyz', [DOWN_ROLL, 0, self.goal_yaw]).as_matrix()
        offset = np.array([0, 0, .146])  # pad centres in the flange frame (gripper half closed)
        lim = self.robot.data.joint_pos_limits.torch[0, self.arm_ids].cpu().numpy()
        def res(q):
            f = base@forward_chain(q, parents, children, axes)
            return np.r_[(f@np.r_[offset, 1])[:3]-target, .2*rotation_error(R, f[:3, :3])]
        q0 = self.robot.data.default_joint_pos.torch[0, self.arm_ids].cpu().numpy()
        sol = least_squares(res, q0, bounds=(lim[:, 0]+.03, lim[:, 1]-.03), max_nfev=500)
        if np.linalg.norm(res(sol.x)) > .003:
            raise RuntimeError(f'Home IK failed, residual {np.linalg.norm(res(sol.x)):.4f}')
        return torch.tensor(sol.x, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------ kinematics
    def _fingertip(self):
        pos = self.robot.data.body_link_pos_w.torch[:, self.pad_ids].mean(1)-self.scene.env_origins
        quat = self.robot.data.body_link_quat_w.torch[:, self.flange_id]
        return pos, quat

    def _fingertip_jacobian(self):
        J = self.robot.data.body_link_jacobian_w.torch
        ids = [i-self.jac_offset for i in self.pad_ids]
        return J[:, ids].mean(1)[:, :, self.arm_ids]

    def _target_quat(self, yaw):
        z = torch.zeros_like(yaw)
        return tm.quat_from_euler_xyz(z+DOWN_ROLL, z, yaw)

    def _ik_targets(self, pos_target, yaw_target):
        pos, quat = self._fingertip()
        err_q = tm.quat_mul(self._target_quat(yaw_target), tm.quat_conjugate(quat))
        err = torch.cat([pos_target-pos, tm.axis_angle_from_quat(err_q)], -1)
        J = self._fingertip_jacobian()
        JT = J.transpose(1, 2)
        lam = self.cfg.ik_damping**2*torch.eye(6, device=self.device)
        dq = (JT@torch.linalg.solve(J@JT+lam, err.unsqueeze(-1))).squeeze(-1)
        dq = dq*torch.clamp(self.cfg.max_joint_step/dq.abs().amax(-1, keepdim=True).clamp_min(1e-9), max=1.)
        return self.robot.data.joint_pos.torch[:, self.arm_ids]+dq

    def _yaw(self, quat):
        return tm.euler_xyz_from_quat(quat)[2]

    def _grip_positions(self, q):
        return q.unsqueeze(-1)*self.grip_mult

    # --------------------------------------------------------------- control
    def _pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clamp(-1, 1)
        pos, quat = self._fingertip()
        target = pos+self.actions[:, :3]*self.cfg.pos_action_scale
        lo = self.hole_top-torch.tensor([self.cfg.pos_bounds[0], self.cfg.pos_bounds[1], -self.cfg.min_tip_height], device=self.device)
        hi = self.hole_top+torch.tensor([*self.cfg.pos_bounds[:2], self.cfg.pos_bounds[2]], device=self.device)
        target = torch.max(torch.min(target, hi), lo)
        yaw = tm.wrap_to_pi(self._yaw(quat)+self.actions[:, 3]*self.cfg.yaw_action_scale)
        self.arm_target = self._ik_targets(target, yaw)

    def _apply_action(self):
        target = self.robot.data.joint_pos.torch.clone()
        target[:, self.arm_ids] = self.arm_target
        self.robot.set_joint_position_target_index(target=target)
        self._hold_grip()

    def _hold_grip(self, effort=None):
        e = torch.zeros_like(self.robot.data.joint_pos.torch)
        e[:, self.driver] = self.cfg.grip_effort if effort is None else effort
        self.robot.set_joint_effort_target_index(target=e)

    # ---------------------------------------------------------- observations
    def _peg_state(self):
        p = self.peg.data.root_pos_w.torch-self.scene.env_origins
        q = self.peg.data.root_quat_w.torch
        return p, q

    def _get_observations(self):
        pos, quat = self._fingertip()
        peg_p, peg_q = self._peg_state()
        yaw_err = tm.wrap_to_pi(self._yaw(quat)-self.goal_yaw)
        peg_yaw_err = tm.wrap_to_pi(self._yaw(peg_q)-self.hole_yaw)
        lin = self.robot.data.body_link_lin_vel_w.torch[:, self.pad_ids].mean(1)
        obs = torch.cat([pos-self.hole_top, torch.sin(yaw_err)[:, None], torch.cos(yaw_err)[:, None], lin,
                         peg_p-self.hole_top, torch.sin(peg_yaw_err)[:, None], torch.cos(peg_yaw_err)[:, None],
                         self.actions], -1)
        return {'policy': obs}

    # --------------------------------------------------------------- rewards
    def _keypoints(self, pos, yaw):
        """Peg bottom centre and two points 1 cm out along the peg x/y axes, for a peg at pos with yaw."""
        offs = torch.tensor([[0, 0, 0], [.01, 0, 0], [0, .01, 0]], device=self.device, dtype=torch.float32)
        z = torch.zeros_like(yaw)
        q = tm.quat_from_euler_xyz(z, z, yaw)
        return pos[:, None]+tm.quat_apply(q[:, None].expand(-1, 3, -1), offs[None].expand(len(pos), -1, -1))

    def _get_rewards(self):
        peg_p, peg_q = self._peg_state()
        goal = self.hole_top-torch.tensor([0, 0, pih.BOARD_THICKNESS], device=self.device)
        kp = self._keypoints(peg_p, self._yaw(peg_q))
        kg = self._keypoints(goal.expand(self.num_envs, -1), torch.full((self.num_envs,), self.hole_yaw, device=self.device))
        d = torch.linalg.norm(kp-kg, dim=-1).mean(-1)
        xy = torch.linalg.norm(peg_p[:, :2]-self.hole_top[:2], dim=-1)
        depth = self.hole_top[2]-peg_p[:, 2]
        self.success = (depth >= self.cfg.success_depth) & (xy < .01)
        self.ever_success |= self.success
        rew = squash(d, 5., 4.)+squash(d, 50., 2.)+squash(d, 100., 0.)
        rew = rew+self.cfg.success_bonus*self.success.float()
        rew = rew-self.cfg.action_penalty*(self.actions**2).sum(-1)
        self.extras.setdefault('log', {}).update({'Metrics/keypoint_dist_m': d.mean().item(),
                                                   'Metrics/depth_m': depth.mean().item(),
                                                   'Metrics/inserted_now': self.success.float().mean().item()})
        return rew

    def _get_dones(self):
        # One shared episode clock: every env times out on the same step, because the reset
        # steps the whole simulation (trainers that randomise initial episode lengths are ignored).
        self.sync_step += 1
        done = self.sync_step >= self.max_episode_length
        time_out = torch.full((self.num_envs,), bool(done), device=self.device)
        return torch.zeros_like(time_out), time_out

    # ----------------------------------------------------------------- reset
    def _step_no_action(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

    def _write_joints(self, arm_q, grip_q, env_ids):
        q = self.robot.data.joint_pos.torch[env_ids].clone()
        q[:, self.arm_ids] = arm_q
        q[:, self.grip_ids] = self._grip_positions(grip_q)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=env_ids)

    def _reset_idx(self, env_ids):
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device).long()
        if len(env_ids) and self.episode_length_buf.numel():
            self.extras.setdefault('log', {})['Metrics/success_rate'] = self.ever_success[env_ids].float().mean().item()
        super()._reset_idx(env_ids)
        n = len(env_ids)
        view = sim_utils.SimulationContext.instance().physics_manager.get_physics_sim_view()
        view.set_gravity(carb.Float3(0., 0., 0.))
        # Park the peg out of the way, put the arm at home with the gripper opened for the peg.
        park = torch.zeros(n, 7, device=self.device)
        park[:, :3] = self.scene.env_origins[env_ids]+self.hole_top+torch.tensor([0., .25, .3], device=self.device)
        park[:, 6] = 1.
        self.peg.write_root_pose_to_sim_index(root_pose=park, env_ids=env_ids)
        self.peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        self._write_joints(self.q_home.expand(n, -1), torch.full((n,), self.q_open, device=self.device), env_ids)
        self._step_no_action()
        # Randomised start: peg bottom 1-3 cm above the board, xy and yaw noise.
        rnd = lambda *s: 2*torch.rand(*s, device=self.device)-1
        lift = self.cfg.start_height[0]+(self.cfg.start_height[1]-self.cfg.start_height[0])*torch.rand(self.num_envs, device=self.device)
        tgt = self.hole_top.expand(self.num_envs, -1).clone()
        tgt[:, :2] += self.cfg.start_xy_noise*rnd(self.num_envs, 2)
        tgt[:, 2] += lift+self.peg_len-self.peg_top_above_pads
        yaw = tm.wrap_to_pi(torch.full((self.num_envs,), self.goal_yaw, device=self.device)+self.cfg.start_yaw_noise*rnd(self.num_envs))
        for _ in range(40):  # ~0.33 s of damped-least-squares servoing
            q = self._ik_targets(tgt, yaw)
            self._write_joints(q[env_ids], torch.full((n,), self.q_open, device=self.device), env_ids)
            self._step_no_action()
        # Put the peg between the pads, aligned with the closing direction, then close.
        pos, quat = self._fingertip()
        peg_yaw = tm.wrap_to_pi(self._yaw(quat)-self.grip_yaw)
        z = torch.zeros_like(peg_yaw)
        pose = torch.cat([pos+self.scene.env_origins, tm.quat_from_euler_xyz(z, z, peg_yaw)], -1)
        pose[:, 2] += self.peg_top_above_pads-self.peg_len
        self.peg.write_root_pose_to_sim_index(root_pose=pose[env_ids], env_ids=env_ids)
        self.peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        arm_q = self.robot.data.joint_pos.torch[env_ids][:, self.arm_ids]
        self._write_joints(arm_q, torch.full((n,), self.q_grasp, device=self.device), env_ids)
        for _ in range(30):  # 0.25 s squeeze
            self._hold_grip()
            self._step_no_action()
        view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        self.actions.zero_(); self.prev_actions.zero_()
        self.ever_success[env_ids] = False
        self.sync_step = 0
        self.arm_target = self.robot.data.joint_pos.torch[:, self.arm_ids].clone()


    def screenshot(self, warmup=10):
        """RGB uint8 image of env 0 from cfg.viewer eye/lookat (requires cfg.screenshot_camera)."""
        o = self.scene.env_origins[0]
        eye = torch.tensor(self.cfg.viewer.eye, device=self.device)+o
        look = torch.tensor(self.cfg.viewer.lookat, device=self.device)+o
        self.shot_cam.set_world_poses_from_view(eye[None], look[None])
        for _ in range(warmup):
            self.sim.render()
            self.shot_cam.update(self.physics_dt)
        rgb = self.shot_cam.data.output['rgb']
        rgb = rgb.torch if hasattr(rgb, 'torch') else rgb
        return rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
