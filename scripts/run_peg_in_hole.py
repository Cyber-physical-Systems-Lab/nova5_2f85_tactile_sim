"""Run a ManipulationNet Peg-in-Hole env with a scripted or zero policy (no training).

  bash scripts/isaac_python.sh scripts/run_peg_in_hole.py --task A1 --num_envs 16 --headless
  bash scripts/isaac_python.sh scripts/run_peg_in_hole.py --task A1 --num_envs 4 --headless --enable_cameras --screenshot docs/media/a1.png

The scripted policy servos the peg over the hole (from the env's own
observations), turns the peg to the hole's yaw, then pushes down. It is a sanity check and
baseline, not a benchmark result.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--task', default='A1', help='shape A-E + level 1-4')
p.add_argument('--num_envs', type=int, default=16)
p.add_argument('--episodes', type=int, default=1)
p.add_argument('--policy', choices=['scripted', 'zero'], default='scripted')
p.add_argument('--screenshot', help='save an RGB frame here (needs --enable_cameras)')
p.add_argument('--screenshot_step', type=int, default=60)
p.add_argument('--trace', type=int, default=0, help='print env 0 state for the first N steps')
p.add_argument('--grip_effort', type=float, default=None, help='override cfg.grip_effort (N m)')
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from nova5_tactile import tasks
from nova5_tactile.mnet.peg_in_hole_env_cfg import PegInHoleEnvCfg


def scripted(obs, cfg):
    # Align the peg itself (it can turn in the hand), not the fingertip.
    peg = obs[:, 8:11]; s, c = obs[:, 11], obs[:, 12]
    a = torch.zeros(len(obs), 4, device=obs.device)
    a[:, :2] = (-peg[:, :2]/cfg.pos_action_scale).clamp(-1, 1)
    yaw_err = torch.atan2(s, c)
    a[:, 3] = (-yaw_err/cfg.yaw_action_scale).clamp(-1, 1)
    aligned = (peg[:, :2].norm(dim=-1) < .0008) & (yaw_err.abs() < .005)
    a[:, 2] = torch.where(aligned, -.6, torch.where(peg[:, 2] < .004, .3, 0.))
    return a


def main():
    tasks.register()
    cfg = PegInHoleEnvCfg(task=args.task)
    cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device
    if args.grip_effort is not None:
        cfg.grip_effort = args.grip_effort
    cfg.screenshot_camera = bool(args.screenshot)
    env = gym.make(tasks.task_id(args.task), cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    if args.trace:
        tip, _ = u._fingertip(); pp, _ = u._peg_state()
        print('AFTER_RESET tip', (tip[0]*1e3).round().tolist(), 'peg', (pp[0]*1e3).round().tolist(), 'hole_top', (u.hole_top*1e3).round().tolist(),
              'grip_q', round(float(u.robot.data.joint_pos.torch[0, u.driver]), 3), 'q_open', round(u.q_open, 3), 'q_grasp', round(u.q_grasp, 3), flush=True)
    for ep in range(args.episodes):
        for step in range(int(u.max_episode_length)):
            act = scripted(obs['policy'], cfg) if args.policy == 'scripted' else torch.zeros(u.num_envs, 4, device=u.device)
            if step == 0:
                st = obs['policy']
                print('START peg rel hole (mm) mean', (st[:, 8:11].mean(0)*1e3).tolist(), flush=True)
            prev = {'tip': u._fingertip()[0].clone(), 'peg': u._peg_state()[0].clone(), 'q': u.robot.data.joint_pos.torch.clone(), 'act': act.clone()}
            obs, rew, term, trunc, extras = env.step(act)
            bad = ~torch.isfinite(u.robot.data.joint_pos.torch).all(-1) | ~torch.isfinite(u._peg_state()[0]).all(-1)
            if bad.any() and not getattr(u, '_nan_reported', False):
                u._nan_reported = True; i = int(bad.nonzero()[0])
                print('NAN step', step, 'envs', bad.nonzero().flatten().tolist(), 'env', i, 'prev tip', (prev['tip'][i]*1e3).round().tolist(),
                      'prev peg', (prev['peg'][i]*1e3).round().tolist(), 'prev q', prev['q'][i].round(decimals=3).tolist(), 'act', prev['act'][i].tolist(), flush=True)
            ever = u.success.clone() if step == 0 else ever | u.success
            if args.trace and step < args.trace:
                tip, _ = u._fingertip(); pp, _ = u._peg_state()
                print('TRACE', step, 'tip', (tip[0]*1e3).round().tolist(), 'peg', (pp[0]*1e3).round().tolist(),
                      'grip_q', round(float(u.robot.data.joint_pos.torch[0, u.driver]), 3), 'tau', [round(float(x), 3) for x in u.robot.data.applied_torque.torch[0, u.grip_ids]], 'act', act[0].round(decimals=2).tolist(),
                      'rew', float(rew[0]), 'q-target', (u.robot.data.joint_pos.torch[0, u.arm_ids]-u.arm_target[0]).abs().max().item(), flush=True)
            if args.screenshot and ep == 0 and step == args.screenshot_step:
                from PIL import Image
                img = u.screenshot()
                print('SCREENSHOT', args.screenshot, 'shape', getattr(img, 'shape', None), 'mean', float(np.asarray(img).mean()), flush=True)
                Image.fromarray(img).save(args.screenshot)
            if step % 30 == 0 or step == u.max_episode_length-2:
                peg = obs['policy'][:, 8:11]
                if args.trace:
                    o = obs['policy'][0]
                    print('YAW tip_err_deg', round(float(torch.atan2(o[3], o[4]).rad2deg()), 2), 'peg_err_deg', round(float(torch.atan2(o[11], o[12]).rad2deg()), 2), flush=True)
                print(f'EP {ep} step {step} reward {rew.mean():.3f} peg rel hole mm xy {peg[:, :2].norm(dim=-1).mean()*1e3:.2f} '
                      f'z {peg[:, 2].mean()*1e3:.1f} inserted {u.success.float().mean():.2f} ever {u.ever_success.float().mean():.2f}', flush=True)
        print('EPISODE', ep, 'success_rate', float(ever.float().mean()), 'non_finite_envs', int((~torch.isfinite(u._peg_state()[0]).all(-1)).sum()), flush=True)
    env.close()
    return 0


code = 1
try:
    code = main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close(exit_code=code)
raise SystemExit(code)
