"""Gymnasium registration of this repo's Isaac Lab tasks.

ManipulationNet Peg-in-Hole, shapes A-E (circle, hexagon, triangle, rectangle, L) and
clearance levels 1-4 (3, 1, 0.1, 0.02 mm):

  Nova5-MNet-PegInHole-<shape><level>-v0        NOVA5 + CPS-lab tactile gripper, e.g. ...-A1-v0
  Nova5-MNet-PegInHole-<shape><level>-2F85-v0   NOVA5 + Robotiq 2F-85

Pass `--external_callback nova5_tactile.tasks.register` to Isaac Lab's
scripts/reinforcement_learning/train.py or play.py, or use scripts/train.py.
"""
import gymnasium as gym

TASKS = [f'{s}{l}' for s in 'ABCDE' for l in '1234']
VARIANTS = {'': 'cps_tactile', '2F85': 'robotiq_2f85'}


def task_id(task, variant=''):
    return f'Nova5-MNet-PegInHole-{task}-{variant+"-" if variant else ""}v0'


def register():
    for t in TASKS:
        for v in VARIANTS:
            if task_id(t, v) in gym.registry:
                continue
            gym.register(id=task_id(t, v), entry_point='nova5_tactile.mnet.peg_in_hole_env:PegInHoleEnv',
                         disable_env_checker=True,
                         kwargs={'env_cfg_entry_point': f'nova5_tactile.tasks:cfg_{t}{"_"+v if v else ""}',
                                 'rsl_rl_cfg_entry_point': 'nova5_tactile.mnet.agents.rsl_rl_ppo_cfg:PegInHolePPORunnerCfg'})
    return None


def __getattr__(name):
    # cfg_A1 ... cfg_E4 and cfg_A1_2F85 ...: env config classes resolved lazily (they import Isaac Lab).
    parts = name.split('_')
    if parts[0] == 'cfg' and len(parts) in (2, 3) and parts[1] in TASKS and (len(parts) == 2 or parts[2] in VARIANTS):
        from .mnet.peg_in_hole_env_cfg import PegInHoleEnvCfg
        from isaaclab.utils.configclass import configclass
        task, gripper = parts[1], VARIANTS[parts[2] if len(parts) == 3 else '']
        return configclass(type(f'PegInHole{task}{gripper}EnvCfg', (PegInHoleEnvCfg,),
                                {'__annotations__': {'task': str, 'gripper': str}, 'task': task, 'gripper': gripper}))
    raise AttributeError(name)
