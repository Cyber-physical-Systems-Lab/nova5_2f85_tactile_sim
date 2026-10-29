"""Gymnasium registration of this repo's Isaac Lab tasks.

Nova5-MNet-PegInHole-<shape><level>-v0, e.g. Nova5-MNet-PegInHole-A1-v0, for
ManipulationNet Peg-in-Hole shapes A-E (circle, hexagon, triangle, rectangle,
L) and clearance levels 1-4 (3, 1, 0.1, 0.02 mm).

Pass `--external_callback nova5_tactile.tasks.register` to Isaac Lab's
scripts/reinforcement_learning/train.py or play.py, or use scripts/train.py.
"""
import gymnasium as gym

TASKS = [f'{s}{l}' for s in 'ABCDE' for l in '1234']


def task_id(task):
    return f'Nova5-MNet-PegInHole-{task}-v0'


def register():
    for t in TASKS:
        if task_id(t) in gym.registry:
            continue
        gym.register(id=task_id(t), entry_point='nova5_tactile.mnet.peg_in_hole_env:PegInHoleEnv', disable_env_checker=True,
                     kwargs={'env_cfg_entry_point': f'nova5_tactile.tasks:cfg_{t}',
                             'rsl_rl_cfg_entry_point': 'nova5_tactile.mnet.agents.rsl_rl_ppo_cfg:PegInHolePPORunnerCfg'})
    return None


def __getattr__(name):
    # cfg_A1 ... cfg_E4: env config classes resolved lazily (they import Isaac Lab).
    if name.startswith('cfg_') and name[4:] in TASKS:
        from .mnet.peg_in_hole_env_cfg import PegInHoleEnvCfg
        from isaaclab.utils.configclass import configclass
        task = name[4:]
        return configclass(type(f'PegInHole{task}EnvCfg', (PegInHoleEnvCfg,), {'__annotations__': {'task': str}, 'task': task}))
    raise AttributeError(name)
