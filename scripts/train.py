"""Run Isaac Lab's RL train script with this repo's tasks registered.

  bash scripts/isaac_python.sh scripts/train.py --rl_library rsl_rl --task Nova5-MNet-PegInHole-A1-v0 --headless [...]

All arguments are passed to IsaacLab/scripts/reinforcement_learning/train.py.
"""
import os
import runpy
import sys

LAB = os.path.join(os.environ.get('ISAACLAB_PATH', os.path.expanduser('~/IsaacLab')), 'scripts/reinforcement_learning')
sys.path.insert(0, LAB)
argv = sys.argv[1:]
if not any(a.startswith('--external_callback') for a in argv):
    argv += ['--external_callback', 'nova5_tactile.tasks.register']
sys.argv = [os.path.join(LAB, 'train.py')]+argv
runpy.run_path(sys.argv[0], run_name='__main__')
