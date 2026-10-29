"""NOVA5 + Robotiq 2F-85 with fingertip tactile pads for Isaac Sim / Isaac Lab.

Modules that import ``pxr`` or ``isaaclab`` must be imported after AppLauncher.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
NOVA5_USD = ASSETS / 'nova5_usd/nova5/nova5.usda'
GRIPPER_URDF = ASSETS / 'robot_description/urdf/nova5_robot_2F85.urdf'
