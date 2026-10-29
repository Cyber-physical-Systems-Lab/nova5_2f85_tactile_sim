"""Nova5 + Robotiq 2F-85 + fingertip tactile sensing for Isaac Sim.

Sub-packages:
  tactile  - taxel model (pure numpy) + Isaac Sim runtime glue
  usd      - pxr authoring/query helpers used by the asset pipeline
  sim      - scene, joint commands and pose helpers for standalone runs

Modules that need Isaac Sim (``omni.*``, ``isaacsim.*``) import it lazily,
so the pure parts (config, math3d, tactile.taxel_grid, ...) can be unit
tested with a plain Python interpreter.
"""

__version__ = "0.1.0"
