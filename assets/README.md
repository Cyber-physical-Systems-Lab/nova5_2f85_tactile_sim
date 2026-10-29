# Assets

Copied on 2026-10-29 from `~/go2_mycobot_ral` (commit `bb30a8c`), unchanged.

| Path | Source in go2_mycobot_ral | Notes |
|---|---|---|
| `nova5_usd/` | `data/assets/nova5_isaac/` | NOVA5 USD converted from Dobot `DOBOT_6Axis_ROS2_V4` (commit `ec201c4`) `nova5_robot.xacro`. Vendor inertias/meshes, fixed base. Vendor zero effort limits are replaced in code by synthetic 60/15 Nm and 1.5 rad/s; not hardware specs. `config.yaml` records the original conversion (absolute paths refer to the old location). Upstream licence: MIT. |
| `robot_description/urdf/nova5_robot_2F85.urdf` | `data/assets/all_robot/robot_description/urdf/` | Source of the Robotiq 2F-85 links, inertias, joints and mimic relations. Only the gripper part is used at runtime; the arm comes from the USD. |
| `robot_description/meshes/robotiq_2f85/` | same package | Fine visual and coarse collision STLs. |
| `robot_description/meshes/nova5/` | same package | Arm STLs referenced by the URDF (kept so the URDF resolves; not used by the Isaac scene). |
| `robot_description/LICENSE` | same package | BSD-style license of the `robot_description` package. |

Mimic-joint limit signs in the URDF are inconsistent for negative multipliers;
`nova5_tactile/robotiq.py` repairs them at load time and records the repair in
each run's `metadata.json`.

## ManipulationNet Peg-in-Hole CAD

`manipulationnet/peg_in_hole/` holds the NIST ATB M1 task board and the five
pegs, downloaded unchanged on 2026-10-29 from
[ManipulationNet/mnet_client](https://github.com/ManipulationNet/mnet_client)
(branch `ros_2`, commit `2745ccc`, `assets/peg_in_hole/cad_files`), with that
repository's Apache-2.0 `LICENSE`. The STLs are in centimetres.

## Licences

| Asset | Licence |
|---|---|
| `nova5_usd/` (from Dobot-Arm/DOBOT_6Axis_ROS2_V4) | MIT |
| `robot_description/` | BSD-3-Clause (`robot_description/LICENSE`) |
| `manipulationnet/` | Apache-2.0 (`manipulationnet/peg_in_hole/LICENSE`) |

`generated/` is built from these by `scripts/build_assets.py` and is not committed.
