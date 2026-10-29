# Assets

Copied on 2026-10-29 from `~/go2_mycobot_ral` (commit `bb30a8c`), unchanged.

| Path | Source in go2_mycobot_ral | Notes |
|---|---|---|
| `nova5_usd/` | `data/assets/nova5_isaac/` | NOVA5 USD converted from Dobot `DOBOT_6Axis_ROS2_V4` (commit `ec201c4`) `nova5_robot.xacro`. Vendor inertias/meshes, fixed base. Vendor zero effort limits are replaced in code by synthetic 60/15 Nm and 1.5 rad/s; not hardware specs. `config.yaml` records the original conversion (absolute paths refer to the old location). |
| `robot_description/urdf/nova5_robot_2F85.urdf` | `data/assets/all_robot/robot_description/urdf/` | Source of the Robotiq 2F-85 links, inertias, joints and mimic relations. Only the gripper part is used at runtime; the arm comes from the USD. |
| `robot_description/meshes/robotiq_2f85/` | same package | Fine visual and coarse collision STLs. |
| `robot_description/meshes/nova5/` | same package | Arm STLs referenced by the URDF (kept so the URDF resolves; not used by the Isaac scene). |
| `robot_description/LICENSE` | same package | BSD-style license of the `robot_description` package. |

Mimic-joint limit signs in the URDF are inconsistent for negative multipliers;
`nova5_tactile/robotiq.py` repairs them at load time and records the repair in
each run's `metadata.json`.
