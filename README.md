# nova5_2f85_tactile_sim

Isaac Sim development workspace for a **Dobot Nova5** arm with a **Robotiq 2F-85**
gripper and **taxel tactile sensors on both fingertips**.

Target: **Isaac Sim 6.2** (Core *Experimental* API, PhysX backend, Python 3.12).
The URDF importer step also has a 4.5/5.x fallback.

> **Status.** The pure-Python core (taxel model, frames, sign handling, pad
> placement, config, URDF prep, recorder) is covered by unit tests (`make test`).
> The Isaac Sim / USD code has **not yet been run against a live Isaac Sim**.
> Expect to adjust asset paths, link names and possibly small API details on
> first run; every such value lives in `config/default.toml`.

## Quick start

```bash
cd ~/nova5_2f85_tactile_sim
export ISAACSIM_PATH=~/isaacsim          # folder containing python.sh (or set ISAACSIM_PYTHON)

make test                # 1. pure-Python tests, no GPU needed
make urdf                # 2. clone Dobot ROS 2 repo -> assets/urdf/nova5.urdf
make import              # 3. URDF -> USD (assets/usd/nova5/)
make inspect-nova5       # 4. check arm link/joint names
make inspect-gripper     #    check 2F-85 link/joint names, fix config if needed
make assemble            # 5. arm + gripper + pads -> assets/usd/nova5_2f85_tactile.usd
make inspect-assembled   #    pad frames + warnings
make demo                # 6. grasp demo (GUI); or: make demo-headless
make plot                # 7. plot the newest log in logs/
```

Pass extra arguments with `ARGS`, e.g.
`make assemble ARGS="--set tactile.rows=16 --set tactile.cols=8"`.

**No Nova5 URDF in the Dobot repo?** Use any Nova5 URDF/xacro you have:
`NOVA_URDF=/path/to/nova5.urdf make urdf`, or
`python3 scripts/prepare_urdf.py --input X --search-root ~/ros2_ws/src`.

## What the pipeline builds

```
/nova5_2f85                         ArticulationRootAPI (one articulation)
  nova5/       reference -> URDF-imported Nova5 (fixed base)
  gripper/     reference -> Robotiq 2F-85 USD, re-posed on the flange
  gripper_mount_joint               FixedJoint  Link6 -> gripper base (mount offset in config)
  Looks/tactile_pad_material        compliant-contact physics material
  .../left_inner_finger/tactile_pad/taxel_RR_CC    8x4 box colliders (default)
  .../right_inner_finger/tactile_pad/taxel_RR_CC
```

The assembler removes the gripper's own articulation root and world joints,
places the gripper so its base sits at `flange * mount`, enables PhysX contact
reports on the finger links, and writes `nova5_2f85_tactile.meta.json` with all
resolved prim paths so runtime code never searches the stage.

## Repository layout

```
config/default.toml          every tunable: paths, link names, mount, taxel grid, materials, demo
scripts/
  isaac_python.sh            runs scripts with Isaac Sim's Python + src/ on PYTHONPATH
  fetch_nova5_description.sh / prepare_urdf.py   vendor URDF -> importer-ready URDF
  import_nova5.py            URDF -> USD
  inspect_usd.py             bodies / joints / pads of any USD
  assemble_robot.py          build the combined asset
  run_grasp_demo.py          settle -> close on a cube -> release to gravity; prints, colours, logs
  plot_tactile_log.py        offline plots of logs/*.npz
src/nova5_tactile/
  config.py, math3d.py, urdf_tools.py            pure Python
  tactile/taxel_grid.py, sensor_model.py, pad_placement.py, recorder.py, ascii_viz.py   pure numpy
  tactile/contact_source.py, fingertip_sensor.py, colorize.py                        Isaac runtime
  usd/query.py, author.py, assemble.py           pxr asset pipeline
  sim/app.py, scene.py, pose.py, commander.py    standalone-sim helpers
tests/                       unittest (also runs under pytest)
docs/tactile_model.md        sensor model, sign conventions, validation checklist
```

## Using the sensors in your own scripts

```python
from nova5_tactile.tactile.fingertip_sensor import TwoFingerTactileRig, load_meta
from nova5_tactile.sim.pose import RigidBodyPoseSource

meta = load_meta("assets/usd/nova5_2f85_tactile.meta.json")
rig = TwoFingerTactileRig(meta, "/World/robot", RigidBodyPoseSource(stage), physics_dt=1/240)
rig.calibrate_pose_source()            # once, right after timeline.play()
...
simulation_app.update()
frames = rig.update(timestamp)         # {"left": TactileFrame, "right": TactileFrame}
frames["left"].normal                  # (rows, cols) N, row index increases towards the fingertip
frames["left"].shear                   # (rows, cols, 2) N in the pad frame
```

Frame conventions are in `docs/tactile_model.md`; read its validation
checklist before using the data for research.

## Things you will likely need to adjust

- **Gripper asset path** (`paths.gripper_usd[_candidates]`): asset locations
  change between Isaac Sim releases. Find the 2F-85 in the Content Browser and
  paste the URL into `paths.gripper_usd`.
- **Link names** (`arm.flange_link_candidates`, `gripper.base_link_candidates`,
  `tactile.left/right.link_candidates`): check with `make inspect-*`.
- **Mount offset** (`gripper.mount_xyz/rpy_deg`): your flange adapter / coupling.
- **Pad size and grid** (`tactile.size_x/size_y/rows/cols`): match your real sensor.
- **Home pose** (`arm.home_deg`): placeholder values.
- **Mimic joints**: Isaac Sim 6.0 changed how URDF mimic joints are authored
  (Newton mimic API). The Robotiq asset from the Isaac library is used as-is;
  if the fingers do not move symmetrically, inspect its joints first.

## Roadmap ideas

Isaac Lab environment (TacSL sensor as an alternative backend), ROS 2 bridge
publishing `TactileFrame`s, cuMotion/IK for approach motions, domain
randomisation of pad stiffness/friction/noise.

## Notes

- Generated assets, logs and `third_party/` are git-ignored; regenerate with the
  `make` targets. `.gitattributes` prepares Git LFS for assets you choose to commit.
- No licence file is included yet — add one before sharing.
- Parts of this scaffold were drafted with AI assistance (Claude). Review the
  code before relying on it, and follow your institution's guidelines on
  generative AI (Uppsala University: https://www.uu.se/en/staff/service-and-tools/ai).
