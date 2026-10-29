# nova5_2f85_tactile_sim

Isaac Sim / Isaac Lab simulation of a **Dobot NOVA5** arm with a **Robotiq 2F-85**
gripper, set up for the **[ManipulationNet](https://manipulation-net.org) Peg-in-Hole**
benchmark task (NIST ATB M1 board and pegs) and for RL training with Isaac Lab.
Fingertip tactile sensing is included but is a separate, later work package
(see [Tactile](#tactile-work-in-progress)).

![NOVA5 + 2F-85 inserting the ManipulationNet cylinder peg](docs/media/peg_in_hole_A1.png)

## What is simulated

| | |
|---|---|
| Arm | NOVA5 from Dobot's ROS 2 description, fixed on a 0.70 m pedestal |
| Gripper | Robotiq 2F-85 built from its URDF CAD: 6 revolute finger joints coupled by PhysX mimic joints, one driven joint `finger_joint` (0–0.725 rad) |
| Task board | NIST ATB M1 from the ManipulationNet CAD: 5 shapes (A circle, B hexagon, C triangle, D rectangle, E L-shape) × 4 clearance levels (3, 1, 0.1, 0.02 mm), 304 steel pegs |
| RL tasks | `Nova5-MNet-PegInHole-<shape><level>-v0`, e.g. `Nova5-MNet-PegInHole-A1-v0` (20 tasks, A1…E4) |

The hole walls are taken from a cross-section of the official board CAD, so hole
size, position and orientation are the benchmark's. Their collision geometry is
built from exact convex pieces, because PhysX SDF voxels are much coarser than
the 0.02–0.1 mm clearances. Measured hole-minus-peg gaps (per side) are written
to `assets/generated/peg_in_hole.json` when the assets are built.

## 1. Install

Tested on Ubuntu 22.04 with an NVIDIA RTX A4500 (20 GB).

1. **NVIDIA driver** recent enough for Isaac Sim (check with `nvidia-smi`).
2. **Isaac Sim** (the version Isaac Lab 3.0 expects), unpacked to `~/isaacsim`.
   Follow NVIDIA's Isaac Sim workstation installation guide, then check it starts:
   ```bash
   ~/isaacsim/isaac-sim.sh
   ```
3. **Isaac Lab 3.x** cloned to `~/IsaacLab` and installed into a conda env.
   Follow the Isaac Lab "pip / conda" installation guide; this repo was developed
   with a conda env named `locomanipu` (Python 3.12) that has `isaaclab`,
   `isaaclab_rl`, `isaaclab_tasks`, `isaaclab_contrib` and `rsl_rl` installed.
   Check with:
   ```bash
   conda run -n locomanipu python -c "import isaaclab, isaaclab_rl, rsl_rl; print('ok')"
   ```
4. **This repository**:
   ```bash
   git clone https://github.com/Cyber-physical-Systems-Lab/nova5_2f85_tactile_sim.git ~/nova5_tactile_sim
   cd ~/nova5_tactile_sim
   ```

Other locations work through environment variables: `ISAAC_PATH`, `ISAACLAB_PATH`,
`TASK_CONDA_ROOT` (default `~/miniconda3`) and `TASK_CONDA_ENV` (default `locomanipu`).
Every Isaac command below goes through `scripts/isaac_python.sh`, which activates
the env, sources Isaac Sim's Python setup and puts this repo on `PYTHONPATH`.

## 2. Build the simulation assets (once, ~1 min)

```bash
bash scripts/isaac_python.sh scripts/build_assets.py
```

This writes `assets/generated/` (git-ignored): the robot as one clonable
articulation (`nova5_2f85.usd`), the five pegs, the board visual, one board per
target hole with its exact colliders, and `peg_in_hole.json`.

## 3. Run the benchmark scene

```bash
# scripted insertion, 16 parallel envs, headless
bash scripts/isaac_python.sh scripts/run_peg_in_hole.py --task A1 --num_envs 16 --headless
# watch it in the Isaac Sim GUI
bash scripts/isaac_python.sh scripts/run_peg_in_hole.py --task A1 --num_envs 4
# save a rendered frame
bash scripts/isaac_python.sh scripts/run_peg_in_hole.py --task A1 --num_envs 4 --headless --enable_cameras --screenshot docs/media/peg_in_hole_A1.png
```

The scripted policy uses the env's own observations to centre the peg over the
hole, align its yaw and push down. It is a sanity check and baseline, not a
benchmark result.

## 4. Train with RL (Isaac Lab + RSL-RL PPO)

```bash
bash scripts/isaac_python.sh scripts/train.py --rl_library rsl_rl --task Nova5-MNet-PegInHole-A1-v0 --headless --num_envs 256
bash scripts/isaac_python.sh scripts/play.py  --rl_library rsl_rl --task Nova5-MNet-PegInHole-A1-v0 --num_envs 16
```

`scripts/train.py` / `play.py` are thin wrappers around Isaac Lab's
`scripts/reinforcement_learning/{train,play}.py` that register this repo's tasks
(`--external_callback nova5_tactile.tasks.register`); all Isaac Lab options work.
Logs and checkpoints go to `logs/rsl_rl/mnet_peg_in_hole/`; watch with
`tensorboard --logdir logs/rsl_rl`.

### Environment

`nova5_tactile/mnet/peg_in_hole_env.py` (`DirectRLEnv`), config in
`peg_in_hole_env_cfg.py`, PPO settings in `mnet/agents/rsl_rl_ppo_cfg.py`.

- **Start**: the matching peg is already grasped, 1–3 cm above its hole, with
  ±12 mm position and ±10° yaw error (Factory-style). Reset steps the
  simulation to servo the arm and settle the grasp, so all envs reset together.
- **Action** (4, 15 Hz): fingertip Δx, Δy, Δz (≤ 6 mm) and Δyaw (≤ 0.05 rad),
  turned into arm joint targets by damped least squares; the hand stays pointing down.
- **Observation** (17): fingertip and peg position relative to the hole top,
  fingertip and peg yaw errors (sin, cos), fingertip velocity, last action.
- **Reward**: Factory-style squashed keypoint distance between the peg and its
  fully inserted pose, plus a success bonus, minus an action penalty.
- **Success**: peg bottom ≥ 20 mm below the board top (board 25 mm) and inside the hole.
- **Not modelled**: picking the peg from a feeder, the 20-task sequence and the
  3 h limit of the real benchmark; scoring here is per sub-task insertion rate.

## Repository layout

| Path | Contents |
|---|---|
| `nova5_tactile/mnet/peg_in_hole.py` | Benchmark geometry from the CAD: holes, pegs, convex decompositions (pure numpy) |
| `nova5_tactile/mnet/usd_assets.py` | Writes peg and board USDs |
| `nova5_tactile/mnet/peg_in_hole_env*.py` | Isaac Lab env and config |
| `nova5_tactile/tasks.py` | Gym registration of the 20 tasks |
| `nova5_tactile/robot.py`, `robotiq.py`, `robot_cfg.py` | NOVA5 + 2F-85 authoring and actuators |
| `nova5_tactile/gripper_kinematics.py` | 2F-85 pad gap ↔ driver angle from the URDF |
| `scripts/build_assets.py` | Builds `assets/generated/` |
| `scripts/run_peg_in_hole.py`, `train.py`, `play.py` | Run, train, evaluate |
| `assets/` | Source CAD/USD with provenance and licences ([assets/README.md](assets/README.md)) |
| `tests/` | Isaac-free unit tests (`conda run -n locomanipu python -m pytest -q tests`) |

## Tactile (work in progress)

Each inner finger carries a 22×32×3 mm pad body with a 6×9 taxel grid and two
readouts: contact taxels (PhysX contact forces binned per taxel, in newtons) and
Isaac Lab's TacSL force field. They are validated on a lift-and-hold demo,
not yet wired into the RL env (whose robot uses rigid, non-compliant pads).

```bash
bash scripts/isaac_python.sh scripts/grasp_demo.py --out runs/grasp01 --device cuda:0 --headless
bash scripts/isaac_python.sh scripts/validate_static.py --out runs/static01 --device cuda:0 --headless
```

Results and known issues: [docs/validation.md](docs/validation.md); the original
scaffold's sensor notes: [docs/tactile_model_scaffold.md](docs/tactile_model_scaffold.md).

## Limitations

- Gripper drive gains and its 2 N·m cap are simulation assumptions; the 2F-85
  force controller is not modelled. Arm effort limits are synthetic (the vendor
  URDF has zeros).
- PhysX contact at 0.02 mm clearance is at the edge of what the solver resolves;
  treat levels 3–4 results with care until compared with hardware.
- The board is modelled exactly only within ±3 cm of the target hole; elsewhere
  its collision is plain boxes (other holes filled).

## Credits

ManipulationNet task CAD: [ManipulationNet/mnet_client](https://github.com/ManipulationNet/mnet_client)
(Apache-2.0). NOVA5 description: [Dobot-Arm/DOBOT_6Axis_ROS2_V4](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4) (MIT).
Robotiq 2F-85 URDF/meshes: `robot_description` package (BSD-3). The env follows
the structure of Isaac Lab's Factory peg-insert task. The tactile backend
contains code adapted from an earlier scaffold of this repository.
Parts of this repository were written with AI assistance (Claude); review before
relying on it, and follow your institution's guidelines on generative AI.
