# nova5_tactile_sim

Isaac Sim / Isaac Lab simulation of a Dobot NOVA5 arm with a Robotiq 2F-85
gripper and a tactile sensor on each fingertip.

- **Arm**: NOVA5 USD (`assets/nova5_usd`), fixed on a 0.70 m table.
- **Gripper**: Robotiq 2F-85 built at load time from the URDF CAD: six revolute
  finger joints coupled by PhysX mimic joints, one driven joint (`finger_joint`,
  0–0.725 rad).
- **Tactile**: a 22×32×3 mm elastomer pad, a rigid body fixed to each inner
  finger (`left_tactile_pad`, `right_tactile_pad`). Each has an Isaac Lab TacSL
  `VisuoTactileSensor` in force-field mode: a 6×9 taxel grid on the contact face
  that reports per-taxel penetration, normal force and 2-D shear force against
  the contact object's SDF. Pad contact is compliant (default 1e4 N/m) so the
  object can press into it. Pads also carry an ordinary `ContactSensor` for
  net force.

Asset provenance: [assets/README.md](assets/README.md).

## Requirements

- Isaac Sim at `~/isaacsim`, Isaac Lab 3.x at `~/IsaacLab` (uses `isaaclab_contrib` TacSL).
- Conda env `locomanipu` (override with `TASK_CONDA_ENV`).
- GPU (`--device cuda:0`); TacSL's SDF queries use the PhysX GPU tensor API.

## Run

```bash
bash scripts/isaac_python.sh scripts/grasp_demo.py --out runs/grasp01 --device cuda:0 --headless
```

Drop `--headless` to watch in the GUI (add `--debug-taxels` to draw taxels).
`--keep-open` is a negative control that never closes the gripper (exits 3).
The demo approaches a 50 g, 4×4×5 cm box, closes, lifts 14 cm and holds.
Output in `runs/<name>/`:

- `metadata.json`: plant spec, CAD hashes, limit repairs, taxel positions in the pad frame
- `result.json`: 20 Hz trace with pad contact forces and per-pad tactile summaries
- `tactile/NNNNNN.npz`: full taxel fields `{left,right}_{normal,shear,depth}`
- `scene.usda`: the composed stage

## Tests

```bash
conda run -n locomanipu python -m pytest -q tests
```

Isaac-free checks of asset paths and pad/box mesh geometry.

## Layout

| Path | Contents |
|---|---|
| `nova5_tactile/robot.py` | `build_nova5_2f85()` table + arm + gripper articulation; `spawn_sdf_box()` |
| `nova5_tactile/robotiq.py` | 2F-85 builder from URDF; `attach_tactile_pads()` |
| `nova5_tactile/tactile.py` | TacSL sensor factory, readout, taxel image rendering |
| `nova5_tactile/pad_geometry.py` | Pad mesh and placement (pure numpy) |
| `nova5_tactile/kinematics.py` | FK helpers for IK |
| `scripts/grasp_demo.py` | Pick-and-lift demo with tactile logging |

## Limitations

- Pad compliance, TacSL stiffness and friction values are simulation choices,
  not calibrated to any physical sensor. Tactile force magnitudes are model
  outputs; compare shapes and timing, not absolute newtons, until calibrated.
- TacSL force field only senses the one object named in
  `contact_object_prim_path_expr`, and that object needs an SDF collider.
- GelSight-style RGB tactile images (TacSL camera mode) are not enabled.
- Gripper drive gains and the 2 N·m cap are assumptions; the 2F-85 hardware
  force controller is not modelled.
