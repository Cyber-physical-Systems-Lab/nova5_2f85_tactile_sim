# nova5_tactile_sim

Isaac Sim / Isaac Lab simulation of a Dobot NOVA5 arm with a Robotiq 2F-85
gripper and a tactile sensor on each fingertip.

- **Arm**: NOVA5 USD (`assets/nova5_usd`), fixed on a 0.70 m table.
- **Gripper**: Robotiq 2F-85 built at load time from the URDF CAD: six revolute
  finger joints coupled by PhysX mimic joints, one driven joint (`finger_joint`,
  0–0.725 rad).
- **Tactile**: a 22×32×3 mm elastomer pad, a rigid body fixed to each inner
  finger (`left_tactile_pad`, `right_tactile_pad`), with a 6×9 taxel grid read
  by two backends side by side:
  - **Contact taxels** (`contact_taxels.py`, default to rely on): the contact
    face is tiled by 54 box colliders; PhysX's GPU contact view gives every
    contact point's normal force and every friction anchor's force, binned onto
    the taxels. Sums are the solver's forces in newtons.
  - **TacSL** (`tactile.py`): Isaac Lab's `VisuoTactileSensor` in force-field
    mode, per-taxel penetration and penalty normal/shear against the contact
    object's SDF. Good contact location for large contacts; magnitudes are
    penalty units and its shear is velocity-based.
  Both report in the same pad frame (`taxels.py`): z into the finger, rows
  across the finger width, columns towards the fingertip, force on the pad.
  An optional per-taxel noise/deadband/saturation/quantisation model applies
  to the contact backend. Pad contact is compliant (default 1e4 N/m).
  Validation results: [docs/validation.md](docs/validation.md).

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
`--noise-std`, `--deadband`, `--saturation`, `--quantization` (N) shape the
contact backend; `--pad-colliders single` uses one pad collider instead of 54.
The demo approaches a 50 g, 4×4×5 cm box, closes, lifts 14 cm and holds.
Output in `runs/<name>/`:

- `metadata.json`: plant spec, CAD hashes, limit repairs, taxel positions in the pad frame
- `result.json`: 20 Hz trace with pad net contact forces and per-pad summaries of both backends
- `tactile/NNNNNN.npz`: full taxel fields `tacsl_{left,right}_{normal,shear,depth}`, `contact_{left,right}_{normal,shear}`
- `scene.usda`: the composed stage

## Validation

```bash
bash scripts/isaac_python.sh scripts/validate_static.py --out runs/static01 --device cuda:0 --headless
conda run -n locomanipu python scripts/validation_report.py --static runs/static01 --grasp runs/grasp01 --open runs/open01
```

`validate_static.py` sets known weights on standalone face-up pads (weight,
location, empty pad); `validation_report.py` grades the five checks against
a static run plus a grasp run and a `--keep-open` run.

## Tests

```bash
conda run -n locomanipu python -m pytest -q tests
```

Isaac-free checks of asset paths, pad/box meshes, taxel binning and the sensor model.

## Layout

| Path | Contents |
|---|---|
| `nova5_tactile/robot.py` | `build_nova5_2f85()` table + arm + gripper articulation; `spawn_sdf_box()` |
| `nova5_tactile/robotiq.py` | 2F-85 builder from URDF; `make_pad()`, `attach_tactile_pads()` |
| `nova5_tactile/contact_taxels.py` | Contact-taxel backend (PhysX GPU contact view) |
| `nova5_tactile/tactile.py` | TacSL backend: sensor factory, readout, taxel image rendering |
| `nova5_tactile/taxels.py` | Taxel grid, contact binning, sensor noise model (pure numpy) |
| `nova5_tactile/pad_geometry.py` | Pad mesh, taxel colliders, placement (pure numpy) |
| `nova5_tactile/kinematics.py` | FK helpers for IK |
| `scripts/grasp_demo.py` | Pick-and-lift demo with tactile logging |
| `scripts/validate_static.py`, `scripts/validation_report.py` | Tactile validation checklist |

## Limitations

- Pad compliance, friction and noise values are simulation choices, not
  calibrated to any physical sensor. The contact backend's newtons are the
  solver's; TacSL magnitudes are penalty units (see docs/validation.md).
- Both backends sense only the one object they are pointed at; TacSL also
  needs it to have an SDF collider. TacSL is GPU-only.
- Contact force reporting is wrong (3×) on kinematic pads spanning many taxel
  colliders; keep pads dynamic (jointed), as the gripper does.
- GelSight-style RGB tactile images (TacSL camera mode) are not enabled.
- Gripper drive gains and the 2 N·m cap are assumptions; the 2F-85 hardware
  force controller is not modelled.
