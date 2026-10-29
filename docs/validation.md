# Tactile validation — 2026-10-29

Checklist taken from the `nova5_2f85_tactile_sim` scaffold (`docs/tactile_model.md`),
run on both backends with default settings (6×9 taxel colliders, compliant pad
1e4 N/m, TacSL normal stiffness 1, GPU `cuda:0`, dt 2.5 ms).

Runs (in `runs/`, not committed):

```bash
bash scripts/isaac_python.sh scripts/validate_static.py --out runs/static14 --device cuda:0 --headless
bash scripts/isaac_python.sh scripts/grasp_demo.py --out runs/grasp04 --device cuda:0 --headless
bash scripts/isaac_python.sh scripts/grasp_demo.py --out runs/open04 --keep-open --device cuda:0 --headless
conda run -n locomanipu python scripts/validation_report.py --static runs/static14 --grasp runs/grasp04 --open runs/open04
```

Static values are means over the last 0.5 s of 2 s; grasp values are means over the hold, t = 10–13.5 s.

## Results

| Check | Backend | Result | Value | Note |
|---|---|---|---|---|
| 1 static weight | contact | PASS | weight: 1.9515 N vs 1.9620 |  |
| 1 static weight | tacsl | info | weight: 6.89e-03 (3.51e-03 x m g) | penalty units, not N |
| 1 static weight | contact | PASS | tip: 0.4902 N vs 0.4905 |  |
| 1 static weight | tacsl | info | tip: 0.00e+00 (0.00e+00 x m g) | penalty units, not N |
| 1 static weight | contact | PASS | side: 0.4896 N vs 0.4905 |  |
| 1 static weight | tacsl | info | side: 0.00e+00 (0.00e+00 x m g) | penalty units, not N |
| 1 static weight | contact | PASS | heavy_small: 1.9605 N vs 1.9620 |  |
| 1 static weight | tacsl | info | heavy_small: 1.48e-06 (7.53e-07 x m g) | penalty units, not N |
| 1 static weight | contact | PASS | light_big: 0.4905 N vs 0.4905 |  |
| 1 static weight | tacsl | info | light_big: 6.66e-04 (1.36e-03 x m g) | penalty units, not N |
| 2 location | contact | PASS | tip peak [3, 8], side peak [5, 5] (grid 6x9) |  |
| 2 location | tacsl | FAIL | tip peak None, side peak None (grid 6x9) |  |
| 3 symmetric squeeze | contact | PASS | L 7.448, R 7.449, diff 0.0% |  |
| 3 symmetric squeeze | tacsl | PASS | L 0.01879, R 0.02021, diff 7.3% |  |
| 3 symmetric squeeze | contact | PASS | left: taxel sum 7.448 N vs pad net 7.448 N |  |
| 3 symmetric squeeze | contact | PASS | right: taxel sum 7.449 N vs pad net 7.449 N |  |
| 4 shear sign | contact | PASS | left (-0.13, +0.0742), right (-0.13, +0.426); sum y +0.5005 | sum y should be +m g = 0.4905 N |
| 4 shear sign | tacsl | FAIL | left (+0.000566, -0.0375), right (+0.00304, -0.0402); sum y -0.07774 | velocity-based friction model, penalty units |
| 5 noise floor | contact | PASS | open run max 0, empty pad 0 |  |
| 5 noise floor | tacsl | PASS | open run max 0, empty pad 0 |  |

Centre of pressure on the static pads (contact backend): `weight` box centred,
measured (−0.72, +0.37) mm; `tip` box at y = +12.5 mm, measured y = +12.67 mm;
`side` box at x = +7.5 mm, measured x = +7.94 mm. TacSL: `weight` (+4.42, +2.14) mm,
no reading for `tip` and `side`.

## Findings

**Contact backend: passes every check.** Taxel normal sums equal the pad's net
contact force from PhysX (7.448 N vs 7.448 N in the grasp hold) and m g on
static pads (within 0.6 %). Friction along pad +y (towards the fingertips)
summed over both pads is 0.5005 N against the box weight 0.4905 N. The split
between pads is uneven (0.07 / 0.43 N); static friction is indeterminate, so
only the sum is checked.

**TacSL: right place for large contacts, not usable for small loads or holding shear.**
- Penetration is microns, so readings are ~1e-3 of m g in penalty units, and
  6×6 mm, 50 g contacts read zero. Compliant stiffness applies per contact
  point, so the pad is effectively much stiffer than 1e4 N/m and the object
  barely sinks in. Lowering `--pad-stiffness` is the obvious lever; not yet tested.
- TacSL shear is a velocity-based friction model (k_t·|v_t|, capped at μ·F_n).
  In the hold it sits at the cap and points to −y, against the weight. It
  measures slip velocity, not static holding load.
- Left/right totals differ by 7 % while the true forces are equal.

**Simulation pitfalls found while porting:**
- `omni.physx` `get_contact_report()`, which the scaffold used, returns no
  headers under Isaac Lab's GPU pipeline. The backend reads the PhysX tensor
  rigid-contact view (`get_contact_data`, `get_friction_data`) instead.
- Kinematic pads make PhysX over-report contact force: exactly 3× m g for
  footprints spanning ~20–30 taxel colliders (both reporting paths agree, the
  object is at rest). Dynamic pads on a fixed joint, as on the gripper, read
  correctly. `validate_static.py --pad-mount kinematic` reproduces it.
- Contacts on the pad's outer edge fall just outside the taxel grid; the
  contact backend clips them to the nearest taxel (`bin_contacts(clip=True)`).
  Before this the taxel sum was 7 % low.
