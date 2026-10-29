# Fingertip tactile model

This document describes what the simulated sensor measures, how, and how to
check it before trusting the numbers.

## Geometry and frames

Each Robotiq 2F-85 fingertip link gets a child Xform `tactile_pad`. Its frame:

- origin at the centre of the pad box,
- **+Z** = sensing normal, pointing out of the finger towards the object,
- **+X** = along the finger towards the tip (taxel *rows*), **+Y** = Z × X (taxel *cols*),
- the sensing surface is the plane `z = thickness / 2`.

`T_body_pad` (pad frame in the finger-link frame) is stored in the asset's
`.meta.json`. It comes from automatic placement (`tactile/pad_placement.py`:
inner face of the link's collider AABB, distal edge `tip_margin` from the tip,
surface `protrusion` proud of the finger) or from manual `pad_xyz` / `pad_rpy_deg`.

Taxel `[r, c]` is centred at `x = -size_x/2 + (r+0.5)·size_x/rows`,
`y = -size_y/2 + (c+0.5)·size_y/cols`.

## Two pad modes

**`taxel_colliders` (default).** The pad is `rows × cols` coplanar box
colliders on the finger rigid body. Every contact is attributed to the taxel
whose collider produced it, so spatial resolution is genuine: PhysX generates
contacts per shape pair. Cost: `rows·cols` extra shapes per finger (64 for the
default 8×4 ×2), which is cheap for PhysX. Possible artefact: tiny "ghost"
contacts at the seams between boxes when an object slides across them.

**`single_collider`.** One box per pad; contact points are binned by position
(`nearest`, `bilinear`, or `gaussian` spreading with `spread_sigma`). PhysX
returns only a handful of points per shape pair (often ≤ 4 for box/box), so the
raw map is sparse; Gaussian spreading approximates elastomer load spreading.

## From PhysX contacts to taxel forces

1. `PhysxContactReportAPI` is applied to each finger link at assembly time.
2. Every app frame, `PhysxContactSource.poll()` reads `get_contact_report()`
   (contacts of the most recent physics step) and keeps pairs whose collider is
   one of our pad colliders.
3. Finger pose comes from PhysX (`get_rigidbody_transformation`) with a USD
   fallback; the quaternion layout is self-calibrated against USD at start-up.
4. For each contact pair: `force = impulse / physics_dt`, transformed into the
   pad frame.
5. **Sign.** PhysX uses one impulse convention per pair (actor0/actor1) and the
   pad may be either actor. For non-adhesive contact every point pushes *into*
   the pad, so `orient_pair_forces` flips the whole pair if its summed normal
   component points outwards. This is exact per pair and keeps friction (shear)
   consistent with the normal.
6. Normal force per taxel = `max(-f_z, 0)`; shear = `(f_x, f_y)` in the pad frame.
7. Optional non-idealities (`sensor_model.py`): Gaussian noise → deadband →
   saturation → quantisation.

Outputs per pad (`TactileFrame`): `normal (rows, cols)` [N], `shear (rows, cols, 2)`
[N], plus helpers `total_normal`, `total_shear`, `center_of_pressure(spec)`,
`pressure(spec)` [Pa].

## Contact physics

Pads use a physics material with PhysX **compliant contact**
(`compliant_stiffness` [N/m], `compliant_damping` [N·s/m]) instead of rigid
contact. This gives a small, force-dependent penetration that behaves like a
soft skin and spreads load over neighbouring taxels. Set the stiffness to 0 for
rigid contact. Friction coefficients are on the same material.

Note that the contact force comes from one physics step. If the app runs
several physics steps per frame, the reading is the last step's (impulses are
not summed across substeps).

## Validation checklist (do these before using the data)

1. **Static weight.** Make the gripper kinematic or very stiff, pads facing up,
   drop a 1.0 kg cube on one pad. After settling, `total_normal ≈ 9.81 N`
   (within a few %). Wrong by a constant factor → check `physics_hz` vs the
   PhysicsScene `timeStepsPerSecond`.
2. **Location / orientation.** Press a small sphere on the distal end of the
   pad: the load must appear in the highest row index (fingertip), and moving
   the sphere along +Y must move the load to higher column indices.
3. **Symmetric squeeze.** In the grasp demo before release, left and right
   totals should match closely (Newton's third law through the object).
4. **Shear sign.** After release (gravity on), shear on both pads should point
   in the direction that supports the object (towards gravity's opposite in the
   pad frames). If it is inverted on one pad only, report it — the per-pair
   orientation should prevent this.
5. **Unloaded noise floor.** With the gripper open and nothing touching, every
   taxel must read 0 (ideal model) — any non-zero reading means a pad collider
   is touching the finger geometry of the *other* finger or the palm.

## Limitations

- PhysX rigid-body contact is not a deformable skin; there is no hysteresis,
  creep or shape-dependent spreading beyond what compliant contact gives.
- `get_contact_report` requires the PhysX backend (not Newton).
- Auto-placement is a bounding-box heuristic; always look at the pads.

## Upgrade paths

- **Isaac Lab TacSL** (`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor`):
  penalty-based force field + GelSight-style tactile images; best if you move
  to Isaac Lab for learning.
- **Vendor digital twins** in the Isaac Sim asset library (Synaptics CTS,
  Tashan TS-F-A) if your real hardware is one of those.
- **Deformable / FEM pads** (`isaacsim.core.experimental.prims.DeformablePrim`)
  for elastomer physics, at a much higher compute cost.
- Keep `TactileFrame` as the interface so downstream code does not change when
  the backend does.
