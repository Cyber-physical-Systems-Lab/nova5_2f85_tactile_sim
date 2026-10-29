"""Contact-point taxel backend. Import after AppLauncher.

Every step, PhysX's GPU rigid-contact view (behind an Isaac Lab ContactSensor
on the pad body, filtered to the contact object) returns each contact point
with its normal force, and each friction anchor with its friction force.
They are moved to the pad frame and binned onto the taxel grid (taxels.py),
so taxel sums are the solver's contact forces in newtons, unlike TacSL's
penalty field. With colliders='taxels' every taxel is its own box collider,
which spreads contact points over the face instead of the few that one
convex hull produces.

Adapted from the nova5_2f85_tactile_sim scaffold (contact_source.py,
fingertip_sensor.py). That version read omni.physx get_contact_report(),
which returns nothing under Isaac Lab's GPU pipeline (tested 2026-10-29),
so this one uses the tensor contact view instead.
"""
import numpy as np
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from .taxels import TaxelGridSpec, TactileFrame, bin_contacts, orient_pair_forces, apply_to_frame, SensorModelParams

# Sign that turns PhysX friction-anchor forces into force on the pad. Checked with the
# gravity hold of grasp_demo.py: pad-frame y friction summed over both pads = +m g (towards the fingertips).
FRICTION_SIGN = 1.


def _np(x):
    return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)


class ContactTaxelSensors:
    """pads: {side: {'body': path, ...}}; contact_objects: one prim path or {side: path}. Create before sim.reset()."""
    def __init__(self, pads, contact_objects, spec=None, model=None, seed=0, max_contacts=1024):
        self.spec = spec or TaxelGridSpec()
        self.model = model or SensorModelParams()
        self.rng = np.random.default_rng(seed)
        self.sensors = {}
        for side, pad in pads.items():
            target = contact_objects[side] if isinstance(contact_objects, dict) else contact_objects
            self.sensors[side] = ContactSensor(ContactSensorCfg(prim_path=pad['body'], filter_prim_paths_expr=[target],
                                                                update_period=0., max_contact_data_count_per_prim=max_contacts))
        self.last = {side: TactileFrame.empty(self.spec) for side in pads}

    def raw(self, side, dt):
        """World frame: contact points, normal force vectors, friction points, friction forces (PhysX signs)."""
        view = self.sensors[side].contact_view
        force, pts, normal, _, count, start = view.get_contact_data(dt)
        n, s = int(_np(count).ravel()[0]), int(_np(start).ravel()[0])
        P = _np(pts).reshape(-1, 3)[s:s+n]; F = _np(force).reshape(-1, 1)[s:s+n]*_np(normal).reshape(-1, 3)[s:s+n]
        fr, fpts, fcount, fstart = view.get_friction_data(dt)
        m, t = int(_np(fcount).ravel()[0]), int(_np(fstart).ravel()[0])
        return P, F, _np(fpts).reshape(-1, 3)[t:t+m], _np(fr).reshape(-1, 3)[t:t+m]

    def update(self, pad_poses_w, dt):
        """pad_poses_w: {side: 4x4 world pose of the pad body}. Returns {side: TactileFrame}."""
        for side in self.sensors:
            P, F, FP, FF = self.raw(side, dt)
            T = np.linalg.inv(pad_poses_w[side]); R = T[:3, :3]
            normal = bin_contacts(P@R.T+T[:3, 3], orient_pair_forces(F@R.T), self.spec, clip=True)
            friction = FRICTION_SIGN*FF@R.T; friction[:, 2] = 0.
            shear = bin_contacts(FP@R.T+T[:3, 3], friction, self.spec, clip=True)
            frame = TactileFrame(normal.normal, normal.shear+shear.shear, num_contacts=len(P))
            # Diagnostics in the pad frame: total normal-force vector and total friction vector.
            frame.meta = {'normal_vec_sum': orient_pair_forces(F@R.T).sum(0).tolist(), 'friction_sum': friction.sum(0).tolist(),
                          'num_friction': len(FP)}
            self.last[side] = apply_to_frame(frame, self.model, self.rng)
        return self.last
