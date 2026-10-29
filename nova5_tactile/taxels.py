"""Contact-report taxel model and sensor non-idealities (pure numpy).

Ported from the nova5_2f85_tactile_sim scaffold (taxel_grid.py,
sensor_model.py) and adapted to this repo's pad frame (see pad_geometry):
+z points from the contact face into the finger, the contact face is the
plane z = -THICKNESS, x spans the finger width (rows) and y runs towards the
fingertip (cols). Forces are the force exerted on the pad by the object, in
the pad frame, so a pushing contact has f_z > 0 and normal = max(f_z, 0).
Shear is (f_x, f_y). TacSL readouts are converted to the same frame in
tactile.read(), so both backends share one convention.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from . import pad_geometry

SPREAD_MODES = ('nearest', 'bilinear', 'gaussian')


@dataclass(frozen=True)
class TaxelGridSpec:
    rows: int = 6
    cols: int = 9
    size_x: float = pad_geometry.CONTACT_SIZE[0]
    size_y: float = pad_geometry.CONTACT_SIZE[1]
    surface_z: float = -pad_geometry.THICKNESS
    mode: str = 'nearest'
    spread_sigma: float = 0.
    contact_z_tolerance: float = .003

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1 or self.size_x <= 0 or self.size_y <= 0:
            raise ValueError('positive grid size required')
        if self.mode not in SPREAD_MODES:
            raise ValueError(f'mode must be one of {SPREAD_MODES}, got {self.mode!r}')
        if self.mode == 'gaussian' and self.spread_sigma <= 0:
            raise ValueError('gaussian mode needs spread_sigma > 0')

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def pitch(self):
        return self.size_x/self.rows, self.size_y/self.cols

    def centers(self):
        """(rows, cols, 2) taxel centre (x, y) in the pad frame."""
        px, py = self.pitch
        xs = -self.size_x/2+(np.arange(self.rows)+.5)*px
        ys = -self.size_y/2+(np.arange(self.cols)+.5)*py
        return np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1)


@dataclass
class TactileFrame:
    normal: np.ndarray            # (rows, cols) [N], compressive >= 0
    shear: np.ndarray             # (rows, cols, 2) [N], pad x/y
    num_contacts: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, spec):
        return cls(np.zeros(spec.shape), np.zeros(spec.shape+(2,)))

    @property
    def total_normal(self):
        return float(self.normal.sum())

    @property
    def total_shear(self):
        return self.shear.reshape(-1, 2).sum(0)

    def center_of_pressure(self, centers):
        """(x, y) in the pad frame from (rows, cols, 2) taxel centres; NaN without load."""
        total = self.total_normal
        if total <= 1e-12:
            return np.full(2, np.nan)
        return (self.normal[..., None]*centers).reshape(-1, 2).sum(0)/total


def orient_pair_forces(forces_local):
    """Make one contact pair's forces 'force on the pad' (pushing => f_z > 0).

    PhysX uses one impulse convention per pair (actor0/actor1) and the pad may
    be either actor. Non-adhesive contact pushes into the pad at every point,
    so flip the whole pair when its summed normal component points outwards.
    """
    F = np.asarray(forces_local, dtype=float).reshape(-1, 3)
    return -F if F.size and F[:, 2].sum() < 0 else F


def _in_area(P, spec):
    return ((np.abs(P[:, 0]) <= spec.size_x/2+1e-9) & (np.abs(P[:, 1]) <= spec.size_y/2+1e-9)
            & (np.abs(P[:, 2]-spec.surface_z) <= spec.contact_z_tolerance))


def _bilinear(coord, size, n, pitch):
    u = np.clip((coord+size/2)/pitch-.5, 0., n-1.)
    i0 = np.minimum(np.floor(u).astype(int), max(n-2, 0)); i1 = np.minimum(i0+1, n-1)
    f = u-i0 if n > 1 else np.zeros_like(u)
    return i0, i1, 1.-f, f


def _splat(P, values, spec):
    out = np.zeros(spec.shape+(values.shape[1],))
    if not len(P):
        return out
    px, py = spec.pitch
    if spec.mode == 'nearest':
        r = np.clip(np.floor((P[:, 0]+spec.size_x/2)/px).astype(int), 0, spec.rows-1)
        c = np.clip(np.floor((P[:, 1]+spec.size_y/2)/py).astype(int), 0, spec.cols-1)
        np.add.at(out, (r, c), values)
    elif spec.mode == 'bilinear':
        r0, r1, wr0, wr1 = _bilinear(P[:, 0], spec.size_x, spec.rows, px)
        c0, c1, wc0, wc1 = _bilinear(P[:, 1], spec.size_y, spec.cols, py)
        for rr, wr in ((r0, wr0), (r1, wr1)):
            for cc, wc in ((c0, wc0), (c1, wc1)):
                np.add.at(out, (rr, cc), values*(wr*wc)[:, None])
    else:  # gaussian, normalised per contact so total force is conserved
        centers = spec.centers().reshape(-1, 2)
        w = np.exp(-.5*((P[:, None, :2]-centers[None])**2).sum(-1)/spec.spread_sigma**2)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-300)
        out = (w.T@values).reshape(spec.shape+(values.shape[1],))
    return out


def bin_contacts(points_local, forces_on_pad_local, spec, taxel_index=None, clip=False):
    """Contact points/forces in the pad frame -> TactileFrame.

    taxel_index: optional (N, 2) ints; rows with r, c >= 0 go straight to that
    taxel (each taxel is its own collider), -1 rows fall back to position binning.
    clip: every contact is known to be on the pad (e.g. a per-body contact
    view), so contacts just outside the face (edges, sides) go to the nearest
    taxel instead of being dropped; that keeps the taxel sum equal to the total.
    """
    P = np.asarray(points_local, dtype=float).reshape(-1, 3)
    F = np.asarray(forces_on_pad_local, dtype=float).reshape(-1, 3)
    if len(P) != len(F):
        raise ValueError('points and forces must have the same length')
    n = len(P)
    values = np.column_stack([np.maximum(F[:, 2], 0.), F[:, 0], F[:, 1]]) if n else np.zeros((0, 3))
    grid = np.zeros(spec.shape+(3,))
    if taxel_index is not None and n:
        idx = np.asarray(taxel_index, dtype=int).reshape(-1, 2)
        direct = (idx >= 0).all(1)
        np.add.at(grid, (np.clip(idx[direct, 0], 0, spec.rows-1), np.clip(idx[direct, 1], 0, spec.cols-1)), values[direct])
        P, values = P[~direct], values[~direct]
    if len(P):
        mask = np.ones(len(P), bool) if clip else _in_area(P, spec)
        grid += _splat(P[mask], values[mask], spec)
    return TactileFrame(grid[..., 0], grid[..., 1:], num_contacts=n)


@dataclass(frozen=True)
class SensorModelParams:
    """Per-taxel non-idealities: noise -> deadband -> saturation -> quantisation. All off by default."""
    noise_std: float = 0.     # [N] per taxel, normal and each shear axis
    deadband: float = 0.      # [N] normal below this reads 0
    saturation: float = 0.    # [N] per taxel; <= 0 disables
    quantization: float = 0.  # [N] LSB; <= 0 disables

    @property
    def is_ideal(self):
        return self.noise_std <= 0 and self.deadband <= 0 and self.saturation <= 0 and self.quantization <= 0


def apply_sensor_model(normal, shear, params, rng):
    """Returns (normal, shear) after the non-idealities; inputs are not modified."""
    if params.is_ideal:
        return normal, shear
    n = np.asarray(normal, dtype=float).copy(); s = np.asarray(shear, dtype=float).copy()
    if params.noise_std > 0:
        n += rng.normal(0., params.noise_std, n.shape); s += rng.normal(0., params.noise_std, s.shape)
    n = np.maximum(n, 0.)
    if params.deadband > 0:
        dead = n < params.deadband; n[dead] = 0.; s[dead] = 0.
    if params.saturation > 0:
        n = np.minimum(n, params.saturation)
        mag = np.linalg.norm(s, axis=-1, keepdims=True)
        s = np.where(mag > params.saturation, s*(params.saturation/np.maximum(mag, 1e-12)), s)
    if params.quantization > 0:
        q = params.quantization; n = np.round(n/q)*q; s = np.round(s/q)*q
    return n, s


def apply_to_frame(frame, params, rng):
    n, s = apply_sensor_model(frame.normal, frame.shear, params, rng)
    return replace(frame, normal=n, shear=s)
