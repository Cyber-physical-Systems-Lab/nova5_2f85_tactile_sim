"""Taxel-grid tactile model (pure numpy, no Isaac Sim dependency).

Pad frame (the frame of the ``tactile_pad`` Xform authored on each fingertip):

* origin at the centre of the pad box,
* +Z is the sensing normal, pointing *out of the finger* towards the object,
* X runs along the finger towards the tip (``rows``), Y across it (``cols``),
* the sensing surface is the plane ``z = surface_z`` (``thickness / 2``).

Forces passed to this module are the force exerted **on the pad by the
object**, expressed in the pad frame. A compressive (pushing) contact
therefore has ``f_z < 0``; the reported normal force is ``-f_z`` (>= 0).

Taxel ``[r, c]`` is centred at::

    x_r = -size_x/2 + (r + 0.5) * size_x / rows
    y_c = -size_y/2 + (c + 0.5) * size_y / cols
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPREAD_MODES = ("nearest", "bilinear", "gaussian")


@dataclass(frozen=True)
class TaxelGridSpec:
    rows: int
    cols: int
    size_x: float
    size_y: float
    thickness: float = 0.002
    mode: str = "nearest"
    spread_sigma: float = 0.0
    contact_z_tolerance: float = 0.003

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows/cols must be >= 1")
        if self.size_x <= 0 or self.size_y <= 0:
            raise ValueError("pad size must be positive")
        if self.mode not in SPREAD_MODES:
            raise ValueError(f"mode must be one of {SPREAD_MODES}, got {self.mode!r}")
        if self.mode == "gaussian" and self.spread_sigma <= 0:
            raise ValueError("gaussian mode needs spread_sigma > 0")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def pitch_x(self) -> float:
        return self.size_x / self.rows

    @property
    def pitch_y(self) -> float:
        return self.size_y / self.cols

    @property
    def taxel_area(self) -> float:
        return self.pitch_x * self.pitch_y

    @property
    def surface_z(self) -> float:
        return 0.5 * self.thickness

    def centers(self) -> np.ndarray:
        """(rows, cols, 2) array of taxel centre (x, y) in the pad frame."""
        xs = -0.5 * self.size_x + (np.arange(self.rows) + 0.5) * self.pitch_x
        ys = -0.5 * self.size_y + (np.arange(self.cols) + 0.5) * self.pitch_y
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([X, Y], axis=-1)


@dataclass
class TactileFrame:
    """One tactile reading of one fingertip pad."""

    normal: np.ndarray  # (rows, cols) [N], compressive >= 0
    shear: np.ndarray  # (rows, cols, 2) [N], pad x/y
    timestamp: float = 0.0
    num_contacts: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, spec: TaxelGridSpec, timestamp: float = 0.0) -> "TactileFrame":
        return cls(
            normal=np.zeros(spec.shape),
            shear=np.zeros(spec.shape + (2,)),
            timestamp=timestamp,
        )

    @property
    def total_normal(self) -> float:
        return float(self.normal.sum())

    @property
    def total_shear(self) -> np.ndarray:
        return self.shear.reshape(-1, 2).sum(axis=0)

    def in_contact(self, threshold: float = 1e-3) -> bool:
        return self.total_normal > threshold

    def center_of_pressure(self, spec: TaxelGridSpec) -> np.ndarray:
        """(x, y) in the pad frame, NaN when there is no load."""
        tot = self.total_normal
        if tot <= 1e-12:
            return np.array([np.nan, np.nan])
        c = spec.centers()
        return (self.normal[..., None] * c).reshape(-1, 2).sum(axis=0) / tot

    def pressure(self, spec: TaxelGridSpec) -> np.ndarray:
        """Normal pressure per taxel [Pa]."""
        return self.normal / spec.taxel_area


def orient_pair_forces(forces_local: np.ndarray) -> np.ndarray:
    """Make one contact pair's forces "force on the pad" (compressive => f_z < 0).

    PhysX reports one impulse direction convention per contact *pair*
    (actor0/actor1), and the pad can be either actor. For a non-adhesive
    contact every point of the pair pushes *into* the pad, so if the summed
    normal component points outwards we flip the whole pair. This is exact
    per pair and keeps the friction (shear) sign consistent with the normal.
    """
    F = np.asarray(forces_local, dtype=float).reshape(-1, 3)
    if F.size and F[:, 2].sum() > 0.0:
        return -F
    return F


def _in_area_mask(points: np.ndarray, spec: TaxelGridSpec) -> np.ndarray:
    hx, hy = 0.5 * spec.size_x, 0.5 * spec.size_y
    eps = 1e-9
    return (
        (np.abs(points[:, 0]) <= hx + eps)
        & (np.abs(points[:, 1]) <= hy + eps)
        & (np.abs(points[:, 2] - spec.surface_z) <= spec.contact_z_tolerance)
    )


def _nearest_index(points: np.ndarray, spec: TaxelGridSpec):
    r = np.floor((points[:, 0] + 0.5 * spec.size_x) / spec.pitch_x).astype(int)
    c = np.floor((points[:, 1] + 0.5 * spec.size_y) / spec.pitch_y).astype(int)
    return np.clip(r, 0, spec.rows - 1), np.clip(c, 0, spec.cols - 1)


def _bilinear_weights(coord, size, n, pitch):
    """Continuous taxel coordinate -> (i0, i1, w0, w1) along one axis."""
    u = (coord + 0.5 * size) / pitch - 0.5
    u = np.clip(u, 0.0, n - 1.0)
    i0 = np.floor(u).astype(int)
    if n > 1:
        i0 = np.minimum(i0, n - 2)
    i1 = np.minimum(i0 + 1, n - 1)
    f = u - i0
    if n == 1:
        f = np.zeros_like(u)
    return i0, i1, 1.0 - f, f


def _splat(points, values, spec: TaxelGridSpec) -> np.ndarray:
    """Distribute per-contact values (N, K) onto the grid -> (rows, cols, K)."""
    K = values.shape[1]
    out = np.zeros(spec.shape + (K,))
    if len(points) == 0:
        return out
    if spec.mode == "nearest":
        r, c = _nearest_index(points, spec)
        np.add.at(out, (r, c), values)
    elif spec.mode == "bilinear":
        r0, r1, wr0, wr1 = _bilinear_weights(points[:, 0], spec.size_x, spec.rows, spec.pitch_x)
        c0, c1, wc0, wc1 = _bilinear_weights(points[:, 1], spec.size_y, spec.cols, spec.pitch_y)
        for rr, wr in ((r0, wr0), (r1, wr1)):
            for cc, wc in ((c0, wc0), (c1, wc1)):
                np.add.at(out, (rr, cc), values * (wr * wc)[:, None])
    else:  # gaussian, normalised over the grid so total force is conserved
        centers = spec.centers().reshape(-1, 2)  # (M, 2)
        d2 = ((points[:, None, :2] - centers[None, :, :]) ** 2).sum(-1)  # (N, M)
        w = np.exp(-0.5 * d2 / spec.spread_sigma**2)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        out = (w.T @ values).reshape(spec.shape + (K,))
    return out


def bin_contacts(
    points_local,
    forces_on_pad_local,
    spec: TaxelGridSpec,
    taxel_index=None,
    timestamp: float = 0.0,
) -> TactileFrame:
    """Convert contact points/forces (pad frame) into a :class:`TactileFrame`.

    Args:
        points_local: (N, 3) contact positions in the pad frame.
        forces_on_pad_local: (N, 3) force on the pad (pad frame), already
            oriented with :func:`orient_pair_forces`.
        spec: grid description.
        taxel_index: optional (N, 2) int array. Rows with valid ``(r, c)``
            (>= 0) are assigned directly to that taxel (used when every taxel
            is its own collider); rows with ``-1`` fall back to position
            binning with ``spec.mode``.
    """
    P = np.asarray(points_local, dtype=float).reshape(-1, 3)
    F = np.asarray(forces_on_pad_local, dtype=float).reshape(-1, 3)
    if len(P) != len(F):
        raise ValueError("points and forces must have the same length")

    values = np.column_stack([np.maximum(-F[:, 2], 0.0), F[:, 0], F[:, 1]]) if len(F) else np.zeros((0, 3))
    grid = np.zeros(spec.shape + (3,))

    if taxel_index is not None and len(P):
        idx = np.asarray(taxel_index, dtype=int).reshape(-1, 2)
        direct = (idx[:, 0] >= 0) & (idx[:, 1] >= 0)
        if direct.any():
            r = np.clip(idx[direct, 0], 0, spec.rows - 1)
            c = np.clip(idx[direct, 1], 0, spec.cols - 1)
            np.add.at(grid, (r, c), values[direct])
        P, values = P[~direct], values[~direct]

    if len(P):
        mask = _in_area_mask(P, spec)
        grid += _splat(P[mask], values[mask], spec)

    return TactileFrame(
        normal=grid[..., 0],
        shear=grid[..., 1:3],
        timestamp=timestamp,
        num_contacts=int(len(np.asarray(points_local).reshape(-1, 3))),
    )
