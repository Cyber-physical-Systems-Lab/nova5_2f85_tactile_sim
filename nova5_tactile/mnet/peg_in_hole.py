"""ManipulationNet Peg-in-Hole (NIST ATB M1) geometry, pure numpy.

Source CAD: github.com/ManipulationNet/mnet_client (branch ros_2,
assets/peg_in_hole/cad_files, Apache-2.0), copied to
assets/manipulationnet/peg_in_hole. The STLs are in centimetres.

Task naming follows the benchmark: shape A-E (circle, hexagon, triangle,
rectangle, L) and clearance level 1-4 (3, 1, 0.1, 0.02 mm), e.g. 'C3'.
The board's hole walls are read from a cross-section of the board STL, so
hole size, position and yaw are the CAD's, not re-modelled. Collision
geometry is built from convex prisms (exact for the sectioned polygons),
because PhysX SDF voxels are far coarser than the 0.02-0.1 mm clearances.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .. import ASSETS

CAD = ASSETS/'manipulationnet/peg_in_hole'
CM = .01
BOARD_STL = CAD/'nist_atb_m1.stl'
BOARD_THICKNESS = 2.5*CM
PEG_LENGTH = 5.*CM
SHAPES = {'A': 'cylinder', 'B': 'hexagonal', 'C': 'triangular', 'D': 'rectangular', 'E': 'l-shaped'}
CLEARANCE_MM = {1: 3., 2: 1., 3: .1, 4: .02}
# Hole grid of the board (cm): columns are shapes A-E, rows are levels 1-4 (board y increases from level 1 to 4).
COLUMN_X_CM = {'A': -12.78, 'B': -6.76, 'C': 0., 'D': 6.09, 'E': 12.}
ROW_Y_CM = {1: -9.5, 2: -3.87, 3: 1.25, 4: 6.5}


def peg_stl(shape):
    return CAD/f'atb_m1_{SHAPES[shape]}_peg.stl'


def read_stl(path):
    """Binary STL -> (triangles[N, 3, 3]) in the file's units."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84+50*n:
        raise ValueError(f'Expected binary STL: {path}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    return np.frombuffer(raw, dtype=dtype, offset=84)['v'].astype(float)


def section_loops(tris, z):
    """Closed polygons where the plane at height z cuts the mesh, as ordered (N, 2) arrays."""
    d = tris[:, :, 2]-z
    segs = []
    for t, s in zip(tris, d):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (s[i] < 0) != (s[j] < 0):
                pts.append(t[i]+(t[j]-t[i])*(s[i]/(s[i]-s[j])))
        if len(pts) == 2:
            segs.append((pts[0][:2], pts[1][:2]))
    key = lambda p: (round(p[0], 6), round(p[1], 6))
    nbr = {}
    for a, b in segs:
        nbr.setdefault(key(a), []).append(key(b)); nbr.setdefault(key(b), []).append(key(a))
    loops, seen = [], set()
    for start in nbr:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in nbr[cur] if n != prev and (n not in seen or (n == start and len(loop) > 2))]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]; loop.append(cur); seen.add(cur)
        loops.append(_dedupe(np.array(loop)))
    return loops


def _dedupe(poly, tol=1e-7):
    """Drop repeated and collinear vertices; return counter-clockwise order."""
    keep = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i-1], poly[i], poly[(i+1) % n]
        if np.linalg.norm(b-a) > tol and abs(cross2(b-a, c-b)) > tol*np.linalg.norm(c-b):
            keep.append(b)
    poly = np.array(keep)
    return poly if signed_area(poly) > 0 else poly[::-1]


def cross2(a, b):
    """z component of the 2-D cross product (np.cross on 2-D vectors is deprecated in numpy 2)."""
    a, b = np.asarray(a), np.asarray(b)
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def simplify(poly, tol=2e-6):
    """Ramer-Douglas-Peucker on a closed polygon; every removed point lies within tol of the result."""
    n = len(poly)
    i0 = 0; i1 = int(np.argmax(np.linalg.norm(poly-poly[0], axis=1)))
    def rdp(idx):
        a, b = poly[idx[0]], poly[idx[-1]]
        ab = b-a; L = np.linalg.norm(ab)
        pts = poly[idx[1:-1]]
        if not len(pts):
            return [idx[0], idx[-1]]
        d = np.abs(cross2(ab, pts-a))/L if L > 0 else np.linalg.norm(pts-a, axis=1)
        k = int(np.argmax(d))
        if d[k] <= tol:
            return [idx[0], idx[-1]]
        left = rdp(idx[:k+2]); right = rdp(idx[k+1:])
        return left[:-1]+right
    first = rdp(list(range(i0, i1+1))); second = rdp(list(range(i1, n))+[0])
    return poly[first[:-1]+second[:-1]]


def signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return .5*float(np.dot(x, np.roll(y, -1))-np.dot(y, np.roll(x, -1)))


@dataclass
class Peg:
    shape: str
    tris: np.ndarray        # (N, 3, 3) metres, peg frame: section centred, bottom at z=0, axis +z
    polygon: np.ndarray     # (M, 2) metres, counter-clockwise cross-section in the peg frame
    grip_yaw: float         # peg-frame yaw of the narrowest width (finger closing direction)
    grip_width: float       # metres between the pads at that yaw


@dataclass
class Hole:
    shape: str
    level: int
    polygon: np.ndarray     # (M, 2) metres, board frame (board centre, top face at z=BOARD_THICKNESS)
    center: np.ndarray      # (2,) metres, board frame: where the peg centre goes when inserted
    yaw: float              # board-frame peg yaw that fits the hole
    clearance_mm: float     # nominal (benchmark)
    fit_gap_mm: float       # measured: mean peg-to-wall gap at that pose, from the CAD sections
    min_gap_mm: float       # measured: smallest peg-to-wall gap at that pose (> 0: the peg fits)


def load_peg(shape):
    tris = read_stl(peg_stl(shape))*CM
    poly = simplify(max(section_loops(tris, PEG_LENGTH/2), key=lambda p: abs(signed_area(p))))
    c = _centroid(poly)
    tris = tris-[c[0], c[1], tris[..., 2].min()]
    poly = poly-c
    ang = np.deg2rad(np.arange(0, 180, .25))
    widths = [np.ptp(poly@[np.cos(a), np.sin(a)]) for a in ang]
    k = int(np.argmin(widths))
    return Peg(shape, tris, poly, float(ang[k]), float(widths[k]))


def _centroid(poly):
    x, y = poly[:, 0], poly[:, 1]; xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x*ys-xs*y; a = cross.sum()/2
    return np.array([((x+xs)*cross).sum(), ((y+ys)*cross).sum()])/(6*a)


def board_holes():
    """{'A1': Hole, ...} from the board CAD."""
    tris = read_stl(BOARD_STL)
    loops = section_loops(tris, 1.25)
    out = {}
    for loop in loops:
        c = _centroid(loop)
        if np.ptp(loop[:, 0]) > 10:  # outer board boundary
            continue
        shape = min(COLUMN_X_CM, key=lambda s: abs(COLUMN_X_CM[s]-c[0]))
        level = min(ROW_Y_CM, key=lambda l: abs(ROW_Y_CM[l]-c[1]))
        if abs(COLUMN_X_CM[shape]-c[0]) > 1.5 or abs(ROW_Y_CM[level]-c[1]) > 1.5:
            raise ValueError(f'Unexpected hole at {c} cm')
        out[f'{shape}{level}'] = (simplify(loop*CM), c*CM)
    if len(out) != 20:
        raise ValueError(f'Expected 20 holes, found {sorted(out)}')
    holes = {}
    for name, (poly, c) in sorted(out.items()):
        peg = load_peg(name[0])
        yaw, off, min_gap, mean_gap = match_pose(peg.polygon, poly-c)
        holes[name] = Hole(name[0], int(name[1]), poly, c+off, yaw, CLEARANCE_MM[int(name[1])], mean_gap*1e3, min_gap*1e3)
    return holes


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _dist_to_polygon(pts, poly):
    a = poly; b = np.roll(poly, -1, 0); ab = b-a
    t = np.clip(((pts[:, None]-a[None])*ab[None]).sum(-1)/(ab*ab).sum(-1)[None], 0, 1)
    return np.linalg.norm(pts[:, None]-(a[None]+t[..., None]*ab[None]), axis=-1).min(1)


def _inside(pts, poly):
    x, y = pts[:, 0], pts[:, 1]; res = np.zeros(len(pts), bool)
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        res ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-30)+a[0])
    return res


def match_pose(peg_poly, hole_poly_centred):
    """Peg yaw and centre offset that maximise the smallest peg-to-wall clearance inside the hole.

    Returns (yaw, offset (2,), min gap m, mean gap m). A negative min gap means the peg cannot fit.
    """
    from scipy.optimize import minimize
    pts = _resample(peg_poly, 240)
    def clearance(x):
        q = pts@_rot(x[0]).T+x[1:3]
        d = _dist_to_polygon(q, hole_poly_centred)
        return np.where(_inside(q, hole_poly_centred), d, -d)
    coarse = np.deg2rad(np.arange(0, 360, .5))
    scores = [clearance(np.r_[y, 0, 0]).min() for y in coarse]
    x0 = np.r_[coarse[int(np.argmax(scores))], 0, 0]
    res = minimize(lambda x: -clearance(x).min(), x0, method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-9})
    x = res.x if -res.fun >= max(scores) else x0
    c = clearance(x)
    return float(np.arctan2(np.sin(x[0]), np.cos(x[0]))), x[1:3], float(c.min()), float(np.abs(_dist_to_polygon(_resample(peg_poly)@_rot(x[0]).T+x[1:3], hole_poly_centred)).mean())


def _resample(poly, n=400):
    seg = np.linalg.norm(np.roll(poly, -1, 0)-poly, axis=1)
    s = np.r_[0, np.cumsum(seg)]; q = np.linspace(0, s[-1], n, endpoint=False)
    i = np.searchsorted(s, q, side='right')-1
    return poly[i]+(np.roll(poly, -1, 0)[i]-poly[i])*((q-s[i])/seg[i])[:, None]


def star_center(poly):
    """A point that sees every edge of a star-shaped polygon (max-min distance to edge lines)."""
    a = poly; b = np.roll(poly, -1, 0); e = b-a
    n = np.stack([-e[:, 1], e[:, 0]], 1)/np.linalg.norm(e, axis=1)[:, None]  # inward normals (CCW)
    lo, hi = poly.min(0), poly.max(0)
    best, best_d = None, -np.inf
    for _ in range(3):
        g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 41), np.linspace(lo[1], hi[1], 41)), -1).reshape(-1, 2)
        d = ((g[:, None]-a[None])*n[None]).sum(-1).min(1)
        k = int(np.argmax(d))
        if d[k] > best_d:
            best, best_d = g[k], d[k]
        span = (hi-lo)/8; lo, hi = best-span, best+span
    if best_d <= 0:
        raise ValueError('Polygon is not star-shaped')
    return best


def outside_pieces(hole_poly, center, half):
    """Convex 2-D pieces covering the square [center +- half] minus the hole.

    Rays from a kernel point of the hole through each hole vertex split the
    square annulus into one convex polygon per hole edge.
    """
    k = star_center(hole_poly)
    lo, hi = center-half, center+half
    corners = np.array([[hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]], [hi[0], lo[1]]])
    def hit(v):
        d = v-k
        ts = [t for t in ((lo[0]-k[0])/d[0] if d[0] < 0 else (hi[0]-k[0])/d[0] if d[0] > 0 else np.inf,
                          (lo[1]-k[1])/d[1] if d[1] < 0 else (hi[1]-k[1])/d[1] if d[1] > 0 else np.inf)]
        return k+min(ts)*d
    ang = lambda p: np.arctan2(p[1]-k[1], p[0]-k[0])
    pieces = []
    n = len(hole_poly)
    for i in range(n):
        v0, v1 = hole_poly[i], hole_poly[(i+1) % n]
        p0, p1 = hit(v0), hit(v1)
        a0, a1 = ang(v0), ang(v1)
        sweep = (a1-a0) % (2*np.pi)
        between = [c for c in corners if 0 < (ang(c)-a0) % (2*np.pi) < sweep]
        between.sort(key=lambda c: (ang(c)-a0) % (2*np.pi))
        pieces.append(np.array([v0, p0, *between, p1, v1]))
    return pieces


def inside_pieces(poly):
    """Convex pieces of a simple polygon: the polygon itself if convex, else a vertical-slab split."""
    if _is_convex(poly):
        return [poly]
    xs = np.unique(np.round(poly[:, 0], 9))
    a = poly; b = np.roll(poly, -1, 0)
    pieces = []
    for xa, xb in zip(xs[:-1], xs[1:]):
        xm = (xa+xb)/2
        cross = [(i, a[i], b[i]) for i in range(len(poly)) if (a[i][0]-xm)*(b[i][0]-xm) < 0]
        at = lambda e, x: e[1][1]+(e[2][1]-e[1][1])*(x-e[1][0])/(e[2][0]-e[1][0])
        cross.sort(key=lambda e: at(e, xm))
        for lo_e, hi_e in zip(cross[0::2], cross[1::2]):
            pieces.append(np.array([[xa, at(lo_e, xa)], [xb, at(lo_e, xb)], [xb, at(hi_e, xb)], [xa, at(hi_e, xa)]]))
    return pieces


def _is_convex(poly):
    e = np.roll(poly, -1, 0)-poly
    return bool((cross2(e, np.roll(e, -1, 0)) >= -1e-12).all())


def prism(poly, z0, z1):
    """Convex prism points (2N, 3) from a convex 2-D polygon."""
    return np.vstack([np.c_[poly, np.full(len(poly), z0)], np.c_[poly, np.full(len(poly), z1)]])
