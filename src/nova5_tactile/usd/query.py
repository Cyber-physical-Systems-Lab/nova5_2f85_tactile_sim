"""Read-only OpenUSD queries used by the asset pipeline and inspect script."""

from __future__ import annotations

import logging

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from ..math3d import from_usd_matrix, transform_points

log = logging.getLogger(__name__)

PAD_PRIM_NAME = "tactile_pad"


def iter_prims(root: Usd.Prim, instance_proxies: bool = True):
    if instance_proxies:
        return Usd.PrimRange(root, Usd.TraverseInstanceProxies())
    return Usd.PrimRange(root)


def is_rigid_body(prim: Usd.Prim) -> bool:
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def rigid_bodies(root: Usd.Prim) -> list[Usd.Prim]:
    return [p for p in iter_prims(root) if is_rigid_body(p)]


def joints(root: Usd.Prim) -> list[Usd.Prim]:
    return [p for p in iter_prims(root) if p.IsA(UsdPhysics.Joint)]


def joint_kind(prim: Usd.Prim) -> str:
    for cls, name in (
        (UsdPhysics.RevoluteJoint, "revolute"),
        (UsdPhysics.PrismaticJoint, "prismatic"),
        (UsdPhysics.FixedJoint, "fixed"),
        (UsdPhysics.SphericalJoint, "spherical"),
    ):
        if prim.IsA(cls):
            return name
    return "d6/other"


def joint_bodies(prim: Usd.Prim) -> tuple[str, str]:
    j = UsdPhysics.Joint(prim)

    def first(rel):
        t = rel.GetTargets() if rel else []
        return str(t[0]) if t else ""

    return first(j.GetBody0Rel()), first(j.GetBody1Rel())


def find_by_name(root: Usd.Prim, candidates, predicate=None) -> Usd.Prim | None:
    """First prim (in candidate order) whose *name* matches and satisfies
    ``predicate``. Ties are broken by the shortest path."""
    pool = [p for p in iter_prims(root) if predicate is None or predicate(p)]
    for name in candidates:
        hits = [p for p in pool if p.GetName() == name]
        if hits:
            if len(hits) > 1:
                log.warning("several prims named %r; using %s", name, hits[0].GetPath())
            return sorted(hits, key=lambda p: len(str(p.GetPath())))[0]
    return None


def find_rigid_body(root: Usd.Prim, candidates) -> Usd.Prim | None:
    return find_by_name(root, candidates, is_rigid_body)


def root_body(root: Usd.Prim) -> Usd.Prim | None:
    """Heuristic: the rigid body that is never a joint *child* (body1)."""
    bodies = rigid_bodies(root)
    if not bodies:
        return None
    children = set()
    for j in joints(root):
        b0, b1 = joint_bodies(j)
        if b0 and b1:
            children.add(b1)
    roots = [b for b in bodies if str(b.GetPath()) not in children]
    return roots[0] if roots else bodies[0]


def world_matrix(prim: Usd.Prim, cache: UsdGeom.XformCache | None = None) -> np.ndarray:
    cache = cache or UsdGeom.XformCache()
    return from_usd_matrix(cache.GetLocalToWorldTransform(prim))


def _local_extent(p: Usd.Prim):
    b = UsdGeom.Boundable(p)
    ext = b.GetExtentAttr().Get()
    if ext is None:
        ext = UsdGeom.Boundable.ComputeExtentFromPlugins(b, Usd.TimeCode.Default())
    if ext is None or len(ext) != 2:
        return None
    lo, hi = np.array(ext[0], dtype=float), np.array(ext[1], dtype=float)
    return (lo, hi) if np.all(hi >= lo) else None


def collider_aabb(link: Usd.Prim, cache: UsdGeom.XformCache | None = None, colliders_only: bool = True):
    """AABB (min, max) of the link's collision geometry in the link frame.

    Uses authored/computed extents of every Boundable below prims carrying
    ``UsdPhysics.CollisionAPI`` (visibility and purpose are ignored on
    purpose: Isaac collision meshes are often invisible/guide). Falls back to
    all geometry under the link if no collider is found. Our own
    ``tactile_pad`` prims are skipped.
    """
    cache = cache or UsdGeom.XformCache()
    T_w_link = world_matrix(link, cache)
    T_link_w = np.linalg.inv(T_w_link)

    def under_collider(p: Usd.Prim) -> bool:
        while p and p != link.GetParent():
            if p.HasAPI(UsdPhysics.CollisionAPI):
                return True
            p = p.GetParent()
        return False

    def gather(only_colliders: bool):
        pts = []
        it = iter(iter_prims(link))
        for p in it:
            if p != link and (is_rigid_body(p) or p.GetName() == PAD_PRIM_NAME):
                it.PruneChildren()  # nested links and our own pads are not this finger's geometry
                continue
            if not p.IsA(UsdGeom.Boundable) or p.IsA(UsdGeom.PointInstancer):
                continue
            if only_colliders and not under_collider(p):
                continue
            ext = _local_extent(p)
            if ext is None:
                continue
            lo, hi = ext
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pts.append(transform_points(T_link_w @ world_matrix(p, cache), corners))
        return np.vstack(pts) if pts else None

    pts = gather(colliders_only)
    if pts is None and colliders_only:
        log.warning("no colliders under %s; using visual geometry for its AABB", link.GetPath())
        pts = gather(False)
    if pts is None:
        raise RuntimeError(f"no geometry found under {link.GetPath()}")
    return pts.min(axis=0), pts.max(axis=0)


def describe(root: Usd.Prim, grep: str = "") -> str:
    """Human-readable summary of articulation roots, bodies and joints."""
    g = grep.lower()
    lines = [f"== {root.GetPath()}"]
    cache = UsdGeom.XformCache()
    arts = [p for p in iter_prims(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    lines.append(f"articulation roots: {[str(p.GetPath()) for p in arts] or 'none'}")
    lines.append("rigid bodies:")
    for b in rigid_bodies(root):
        if g and g not in str(b.GetPath()).lower():
            continue
        t = world_matrix(b, cache)[:3, 3]
        ncol = sum(1 for p in iter_prims(b) if p.HasAPI(UsdPhysics.CollisionAPI))
        lines.append(f"  {b.GetPath()}  pos=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})  colliders={ncol}")
    lines.append("joints:")
    for j in joints(root):
        if g and g not in str(j.GetPath()).lower():
            continue
        b0, b1 = joint_bodies(j)
        extra = ""
        if j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint):
            api = UsdPhysics.RevoluteJoint(j) if j.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(j)
            lo, hi = api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()
            extra = f" limits=[{lo}, {hi}]"
        drives = [s for s in j.GetAppliedSchemas() if "Drive" in s]
        lines.append(f"  {j.GetName():<32} {joint_kind(j):<10} {b0 or '<world>'} -> {b1 or '<world>'}{extra} {drives}")
    return "\n".join(lines)
