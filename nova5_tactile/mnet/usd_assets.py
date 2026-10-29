"""Write the peg-in-hole USD assets (pxr only). Called by scripts/build_assets.py."""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt
from . import peg_in_hole as pih

STEEL_DENSITY = 7900.  # kg/m^3, 304 stainless
PEG_FRICTION = (.6, .5)
BOARD_FRICTION = (.3, .25)
TILE_HALF = .03        # m, half size of the exactly modelled square around the target hole


def _stage(root):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage


def _material(stage, path, friction, color, opacity=1.):
    mat = UsdShade.Material.Define(stage, path)
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(friction[0]); phys.CreateDynamicFrictionAttr(friction[1]); phys.CreateRestitutionAttr(0.)
    shader = UsdShade.Shader.Define(stage, path+'/Shader'); shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.3)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.9 if opacity == 1. else 0.)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def _mesh(stage, path, points, tris):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(tris)); mesh.CreateFaceVertexIndicesAttr(np.asarray(tris).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    return mesh


def _soup(tris):
    return tris.reshape(-1, 3), np.arange(len(tris)*3).reshape(-1, 3)


def _prism_mesh(poly, z0, z1):
    n = len(poly); pts = pih.prism(poly, z0, z1)
    tris = [[0, i+1, i] for i in range(1, n-1)]+[[n, n+i, n+i+1] for i in range(1, n-1)]
    for i in range(n):
        j = (i+1) % n; tris += [[i, j, n+j], [i, n+j, n+i]]
    return pts, np.array(tris)


def _colliders(stage, parent, polys, z0, z1, material, offset=.001):
    for k, poly in enumerate(polys):
        pts, tris = _prism_mesh(poly, z0, z1)
        mesh = _mesh(stage, f'{parent}/c{k:03d}', pts, tris)
        UsdGeom.Imageable(mesh).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()); api.CreateContactOffsetAttr(offset); api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')


def write_peg(path, shape):
    peg = pih.load_peg(shape)
    stage = _stage('/Peg'); root = stage.GetPrimAtPath('/Peg')
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16); rb.CreateSolverVelocityIterationCountAttr(4); rb.CreateSleepThresholdAttr(0.)
    mass = STEEL_DENSITY*pih.signed_area(peg.polygon)*pih.PEG_LENGTH
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    look = _material(stage, '/Peg/Material', PEG_FRICTION, (.75, .76, .78))
    pts, tris = _soup(peg.tris)
    vis = _mesh(stage, '/Peg/visual', pts, tris)
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(look)
    UsdGeom.Xform.Define(stage, '/Peg/collision')
    _colliders(stage, '/Peg/collision', pih.inside_pieces(peg.polygon), 0., pih.PEG_LENGTH, look)
    stage.Export(str(path))
    return {'mass_kg': mass, 'grip_yaw': peg.grip_yaw, 'grip_width_m': peg.grip_width,
            'length_m': pih.PEG_LENGTH, 'polygon_m': peg.polygon.tolist()}


def write_board_visual(path):
    stage = _stage('/BoardVisual')
    tris = pih.read_stl(pih.BOARD_STL)*pih.CM
    pts, idx = _soup(tris)
    mesh = _mesh(stage, '/BoardVisual/mesh', pts, idx)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(_material(stage, '/BoardVisual/Acrylic', BOARD_FRICTION, (.35, .62, .85), .7))
    stage.Export(str(path))


def write_board(path, hole, visual_name):
    """Static board for one target hole: CAD visual + exact convex colliders around that hole.

    Elsewhere the board is three-sided boxes (the other holes are filled for collision).
    """
    stage = _stage('/Board')
    UsdGeom.Xform.Define(stage, '/Board/visual').GetPrim().GetReferences().AddReference('./'+visual_name)
    mat = _material(stage, '/Board/Physics', BOARD_FRICTION, (.8, .9, .95))
    UsdGeom.Xform.Define(stage, '/Board/collision')
    c = hole.center; h = TILE_HALF
    polys = pih.outside_pieces(hole.polygon, c, np.array([h, h]))
    bx, by = 16.*pih.CM, 12.5*pih.CM
    lo, hi = c-h, c+h
    rect = lambda x0, x1, y0, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    polys += [p for p in (rect(-bx, lo[0], -by, by), rect(hi[0], bx, -by, by),
                          rect(lo[0], hi[0], -by, lo[1]), rect(lo[0], hi[0], hi[1], by))
              if np.ptp(p[:, 0]) > 1e-6 and np.ptp(p[:, 1]) > 1e-6]
    _colliders(stage, '/Board/collision', polys, 0., pih.BOARD_THICKNESS, mat)
    # Export the layer, not the flattened stage, so the relative reference to the shared visual survives.
    stage.GetRootLayer().Export(str(path))
    return len(polys)
