"""Isaac-free checks: asset files resolve and pad/box meshes are well formed."""
import xml.etree.ElementTree as ET
import numpy as np
from nova5_tactile import ASSETS, NOVA5_USD, GRIPPER_URDF
from nova5_tactile import pad_geometry as g


def test_assets_present():
    assert NOVA5_USD.is_file()
    for e in ET.parse(GRIPPER_URDF).getroot().iter('mesh'):
        assert (ASSETS/e.get('filename').removeprefix('package://')).is_file(), e.get('filename')


def test_meshes_closed_and_outward():
    pts, tris = g.frustum()
    expected = g.THICKNESS/3*(np.prod(g.BACK_SIZE)+np.prod(g.CONTACT_SIZE)+np.sqrt(np.prod(g.BACK_SIZE)*np.prod(g.CONTACT_SIZE)))
    assert np.isclose(g.signed_volume(pts, tris), expected)
    pts, tris = g.box((.04, .04, .05))
    assert np.isclose(g.signed_volume(pts, tris), .04*.04*.05)


def test_tacsl_senses_contact_face():
    # TacSL picks the sensing side as the one the centre of mass leans towards.
    pts, tris = g.frustum()
    assert g.centre_of_mass(pts, tris)[2] < (pts[:, 2].min()+pts[:, 2].max())/2


def test_pad_faces_opposite_finger():
    pose = g.finger_to_pad()
    # Left inner finger: the opposite finger lies along finger -x, so pad z (into the finger) is +x.
    assert np.allclose(pose[:3, 2], [1, 0, 0]) and np.isclose(np.linalg.det(pose[:3, :3]), 1)


def test_taxel_grid_inside_contact_face():
    dx, x, y = g.taxel_grid((6, 9), .002)
    assert dx > 0 and len(x) == 6 and len(y) == 9
    assert np.abs(x).max() < g.CONTACT_SIZE[0]/2 and np.abs(y).max() < g.CONTACT_SIZE[1]/2
