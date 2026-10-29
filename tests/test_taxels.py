"""Contact-taxel binning and sensor model (ported tests, pad-frame convention of taxels.py)."""
import numpy as np
import pytest
from nova5_tactile import pad_geometry as g
from nova5_tactile.taxels import (TaxelGridSpec, TactileFrame, bin_contacts, orient_pair_forces,
                                  SensorModelParams, apply_sensor_model)

SURF = -g.THICKNESS


def test_orient_flips_outward_pair_only():
    pushing = np.array([[.1, 0, 2.], [0, .1, 1.]])
    assert np.allclose(orient_pair_forces(pushing), pushing)
    assert np.allclose(orient_pair_forces(-pushing), pushing)


def test_nearest_binning_location_and_totals():
    spec = TaxelGridSpec()
    c = spec.centers()
    P = np.array([[*c[5, 8], SURF], [*c[0, 0], SURF]])
    F = np.array([[.2, -.1, 3.], [0, 0, 1.]])
    f = bin_contacts(P, F, spec)
    assert f.normal[5, 8] == 3. and f.normal[0, 0] == 1. and np.isclose(f.total_normal, 4.)
    assert np.allclose(f.shear[5, 8], [.2, -.1]) and np.allclose(f.total_shear, [.2, -.1])


def test_direct_index_overrides_position():
    spec = TaxelGridSpec()
    f = bin_contacts([[0, 0, SURF]], [[0, 0, 2.]], spec, taxel_index=[[1, 7]])
    assert f.normal[1, 7] == 2. and f.total_normal == 2.


def test_pulling_and_off_pad_contacts_ignored():
    spec = TaxelGridSpec()
    f = bin_contacts([[0, 0, SURF], [.05, 0, SURF], [0, 0, .02]], [[0, 0, -1.], [0, 0, 5.], [0, 0, 5.]], spec)
    assert f.total_normal == 0.


@pytest.mark.parametrize('mode,sigma', [('bilinear', 0.), ('gaussian', .003)])
def test_spreading_conserves_force(mode, sigma):
    spec = TaxelGridSpec(mode=mode, spread_sigma=sigma)
    f = bin_contacts([[.001, .004, SURF]], [[.3, .1, 2.]], spec)
    assert np.isclose(f.total_normal, 2.) and np.allclose(f.total_shear, [.3, .1])
    assert (f.normal > 0).sum() > 1


def test_center_of_pressure():
    spec = TaxelGridSpec(); c = spec.centers()
    f = bin_contacts([[*c[2, 6], SURF]], [[0, 0, 1.]], spec)
    assert np.allclose(f.center_of_pressure(c), c[2, 6])
    assert np.isnan(TactileFrame.empty(spec).center_of_pressure(c)).all()


def test_sensor_model():
    rng = np.random.default_rng(0)
    n = np.array([[.01, .5, 3.]]); s = np.zeros((1, 3, 2)); s[0, 2] = [4., 0.]
    out_n, out_s = apply_sensor_model(n, s, SensorModelParams(deadband=.05, saturation=2., quantization=.25), rng)
    assert np.allclose(out_n, [[0., .5, 2.]]) and np.allclose(out_s[0, 2], [2., 0.])
    same_n, _ = apply_sensor_model(n, s, SensorModelParams(), rng)
    assert same_n is n
    noisy, _ = apply_sensor_model(np.zeros((50, 50)), np.zeros((50, 50, 2)), SensorModelParams(noise_std=.1), rng)
    assert (noisy >= 0).all() and noisy.std() > .03


def test_taxel_boxes_tile_contact_face():
    boxes = g.taxel_boxes((6, 9))
    assert len(boxes) == 54
    vol = sum(g.signed_volume(p, t) for _, (p, t) in boxes)
    assert np.isclose(vol, g.CONTACT_SIZE[0]*g.CONTACT_SIZE[1]*g.THICKNESS)
    (r, c), (p, _) = boxes[-1]
    assert (r, c) == (5, 8) and np.isclose(p[:, 0].max(), g.CONTACT_SIZE[0]/2) and np.isclose(p[:, 2].min(), SURF)
    assert np.allclose(TaxelGridSpec().centers()[5, 8], p[:, :2].mean(0))


def test_clip_keeps_edge_contacts():
    spec = TaxelGridSpec()
    P = [[spec.size_x/2+.0005, 0, SURF-.0001], [0, -spec.size_y/2-.001, SURF+.004]]
    F = [[0, 0, 1.], [.1, .2, 2.]]
    assert bin_contacts(P, F, spec).total_normal == 0.
    f = bin_contacts(P, F, spec, clip=True)
    assert np.isclose(f.total_normal, 3.) and f.normal[5, 4] == 1. and f.normal[3, 0] == 2.
