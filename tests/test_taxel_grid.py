import unittest

import numpy as np

from nova5_tactile.tactile import (
    SensorModelParams,
    TactileFrame,
    TaxelGridSpec,
    apply_sensor_model,
    bin_contacts,
    orient_pair_forces,
)
from nova5_tactile.tactile.ascii_viz import side_by_side


def spec(**kw):
    base = dict(rows=4, cols=2, size_x=0.04, size_y=0.02, thickness=0.002)
    base.update(kw)
    return TaxelGridSpec(**base)


def push(z_force=-1.0, shear=(0.0, 0.0)):
    return np.array([[shear[0], shear[1], z_force]])


class TestSpec(unittest.TestCase):
    def test_centers(self):
        s = spec()
        c = s.centers()
        self.assertEqual(c.shape, (4, 2, 2))
        np.testing.assert_allclose(c[0, 0], [-0.015, -0.005])
        np.testing.assert_allclose(c[3, 1], [0.015, 0.005])
        self.assertAlmostEqual(s.taxel_area, 0.01 * 0.01)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            spec(mode="gaussian", spread_sigma=0.0)
        with self.assertRaises(ValueError):
            spec(rows=0)


class TestBinning(unittest.TestCase):
    def test_nearest_hits_single_taxel(self):
        s = spec()
        p = np.array([[0.015, 0.005, s.surface_z]])
        fr = bin_contacts(p, push(-2.0), s)
        self.assertAlmostEqual(fr.normal[3, 1], 2.0)
        self.assertAlmostEqual(fr.total_normal, 2.0)

    def test_force_conserved_all_modes(self):
        rng = np.random.default_rng(0)
        P = np.column_stack([rng.uniform(-0.02, 0.02, 50), rng.uniform(-0.01, 0.01, 50), np.full(50, 0.001)])
        F = np.column_stack([rng.normal(0, 0.1, 50), rng.normal(0, 0.1, 50), -rng.uniform(0, 1, 50)])
        for mode, sigma in (("nearest", 0.0), ("bilinear", 0.0), ("gaussian", 0.004)):
            fr = bin_contacts(P, F, spec(mode=mode, spread_sigma=sigma))
            self.assertAlmostEqual(fr.total_normal, float(-F[:, 2].sum()), places=9, msg=mode)
            np.testing.assert_allclose(fr.total_shear, F[:, :2].sum(0), atol=1e-12, err_msg=mode)

    def test_bilinear_at_center_is_exact(self):
        s = spec(mode="bilinear")
        c = s.centers()[1, 0]
        fr = bin_contacts([[c[0], c[1], s.surface_z]], push(-1.0), s)
        self.assertAlmostEqual(fr.normal[1, 0], 1.0)

    def test_bilinear_midpoint_splits(self):
        s = spec(mode="bilinear")
        c = s.centers()
        mid = 0.5 * (c[1, 0] + c[2, 0])
        fr = bin_contacts([[mid[0], mid[1], s.surface_z]], push(-1.0), s)
        self.assertAlmostEqual(fr.normal[1, 0], 0.5)
        self.assertAlmostEqual(fr.normal[2, 0], 0.5)

    def test_outside_area_and_far_contacts_ignored(self):
        s = spec()
        P = [[0.05, 0.0, s.surface_z], [0.0, 0.0, s.surface_z + 0.01]]
        fr = bin_contacts(P, np.vstack([push(), push()]), s)
        self.assertEqual(fr.total_normal, 0.0)

    def test_tensile_normal_clipped(self):
        s = spec()
        fr = bin_contacts([[0, 0, s.surface_z]], push(+1.0), s)
        self.assertEqual(fr.total_normal, 0.0)

    def test_direct_taxel_index(self):
        s = spec()
        P = [[0.0, 0.0, 0.0], [0.015, 0.005, s.surface_z]]
        F = np.vstack([push(-3.0), push(-1.0)])
        idx = [[2, 1], [-1, -1]]  # first contact assigned directly, second binned
        fr = bin_contacts(P, F, s, taxel_index=idx)
        self.assertAlmostEqual(fr.normal[2, 1], 3.0)
        self.assertAlmostEqual(fr.normal[3, 1], 1.0)

    def test_center_of_pressure(self):
        s = spec()
        c = s.centers()
        P = [[c[0, 0, 0], c[0, 0, 1], s.surface_z], [c[3, 1, 0], c[3, 1, 1], s.surface_z]]
        fr = bin_contacts(P, np.vstack([push(-1.0), push(-3.0)]), s)
        cop = fr.center_of_pressure(s)
        expected = (1.0 * c[0, 0] + 3.0 * c[3, 1]) / 4.0
        np.testing.assert_allclose(cop, expected)
        self.assertTrue(np.isnan(TactileFrame.empty(s).center_of_pressure(s)).all())

    def test_empty(self):
        fr = bin_contacts(np.zeros((0, 3)), np.zeros((0, 3)), spec())
        self.assertEqual(fr.total_normal, 0.0)
        self.assertFalse(fr.in_contact())


class TestOrientation(unittest.TestCase):
    def test_flip_when_pointing_out(self):
        F = np.array([[0.1, 0.0, 1.0], [0.1, 0.0, 2.0]])
        G = orient_pair_forces(F)
        np.testing.assert_allclose(G, -F)

    def test_keep_when_compressive(self):
        F = np.array([[0.1, 0.0, -1.0]])
        np.testing.assert_allclose(orient_pair_forces(F), F)


class TestSensorModel(unittest.TestCase):
    def setUp(self):
        self.s = spec()
        self.fr = TactileFrame(normal=np.full((4, 2), 5.0), shear=np.full((4, 2, 2), 3.0))

    def test_ideal_passthrough(self):
        self.assertIs(apply_sensor_model(self.fr, SensorModelParams()), self.fr)

    def test_saturation_and_deadband(self):
        fr = TactileFrame(normal=np.array([[0.1, 5.0]]), shear=np.array([[[0.0, 0.0], [30.0, 40.0]]]))
        out = apply_sensor_model(fr, SensorModelParams(deadband=0.5, saturation=2.0))
        np.testing.assert_allclose(out.normal, [[0.0, 2.0]])
        self.assertAlmostEqual(float(np.linalg.norm(out.shear[0, 1])), 2.0)

    def test_noise_is_seeded(self):
        p = SensorModelParams(noise_std=0.1)
        a = apply_sensor_model(self.fr, p, np.random.default_rng(1))
        b = apply_sensor_model(self.fr, p, np.random.default_rng(1))
        np.testing.assert_allclose(a.normal, b.normal)
        self.assertTrue((a.normal >= 0).all())


class TestAscii(unittest.TestCase):
    def test_render(self):
        s = spec()
        fr = bin_contacts([[0.015, 0.005, s.surface_z]], push(-2.0), s)
        txt = side_by_side({"left": fr, "right": TactileFrame.empty(s)})
        lines = txt.splitlines()
        self.assertIn("left:", lines[0])
        self.assertEqual(len(lines), 1 + s.rows)
        self.assertIn("@@", lines[1])  # fingertip row (highest index) printed first


if __name__ == "__main__":
    unittest.main()
