import tempfile
import unittest
from pathlib import Path

import numpy as np

from nova5_tactile import math3d as m
from nova5_tactile.config import DEFAULT_CONFIG, load_config, parse_override
from nova5_tactile.tactile import auto_place_pad


class TestMath(unittest.TestCase):
    def test_quat_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            q = q if q[0] >= 0 else -q
            R = m.quat_wxyz_to_matrix(q)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(m.matrix_to_quat_wxyz(R), q, atol=1e-9)

    def test_rpy(self):
        R = m.rpy_deg_to_matrix([0, 0, 90])
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        R = m.rpy_deg_to_matrix([90, 0, 0])
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-12)

    def test_invert_and_points(self):
        T = m.make_transform(m.rpy_deg_to_matrix([10, 20, 30]), [1, 2, 3])
        np.testing.assert_allclose(T @ m.invert_transform(T), np.eye(4), atol=1e-12)
        p = np.array([[0.1, -0.2, 0.3]])
        np.testing.assert_allclose(m.transform_points(m.invert_transform(T), m.transform_points(T, p)), p, atol=1e-12)

    def test_usd_matrix_convention(self):
        # A Gf.Matrix4d for "translate (1,2,3)" has the translation in the last row.
        rows = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]]
        T = m.from_usd_matrix(rows)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(m.to_usd_rows(T), rows)

    def test_split_scale(self):
        T = m.make_transform(m.rpy_deg_to_matrix([0, 0, 45]), [1, 0, 0])
        S = T.copy()
        S[:3, :3] = S[:3, :3] @ np.diag([0.01, 0.01, 0.01])
        rigid, scale = m.split_scale(S)
        np.testing.assert_allclose(rigid, T, atol=1e-12)
        np.testing.assert_allclose(scale, [0.01] * 3)


class TestPadPlacement(unittest.TestCase):
    def test_inner_face_and_tip(self):
        # Finger link: 10 mm thick (x), 20 mm wide (y), 50 mm long (z, tip at +z).
        lo, hi = [-0.005, -0.01, 0.0], [0.005, 0.01, 0.05]
        place = auto_place_pad(lo, hi, toward_other_finger=[1, 0, 0], toward_base=[0, 0, -1],
                               size_x=0.03, size_y=0.016, thickness=0.002, protrusion=0.0005, tip_margin=0.002)
        T = place.T_link_pad
        R = T[:3, :3]
        np.testing.assert_allclose(R[:, 2], [1, 0, 0])  # normal towards the other finger
        np.testing.assert_allclose(R[:, 0], [0, 0, 1])  # rows towards the tip
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        # surface (z = +t/2 in pad frame) must sit `protrusion` beyond the finger face
        surface = T @ np.array([0, 0, 0.001, 1.0])
        self.assertAlmostEqual(surface[0], 0.005 + 0.0005)
        # distal edge of the pad = 0.05 - tip_margin
        distal = T @ np.array([0.015, 0, 0, 1.0])
        self.assertAlmostEqual(distal[2], 0.05 - 0.002)
        self.assertTrue(place.fits)

    def test_negative_side(self):
        lo, hi = [-0.005, -0.01, 0.0], [0.005, 0.01, 0.05]
        place = auto_place_pad(lo, hi, [-1, 0.1, 0], [0, 0, -1], 0.03, 0.016, 0.002)
        np.testing.assert_allclose(place.T_link_pad[:3, 2], [-1, 0, 0])
        self.assertAlmostEqual(np.linalg.det(place.T_link_pad[:3, :3]), 1.0)

    def test_does_not_fit(self):
        place = auto_place_pad([0, 0, 0], [0.01, 0.01, 0.01], [1, 0, 0], [0, 0, -1], 0.03, 0.016, 0.002)
        self.assertFalse(place.fits)


class TestConfig(unittest.TestCase):
    def test_default_loads(self):
        cfg = load_config()
        self.assertEqual(cfg.tactile.left.name, "left")
        self.assertEqual(len(cfg.arm.home_deg), 6)
        self.assertIsInstance(cfg.sim.physics_hz, float)

    def test_overrides(self):
        cfg = load_config(overrides=["tactile.rows=16", "gripper.mount_xyz=[0, 0, 0.01]", "tactile.pad_mode=single_collider"])
        self.assertEqual(cfg.tactile.rows, 16)
        self.assertEqual(cfg.gripper.mount_xyz, [0, 0, 0.01])
        self.assertEqual(cfg.tactile.pad_mode, "single_collider")

    def test_unknown_key_rejected(self):
        with self.assertRaises(KeyError):
            load_config(overrides=["tactile.rowz=3"])

    def test_parse_override(self):
        self.assertEqual(parse_override("a.b=1.5"), ("a.b", 1.5))
        self.assertEqual(parse_override('a.b="x"'), ("a.b", "x"))

    def test_custom_file(self):
        txt = Path(DEFAULT_CONFIG).read_text().replace("rows = 8", "rows = 2")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text(txt)
            self.assertEqual(load_config(p).tactile.rows, 2)


if __name__ == "__main__":
    unittest.main()
