import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from nova5_tactile.math3d import make_transform, rpy_deg_to_matrix, transform_points
from nova5_tactile.tactile.contact_source import PairContacts
from nova5_tactile.tactile.fingertip_sensor import FingertipTactileSensor, PadInfo, pads_from_meta
from nova5_tactile.tactile.recorder import TactileRecorder
from nova5_tactile.tactile.taxel_grid import TaxelGridSpec

SPEC = TaxelGridSpec(rows=4, cols=2, size_x=0.04, size_y=0.02, thickness=0.002)
DT = 1.0 / 240.0


def make_sensor(colliders=None):
    T_body_pad = make_transform(rpy_deg_to_matrix([0, 90, 0]), [0.01, 0.0, 0.05])
    pad = PadInfo("left", "/World/robot/finger", "/World/robot/finger/tactile_pad",
                  colliders or {"/pad/surface": None}, T_body_pad)
    return FingertipTactileSensor(pad, SPEC), T_body_pad


class TestFingertipSensor(unittest.TestCase):
    def setUp(self):
        self.sensor, self.T_body_pad = make_sensor()
        self.T_world_body = make_transform(rpy_deg_to_matrix([30, -20, 75]), [0.3, -0.1, 0.4])
        self.T_world_pad = self.T_world_body @ self.T_body_pad
        R = self.T_world_pad[:3, :3]
        self.n_world = R[:, 2]  # pad outward normal in world
        # contact on the surface, over taxel [3, 1]
        c = SPEC.centers()[3, 1]
        self.p_world = transform_points(self.T_world_pad, [[c[0], c[1], SPEC.surface_z]])

    def _pair(self, impulse_world, collider="/pad/surface"):
        return PairContacts(collider, "/World/obj", self.p_world, np.atleast_2d(impulse_world), np.zeros(1))

    def test_push_maps_to_taxel_and_newtons(self):
        # 2 N pushing into the pad (force on pad = -2 * n) plus 0.5 N shear along pad +x
        shear_dir = self.T_world_pad[:3, 0]
        f_world = -2.0 * self.n_world + 0.5 * shear_dir
        fr = self.sensor.compute([self._pair(f_world * DT)], self.T_world_body, DT, 1.0)
        self.assertAlmostEqual(fr.total_normal, 2.0, places=9)
        self.assertAlmostEqual(fr.normal[3, 1], 2.0, places=9)
        np.testing.assert_allclose(fr.shear[3, 1], [0.5, 0.0], atol=1e-9)
        self.assertEqual(fr.timestamp, 1.0)

    def test_opposite_physx_sign_is_corrected(self):
        f_world = -2.0 * self.n_world + 0.5 * self.T_world_pad[:3, 0]
        fr = self.sensor.compute([self._pair(-f_world * DT)], self.T_world_body, DT, 0.0)
        self.assertAlmostEqual(fr.total_normal, 2.0, places=9)
        np.testing.assert_allclose(fr.shear[3, 1], [0.5, 0.0], atol=1e-9)

    def test_taxel_collider_identity_wins(self):
        sensor, _ = make_sensor({"/pad/taxel_00_00": [0, 0]})
        fr = sensor.compute([self._pair(-1.0 * self.n_world * DT, "/pad/taxel_00_00")], self.T_world_body, DT, 0.0)
        self.assertAlmostEqual(fr.normal[0, 0], 1.0, places=9)  # assigned by collider, not by position

    def test_no_contacts(self):
        fr = self.sensor.compute([], self.T_world_body, DT, 0.0)
        self.assertEqual(fr.total_normal, 0.0)


class TestMetaAndRecorder(unittest.TestCase):
    def test_pads_from_meta(self):
        meta = {"tactile": {"spec": asdict(SPEC), "pads": {
            "left": {"body": "gripper/left_inner_finger", "frame": "gripper/left_inner_finger/tactile_pad",
                     "colliders": {"gripper/left_inner_finger/tactile_pad/taxel_00_00": [0, 0]},
                     "T_body_pad": np.eye(4).tolist()}}}}
        spec, pads = pads_from_meta(meta, "/World/robot/")
        self.assertEqual(spec, SPEC)
        self.assertEqual(pads[0].body_path, "/World/robot/gripper/left_inner_finger")
        self.assertIn("/World/robot/gripper/left_inner_finger/tactile_pad/taxel_00_00", pads[0].colliders)

    def test_recorder_roundtrip(self):
        sensor, _ = make_sensor()
        rec = TactileRecorder(SPEC)
        for k in range(3):
            fr = sensor.compute([], np.eye(4), DT, k * DT)
            rec.add(k * DT, {"left": fr, "right": fr}, finger_joint=0.1 * k)
        with tempfile.TemporaryDirectory() as d:
            p = rec.save(Path(d) / "log.npz")
            z = np.load(p)
            self.assertEqual(z["left/normal"].shape, (3, 4, 2))
            self.assertEqual(z["right/shear"].shape, (3, 4, 2, 2))
            np.testing.assert_allclose(z["finger_joint"], [0.0, 0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
