import tempfile
import unittest
from pathlib import Path

from nova5_tactile.urdf_tools import prepare_urdf

URDF = """<?xml version="1.0"?>
<robot name="nova5">
  <link name="base_link"><visual><geometry><mesh filename="package://nova5_description/meshes/base.stl"/></geometry></visual></link>
  <link name="Link1"><visual><geometry><mesh filename="package://missing_pkg/meshes/l1.stl"/></geometry></visual></link>
  <joint name="joint1" type="revolute"><parent link="base_link"/><child link="Link1"/></joint>
</robot>
"""


class TestPrepareUrdf(unittest.TestCase):
    def test_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pkg = d / "src" / "nova5_description"
            (pkg / "meshes").mkdir(parents=True)
            (pkg / "meshes" / "base.stl").write_text("solid x\nendsolid x\n")
            (pkg / "package.xml").write_text("<package><name>nova5_description</name></package>")
            src = pkg / "urdf" / "nova5.urdf"
            src.parent.mkdir()
            src.write_text(URDF)
            out = d / "out" / "nova5.urdf"
            rep = prepare_urdf(src, out, [d / "src"])
            text = out.read_text()
            self.assertIn(str((pkg / "meshes" / "base.stl").resolve()), text)
            self.assertIn("package://missing_pkg", text)  # left untouched, reported
            self.assertEqual(rep.unresolved_packages, {"missing_pkg"})
            self.assertEqual(rep.links, ["base_link", "Link1"])
            self.assertEqual(rep.joints, [("joint1", "revolute")])
            self.assertEqual(rep.missing_files, [])


if __name__ == "__main__":
    unittest.main()
