#!/usr/bin/env python3
"""Expand xacro (if needed) and rewrite package:// URIs to absolute paths.

Plain Python (no Isaac Sim needed):
    python3 scripts/prepare_urdf.py --input nova5.urdf --search-root ~/ros2_ws/src
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nova5_tactile.urdf_tools import prepare_urdf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help=".urdf or .xacro")
    ap.add_argument("--search-root", action="append", default=[], type=Path,
                    help="directory searched for package.xml files (repeatable)")
    ap.add_argument("--output", type=Path, default=REPO / "assets/urdf/nova5.urdf")
    ap.add_argument("--xacro-arg", action="append", default=[], help="extra xacro arg, e.g. name:=value")
    a = ap.parse_args()

    rep = prepare_urdf(a.input, a.output, a.search_root, a.xacro_arg)
    print(f"wrote {rep.output}")
    print(f"links  ({len(rep.links)}): {rep.links}")
    print(f"joints ({len(rep.joints)}): {[n for n, t in rep.joints if t != 'fixed']} (+fixed)")
    for pkg, path in sorted(rep.packages.items()):
        print(f"package://{pkg} -> {path}")
    ok = True
    if rep.unresolved_packages:
        ok = False
        print(f"UNRESOLVED packages: {sorted(rep.unresolved_packages)} (add --search-root)", file=sys.stderr)
    if rep.missing_files:
        ok = False
        print("MISSING mesh files:\n  " + "\n  ".join(sorted(set(rep.missing_files))), file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
