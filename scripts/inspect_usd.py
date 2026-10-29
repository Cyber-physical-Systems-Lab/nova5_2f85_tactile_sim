"""Print articulation roots, rigid bodies and joints of a robot USD.

    scripts/isaac_python.sh scripts/inspect_usd.py gripper          # resolved Robotiq 2F-85 asset
    scripts/isaac_python.sh scripts/inspect_usd.py nova5            # URDF-imported arm
    scripts/isaac_python.sh scripts/inspect_usd.py assembled --pads # final asset + pad frames
    scripts/isaac_python.sh scripts/inspect_usd.py /any/file.usd --grep finger

Use this to fill in link/joint names in config/default.toml.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nova5_tactile.config import load_config  # noqa: E402
from nova5_tactile.sim.app import launch, resolve_gripper_usd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("target", help="gripper | nova5 | assembled | <path or URL>")
ap.add_argument("--grep", default="", help="only show bodies/joints whose path contains this")
ap.add_argument("--pads", action="store_true", help="also print pad frames from the meta file")
ap.add_argument("--config", default=None)
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
args = ap.parse_args()
cfg = load_config(args.config, args.set)

app = launch(headless=True)
try:
    from pxr import Usd, UsdGeom

    from nova5_tactile.usd.query import describe

    if args.target == "gripper":
        url = resolve_gripper_usd(cfg)
    elif args.target == "nova5":
        res = cfg.paths.resolve(cfg.paths.nova5_usd_dir) / "import_result.json"
        url = json.loads(res.read_text())["usd"]
    elif args.target == "assembled":
        url = str(cfg.paths.resolve(cfg.paths.assembled_usd))
    else:
        url = args.target
    stage = Usd.Stage.Open(url)
    if stage is None:
        raise SystemExit(f"could not open {url}")
    print(f"file: {url}")
    print(f"upAxis={UsdGeom.GetStageUpAxis(stage)} metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}")
    print(describe(stage.GetDefaultPrim() or stage.GetPseudoRoot(), args.grep))

    if args.pads:
        meta = json.loads(Path(url).with_suffix(".meta.json").read_text())
        np.set_printoptions(precision=4, suppress=True)
        for name, p in meta["tactile"]["pads"].items():
            print(f"\npad '{name}' on {p['body']}  ({len(p['colliders'])} colliders)")
            print(f"  link AABB min={p['link_aabb'][0]} max={p['link_aabb'][1]}")
            print("  T_body_pad =\n" + str(np.asarray(p["T_body_pad"])))
        for w in meta["report"]["warnings"]:
            print(f"WARNING: {w}")
finally:
    app.close()
