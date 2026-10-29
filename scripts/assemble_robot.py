"""Build assets/usd/nova5_2f85_tactile.usd (+ .meta.json).

    scripts/isaac_python.sh scripts/assemble_robot.py [--gripper-usd URL] [--set key=value ...]

Inputs: the URDF-imported Nova5 USD (import_result.json) and the Robotiq
2F-85 USD (config paths.gripper_usd or the first existing candidate).
Re-run whenever you change [gripper] mount or [tactile] settings.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nova5_tactile.config import load_config  # noqa: E402
from nova5_tactile.sim.app import launch, resolve_gripper_usd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
ap.add_argument("--nova5-usd", default=None, help="defaults to <nova5_usd_dir>/import_result.json")
ap.add_argument("--gripper-usd", default=None, help="overrides paths.gripper_usd")
args = ap.parse_args()
cfg = load_config(args.config, args.set)
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s %(message)s")

app = launch(headless=True)
try:
    from nova5_tactile.usd.assemble import assemble

    nova5_usd = args.nova5_usd
    if not nova5_usd:
        res = cfg.paths.resolve(cfg.paths.nova5_usd_dir) / "import_result.json"
        if not res.exists():
            raise SystemExit(f"{res} missing: run scripts/import_nova5.py first")
        nova5_usd = json.loads(res.read_text())["usd"]
    if args.gripper_usd:
        cfg.paths.gripper_usd = args.gripper_usd
    gripper_usd = resolve_gripper_usd(cfg)

    out = cfg.paths.resolve(cfg.paths.assembled_usd)
    meta = assemble(cfg, nova5_usd, gripper_usd, out)

    print(f"\n[assemble] wrote {out}")
    print(f"  flange       : {meta['flange']}")
    print(f"  gripper base : {meta['gripper_base']}")
    print(f"  removed articulation roots: {meta['report']['removed_articulation_roots']}")
    print(f"  deactivated world joints  : {meta['report']['deactivated_world_joints']}")
    for name, p in meta["tactile"]["pads"].items():
        print(f"  pad {name:<5}: {p['frame']}  ({len(p['colliders'])} colliders)")
    for w in meta["report"]["warnings"]:
        print(f"  WARNING: {w}")
    print("Open the USD in Isaac Sim and check the orange pads sit on the inner finger faces.")
finally:
    app.close()
