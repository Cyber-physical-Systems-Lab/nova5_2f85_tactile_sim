"""Convert the prepared Nova5 URDF to USD with the Isaac Sim URDF importer.

    scripts/isaac_python.sh scripts/import_nova5.py [--urdf path] [--set key=value ...]

Uses the Isaac Sim 6.x ``URDFImporter`` API and falls back to the 4.5/5.x Kit
commands. Writes ``<nova5_usd_dir>/import_result.json`` for the assembly step.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nova5_tactile.config import load_config  # noqa: E402
from nova5_tactile.sim.app import enable_extension, launch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
ap.add_argument("--urdf", default=None, help="defaults to paths.nova5_urdf")
args = ap.parse_args()
cfg = load_config(args.config, args.set)

app = launch(headless=True)


def _pick_usd(result, out_dir: Path, stem: str) -> Path:
    p = Path(str(result)) if result else out_dir
    if p.is_file():
        return p
    hits = sorted((p if p.is_dir() else out_dir).glob(f"{stem}*.usd*")) or sorted(out_dir.rglob("*.usd*"))
    if not hits:
        raise RuntimeError(f"importer produced no USD in {out_dir}")
    return hits[0]


def import_urdf(urdf: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    except ImportError:
        URDFImporter = None
    if URDFImporter is not None:  # Isaac Sim 6.x
        conf = URDFImporterConfig(
            urdf_path=str(urdf),
            usd_path=str(out_dir),
            fix_base=True,  # bolted to the table
            merge_fixed_joints=False,  # keep Link6/flange frames addressable
            merge_mesh=False,
        )
        return _pick_usd(URDFImporter(conf).import_urdf(), out_dir, urdf.stem)

    import omni.kit.commands  # Isaac Sim 4.5 / 5.x

    _, ic = omni.kit.commands.execute("URDFCreateImportConfig")
    ic.fix_base = True
    ic.merge_fixed_joints = False
    ic.make_default_prim = True
    ic.create_physics_scene = False
    dest = out_dir / f"{urdf.stem}.usd"
    omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf), import_config=ic, dest_path=str(dest))
    return _pick_usd(dest, out_dir, urdf.stem)


try:
    enable_extension("isaacsim.asset.importer.urdf")
    urdf = cfg.paths.resolve(args.urdf or cfg.paths.nova5_urdf)
    if not urdf.exists():
        raise SystemExit(f"URDF not found: {urdf} (run scripts/fetch_nova5_description.sh first)")
    out_dir = cfg.paths.resolve(cfg.paths.nova5_usd_dir)
    usd = import_urdf(urdf, out_dir)
    (out_dir / "import_result.json").write_text(json.dumps({"usd": str(usd), "urdf": str(urdf)}, indent=2))
    print(f"[import_nova5] USD: {usd}")

    from pxr import Usd

    from nova5_tactile.usd.query import describe

    stage = Usd.Stage.Open(str(usd))
    print(describe(stage.GetDefaultPrim() or stage.GetPseudoRoot()))
finally:
    app.close()
