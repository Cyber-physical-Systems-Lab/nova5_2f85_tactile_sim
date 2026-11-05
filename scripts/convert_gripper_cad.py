"""Convert the tactile gripper's Solid Edge CAD to USD with Isaac Sim's HOOPS CAD converter.

  bash scripts/isaac_python.sh scripts/convert_gripper_cad.py

Source: github.com/Cyber-physical-Systems-Lab/master-thesis-tactile-gripper, cloned into
third_party/tactile_gripper. HOOPS does not find the parts an assembly references on Linux,
so the assembly is converted for its placements and the STEP exports for part geometry. Output: assets/generated/tactile_gripper/{assembly.usd, parts/<name>.usd}.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--cad', default='third_party/tactile_gripper/CAD')
p.add_argument('--out', default='assets/generated/tactile_gripper')
p.add_argument('--sensor', default='third_party/daimon/DM-Tac WL.STEP', help='Daimon DM-Tac WL STEP (from Daimon)')
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.headless = True
app = AppLauncher(args).app

import asyncio
from pathlib import Path
import omni.kit.app


def main():
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in ('omni.kit.converter.hoops_core',):
        mgr.set_extension_enabled_immediate(ext, True)
    from omni.kit.converter.hoops_core import HoopsConverterHelper
    cad = Path(args.cad).resolve(); out = Path(args.out).resolve(); (out/'parts').mkdir(parents=True, exist_ok=True)
    step = cad.parent/'STEP'
    # Assembly: placements only (part geometry inside Solid Edge files is not read on Linux).
    # Geometry: the STEP exports of the parts.
    jobs = [(cad/'ASM_Tactile_gripper.asm', out/'assembly.usd')]+[(f, out/'parts'/(f.stem+'.usd')) for f in sorted(step.glob('*.stp'))]
    # Daimon DM-Tac WL sensor (vendor STEP, supplied separately into third_party/daimon/).
    sensor = Path(args.sensor).resolve()
    if sensor.exists():
        jobs.append((sensor, out/'parts'/'DM_Tac_WL.usd'))
    else:
        print('WARNING', sensor, 'missing: the sensor fingers will be boxes', flush=True)
    helper = HoopsConverterHelper(); failed = 0
    for src, dst in jobs:
        task = asyncio.ensure_future(helper.create_import_task(str(src), str(dst), {'bOptimize': 'false'}))
        while not task.done():
            app.update()
        url, status = task.result()
        failed += status[0] != 0
        print('CONVERT', src.name, '->', dst.name, status[0], status[1], flush=True)
    return 2 if failed else 0

code = 1
try:
    code = main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close(exit_code=code)
raise SystemExit(code)
