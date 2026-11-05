"""Build the generated USD assets into assets/generated/ (run once, ~1-2 min).

Needs assets/generated/tactile_gripper/ from scripts/convert_gripper_cad.py for the CPS gripper.

  nova5_cps_tactile.usd     NOVA5 + CPS-lab tactile gripper, one clonable articulation
  nova5_2f85.usd            NOVA5 + Robotiq 2F-85 (+ fingertip pads), one clonable articulation
  peg_{A..E}.usd            ManipulationNet steel pegs (visual CAD + convex colliders)
  board_visual.usd          NIST ATB M1 task board CAD (visual only)
  board_{A1..E4}.usd        board with exact colliders around that target hole
  peg_in_hole.json          hole centres/yaws/measured gaps, peg grip data, per-gripper data
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from nova5_tactile import ASSETS
from nova5_tactile.robot import author_nova5_2f85
from nova5_tactile.mnet import peg_in_hole as pih
from nova5_tactile.mnet import usd_assets
from nova5_tactile import gripper_kinematics
from nova5_tactile.grippers import GRIPPERS, cps_tactile

OUT = ASSETS/'generated'


def build_robot(name):
    """Write the RL robot USD for one gripper and return its metadata."""
    spec = GRIPPERS[name]
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = '/nova5'
    # Rigid (non-compliant) pad contact for RL; tactile work uses the compliant pads of build_nova5_2f85().
    robot = author_nova5_2f85(stage, root, pad_colliders='single', pad_stiffness=None, gripper=name)
    # RL asset: gravity off on every body (Isaac Lab's rigid_props stop at the first, outermost body).
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    stage.SetDefaultPrim(stage.GetPrimAtPath(root))
    stage.Export(str(OUT/spec.usd))
    g = robot['gripper']
    if name == 'robotiq_2f85':
        gap = [[float(q), gripper_kinematics.pad_gap(q)] for q in np.linspace(0, .725, 59)]
        extra = {}
    else:
        gap = cps_tactile.pad_gap_table(g['gap0_m'], g['q_range_m'])
        extra = {k: g[k] for k in ('gap0_m', 'tool_offset_m', 'tip_below_contact_m', 'q_range_m', 'sensor', 'sensor_m', 'assumptions')}
    print('robot', name, OUT/spec.usd, extra, flush=True)
    return {'usd': spec.usd, 'arm_joints': robot['arm_joints'], 'flange': robot['flange'],
            'gripper_joints': list(spec.joints), 'driver': spec.driver, 'fingers': spec.fingers,
            'pad_gap_table': gap, **extra}


def main():
    OUT.mkdir(exist_ok=True)
    meta = {'robots': {}, 'pegs': {}, 'holes': {}, 'board': {'thickness_m': pih.BOARD_THICKNESS, 'size_m': [.32, .25]}}
    for name in ('cps_tactile', 'robotiq_2f85'):
        meta['robots'][name] = build_robot(name)
    for shape in pih.SHAPES:
        meta['pegs'][shape] = usd_assets.write_peg(OUT/f'peg_{shape}.usd', shape)
        print('peg', shape, round(meta['pegs'][shape]['mass_kg']*1e3, 1), 'g', flush=True)
    usd_assets.write_board_visual(OUT/'board_visual.usd')
    for name, hole in pih.board_holes().items():
        n = usd_assets.write_board(OUT/f'board_{name}.usd', hole, 'board_visual.usd')
        meta['holes'][name] = {'center_m': hole.center.tolist(), 'yaw': hole.yaw, 'clearance_mm': hole.clearance_mm,
                               'fit_gap_mm': hole.fit_gap_mm, 'min_gap_mm': hole.min_gap_mm, 'colliders': n}
        print('board', name, n, 'colliders, gap', round(hole.fit_gap_mm, 4), 'mm', flush=True)
    (OUT/'peg_in_hole.json').write_text(json.dumps(meta, indent=1)+'\n')
    return 0


code = 1
try:
    code = main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close(exit_code=code)
raise SystemExit(code)
