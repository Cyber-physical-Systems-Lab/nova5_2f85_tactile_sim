"""Render one RGB image of a USD file (asset checks).

  bash scripts/isaac_python.sh scripts/render_usd.py FILE.usd OUT.png [--eye x y z] [--target x y z] --enable_cameras
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('usd'); p.add_argument('png')
p.add_argument('--eye', type=float, nargs=3, default=(.25, -.30, .15))
p.add_argument('--target', type=float, nargs=3, default=(0., 0., 0.))
p.add_argument('--res', type=int, nargs=2, default=(1280, 720))
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device='cuda:0'))
    sim_utils.DomeLightCfg(intensity=2500.).func('/World/Light', sim_utils.DomeLightCfg(intensity=2500.))
    sim.stage.DefinePrim('/World/Asset', 'Xform').GetReferences().AddReference(str(__import__('pathlib').Path(args.usd).resolve()))
    w, h = args.res
    cam = Camera(CameraCfg(prim_path='/World/Cam', height=h, width=w, data_types=['rgb'], update_period=0.,
                           renderer_cfg=IsaacRtxRendererCfg(),
                           spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955, clipping_range=(.005, 50.))))
    sim.reset()
    cam.set_world_poses_from_view(torch.tensor([args.eye], device='cuda:0'), torch.tensor([args.target], device='cuda:0'))
    for _ in range(15):
        sim.render(); cam.update(.01)
    rgb = cam.data.output['rgb']; rgb = rgb.torch if hasattr(rgb, 'torch') else rgb
    Image.fromarray(rgb[0, ..., :3].cpu().numpy().astype(np.uint8)).save(args.png)
    print('RENDERED', args.png, flush=True)
    return 0


code = 1
try:
    code = main()
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close(exit_code=code)
raise SystemExit(code)
