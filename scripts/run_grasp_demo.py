"""Grasp demo for the assembled Nova5 + 2F-85 + tactile pads.

    scripts/isaac_python.sh scripts/run_grasp_demo.py            # GUI
    scripts/isaac_python.sh scripts/run_grasp_demo.py --headless --log logs/run.npz

Phases (durations in the [demo] config table):
  settle  arm -> arm.home_deg, gripper open
  close   a *kinematic* cube is spawned between the pads, gripper ramps to close_deg
  hold    the cube becomes dynamic (gravity on): it must be held by friction,
          so shear appears on the taxels in addition to the squeeze

No IK/motion planning on purpose: the object is placed relative to the
fingertips, which keeps this script independent of the arm pose.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from nova5_tactile.config import load_config  # noqa: E402
from nova5_tactile.sim.app import launch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--log", default=None, help="output .npz (default logs/grasp_<timestamp>.npz)")
ap.add_argument("--max-frames", type=int, default=0, help="stop after N app frames (0 = run all phases)")
args = ap.parse_args()
cfg = load_config(args.config, args.set)
headless = args.headless or cfg.sim.headless
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s %(message)s")

app = launch(headless=headless)

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

from nova5_tactile.math3d import make_transform, transform_points  # noqa: E402
from nova5_tactile.sim import scene  # noqa: E402
from nova5_tactile.sim.commander import JointCommander  # noqa: E402
from nova5_tactile.sim.pose import RigidBodyPoseSource  # noqa: E402
from nova5_tactile.tactile.ascii_viz import side_by_side  # noqa: E402
from nova5_tactile.tactile.colorize import TaxelColorizer  # noqa: E402
from nova5_tactile.tactile.fingertip_sensor import TwoFingerTactileRig, load_meta  # noqa: E402
from nova5_tactile.tactile.recorder import TactileRecorder  # noqa: E402
from nova5_tactile.tactile.sensor_model import SensorModelParams  # noqa: E402

PHASES = ("settle", "close", "hold")
log = logging.getLogger("grasp_demo")


def build_stage(usd: Path):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    app.update()
    stage = ctx.get_stage()
    scene.init_stage(stage)
    scene.physics_scene(stage, "/World/physicsScene", cfg.sim.gravity, cfg.sim.physics_hz)
    scene.ground(stage)
    scene.dome_light(stage)
    scene.add_reference(stage, cfg.demo.robot_prim, str(usd))
    for _ in range(5):
        app.update()
    return stage


def pad_surfaces(rig, pose):
    """World pose of each pad and the world point at the centre of its sensing surface."""
    out = {}
    for name, s in rig.sensors.items():
        T_wp = pose.get(s.pad.body_path) @ s.pad.T_body_pad
        out[name] = (T_wp, transform_points(T_wp, [[0.0, 0.0, rig.spec.surface_z]])[0])
    return out


def spawn_object(stage, rig, pose):
    surf = pad_surfaces(rig, pose)
    (T_a, p_a), (_, p_b) = list(surf.values())[:2]
    gap = float(np.linalg.norm(p_a - p_b))
    size = cfg.demo.object_size
    if size > 0.9 * gap:
        log.warning("object_size %.3f m does not fit the %.3f m opening; shrinking", size, gap)
        size = 0.8 * gap
    T_obj = make_transform(T_a[:3, :3], 0.5 * (p_a + p_b))  # faces parallel to the pads
    return scene.test_cube(stage, "/World/object", size, cfg.demo.object_mass, T_obj, kinematic=True), gap, size


def main():
    usd = cfg.paths.resolve(cfg.paths.assembled_usd)
    meta_path = usd.with_suffix(".meta.json")
    if not usd.exists() or not meta_path.exists():
        raise SystemExit(f"{usd} (+ .meta.json) missing: run scripts/assemble_robot.py first")
    meta = load_meta(meta_path)
    stage = build_stage(usd)

    robot = cfg.demo.robot_prim.rstrip("/")
    joint_paths = {n: f"{robot}/{j['path']}" for n, j in meta["joints"].items()}
    joint_types = {n: j["type"] for n, j in meta["joints"].items()}

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        app.update()

    t = cfg.tactile
    pose = RigidBodyPoseSource(stage)
    rig = TwoFingerTactileRig(meta, robot, pose, 1.0 / cfg.sim.physics_hz,
                              SensorModelParams(t.noise_std, t.deadband, t.saturation, t.quantization))
    rig.calibrate_pose_source()
    cmd = JointCommander(stage, robot, joint_paths, joint_types)
    vmax = t.saturation if t.saturation > 0 else None
    colorizer = TaxelColorizer(stage, rig.sensors, vmax=vmax or 5.0) if (cfg.demo.live_colors and not headless) else None
    rec = TactileRecorder(rig.spec)

    home = dict(zip(cfg.arm.joint_names, np.radians(cfg.arm.home_deg)))
    g = cfg.gripper.drive_joint
    g_open, g_close = np.radians(cfg.gripper.open_deg), np.radians(cfg.gripper.close_deg)

    t0 = timeline.get_current_time()
    phase, t_phase, frame, obj = "settle", 0.0, 0, None
    wall = time.time()
    try:
        while app.is_running():
            now = timeline.get_current_time() - t0
            if phase == "settle":
                cmd.set_targets({**home, g: g_open})
                if now >= cfg.demo.settle_s:
                    obj, gap, size = spawn_object(stage, rig, pose)
                    log.info("pad gap %.4f m, object %.4f m -> closing", gap, size)
                    phase, t_phase = "close", now
            elif phase == "close":
                a = min(1.0, (now - t_phase) / max(cfg.demo.close_s, 1e-6))
                cmd.set_targets({g: g_open + a * (g_close - g_open)})
                if now - t_phase >= cfg.demo.close_s + 0.3:
                    scene.set_kinematic(obj, False)
                    log.info("object released to gravity -> hold")
                    phase, t_phase = "hold", now
            elif phase == "hold" and now - t_phase >= cfg.demo.hold_s:
                break

            app.update()
            frames = rig.update(now)
            q = cmd.get_positions()
            rec.add(now, frames, finger_joint=(q or {}).get(g), phase=PHASES.index(phase))
            if colorizer and frame % 3 == 0:
                colorizer.update(frames)
            if cfg.demo.print_every and frame % cfg.demo.print_every == 0:
                print(f"\n t={now:6.3f}s  phase={phase:<6}  backend={cmd.backend}")
                print(side_by_side(frames, vmax))
            frame += 1
            if args.max_frames and frame >= args.max_frames:
                break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        out = Path(args.log) if args.log else REPO / "logs" / time.strftime("grasp_%Y%m%d_%H%M%S.npz")
        if len(rec):
            print(f"\n[grasp_demo] {len(rec)} frames in {time.time() - wall:.1f} s wall -> {rec.save(out)}")
        timeline.stop()


try:
    main()
finally:
    app.close()
