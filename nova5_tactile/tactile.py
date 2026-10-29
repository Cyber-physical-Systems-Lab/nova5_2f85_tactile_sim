"""TacSL force-field tactile sensors on the two fingertip pads. Import after AppLauncher.

Each pad is an elastomer rigid body; TacSL ray-casts a taxel grid onto its
contact face and, every step, queries the contact object's SDF to get per-taxel
penetration, normal force and shear force. Camera (GelSight RGB) tactile
rendering is off: it needs Nucleus calibration data and RTX cameras.
"""
import numpy as np
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image
from isaaclab_assets.sensors import GELSIGHT_R15_CFG

ARRAY_SIZE = (6, 9)  # taxels across finger width x along finger
MARGIN = .002


def make_tactile_sensors(pads, contact_object_path, *, array_size=ARRAY_SIZE, margin=MARGIN,
                         normal_stiffness=1., tangential_stiffness=.1, friction=2., debug_vis=False):
    """Returns {'left': VisuoTactileSensor, 'right': ...}. Create before sim.reset()."""
    sensors = {}
    for side, pad in pads.items():
        cfg = VisuoTactileSensorCfg(prim_path=pad+'/tactile_sensor', update_period=0., debug_vis=debug_vis,
            render_cfg=GELSIGHT_R15_CFG, enable_camera_tactile=False, enable_force_field=True,
            tactile_array_size=tuple(array_size), tactile_margin=margin,
            contact_object_prim_path_expr=contact_object_path,
            normal_contact_stiffness=normal_stiffness, tangential_stiffness=tangential_stiffness,
            friction_coefficient=friction)
        sensors[side] = VisuoTactileSensor(cfg)
    return sensors


def read(sensor, array_size=ARRAY_SIZE):
    """Numpy snapshot of env 0: normal (rows, cols), shear (rows, cols, 2), depth (rows, cols)."""
    d = sensor.data
    def grid(x, *tail):
        return x[0].detach().cpu().numpy().reshape(*array_size, *tail)
    return {'normal': grid(d.tactile_normal_force), 'shear': grid(d.tactile_shear_force, 2),
            'depth': grid(d.penetration_depth)}


def image(snapshot, resolution=30, normal_scale=None, shear_scale=None):
    """RGB float image in [0, 1]: normal force colour, shear arrows."""
    kw = {}
    if normal_scale:
        kw['normal_force_threshold'] = normal_scale
    if shear_scale:
        kw['shear_force_threshold'] = shear_scale
    return np.clip(compute_tactile_shear_image(snapshot['normal'], snapshot['shear'], resolution=resolution, **kw), 0, 1)
