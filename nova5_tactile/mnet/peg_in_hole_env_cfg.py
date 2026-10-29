"""Isaac Lab config for the ManipulationNet Peg-in-Hole task on NOVA5 + Robotiq 2F-85."""
from isaaclab_physx.physics import PhysxCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from ..robot_cfg import ARM_JOINTS, actuators
from ..gripper_kinematics import mimic_multipliers
from .. import ASSETS

GENERATED = ASSETS/'generated'
ROBOT_BASE_HEIGHT = .70
TABLE_TOP = .75
# Where the target hole sits relative to the robot base (env frame, m). Reachable top-down.
HOLE_XY = (-.40, -.25)


@configclass
class PegInHoleEnvCfg(DirectRLEnvCfg):
    task: str = 'A1'                      # shape A-E + clearance level 1-4, as in ManipulationNet
    decimation = 8                        # 120 Hz physics, 15 Hz policy
    episode_length_s = 8.
    action_space = 4                      # fingertip dx, dy, dz, dyaw (scaled, clipped to [-1, 1])
    observation_space = 17
    state_space = 0

    # Control
    pos_action_scale = .006               # m per policy step
    yaw_action_scale = .05                # rad per policy step
    pos_bounds = (.05, .05, .10)          # fingertip target box around the hole top (m)
    ik_damping = .05
    max_joint_step = .08                  # rad, cap on one IK update
    # Lowest fingertip (pad-centre) height above the board top. The inner fingertips reach ~18 mm
    # below the pad centres; with the peg top 5 mm above the pads this still allows full insertion.
    min_tip_height = .021
    grip_effort = 2.                      # N m on finger_joint while holding
    # Reset randomisation
    start_height = (.010, .030)           # peg bottom above the board top (m)
    start_xy_noise = .012                 # m
    start_yaw_noise = .17                 # rad (~10 deg)
    # Success: peg bottom this far below the board top, and inside the hole
    success_depth = .020
    # Reward
    success_bonus = 1.
    # Optional RGB camera on env 0 for screenshots/videos (needs --enable_cameras)
    screenshot_camera = False
    action_penalty = .01

    sim: SimulationCfg = SimulationCfg(
        dt=1/120, gravity=(0., 0., -9.81),
        physics=PhysxCfg(solver_type=1, max_position_iteration_count=192, max_velocity_iteration_count=1,
                         bounce_threshold_velocity=.2, friction_offset_threshold=.01,
                         friction_correlation_distance=.00625, gpu_max_rigid_contact_count=2**23,
                         gpu_max_rigid_patch_count=2**23, gpu_collision_stack_size=2**28, gpu_max_num_partitions=1))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=1.5, clone_in_fabric=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(GENERATED/'nova5_2f85.usd'),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.,
                                                         solver_position_iteration_count=192, solver_velocity_iteration_count=1),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=192, solver_velocity_iteration_count=1,
                fix_root_link=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., ROBOT_BASE_HEIGHT),
            joint_pos={**dict(zip(ARM_JOINTS, [.27, 0., 1.52, .04, -1.57, 1.84])), **dict.fromkeys(mimic_multipliers(), 0.)}),
        actuators=actuators(list(mimic_multipliers()), arm_stiffness=800., arm_damping=80.))

    peg: RigidObjectCfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/Peg',
        spawn=sim_utils.UsdFileCfg(usd_path=str(GENERATED/'peg_A.usd')),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(HOLE_XY[0], HOLE_XY[1], TABLE_TOP+.3)))

    def __post_init__(self):
        super().__post_init__()
        if len(self.task) != 2 or self.task[0] not in 'ABCDE' or self.task[1] not in '1234':
            raise ValueError(f'task must be A-E followed by 1-4, got {self.task!r}')
        self.peg.spawn.usd_path = str(GENERATED/f'peg_{self.task[0]}.usd')
        self.viewer.origin_type = 'env'; self.viewer.env_index = 0
        self.viewer.resolution = (1280, 720)
        self.viewer.eye = (HOLE_XY[0]+.30, HOLE_XY[1]-.95, TABLE_TOP+.55)
        self.viewer.lookat = (HOLE_XY[0]+.12, HOLE_XY[1]+.08, TABLE_TOP+.12)
