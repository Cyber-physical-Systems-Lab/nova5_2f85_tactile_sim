"""NOVA5 + 2F-85 joint names and actuator config. Import-safe before Isaac Sim starts (no pxr)."""
from isaaclab.actuators import ImplicitActuatorCfg

ARM_JOINTS = [f'joint{k}' for k in range(1, 7)]
FLANGE = 'Link6'
# Vendor URDF has zero limits; these are synthetic, not hardware specifications.
ARM_EFFORT_NM = {name: 60. if k < 3 else 15. for k, name in enumerate(ARM_JOINTS)}


def actuators(gripper_joint_names, *, arm_stiffness=0., arm_damping=18., driver='finger_joint', grip_effort_limit=2.,
              grip_velocity_limit=1., grip_damping=.02):
    """Arm PD actuators plus an effort-driven gripper driver; the other gripper joints are passive (mimic)."""
    passive = [n for n in gripper_joint_names if n != driver]
    return {
        'arm': ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=arm_stiffness, damping=arm_damping,
                                   velocity_limit_sim=1.5, effort_limit_sim=ARM_EFFORT_NM),
        'gripper_drive': ImplicitActuatorCfg(joint_names_expr=[driver], stiffness=0., damping=grip_damping,
                                             effort_limit_sim=grip_effort_limit, velocity_limit_sim=grip_velocity_limit),
        'gripper_passive': ImplicitActuatorCfg(joint_names_expr=passive, stiffness=0., damping=0.,
                                               effort_limit_sim=grip_effort_limit, velocity_limit_sim=grip_velocity_limit)}


def gripper_actuators(spec, **arm):
    """Actuators for a grippers.GripperSpec."""
    if spec.effort_unit == 'N':   # prismatic jaws: m/s and N s/m
        return actuators(list(spec.joints), driver=spec.driver, grip_effort_limit=spec.effort_limit,
                         grip_velocity_limit=.15, grip_damping=20., **arm)
    return actuators(list(spec.joints), driver=spec.driver, grip_effort_limit=spec.effort_limit, **arm)
