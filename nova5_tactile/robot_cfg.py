"""NOVA5 + 2F-85 joint names and actuator config. Import-safe before Isaac Sim starts (no pxr)."""
from isaaclab.actuators import ImplicitActuatorCfg

ARM_JOINTS = [f'joint{k}' for k in range(1, 7)]
FLANGE = 'Link6'
# Vendor URDF has zero limits; these are synthetic, not hardware specifications.
ARM_EFFORT_NM = {name: 60. if k < 3 else 15. for k, name in enumerate(ARM_JOINTS)}


def actuators(gripper_joint_names, *, arm_stiffness=0., arm_damping=18.):
    passive = [n for n in gripper_joint_names if n != 'finger_joint']
    return {
        'arm': ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=arm_stiffness, damping=arm_damping,
                                   velocity_limit_sim=1.5, effort_limit_sim=ARM_EFFORT_NM),
        'gripper_drive': ImplicitActuatorCfg(joint_names_expr=['finger_joint'], stiffness=0., damping=.02,
                                             effort_limit_sim=2., velocity_limit_sim=1.),
        'gripper_passive': ImplicitActuatorCfg(joint_names_expr=passive, stiffness=0., damping=0.,
                                               effort_limit_sim=2., velocity_limit_sim=1.)}
