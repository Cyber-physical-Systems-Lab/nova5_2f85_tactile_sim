"""Gripper registry: the data the robot builder, the env config and the env need (no pxr / Isaac imports).

'cps_tactile': the parallel-jaw gripper for Daimon vision-based tactile sensors from
               github.com/Cyber-physical-Systems-Lab/master-thesis-tactile-gripper (default).
'robotiq_2f85': Robotiq 2F-85, the original gripper of this repo.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GripperSpec:
    name: str
    usd: str                     # robot file in assets/generated/
    joints: dict                 # joint name -> multiplier on the driver value (q_joint = mult * q)
    driver: str                  # actuated joint; positive = closing
    fingers: dict                # side -> body whose origin is the finger's contact-face centre
    effort_unit: str             # 'N' (prismatic) or 'N m' (revolute)
    grip_effort: float           # closing effort used to hold a peg
    effort_limit: float
    tool_offset: tuple           # contact-face midpoint in the NOVA5 flange frame, gripper half closed (m)
    tip_below_contact: float     # finger tip below the contact-face centres (m)
    peg_top_above_contact: float # how far the held peg's top sticks out above the contact-face centres (m)
    notes: tuple = field(default_factory=tuple)


GRIPPERS = {
    'cps_tactile': GripperSpec(
        name='cps_tactile', usd='nova5_cps_tactile.usd',
        joints={'left_finger_joint': 1., 'right_finger_joint': 1.}, driver='left_finger_joint',
        fingers={'left': 'left_finger', 'right': 'right_finger'},
        effort_unit='N', grip_effort=15., effort_limit=40.,
        # Defaults matching grippers/cps_tactile.geometry() with the DM-Tac WL vendor model; the env uses the
        # values build_assets.py measured when present. The two jaws' level plates put the fingertips 5 mm
        # apart in height; the lower tip is 24.9 mm below the contact midpoint. Holding the 50 mm peg with
        # its top level with the gel centres leaves 25.1 mm below the lower tip: full insertion.
        tool_offset=(0., 0., .159), tip_below_contact=.0249, peg_top_above_contact=0.,
        notes=('Rack-and-pinion parallel jaws; the two carriages are coupled by a mimic joint.',
               'Sensor from the DM-Tac WL vendor STEP; its mounting on the carriage is by rule (see cps_tactile.py).')),
    'robotiq_2f85': GripperSpec(
        name='robotiq_2f85', usd='nova5_2f85.usd',
        joints={'finger_joint': 1., 'left_inner_knuckle_joint': 1., 'left_inner_finger_joint': -1.,
                'right_inner_knuckle_joint': -1., 'right_inner_finger_joint': 1., 'right_outer_knuckle_joint': -1.},
        driver='finger_joint', fingers={'left': 'left_tactile_pad', 'right': 'right_tactile_pad'},
        effort_unit='N m', grip_effort=2., effort_limit=2.,
        tool_offset=(0., 0., .146), tip_below_contact=.020, peg_top_above_contact=.005),
}
DEFAULT = 'cps_tactile'
