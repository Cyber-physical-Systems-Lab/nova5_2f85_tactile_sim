"""Robotiq 2F-85 pad kinematics from the URDF (pure numpy).

Used to pre-shape the gripper to an object width: pad_gap(q) is the distance
between the two pads' contact faces for driver angle q (finger_joint, rad).
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from . import GRIPPER_URDF
from . import pad_geometry

Q_MAX = .725


def _origin(e):
    o = e.find('origin')
    xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' '); rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ')
    T = np.eye(4); T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix(); T[:3, 3] = xyz
    return T


def _joints():
    doc = ET.parse(GRIPPER_URDF).getroot()
    return {j.find('child').get('link'): j for j in doc.findall('joint')}


def pad_face_in_base(q, side='left'):
    """Contact-face centre of the pad, in robotiq_85_base_link coordinates (m)."""
    joints = _joints()
    chain = [f'{side}_inner_knuckle', f'{side}_inner_finger']
    T = np.eye(4)
    for link in chain:
        j = joints[link]
        mimic = j.find('mimic'); mult = float(mimic.get('multiplier')) if mimic is not None else 1.
        axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
        R = np.eye(4); R[:3, :3] = Rotation.from_rotvec(axis*mult*q).as_matrix()
        T = T@_origin(j)@R
    face = T@pad_geometry.finger_to_pad()@np.array([0., 0., -pad_geometry.THICKNESS, 1.])
    return face[:3]


def pad_gap(q):
    """Signed gap along the base x axis (negative once the pads would overlap)."""
    return float(pad_face_in_base(q, 'left')[0]-pad_face_in_base(q, 'right')[0])


def q_for_gap(gap):
    """Driver angle giving the pad gap (clipped to the stroke)."""
    qs = np.linspace(0., Q_MAX, 300); gaps = np.array([pad_gap(q) for q in qs])
    return float(np.interp(gap, gaps[::-1], qs[::-1]))


def mimic_multipliers():
    """{joint name: q_joint / q_finger_joint} for the six revolute gripper joints."""
    doc = ET.parse(GRIPPER_URDF).getroot()
    out = {}
    for j in doc.findall('joint'):
        if j.get('type') != 'revolute' or 'finger' not in j.get('name') and 'knuckle' not in j.get('name'):
            continue
        m = j.find('mimic')
        out[j.get('name')] = float(m.get('multiplier')) if m is not None else 1.
    return out
