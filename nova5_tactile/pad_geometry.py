"""Fingertip tactile pad geometry (pure numpy, testable without Isaac).

Pad frame (TacSL convention: the tactile normal is the body z axis, pointing
from the touched object into the sensor): +z points into the finger, x spans
the finger width and y runs along the finger. The pad is a frustum: the back
face (z=0) sits on the Robotiq inner-finger pad face, the larger contact face
is at z=-thickness. TacSL picks the sensing side from the mesh centre of mass,
which a frustum shifts towards the larger contact face.
"""
import numpy as np

# Inner-finger CAD (inner_finger_fine.STL): flat inward face at x=-0.0078 m,
# y in [-0.011, 0.011], z in about [0.0075, 0.044] in the finger link frame.
FINGER_FACE_X = -.0078
PAD_CENTER_Z = .026
CONTACT_SIZE = (.022, .032)   # (x, y) of the contact face
BACK_SIZE = (.018, .028)
THICKNESS = .003


def frustum(contact_size=CONTACT_SIZE, back_size=BACK_SIZE, thickness=THICKNESS):
    """Closed, outward-oriented triangle mesh: (points[N,3], triangles[M,3])."""
    (cx, cy), (bx, by) = np.asarray(contact_size)/2, np.asarray(back_size)/2
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    back = [(sx*bx, sy*by, 0.) for sx, sy in corners]
    front = [(sx*cx, sy*cy, -thickness) for sx, sy in corners]
    points = np.array(back+front)
    tris = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    for k in range(4):
        a, b = k, (k+1) % 4
        tris += [[a, b+4, b], [a, a+4, b+4]]
    return points, np.array(tris)


def finger_to_pad():
    """4x4 pose of the pad frame in the inner-finger link frame (both fingers)."""
    pose = np.eye(4)
    pose[:3, :3] = np.eye(3)[:, [1, 2, 0]]  # pad x, y, z = finger y, z, x (z into the finger)
    pose[:3, 3] = [FINGER_FACE_X, 0., PAD_CENTER_Z]
    return pose


def taxel_grid(array_size, margin, contact_size=CONTACT_SIZE):
    """Taxel (x, y) layout that TacSL's ray casting reproduces on the contact face."""
    rows, cols = array_size
    dx = min((contact_size[0]-2*margin)/(rows+1), (contact_size[1]-2*margin)/(cols+1))
    x = (np.arange(rows)-(rows-1)/2)*dx
    y = (np.arange(cols)-(cols-1)/2)*dx
    return dx, x, y


def box(size):
    """Closed, outward-oriented box triangle mesh centred at the origin."""
    hx, hy, hz = np.asarray(size)/2
    points = np.array([[sx*hx, sy*hy, sz*hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return points, np.array([q[:3] for q in quads]+[[q[0], q[2], q[3]] for q in quads])


def signed_volume(points, triangles):
    a, b, c = (points[triangles[:, k]] for k in range(3))
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6)


def centre_of_mass(points, triangles):
    a, b, c = (points[triangles[:, k]] for k in range(3))
    v = np.einsum('ij,ij->i', a, np.cross(b, c))/6
    return ((a+b+c)/4*v[:, None]).sum(0)/v.sum()
