"""
Rotation utilities.

Conventions
-----------
* Quaternions are Hamilton, ordered [w, x, y, z] (scalar first), matching the
  convention used in the paper (Tal & Karaman, Doc 3/4).
* A quaternion / rotation matrix R maps BODY-frame vectors to WORLD-frame
  vectors:   v_world = R @ v_body  =  quat_rotate(q, v_body).
* Euler angles use the ZXY sequence (yaw psi about z, roll phi about x,
  pitch theta about y):   R_i_b = Rz(psi) @ Rx(phi) @ Ry(theta).
  With psi = phi = theta = 0 the body frame coincides with NED (wings-level
  forward flight toward north). The ZXY order moves the kinematic singularity
  to +-90 deg roll (which the tailsitter avoids) rather than +-90 deg pitch
  (which it visits in hover) -- see Doc 4, Sec. III-A.
"""
import numpy as np


# ---------------------------------------------------------------- basic axes
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ------------------------------------------------------------ ZXY <-> matrix
def euler_zxy_to_R(psi, phi, theta):
    """Body->world rotation matrix for ZXY Euler angles."""
    return Rz(psi) @ Rx(phi) @ Ry(theta)


def R_to_euler_zxy(R):
    """Recover (psi, phi, theta) from a body->world matrix (ZXY)."""
    # R = Rz(psi) Rx(phi) Ry(theta). phi = asin(R[2,1]).
    phi = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    psi = np.arctan2(-R[0, 1], R[1, 1])
    theta = np.arctan2(-R[2, 0], R[2, 2])
    return psi, phi, theta


# ----------------------------------------------------------- quaternion core
def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_mul(p, q):
    """Hamilton product p (x) q, both [w,x,y,z]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_inv(q):
    return quat_conj(q) / np.dot(q, q)


def quat_rotate(q, v):
    """Rotate vector v (body) into world: v_world = R(q) v."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def euler_zxy_to_quat(psi, phi, theta):
    return R_to_quat(euler_zxy_to_R(psi, phi, theta))


def quat_to_euler_zxy(q):
    return R_to_euler_zxy(quat_to_R(q))


# ------------------------------------------------------ kinematics / helpers
def quat_continuity(q, q_ref):
    """Return whichever of +-q is closer to q_ref (resolves sign ambiguity)."""
    return q if np.dot(q, q_ref) >= 0 else -q


def omega_from_quat_rate(q, qdot):
    """Body angular velocity from xi_dot = 0.5 * xi (x) [0, omega]."""
    return 2.0 * quat_mul(quat_inv(q), qdot)[1:]


def quat_kinematics(q, omega):
    """xi_dot = 0.5 * xi (x) [0, omega]  (body-frame omega)."""
    return 0.5 * quat_mul(q, np.array([0.0, omega[0], omega[1], omega[2]]))


def hat(v):
    """Skew-symmetric matrix so that hat(a) @ b = a x b."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
