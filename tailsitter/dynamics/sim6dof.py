"""
6-DOF rigid-body tailsitter simulation  (Doc 4, Eqs. 1-4).

State x = [pos(3), vel(3), quat(4, [w,x,y,z]), omega(3)]  (13,)
Inputs u = [w1, w2, d1, d2]  (rotor speeds [rad/s], flap deflections [rad]).

    xdot   = v
    vdot   = g*iz + (1/m)(R_ia f_alpha + f_ext)
    xidot  = 0.5 xi (x) [0, omega]
    omdot  = J^-1 (m_body + m_ext - omega x J omega)
"""
import numpy as np
from .phi_theory import PhiTheory
from ..utils.rotation import quat_to_R, quat_kinematics, quat_normalize, hat


class Tailsitter6DOF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phi = PhiTheory(cfg)
        self.J = cfg.vehicle.inertia
        self.Jinv = np.linalg.inv(self.J)
        self.m = cfg.vehicle.mass
        self.g = cfg.sim.g
        self.iz = np.array([0.0, 0.0, 1.0])

    # -------------------------------------------------------------- derivatives
    def deriv(self, x, u, f_ext=None, m_ext=None):
        pos, vel, q, om = x[0:3], x[3:6], x[6:10], x[10:13]
        q = quat_normalize(q)
        w1, w2, d1, d2 = u
        f_alpha, m_body, R_ia = self.phi.forces_moments(quat_to_R(q), vel, w1, w2, d1, d2)
        if f_ext is None:
            f_ext = np.zeros(3)
        if m_ext is None:
            m_ext = np.zeros(3)
        acc = self.g * self.iz + (R_ia @ f_alpha + f_ext) / self.m
        qdot = quat_kinematics(q, om)
        omdot = self.Jinv @ (m_body + m_ext - np.cross(om, self.J @ om))
        return np.concatenate([vel, acc, qdot, omdot])

    # --------------------------------------------------------------- RK4 step
    def step(self, x, u, dt=None, f_ext=None, m_ext=None):
        dt = dt or self.cfg.sim.dt
        k1 = self.deriv(x, u, f_ext, m_ext)
        k2 = self.deriv(x + 0.5 * dt * k1, u, f_ext, m_ext)
        k3 = self.deriv(x + 0.5 * dt * k2, u, f_ext, m_ext)
        k4 = self.deriv(x + dt * k3, u, f_ext, m_ext)
        x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_new[6:10] = quat_normalize(x_new[6:10])
        return x_new

    def rollout(self, x0, u_of_t, T, dt=None):
        """Open-loop rollout. u_of_t(t) -> (w1,w2,d1,d2). Returns (ts, xs)."""
        dt = dt or self.cfg.sim.dt
        n = int(round(T / dt))
        xs = np.zeros((n + 1, 13))
        ts = np.zeros(n + 1)
        xs[0] = x0
        for k in range(n):
            t = k * dt
            xs[k + 1] = self.step(xs[k], np.asarray(u_of_t(t)), dt)
            ts[k + 1] = t + dt
        return ts, xs

    # ------------------------------------------------------- accel readout (IMU-like)
    def specific_force_body(self, x, u):
        """Body-frame specific force (what an accelerometer reads): (R_ia f_alpha)/m
        expressed in body. Useful for Phase-3 INDI feedback."""
        pos, vel, q, om = x[0:3], x[3:6], x[6:10], x[10:13]
        w1, w2, d1, d2 = u
        f_alpha, _, R_ia = self.phi.forces_moments(quat_to_R(q), vel, w1, w2, d1, d2)
        f_world = R_ia @ f_alpha
        return quat_to_R(q).T @ (f_world / self.m)


def hover_state(pos=(0, 0, -1.0), psi=0.0):
    """Build a state at a given position; attitude/omega set later by trim."""
    from ..utils.rotation import euler_zxy_to_quat
    x = np.zeros(13)
    x[0:3] = pos
    x[6:10] = euler_zxy_to_quat(psi, 0.0, -np.pi / 2)   # hover: pitch -90 deg
    return x
