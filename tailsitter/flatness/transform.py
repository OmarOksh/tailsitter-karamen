"""
Differential flatness transform  (Tal-Ryou-Karaman, Doc 4, Sec. III).

Flat output:  sigma = [x (3), psi]  -> position and yaw.
Given sigma and its time derivatives, the full state AND the control inputs are
recovered algebraically.

Implementation note
-------------------
The ALGEBRAIC parts are coded exactly from the paper:
  * attitude + collective thrust   (Eqs. 20-26)
  * control allocation from moment (Eqs. 51-56)
The angular velocity (Eqs. 27-38) and angular acceleration (Eqs. 39-50) are the
time-derivatives of the algebraic attitude map. Rather than transcribe those
long closed forms, we differentiate the exact algebraic map numerically along
the trajectory. This is mathematically equivalent, dramatically less
error-prone, and exactly how the quantities are consumed for offline trajectory
generation / simulation. (For embedded online use one would code the closed
forms for speed.)

Following the paper, the attitude+thrust solve NEGLECTS the direct (non-minimum
phase) flap force; the flaps are then fixed by the moment requirement. The small
residual this introduces is what the INDI controller cancels in closed loop.
"""
import numpy as np
from ..utils.rotation import (Rx, Ry, Rz, R_to_quat, quat_continuity,
                               omega_from_quat_rate)
from ..dynamics.phi_theory import PhiTheory


class FlatTransform:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = cfg.vehicle
        self.phi = PhiTheory(cfg)
        self.g = cfg.sim.g
        self.m = cfg.vehicle.mass
        self.iz = np.array([0.0, 0.0, 1.0])
        self.abar = self.v.alpha0 + self.v.alphaT
        # eta = ratio of lift to forward force due to thrust (Eq. 26)
        self.eta = (np.sin(self.abar) * (self.v.cLT - 1)
                    / (np.cos(self.abar) * (1 - self.v.cDT)))

    # ===================================================== attitude + thrust (Eqs 20-26)
    def attitude_thrust(self, a, psi, vel, phi_ref=0.0):
        v = self.v
        f_i = self.m * (a - self.g * self.iz)                       # Eq. 20

        # --- roll phi (Eq. 21), constraint i_y^T f_alpha = 0
        f_psi = Rz(-psi) @ f_i
        phi_base = -np.arctan2(f_psi[1], f_psi[2])
        phi = self._pick_k(phi_base, phi_ref)                      # choose k in {0,1}*pi

        f_phi = Rx(-phi) @ f_psi
        v_phi = Rx(-phi) @ Rz(-psi) @ vel
        vn = np.linalg.norm(vel)

        # --- pitch thetabar and collective thrust T (Eqs. 24-25)
        num = (self.eta * (f_phi[0] + v.cDV * vn * v_phi[0])
               - v.cLV * vn * v_phi[2] - f_phi[2])
        den = (self.eta * (f_phi[2] + v.cDV * vn * v_phi[2])
               + v.cLV * vn * v_phi[0] + f_phi[0])
        thetabar0 = np.arctan2(num, den)

        def thrust_of(tb):
            return (1.0 / (np.cos(self.abar) * (1 - v.cDT))) * (
                np.cos(tb) * f_phi[0] - np.sin(tb) * f_phi[2]
                + v.cDV * vn * (np.cos(tb) * v_phi[0] - np.sin(tb) * v_phi[2]))

        thetabar, T = thetabar0, thrust_of(thetabar0)
        if T < 0:                                                  # pick k so T >= 0
            thetabar, T = thetabar0 + np.pi, thrust_of(thetabar0 + np.pi)

        theta = thetabar + v.alpha0
        R_ia = Rz(psi) @ Rx(phi) @ Ry(thetabar)                    # zero-lift -> world
        R_ib = Rz(psi) @ Rx(phi) @ Ry(theta)                       # body      -> world
        q = quat_continuity(R_to_quat(R_ib), np.array([1.0, 0, 0, 0]))
        return dict(R_ia=R_ia, R_ib=R_ib, q=q, T=T,
                    phi=phi, thetabar=thetabar, theta=theta, psi=psi, f_i=f_i)

    @staticmethod
    def _pick_k(base, ref):
        cands = [base, base + np.pi, base - np.pi]
        return min(cands, key=lambda c: abs(c - ref))

    # ============================================ control allocation from moment (Eqs 51-56)
    def allocate(self, att, vel, Omega, Omega_dot):
        v = self.v
        J = self.cfg.vehicle.inertia
        m_req = J @ Omega_dot + np.cross(Omega, J @ Omega)         # Eq. 51

        # --- differential thrust (Eq. 53)
        D = (-np.sin(v.alphaT) * v.cmu / v.cT
             + v.lTy * (np.cos(v.alpha0) * np.cos(self.abar) * (1 - v.cDT)
                        - np.sin(v.alpha0) * np.sin(self.abar) * (v.cLT - 1)))
        dT = m_req[2] / D                                          # iz^T m / D
        T = att["T"]
        T1, T2 = 0.5 * (T + dT), 0.5 * (T - dT)                    # Eq. 54

        feas = (T1 >= 0.0) and (T2 >= 0.0)
        w1 = np.sqrt(max(T1, 0.0) / v.cT)
        w2 = np.sqrt(max(T2, 0.0) / v.cT)

        # --- flap deflections (Eqs. 55-56); subtract thrust+torque moment
        R_ib = att["R_ib"]
        _, m_Tmu, _ = self.phi.forces_moments(R_ib, vel, w1, w2, 0.0, 0.0)
        m_d = m_req - m_Tmu
        R_ia = att["R_ia"]
        v_alpha = R_ia.T @ vel
        vn = np.linalg.norm(vel)
        nu1 = -v.cLT_flap * np.cos(self.abar) * T1 - v.cLV_flap * vn * v_alpha[0]
        nu2 = -v.cLT_flap * np.cos(self.abar) * T2 - v.cLV_flap * vn * v_alpha[0]
        A = np.array([[-v.lDy * np.cos(v.alpha0) * nu1,  v.lDy * np.cos(v.alpha0) * nu2],
                      [ v.lDx * nu1,                      v.lDx * nu2]])
        d1, d2 = np.linalg.solve(A, np.array([m_d[0], m_d[1]]))

        within = (v.w_min <= w1 <= v.w_max and v.w_min <= w2 <= v.w_max
                  and v.flap_min <= d1 <= v.flap_max and v.flap_min <= d2 <= v.flap_max)
        return dict(w1=w1, w2=w2, d1=d1, d2=d2, T1=T1, T2=T2,
                    feasible=bool(feas and within), real_thrust=feas, in_bounds=within)

    # =================================================== full reference along a trajectory
    def reference(self, traj, ts):
        """
        traj(t) -> (pos(3), vel(3), acc(3), psi)   evaluated on a UNIFORM grid ts.
        Returns dict of arrays: q, T, Omega, Omega_dot, w1,w2,d1,d2, feasible, ...
        Omega / Omega_dot come from differentiating the algebraic attitude map.
        """
        ts = np.asarray(ts)
        dt = ts[1] - ts[0]
        N = len(ts)
        q = np.zeros((N, 4)); Tcol = np.zeros(N)
        phi = np.zeros(N); theta = np.zeros(N)
        vel = np.zeros((N, 3)); acc = np.zeros((N, 3))
        R_ia = np.zeros((N, 3, 3)); R_ib = np.zeros((N, 3, 3))

        phi_ref = 0.0
        for i, t in enumerate(ts):
            p, vv, aa, psi = traj(t)
            vel[i], acc[i] = vv, aa
            att = self.attitude_thrust(aa, psi, vv, phi_ref=phi_ref)
            phi_ref = att["phi"]
            qi = att["q"] if i == 0 else quat_continuity(att["q"], q[i - 1])
            q[i] = qi; Tcol[i] = att["T"]
            phi[i], theta[i] = att["phi"], att["theta"]
            R_ia[i], R_ib[i] = att["R_ia"], att["R_ib"]

        # angular velocity / acceleration by finite difference of the attitude map
        qdot = np.gradient(q, dt, axis=0)
        Omega = np.array([omega_from_quat_rate(q[i], qdot[i]) for i in range(N)])
        Omega_dot = np.gradient(Omega, dt, axis=0)

        # control allocation per sample
        w1 = np.zeros(N); w2 = np.zeros(N); d1 = np.zeros(N); d2 = np.zeros(N)
        feasible = np.zeros(N, dtype=bool)
        for i in range(N):
            att = dict(R_ia=R_ia[i], R_ib=R_ib[i], T=Tcol[i])
            alloc = self.allocate(att, vel[i], Omega[i], Omega_dot[i])
            w1[i], w2[i] = alloc["w1"], alloc["w2"]
            d1[i], d2[i] = alloc["d1"], alloc["d2"]
            feasible[i] = alloc["feasible"]

        return dict(ts=ts, q=q, T=Tcol, phi=phi, theta=theta,
                    Omega=Omega, Omega_dot=Omega_dot,
                    w1=w1, w2=w2, d1=d1, d2=d2, feasible=feasible)
