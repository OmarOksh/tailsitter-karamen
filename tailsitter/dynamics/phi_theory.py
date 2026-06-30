"""
phi-theory force & moment model  (Tal-Ryou-Karaman, Doc 4, Eqs. 5-14).

Forces are summed in the ZERO-LIFT axis system; the moment is summed in the
BODY frame. The model has no lateral (body-y) force term -- the flying wing has
no fuselage / vertical tail, so any real lateral force is treated as unmodeled
(handled by the controller in closed loop, Phase 3).
"""
import numpy as np
from ..utils.rotation import Ry, quat_to_R


class PhiTheory:
    def __init__(self, cfg):
        self.v = cfg.vehicle
        self.abar = self.v.alpha0 + self.v.alphaT          # = alpha0 + alphaT
        # zero-lift <-> body (R_i_b = R_i_alpha @ Ry(alpha0)  =>  body = Ry(-a0) @ alpha)
        self.R_b_alpha = Ry(-self.v.alpha0)                # maps alpha -> body
        self.R_alpha_b = Ry(self.v.alpha0)                 # maps body  -> alpha

    # ----------------------------------------------------------- per-motor thrust
    def thrust_force_alpha(self, Ti):
        """Per-motor thrust force in zero-lift frame, Eq. 6."""
        a = self.abar
        return np.array([np.cos(a) * (1 - self.v.cDT),
                         0.0,
                         np.sin(a) * (self.v.cLT - 1)]) * Ti

    def flap_force_alpha(self, Ti, di, v_alpha, vnorm):
        """Per-flap force in zero-lift frame, Eq. 8."""
        fz = (self.v.cLT_flap * np.cos(self.abar) * Ti
              + self.v.cLV_flap * vnorm * v_alpha[0])
        return -np.array([0.0, 0.0, fz]) * di

    def wing_force_alpha(self, v_alpha, vnorm):
        """Wing force in zero-lift frame, Eq. 9."""
        return -np.array([self.v.cDV * v_alpha[0],
                          0.0,
                          self.v.cLV * v_alpha[2]]) * vnorm

    # ------------------------------------------------------------- full force/moment
    def forces_moments(self, R_ib, v_world, w1, w2, d1, d2):
        """
        Returns
        -------
        f_alpha : (3,)  total aerodynamic+thrust force, zero-lift frame  (Eq. 5)
        m_body  : (3,)  total moment, body frame                          (Eq. 10)
        R_ia    : (3,3) zero-lift -> world rotation (for the EOM)
        """
        v = self.v
        R_ia = R_ib @ self.R_b_alpha                       # zero-lift -> world
        v_alpha = R_ia.T @ v_world                         # velocity in zero-lift frame
        vnorm = np.linalg.norm(v_world)

        T1, T2 = v.cT * w1 * w1, v.cT * w2 * w2

        # --- force (zero-lift frame)
        fT1, fT2 = self.thrust_force_alpha(T1), self.thrust_force_alpha(T2)
        fd1 = self.flap_force_alpha(T1, d1, v_alpha, vnorm)
        fd2 = self.flap_force_alpha(T2, d2, v_alpha, vnorm)
        fw = self.wing_force_alpha(v_alpha, vnorm)
        f_alpha = fT1 + fT2 + fd1 + fd2 + fw

        # --- moment (body frame)
        Rba = self.R_b_alpha
        fT1_b, fT2_b = Rba @ fT1, Rba @ fT2
        m_T = np.array([
            v.lTy * (fT2_b[2] - fT1_b[2]),                 # roll  (iz comp diff)
            v.cmuT * (T1 + T2),                            # pitch
            v.lTy * (fT1_b[0] - fT2_b[0]),                 # yaw   (ix comp diff)
        ])
        mu1 = +v.cmu * w1 * w1                             # mu_i = -(-1)^i cmu wi^2
        mu2 = -v.cmu * w2 * w2
        m_mu = np.array([np.cos(v.alphaT), 0.0, -np.sin(v.alphaT)]) * (mu1 + mu2)
        # flap moment uses iz-component of per-flap force in zero-lift frame (Eq. 14)
        a0 = v.alpha0
        m_d = np.array([
            v.lDy * np.cos(a0) * (fd2[2] - fd1[2]),
            v.lDx * (fd1[2] + fd2[2]),
            v.lDy * np.sin(a0) * (fd2[2] - fd1[2]),
        ])
        m_body = m_T + m_mu + m_d
        return f_alpha, m_body, R_ia

    # --------------------------------------------------------- convenience wrappers
    def forces_moments_q(self, q, v_world, w1, w2, d1, d2):
        return self.forces_moments(quat_to_R(q), v_world, w1, w2, d1, d2)
