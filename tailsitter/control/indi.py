"""
Tailsitter INDI trajectory-tracking controller  (Doc 3, Tal & Karaman, Sec. IV).

Cascade (their Table 1):

  position/velocity  --PD-->  a_c            (Eq. 37)
  linear accel       --INDI-> xi_c, T_c      (Eq. 42) via the flatness attitude map
  attitude/rate      --PD-->  Omega_dot_c    (Eqs. 43-45)
  angular accel      --INDI-> m_c            (Eq. 48) -> actuators via the allocation

The two INDI stages compare a *sensor-based* estimate (measured linear / angular
acceleration) with a *model-based* estimate (aerodynamic model at the last applied
input) and command the increment that cancels the difference. That difference IS the
unmodelled force/moment f_ext, m_ext (Eqs. 40, 46) -- so the controller absorbs the
A1/A2 flatness approximations, parameter error, and external disturbances with no
integral action. The nonlinear inversion is performed *through* the Phase-1 flatness
transform, exactly as the paper describes.

In simulation the "IMU" is read from the plant (`specific_force_body`, and the gyro
`Omega`); the model-based terms use the controller's own vehicle model, which may be
deliberately mismatched from the plant to exercise INDI's robustness.
"""
import numpy as np
from ..flatness.transform import FlatTransform
from ..dynamics.phi_theory import PhiTheory
from ..dynamics.sim6dof import Tailsitter6DOF
from ..utils.rotation import quat_to_R, quat_mul, quat_inv


def _lpf_alpha(cutoff_hz, dt):
    """First-order low-pass smoothing factor for a given cutoff."""
    if cutoff_hz is None or cutoff_hz <= 0:
        return 1.0                       # pass-through
    tau = 1.0 / (2 * np.pi * cutoff_hz)
    return dt / (tau + dt)


class TailsitterINDI:
    def __init__(self, cfg, model_cfg=None):
        """
        cfg       : plant/config used for gains, limits, dt.
        model_cfg : the *controller's* vehicle model (defaults to cfg). Pass a
                    perturbed config to simulate model mismatch.
        """
        self.cfg = cfg
        self.v = cfg.vehicle
        self.dt = cfg.sim.dt
        self.g = cfg.sim.g
        self.iz = np.array([0.0, 0.0, 1.0])

        mcfg = model_cfg or cfg
        self.flat = FlatTransform(mcfg)          # inversion uses the controller model
        self.phi = PhiTheory(mcfg)
        self.J = mcfg.vehicle.inertia
        self.Jinv = np.linalg.inv(self.J)
        self.m = mcfg.vehicle.mass
        self._plant = Tailsitter6DOF(cfg)        # for reading the "IMU" in sim

        c = cfg.ctrl
        self.Kx = np.atleast_1d(c.Kx) * np.ones(3)
        self.Kv = np.atleast_1d(c.Kv) * np.ones(3)
        self.Ka = np.atleast_1d(c.Ka) * np.ones(3)
        self.Kxi = np.atleast_1d(c.Kq) * np.ones(3)       # attitude P (Kxi)
        self.KOm = np.atleast_1d(c.Komega) * np.ones(3)   # rate P (KOmega)
        self.a_a = _lpf_alpha(c.lpf_cutoff_hz, self.dt)
        self.a_O = _lpf_alpha(c.lpf_cutoff_hz, self.dt)
        self.reset()

    def reset(self, u0=None):
        self.u_prev = np.array([1200.0, 1200.0, 0.0, 0.0]) if u0 is None else np.asarray(u0, float)
        self.a_lpf = None
        self.Om_lpf = None
        self.Om_lpf_prev = None
        self.phi_prev = 0.0

    # -------------------------------------------------------------- one control update
    def update(self, t, x, ref, imu=None):
        """
        x   : plant state [pos(3), vel(3), q(4), Omega(3)]
        ref : object/dict with pos, vel, acc, psi, Omega  (feedforward from Phase 2)
        imu : optional (a_body, Omega) to override the plant readout (real sensors).
        Returns the actuator command u = [w1, w2, d1, d2] (clipped to limits).
        """
        pos, vel = x[0:3], x[3:6]
        q, Om = x[6:10], x[10:13]
        R_ib = quat_to_R(q)
        rp, rv, ra = ref["pos"], ref["vel"], ref["acc"]
        rpsi, rOm = ref["psi"], ref["Omega"]

        # ---- "IMU": measured linear & angular acceleration from the plant ------------
        if imu is None:
            a_b = self._plant.specific_force_body(x, self.u_prev)      # accelerometer
            Om_meas = Om
        else:
            a_b, Om_meas = imu
        a_meas = self.g * self.iz + R_ib @ a_b                        # kinematic accel (Eq. 38)
        if self.a_lpf is None:
            self.a_lpf, self.Om_lpf, self.Om_lpf_prev = a_meas.copy(), Om_meas.copy(), Om_meas.copy()
        self.a_lpf += self.a_a * (a_meas - self.a_lpf)
        self.Om_lpf += self.a_O * (Om_meas - self.Om_lpf)
        Om_dot_lpf = (self.Om_lpf - self.Om_lpf_prev) / self.dt        # numerical (Eq. 46)
        self.Om_lpf_prev = self.Om_lpf.copy()

        # ---- OUTER: PD position/velocity -> acceleration command (Eq. 37) ------------
        ep, ev = rp - pos, rv - vel
        ac = (R_ib @ (self.Kx * (R_ib.T @ ep)
                      + self.Kv * (R_ib.T @ ev)
                      + self.Ka * (R_ib.T @ (ra - self.a_lpf)))
              + ra)

        # ---- INDI linear-acceleration control -> force cmd (Eq. 42) ------------------
        f_alpha, _, R_ia = self.phi.forces_moments(R_ib, vel, *self.u_prev)
        f_model = R_ia @ f_alpha                                       # model force, world (f_ilpf)
        f_ic = self.m * (ac - self.a_lpf) + f_model
        a_equiv = f_ic / self.m + self.g * self.iz                     # so attitude_thrust f_i == f_ic
        att = self.flat.attitude_thrust(a_equiv, rpsi, vel, phi_ref=self.phi_prev)
        self.phi_prev = att["phi"]
        qc, Tc = att["q"], att["T"]

        # ---- ATTITUDE PD -> angular-acceleration command (Eqs. 43-45) ----------------
        xi_e = quat_mul(quat_inv(q), qc)
        if xi_e[0] < 0:
            xi_e = -xi_e                                               # short way (double cover)
        w = np.clip(xi_e[0], -1.0, 1.0)
        s = np.sqrt(max(1.0 - w * w, 0.0))
        zeta_e = (2 * np.arccos(w) / s) * xi_e[1:4] if s > 1e-6 else 2.0 * xi_e[1:4]
        Om_dot_c = self.Kxi * zeta_e + self.KOm * (rOm - self.Om_lpf)

        # ---- INDI angular-acceleration control -> moment cmd (Eq. 48) ----------------
        _, m_model, _ = self.phi.forces_moments(R_ib, vel, *self.u_prev)   # mlpf
        mc = self.J @ (Om_dot_c - Om_dot_lpf) + m_model

        # ---- allocate (Tc, mc) -> actuators (reuse the flatness allocation) ----------
        Om_dot_eff = self.Jinv @ (mc - np.cross(Om, self.J @ Om))      # so m_req == mc
        alloc = self.flat.allocate(att, vel, Om, Om_dot_eff)
        u = np.array([alloc["w1"], alloc["w2"], alloc["d1"], alloc["d2"]])
        u[0:2] = np.clip(u[0:2], self.v.w_min, self.v.w_max)
        u[2:4] = np.clip(u[2:4], self.v.flap_min, self.v.flap_max)
        self.u_prev = u
        return u
