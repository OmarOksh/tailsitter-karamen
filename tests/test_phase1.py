"""
Phase-1 validation.

Run:  python tests/test_phase1.py    (from repo root, after `pip install -e .`)

The flatness transform makes two APPROXIMATIONS that the paper states explicitly:
  (A1) the attitude+thrust solve neglects the direct flap force (non-minimum
       phase) -- Doc 4, Sec. III-A;
  (A2) the control allocation neglects the flap contribution to YAW moment
       (iz . m_delta), "negligible due to multiplication with sin(alpha0)"
       -- Doc 4, Eq. 52.
These tests verify the EXACT parts are exact to machine precision, and quantify
the two approximations so you can see they behave as documented (and shrink with
better alpha0 / cmuT values). The INDI controller (Phase 3) closes both gaps.
"""
import numpy as np
from tailsitter.config import load_config
from tailsitter.dynamics.sim6dof import Tailsitter6DOF
from tailsitter.flatness.transform import FlatTransform
from tailsitter.utils.rotation import quat_kinematics, quat_normalize

np.set_printoptions(precision=4, suppress=True)
cfg = load_config()
sim = Tailsitter6DOF(cfg)
flat = FlatTransform(cfg)
g, m = cfg.sim.g, cfg.vehicle.mass


def ok(name, val, tol):
    s = "PASS" if val < tol else "FAIL"
    print(f"  [{s}] {name:48s} = {val:.3e}  (tol {tol:.0e})")
    return val < tol


def info(name, val):
    print(f"  [info] {name:48s} = {val:.3e}")


def test_hover():
    print("T0  Hover trim (a=0, v=0)")
    a = np.zeros(3); vel = np.zeros(3); psi = 0.0
    att = flat.attitude_thrust(a, psi, vel)
    alloc = flat.allocate(att, vel, np.zeros(3), np.zeros(3))
    u = [alloc["w1"], alloc["w2"], alloc["d1"], alloc["d2"]]
    print(f"     T={att['T']:.3f} N (mg={m*g:.3f})  theta={np.degrees(att['theta']):.1f} deg  "
          f"w={u[0]:.0f},{u[1]:.0f} rad/s  delta={u[2]:.3f},{u[3]:.3f} rad")
    x = np.zeros(13); x[2] = -1.0; x[6:10] = att["q"]
    xd = sim.deriv(x, u)
    vdot, omdot = xd[3:6], xd[10:13]
    R_ia = att["R_ia"]
    fd1 = sim.phi.flap_force_alpha(alloc["T1"], u[2], R_ia.T @ vel, 0.0)
    fd2 = sim.phi.flap_force_alpha(alloc["T2"], u[3], R_ia.T @ vel, 0.0)
    vdot_expected = R_ia @ (fd1 + fd2) / m
    r1 = ok("angular accel == 0 (exact)", np.linalg.norm(omdot), 1e-10)
    r2 = ok("vdot == neglected flap-force term (A1 exact)",
            np.linalg.norm(vdot - vdot_expected), 1e-10)
    info("approximation A1 gap |vdot| [m/s^2]", np.linalg.norm(vdot))
    return r1 and r2


def circle_traj(r=3.0, speed=6.0, h=2.0):
    w = speed / r
    def traj(t):
        pos = np.array([r * np.cos(w * t), r * np.sin(w * t), -h])
        vel = np.array([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0.0])
        acc = np.array([-r * w * w * np.cos(w * t), -r * w * w * np.sin(w * t), 0.0])
        psi = w * t + np.pi / 2
        return pos, vel, acc, psi
    return traj


def test_reconstruction():
    print("T1  Reconstruction along a 6 m/s coordinated circle")
    traj = circle_traj()
    ts = np.linspace(0.0, 2.0, 4001)
    ref = flat.reference(traj, ts)
    idx = np.arange(50, len(ts) - 50, 200)
    f_res, m_rollpitch, m_yaw = [], [], []
    J = cfg.vehicle.inertia
    for i in idx:
        p, vv, aa, psi = traj(ts[i])
        att = flat.attitude_thrust(aa, psi, vv)
        m_req = J @ ref["Omega_dot"][i] + np.cross(ref["Omega"][i], J @ ref["Omega"][i])
        _, m_phi, _ = sim.phi.forces_moments(att["R_ib"], vv,
                                             ref["w1"][i], ref["w2"][i],
                                             ref["d1"][i], ref["d2"][i])
        m_rollpitch.append(np.linalg.norm((m_phi - m_req)[:2]))
        m_yaw.append(abs((m_phi - m_req)[2]))
        fT1 = sim.phi.thrust_force_alpha(cfg.vehicle.cT * ref["w1"][i] ** 2)
        fT2 = sim.phi.thrust_force_alpha(cfg.vehicle.cT * ref["w2"][i] ** 2)
        v_alpha = att["R_ia"].T @ vv
        fw = sim.phi.wing_force_alpha(v_alpha, np.linalg.norm(vv))
        a_rec = g * np.array([0, 0, 1.0]) + att["R_ia"] @ (fT1 + fT2 + fw) / m
        f_res.append(np.linalg.norm(a_rec - aa))
    r1 = ok("force/attitude reconstruction (Eqs 20-26 exact)", max(f_res), 1e-10)
    r2 = ok("roll+pitch moment reconstruction (exact)", max(m_rollpitch), 1e-9)
    info("approximation A2 gap, yaw moment [N m]", max(m_yaw))
    return r1 and r2


def test_omega_kinematics():
    print("T2  Numeric Omega is consistent with attitude kinematics")
    traj = circle_traj()
    ts = np.linspace(0.0, 2.0, 8001)
    ref = flat.reference(traj, ts)
    dt = ts[1] - ts[0]
    res = []
    for i in range(100, len(ts) - 100, 400):
        q_i, q_ip1 = ref["q"][i], ref["q"][i + 1]
        if np.dot(q_i, q_ip1) < 0:
            q_ip1 = -q_ip1
        q_pred = quat_normalize(q_i + dt * quat_kinematics(q_i, ref["Omega"][i]))
        res.append(np.linalg.norm(q_pred - q_ip1))
    return ok("integrate(xi_dot) matches next attitude", max(res), 1e-5)


if __name__ == "__main__":
    print("=" * 74)
    results = [test_hover(), test_reconstruction(), test_omega_kinematics()]
    print("=" * 74)
    print("ALL EXACT-IDENTITY TESTS PASS" if all(results) else "SOME FAILED")
