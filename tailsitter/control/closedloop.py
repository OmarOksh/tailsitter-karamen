"""
Closed-loop flight of a Phase-2 reference with the Phase-3 INDI controller.

`build_reference` turns a (position, yaw) trajectory into the per-step reference the
controller needs (position/velocity/acceleration/yaw plus the flatness feedforward
attitude and body rate). `fly` then integrates the 6-DOF plant at the sim rate,
calling the controller every step, with optional external force/moment disturbances.
"""
import numpy as np
from ..dynamics.sim6dof import Tailsitter6DOF
from ..flatness.transform import FlatTransform
from ..trajgen.feasibility import make_sigma


def build_reference(traj_pos, traj_yaw, cfg, t0=None, t1=None):
    """Precompute reference arrays on the sim grid. Returns dict of arrays + ts."""
    dt = cfg.sim.dt
    t0 = traj_pos.knots[0] if t0 is None else t0
    t1 = traj_pos.knots[-1] if t1 is None else t1
    ts = np.arange(t0, t1 + 0.5 * dt, dt)
    sigma = make_sigma(traj_pos, traj_yaw)
    flat = FlatTransform(cfg)
    ref = flat.reference(sigma, ts)                    # q, Omega (feedforward) + inputs
    pos = np.array([traj_pos(t, 0) for t in ts])
    vel = np.array([traj_pos(t, 1) for t in ts])
    acc = np.array([traj_pos(t, 2) for t in ts])
    psi = np.array([0.0 if traj_yaw is None else float(traj_yaw(t, 0)[0]) for t in ts])
    return dict(ts=ts, pos=pos, vel=vel, acc=acc, psi=psi,
                q=ref["q"], Omega=ref["Omega"],
                w1=ref["w1"], w2=ref["w2"], d1=ref["d1"], d2=ref["d2"])


def fly(cfg, controller, traj_pos, traj_yaw=None, ref=None,
        f_ext=None, m_ext=None, x0=None):
    """
    Integrate the plant closed-loop.

    f_ext, m_ext : optional callables t -> (3,) external force / moment on the plant.
    x0           : optional initial state; default starts exactly on the reference.
    Returns a dict of logged arrays (state, input, reference, tracking error).
    """
    if ref is None:
        ref = build_reference(traj_pos, traj_yaw, cfg)
    ts = ref["ts"]
    dt = cfg.sim.dt
    plant = Tailsitter6DOF(cfg)

    if x0 is None:
        x0 = np.concatenate([ref["pos"][0], ref["vel"][0], ref["q"][0], ref["Omega"][0]])
    controller.reset(u0=np.array([ref["w1"][0], ref["w2"][0], ref["d1"][0], ref["d2"][0]]))

    N = len(ts)
    X = np.zeros((N, 13)); U = np.zeros((N, 4))
    x = x0.copy()
    for k in range(N):
        rk = dict(pos=ref["pos"][k], vel=ref["vel"][k], acc=ref["acc"][k],
                  psi=ref["psi"][k], Omega=ref["Omega"][k])
        u = controller.update(ts[k], x, rk)
        X[k] = x; U[k] = u
        if k < N - 1:
            fe = None if f_ext is None else f_ext(ts[k])
            me = None if m_ext is None else m_ext(ts[k])
            x = plant.step(x, u, dt, f_ext=fe, m_ext=me)

    pos_err = np.linalg.norm(X[:, 0:3] - ref["pos"], axis=1)
    return dict(ts=ts, X=X, U=U, ref=ref,
                pos_err=pos_err,
                rmse=float(np.sqrt(np.mean(pos_err ** 2))),
                max_err=float(pos_err.max()))
