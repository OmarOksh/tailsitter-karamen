"""
Phase-3 validation: closed-loop tracking with the INDI controller, and the
robustness that INDI is supposed to provide.

Run:  python tests/test_phase3.py
"""
import numpy as np
from dataclasses import replace
from tailsitter.config import load_config
from tailsitter.trajgen import min_snap_trajectory, allocate_times, maneuvers
from tailsitter.control import TailsitterINDI, fly, build_reference
from tailsitter.dynamics.sim6dof import Tailsitter6DOF

cfg = load_config()


def ok(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:52s} {extra}")
    return cond


def _loop_ref(scale=2.2):
    m = maneuvers.loop(radius=3.0, entry_alt=3.0, n=16, speed=6.0)
    k = allocate_times(m["waypoints"], m["nominal_speed"]) * scale
    tp, ty = min_snap_trajectory(m["waypoints"], k, yaw=m["yaw"],
                                 v0=m["v0"] / scale, v1=m["v1"] / scale)
    return tp, ty, build_reference(tp, ty, cfg)


def test_tracking():
    print("T0  Closed-loop tracking of an aerobatic loop (matched model)")
    tp, ty, ref = _loop_ref()
    log = fly(cfg, TailsitterINDI(cfg), tp, ty, ref=ref)
    print(f"     duration {log['ts'][-1]:.2f}s   pos RMSE {log['rmse']*100:.2f} cm   "
          f"max {log['max_err']*100:.2f} cm")
    return ok("RMSE < 15 cm on the loop", log["rmse"] < 0.15, f"{log['rmse']*100:.1f} cm")


def test_model_mismatch():
    print("T1  INDI absorbs model error (controller model != plant)")
    tp, ty, ref = _loop_ref()
    vb = cfg.vehicle
    mm = replace(cfg, vehicle=replace(vb, mass=vb.mass * 1.25, inertia=vb.inertia * 1.40,
                                      lTy=vb.lTy * 1.2, lDx=vb.lDx * 0.8, lDy=vb.lDy * 1.2))
    log = fly(cfg, TailsitterINDI(cfg, model_cfg=mm), tp, ty, ref=ref)
    print(f"     +25% mass, +40% inertia, arms off 20%  ->  RMSE {log['rmse']*100:.2f} cm")
    return ok("stays bounded, RMSE < 40 cm", np.isfinite(log["rmse"]) and log["rmse"] < 0.40,
              f"{log['rmse']*100:.1f} cm")


def test_disturbance():
    print("T2  INDI rejects a steady force disturbance (no integral action)")
    tp, ty, ref = _loop_ref()
    Fw = np.array([2.0, 1.0, 0.0])            # ~1/3 of weight
    log = fly(cfg, TailsitterINDI(cfg), tp, ty, ref=ref, f_ext=lambda t: Fw)
    print(f"     2.2 N steady wind  ->  RMSE {log['rmse']*100:.2f} cm")
    return ok("rejected, RMSE < 45 cm", log["rmse"] < 0.45, f"{log['rmse']*100:.1f} cm")


def test_feedback_matters():
    print("T3  Feedback is essential: open-loop feedforward diverges")
    tp, ty, ref = _loop_ref()
    log = fly(cfg, TailsitterINDI(cfg), tp, ty, ref=ref)
    # open-loop: apply the reference inputs directly, no feedback
    plant = Tailsitter6DOF(cfg); ts = ref["ts"]; dt = cfg.sim.dt
    x = np.concatenate([ref["pos"][0], ref["vel"][0], ref["q"][0], ref["Omega"][0]])
    err = []
    for k in range(len(ts)):
        u = np.array([ref["w1"][k], ref["w2"][k], ref["d1"][k], ref["d2"][k]])
        err.append(np.linalg.norm(x[0:3] - ref["pos"][k]))
        if k < len(ts) - 1:
            x = plant.step(x, u, dt)
    ol = np.sqrt(np.mean(np.array(err) ** 2))
    print(f"     closed-loop {log['rmse']*100:.1f} cm   vs   open-loop {ol*100:.0f} cm")
    return ok("closed-loop >20x better than open-loop", ol > 20 * log["rmse"],
              f"ratio {ol/log['rmse']:.0f}x")


if __name__ == "__main__":
    print("=" * 74)
    res = [test_tracking(), test_model_mismatch(), test_disturbance(), test_feedback_matters()]
    print("=" * 74)
    print("ALL PHASE-3 TESTS PASS" if all(res) else "SOME FAILED")
