"""
Phase-2 validation: minimum-snap generation, feasibility via the flatness
transform, and time-optimal scaling.

Run:  python tests/test_phase2.py
"""
import numpy as np
from tailsitter.config import load_config
from tailsitter.trajgen import (min_snap_trajectory, allocate_times,
                                TailsitterFeasibility, time_optimal_scale, maneuvers)

np.set_printoptions(precision=4, suppress=True)
cfg = load_config()


def ok(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:52s} {extra}")
    return cond


def _build(m, speed=None):
    speed = speed or m["nominal_speed"]
    k = allocate_times(m["waypoints"], speed)
    tp, ty = min_snap_trajectory(m["waypoints"], k, yaw=m["yaw"],
                                 v0=m["v0"] * speed / m["nominal_speed"],
                                 v1=m["v1"] * speed / m["nominal_speed"])
    return k, tp, ty


def test_minsnap_interp():
    print("T0  Minimum-snap interpolation & smoothness")
    m = maneuvers.loop(radius=3.0, n=16)
    k, tp, ty = _build(m)
    e_wp = max(np.linalg.norm(tp(k[j]) - m["waypoints"][j]) for j in range(len(k)))
    cont = 0.0
    for j in range(1, len(k) - 1):
        for der in range(4):
            cont = max(cont, np.linalg.norm(tp(k[j] - 1e-7, der) - tp(k[j] + 1e-7, der)))
    r1 = ok("waypoints interpolated", e_wp < 1e-8, f"err={e_wp:.1e}")
    r2 = ok("C3 continuous at interior knots", cont < 1e-2, f"jump={cont:.1e}")
    return r1 and r2


def test_feasibility_monotone():
    print("T1  Feasibility: flying the same shape slower lowers demand")
    m = maneuvers.loop(radius=3.0, n=16)
    feas = TailsitterFeasibility(cfg)
    _, tp_f, ty_f = _build(m, speed=10.0)
    _, tp_s, ty_s = _build(m, speed=4.0)
    rf, rs = feas.evaluate(tp_f, ty_f), feas.evaluate(tp_s, ty_s)
    print(f"     fast: dur={rf['duration']:.2f}s w_util={rf['w_util']:.2f} flap_util={rf['flap_util']:.2f}")
    print(f"     slow: dur={rs['duration']:.2f}s w_util={rs['w_util']:.2f} flap_util={rs['flap_util']:.2f}")
    return ok("rotor demand drops when slower", rs["w_peak"] <= rf["w_peak"])


def test_time_optimal():
    print("T2  Time-optimal scaling rides an actuator limit")
    allok = True
    for name in ["loop", "climbing_turn", "racing_gates"]:
        m = maneuvers.CATALOG[name]()
        k0 = allocate_times(m["waypoints"], m["nominal_speed"])
        res = time_optimal_scale(m["waypoints"], k0, yaw=m["yaw"], cfg=cfg,
                                 v0=m["v0"], v1=m["v1"], N=300)
        f = res["feas"]
        util = max(f["w_util"], f["flap_util"])
        print(f"     {name:16s} alpha*={res['alpha']:.2f} dur={f['duration']:.2f}s "
              f"w_util={f['w_util']:.2f} flap_util={f['flap_util']:.2f}")
        allok &= f["feasible"] and util > 0.9
    return ok("fastest feasible sits at a limit (util>0.9)", allok)


def test_catalog_runs():
    print("T3  All maneuvers generate + evaluate without error")
    feas = TailsitterFeasibility(cfg)
    allok = True
    for name, gen in maneuvers.CATALOG.items():
        try:
            m = gen()
            _, tp, ty = _build(m)
            r = feas.evaluate(tp, ty, N=200)
            print(f"     {name:26s} dur={r['duration']:5.2f}s w_util={r['w_util']:.2f} "
                  f"flap_util={r['flap_util']:6.2f} feasible={r['feasible']}")
        except Exception as e:
            print(f"     {name:26s} ERROR {e}")
            allok = False
    return ok("every maneuver ran", allok)


if __name__ == "__main__":
    print("=" * 74)
    res = [test_minsnap_interp(), test_feasibility_monotone(),
           test_time_optimal(), test_catalog_runs()]
    print("=" * 74)
    print("ALL PHASE-2 TESTS PASS" if all(res) else "SOME FAILED")
