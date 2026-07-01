"""
Time allocation and time-optimal scaling.

* `allocate_times` gives an initial knot schedule from segment geometry
  (segment time ~ segment length, so the nominal speed is roughly uniform).
* `time_optimal_scale` finds the FASTEST feasible version of a trajectory by
  scaling all segment durations by a single factor `alpha` and binary-searching
  the smallest `alpha` for which the tailsitter feasibility check still passes.
  Because minimum-snap is covariant under time scaling, scaling the knots and
  re-solving reproduces the exact time-scaled trajectory.

This mirrors the paper's approach: the flat feasibility check turns "how fast can
we fly this shape" into a 1-D search, with no closed-loop simulation.
"""
import numpy as np
from .minsnap import min_snap_trajectory
from .feasibility import TailsitterFeasibility


def allocate_times(waypoints, nominal_speed=5.0, power=1.0):
    """Initial knot times from cumulative segment length.
    power=1.0 -> time proportional to distance; power=0.5 spreads time toward
    longer segments a little less aggressively."""
    wp = np.asarray(waypoints, float)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    dt = (seg ** power) / nominal_speed
    dt *= seg.sum() / (nominal_speed * dt.sum())      # normalise mean speed
    return np.concatenate([[0.0], np.cumsum(dt)])


def scale_knots(knots, alpha):
    return knots[0] + alpha * (knots - knots[0])


def time_optimal_scale(waypoints, knots0, yaw=None, cfg=None,
                       pos_order=4, yaw_order=3, N=400, v0=None, v1=None,
                       alpha_lo=0.05, alpha_hi=20.0, tol=1e-3, verbose=False):
    """
    Return the fastest feasible time scaling.

    Scans alpha (time multiplier): small alpha = fast/aggressive, large = slow.
    Finds the minimum alpha whose trajectory is feasible everywhere.

    Returns dict: alpha, knots, traj_pos, traj_yaw, feas (feasibility summary).
    """
    assert cfg is not None
    feas = TailsitterFeasibility(cfg)

    def build(alpha):
        k = scale_knots(knots0, alpha)
        # entry/exit speed scales as 1/alpha (same path, slower clock)
        vv0 = None if v0 is None else np.asarray(v0) / alpha
        vv1 = None if v1 is None else np.asarray(v1) / alpha
        tp, ty = min_snap_trajectory(waypoints, k, pos_order=pos_order,
                                     yaw=yaw, yaw_order=yaw_order, v0=vv0, v1=vv1)
        return k, tp, ty

    def is_feasible(alpha):
        k, tp, ty = build(alpha)
        return feas.evaluate(tp, ty, N=N)

    # 1) find a feasible upper bound (slow enough)
    a_hi = 1.0
    res_hi = is_feasible(a_hi)
    while not res_hi["feasible"] and a_hi < alpha_hi:
        a_hi *= 1.5
        res_hi = is_feasible(a_hi)
        if verbose:
            print(f"  expand  alpha={a_hi:.3f}  feasible={res_hi['feasible']}"
                  f"  w_util={res_hi['w_util']:.2f} flap_util={res_hi['flap_util']:.2f}")
    if not res_hi["feasible"]:
        raise RuntimeError("No feasible time scaling found up to alpha_hi; "
                           "loosen the path or check parameters.")

    # 2) find an infeasible lower bound (too fast)
    a_lo = a_hi
    res_lo = res_hi
    while res_lo["feasible"] and a_lo > alpha_lo:
        a_lo *= 0.6
        res_lo = is_feasible(a_lo)
    if res_lo["feasible"]:
        a_lo = alpha_lo                                  # even fastest is feasible

    # 3) bisect for the boundary (smallest feasible alpha)
    while a_hi - a_lo > tol * a_hi:
        a_mid = 0.5 * (a_lo + a_hi)
        r = is_feasible(a_mid)
        if r["feasible"]:
            a_hi = a_mid
        else:
            a_lo = a_mid
        if verbose:
            print(f"  bisect  alpha={a_mid:.4f}  feasible={r['feasible']}")

    k, tp, ty = build(a_hi)
    return dict(alpha=a_hi, knots=k, traj_pos=tp, traj_yaw=ty,
                feas=feas.evaluate(tp, ty, N=N))
