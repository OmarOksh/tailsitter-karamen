"""
Minimum-snap piecewise-polynomial trajectory generation.

Follows the same formulation as MIT-AERA `mfboTrajectory`
(`BaseTrajFunc`/`MinSnapTrajectory`), which is the Richter-Bry-Roy /
Mellinger-Kumar method:

  * each spatial axis is an independent piecewise polynomial;
  * the decision variables are the endpoint DERIVATIVES at the waypoints
    (position fixed at each waypoint, interior derivatives free);
  * sharing endpoint derivatives between adjacent segments makes the
    trajectory C^(min_order-1) continuous automatically;
  * we minimise the integral of the squared `min_order`-th derivative (snap,
    min_order=4, for position; acceleration/jerk for yaw), which is an
    unconstrained QP in the free interior derivatives.

Each segment is parameterised on s in [0,1] with real time t = t_i + T_i s, so
the QP stays well conditioned regardless of segment duration.

Output is a `PolyTrajectory` that can be sampled at ANY time for value +
derivatives -- exactly the smooth flat output sigma(t) the flatness transform
consumes.
"""
import numpy as np
from math import factorial


def _perm(n, r):
    return factorial(n) // factorial(n - r) if n >= r else 0


class PolyTrajectory:
    """Piecewise polynomial in real time. coeffs[i] are for p(s)=sum c_k s^k,
    s=(t-t_i)/T_i on segment i."""

    def __init__(self, knots, coeffs):
        self.knots = np.asarray(knots, float)       # (M+1,)
        self.coeffs = [np.asarray(c, float) for c in coeffs]  # M arrays (D, deg+1)
        self.M = len(coeffs)
        self.D = coeffs[0].shape[0]
        self.deg = coeffs[0].shape[1] - 1

    def _seg(self, t):
        if t <= self.knots[0]:
            return 0, 0.0
        if t >= self.knots[-1]:
            return self.M - 1, 1.0
        i = int(np.searchsorted(self.knots, t) - 1)
        i = min(max(i, 0), self.M - 1)
        T = self.knots[i + 1] - self.knots[i]
        return i, (t - self.knots[i]) / T

    def __call__(self, t, der=0):
        i, s = self._seg(t)
        T = self.knots[i + 1] - self.knots[i]
        c = self.coeffs[i]                          # (D, deg+1)
        out = np.zeros(self.D)
        for k in range(der, self.deg + 1):
            out += c[:, k] * _perm(k, der) * s ** (k - der)
        return out / T ** der

    def sample(self, ts, ders=(0, 1, 2)):
        ts = np.asarray(ts, float)
        return {d: np.array([self(t, d) for t in ts]) for d in ders}


def _endpoint_map(T, deg, n_der):
    """Map polynomial coeffs -> [ders at s=0 (0..n_der-1), ders at s=1 (0..n_der-1)]
    in REAL time. Returns (2*n_der, deg+1) matrix A with d = A c."""
    A = np.zeros((2 * n_der, deg + 1))
    for r in range(n_der):
        A[r, r] = factorial(r) / T ** r                      # s=0: only c_r survives
        for k in range(r, deg + 1):
            A[n_der + r, k] = _perm(k, r) / T ** r            # s=1
    return A


def _snap_hessian_norm(deg, min_order):
    """Normalised cost Hessian: Qbar[j,k]=int_0^1 D^m s^j D^m s^k ds, m=min_order."""
    Q = np.zeros((deg + 1, deg + 1))
    m = min_order
    for j in range(m, deg + 1):
        for k in range(m, deg + 1):
            Q[j, k] = _perm(j, m) * _perm(k, m) / (j + k - 2 * m + 1)
    return Q


def solve_axis(waypoints, knots, min_order=4, bc0=None, bc1=None):
    """
    Minimum-`min_order` polynomial through `waypoints` at times `knots`.

    waypoints : (M+1,) scalar positions
    knots     : (M+1,) times
    bc0, bc1  : optional dict {der: value} boundary derivatives at the two ends
                (default: derivatives 1..min_order-1 = 0)
    Returns coeffs list of (deg+1,) per segment (normalised s form).
    The polynomial degree is fixed by the endpoint-derivative parameterisation:
    deg = 2*min_order - 1 (snap->deg 7, jerk->deg 5, acc->deg 3).
    """
    deg = 2 * min_order - 1
    waypoints = np.asarray(waypoints, float)
    knots = np.asarray(knots, float)
    M = len(waypoints) - 1
    nd = min_order                          # endpoint derivatives per node (0..nd-1)

    # per-segment: coeffs = Ainv @ d_seg ; segment cost in d = Qd
    Ainv, Qd = [], []
    for i in range(M):
        T = knots[i + 1] - knots[i]
        A = _endpoint_map(T, deg, nd)
        Ai = np.linalg.inv(A)
        Qc = _snap_hessian_norm(deg, min_order) / T ** (2 * min_order - 1)
        Ainv.append(Ai)
        Qd.append(Ai.T @ Qc @ Ai)

    # global node-derivative vector D_glob: node j has ders [0..nd-1]
    n_nodes = M + 1
    ncols = nd * n_nodes

    def node_idx(j, r):
        return j * nd + r

    # assemble global Hessian over node derivatives
    H = np.zeros((ncols, ncols))
    for i in range(M):
        # segment i uses node i (rows 0..nd-1) and node i+1 (rows nd..2nd-1)
        gidx = [node_idx(i, r) for r in range(nd)] + [node_idx(i + 1, r) for r in range(nd)]
        for a in range(2 * nd):
            for b in range(2 * nd):
                H[gidx[a], gidx[b]] += Qd[i][a, b]

    # fixed vs free
    fixed = {}
    for j in range(n_nodes):
        fixed[node_idx(j, 0)] = waypoints[j]          # positions fixed everywhere
    bc0 = bc0 or {}
    bc1 = bc1 or {}
    for r in range(1, nd):
        fixed[node_idx(0, r)] = bc0.get(r, 0.0)       # start derivatives
        fixed[node_idx(M, r)] = bc1.get(r, 0.0)       # end derivatives
    free = [i for i in range(ncols) if i not in fixed]

    d = np.zeros(ncols)
    for i, val in fixed.items():
        d[i] = val
    if free:
        Hff = H[np.ix_(free, free)]
        Hfx = H[np.ix_(free, list(fixed.keys()))]
        dfx = np.array([fixed[k] for k in fixed])
        d[free] = np.linalg.solve(Hff, -Hfx @ dfx)

    # recover coeffs per segment
    coeffs = []
    for i in range(M):
        gidx = [node_idx(i, r) for r in range(nd)] + [node_idx(i + 1, r) for r in range(nd)]
        coeffs.append(Ainv[i] @ d[gidx])
    return coeffs


def min_snap_trajectory(waypoints, knots, pos_order=4, yaw=None, yaw_order=3,
                        v0=None, v1=None, a0=None, a1=None):
    """
    Build a PolyTrajectory for position (D=3) and, if given, yaw (D=1, appended
    as a 4th channel).

    waypoints : (M+1, 3) position waypoints
    knots     : (M+1,) times
    yaw       : optional (M+1,) yaw waypoints [rad]; unwrapped internally
    Returns (traj_pos, traj_yaw_or_None).
    """
    waypoints = np.asarray(waypoints, float)
    M = waypoints.shape[0] - 1
    pos_deg = 2 * pos_order - 1
    seg_coeffs = [np.zeros((3, pos_deg + 1)) for _ in range(M)]
    for ax in range(3):
        bc0 = {}
        bc1 = {}
        if v0 is not None:
            bc0[1] = float(v0[ax])
        if v1 is not None:
            bc1[1] = float(v1[ax])
        if a0 is not None:
            bc0[2] = float(a0[ax])
        if a1 is not None:
            bc1[2] = float(a1[ax])
        cax = solve_axis(waypoints[:, ax], knots, pos_order, bc0=bc0, bc1=bc1)
        for i in range(M):
            seg_coeffs[i][ax, :] = cax[i]
    traj_pos = PolyTrajectory(knots, seg_coeffs)

    traj_yaw = None
    if yaw is not None:
        yaw = np.unwrap(np.asarray(yaw, float))
        cyaw = solve_axis(yaw, knots, yaw_order)
        traj_yaw = PolyTrajectory(knots, [c.reshape(1, -1) for c in cyaw])
    return traj_pos, traj_yaw
