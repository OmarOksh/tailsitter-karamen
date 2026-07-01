"""
Trajectory feasibility for the tailsitter.

This is the drop-in replacement for `quadModel.getWs_vector` in the MIT-AERA
pipeline: instead of mapping a trajectory to quadrotor rotor speeds, we push the
flat output through the tailsitter differential-flatness transform (Phase 1) and
read off the implied rotor speeds and flap deflections. A trajectory is feasible
iff those stay within the actuator limits everywhere (and the per-motor thrust
stays non-negative). No time integration is needed -- feasibility is a pure
algebraic evaluation of the flat map, which is exactly why it is cheap enough to
sit inside a time-optimisation loop.
"""
import numpy as np
from ..flatness.transform import FlatTransform


def make_sigma(traj_pos, traj_yaw=None):
    """Adapt (position, yaw) polynomial trajectories into the sigma(t) callable
    the flatness transform consumes: t -> (pos, vel, acc, psi)."""
    def sigma(t):
        psi = 0.0 if traj_yaw is None else float(traj_yaw(t, 0)[0])
        return traj_pos(t, 0), traj_pos(t, 1), traj_pos(t, 2), psi
    return sigma


class TailsitterFeasibility:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = cfg.vehicle
        self.flat = FlatTransform(cfg)

    def evaluate(self, traj_pos, traj_yaw=None, N=400, margin_frac=0.05):
        """
        Sample the trajectory, run the flatness transform, and summarise demand.

        Returns dict with:
          feasible      : bool, all samples within limits and real thrust
          w_peak        : peak rotor speed [rad/s]
          flap_peak     : peak |flap| [rad]
          w_util        : w_peak / w_max        (1.0 = at the limit)
          flap_util     : flap_peak / flap_max
          frac_feasible : fraction of samples that are feasible
          ref           : the full FlatTransform.reference() output (arrays)
        """
        sigma = make_sigma(traj_pos, traj_yaw)
        t0, tf = traj_pos.knots[0], traj_pos.knots[-1]
        ts = np.linspace(t0, tf, N)
        ref = self.flat.reference(sigma, ts)
        # drop the gradient endpoints (Omega/Omega_dot are one-sided there)
        sl = slice(3, -3)
        w_peak = float(np.nanmax(np.maximum(ref["w1"][sl], ref["w2"][sl])))
        flap_peak = float(np.nanmax(np.abs(np.concatenate([ref["d1"][sl], ref["d2"][sl]]))))
        frac = float(ref["feasible"][sl].mean())
        feasible = bool(ref["feasible"][sl].all())
        return dict(
            feasible=feasible,
            w_peak=w_peak, flap_peak=flap_peak,
            w_util=w_peak / self.v.w_max,
            flap_util=flap_peak / self.v.flap_max,
            frac_feasible=frac,
            duration=float(tf - t0),
            ref=ref, ts=ts,
        )
