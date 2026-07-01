"""Trajectory generation (Phase 2): minimum-snap + tailsitter feasibility + time scaling."""
from .minsnap import min_snap_trajectory, solve_axis, PolyTrajectory
from .feasibility import TailsitterFeasibility, make_sigma
from .timeopt import allocate_times, time_optimal_scale, scale_knots
from . import maneuvers
