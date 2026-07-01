"""
Aerobatic maneuver library.

Each generator returns a dict with
  waypoints      : (M+1, 3) position waypoints in NED  (altitude = -z)
  yaw            : (M+1,) flat-output yaw psi [rad]
  nominal_speed  : suggested speed for the initial time allocation
  name           : label

The aerobatic character of each figure comes from the position path plus the yaw
schedule; the differential-flatness transform then turns these into the required
(and possibly extreme) attitudes and actuator commands. Yaw conventions used:
  coordinated : psi = heading + pi/2   (wing banks into the turn)
  forward     : psi = const heading    (vehicle pitches through the figure)
  knife_edge  : psi = heading          (wing edge-on to the airflow)
Feed the result to `allocate_times` then `min_snap_trajectory`.
"""
import numpy as np


def _pack(name, wp, yaw, speed, v0=None, v1=None):
    wp = np.asarray(wp, float)
    # default entry/exit velocity: tangent to the path at the ends, at `speed`
    if v0 is None:
        t0 = wp[1] - wp[0]; v0 = speed * t0 / (np.linalg.norm(t0) + 1e-12)
    if v1 is None:
        t1 = wp[-1] - wp[-2]; v1 = speed * t1 / (np.linalg.norm(t1) + 1e-12)
    return dict(name=name, waypoints=wp, yaw=np.asarray(yaw, float),
                nominal_speed=float(speed), v0=np.asarray(v0, float),
                v1=np.asarray(v1, float))


def line(length=10.0, alt=2.0, n=4, speed=6.0):
    x = np.linspace(0, length, n + 1)
    wp = np.c_[x, np.zeros(n + 1), -alt * np.ones(n + 1)]
    return _pack("line", wp, np.full(n + 1, np.pi / 2), speed)


def loop(radius=3.0, entry_alt=3.0, n=16, speed=7.0):
    """Vertical loop in the x-altitude plane; vehicle pitches through 360 deg."""
    th = np.linspace(0, 2 * np.pi, n + 1)
    x = radius * np.sin(th)
    alt = entry_alt + radius * (1 - np.cos(th))
    wp = np.c_[x, np.zeros(n + 1), -alt]
    return _pack("loop", wp, np.zeros(n + 1), speed)


def knife_edge(length=10.0, alt=3.0, n=6, speed=6.0):
    """Straight pass flown wing-edge-on (yaw aligned with velocity)."""
    x = np.linspace(0, length, n + 1)
    wp = np.c_[x, np.zeros(n + 1), -alt * np.ones(n + 1)]
    heading = 0.0
    return _pack("knife_edge", wp, np.full(n + 1, heading), speed)


def climbing_turn(radius=4.0, turns=1.0, climb=4.0, entry_alt=2.0, n=24, speed=6.0):
    """Helix: constant-radius turn while climbing."""
    phi = np.linspace(0, 2 * np.pi * turns, n + 1)
    x = radius * np.cos(phi) - radius
    y = radius * np.sin(phi)
    alt = entry_alt + climb * phi / (2 * np.pi * turns)
    wp = np.c_[x, y, -alt]
    heading = phi + np.pi / 2
    return _pack("climbing_turn", wp, heading + np.pi / 2, speed)


def immelmann(radius=3.0, entry_alt=2.0, n=14, speed=8.0):
    """Half loop up + roll out: reverse direction and gain ~2R altitude."""
    th = np.linspace(0, np.pi, n + 1)
    x = radius * np.sin(th)
    alt = entry_alt + radius * (1 - np.cos(th))
    wp = np.c_[x, np.zeros(n + 1), -alt]
    # short exit heading -x at the top
    wp = np.vstack([wp, wp[-1] + np.array([-2.0, 0, 0])])
    yaw = np.zeros(n + 2)
    return _pack("immelmann", wp, yaw, speed)


def split_s(radius=3.0, entry_alt=6.0, n=14, speed=8.0):
    """Half loop down: reverse direction and lose ~2R altitude."""
    th = np.linspace(0, np.pi, n + 1)
    x = radius * np.sin(th)
    alt = entry_alt - radius * (1 - np.cos(th))
    wp = np.c_[x, np.zeros(n + 1), -alt]
    wp = np.vstack([wp, wp[-1] + np.array([-2.0, 0, 0])])
    yaw = np.zeros(n + 2)
    return _pack("split_s", wp, yaw, speed)


def differential_thrust_turn(radius=1.5, alt=3.0, n=12, speed=5.0):
    """Tight horizontal U-turn (small radius) -- stresses yaw / differential thrust."""
    th = np.linspace(0, np.pi, n + 1)
    x = radius * np.sin(th)
    y = radius * (1 - np.cos(th))
    wp = np.c_[x, y, -alt * np.ones(n + 1)]
    heading = th + np.pi / 2
    return _pack("differential_thrust_turn", wp, heading + np.pi / 2, speed)


def racing_gates(gates=None, speed=8.0):
    """Pass through a sequence of 3D gate centres (coordinated yaw)."""
    if gates is None:
        gates = [[0, 0, -2], [6, 3, -3], [12, -2, -2.5], [16, 2, -4], [22, 0, -2.5]]
    wp = np.asarray(gates, float)
    d = np.diff(wp, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    heading = np.concatenate([[heading[0]], heading])
    return _pack("racing_gates", wp, heading + np.pi / 2, speed)


CATALOG = {
    "line": line, "loop": loop, "knife_edge": knife_edge,
    "climbing_turn": climbing_turn, "immelmann": immelmann, "split_s": split_s,
    "differential_thrust_turn": differential_thrust_turn, "racing_gates": racing_gates,
}
