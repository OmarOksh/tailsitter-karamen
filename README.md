# tailsitter-aero

Replication of the trajectory-generation pipeline from

> E. Tal, G. Ryou, S. Karaman, *"Aerobatic Trajectory Generation for a VTOL
> Fixed-Wing Aircraft Using Differential Flatness,"* IEEE T-RO 39(6), 2023.

It pairs the `phi`-theory tailsitter aerodynamic model (Lustosa et al. 2019) with
the paper's differential-flatness transform, so any smooth flat output
`sigma(t) = [x(t), psi(t)]` (position + yaw) maps to the full state and the rotor /
flap inputs — and trajectory feasibility becomes a cheap open-loop check.

**Status: Phases 1-3 complete and validated.** The full pipeline runs end to end:
dynamics + flatness, trajectory generation, and closed-loop INDI tracking.

* **Phase 1** — `phi`-theory dynamics, 6-DOF simulator, differential-flatness
  transform (`sigma -> ` state + inputs), open-loop feasibility.
* **Phase 2** — minimum-snap trajectory generation through maneuver waypoints,
  tailsitter feasibility (drop-in for the quadrotor model), and time-optimal
  scaling; a catalog of aerobatic figures (loop, knife-edge, climbing turn,
  Immelmann, Split-S, differential-thrust turn, racing gates).
* **Phase 3** — the INDI trajectory-tracking controller (PD kinematics + INDI
  dynamics) that flies the Phase-2 references closed-loop in the 6-DOF plant,
  tracking to centimetres and rejecting model error and disturbances.

---

## Install

```bash
pip install -e .          # add --break-system-packages on Debian/Ubuntu system Python
```

Requires numpy, scipy, pyyaml, matplotlib (installed automatically). The notebook
additionally needs `jupyter` / `nbconvert`.

## Run

Open the walkthroughs — they drive every piece step by step and render the plots:

```bash
jupyter notebook notebooks/phase1_walkthrough.ipynb   # dynamics, sim, flatness
jupyter notebook notebooks/phase2_walkthrough.ipynb   # min-snap, feasibility, time-opt
jupyter notebook notebooks/phase3_walkthrough.ipynb   # closed-loop INDI tracking
```

Run the validation suites directly:

```bash
python tests/test_phase1.py
python tests/test_phase2.py
python tests/test_phase3.py
```

## What's here

```
tailsitter-aero/
├── config/tailsitter.yaml         # EVERY tunable parameter (vehicle/sim/opt/ctrl)
├── tailsitter/
│   ├── config.py                  # YAML -> typed Config dataclasses
│   ├── utils/rotation.py          # Hamilton quaternions, ZXY Euler, kinematics
│   ├── dynamics/
│   │   ├── phi_theory.py          # force & moment model      (Doc 4, Eqs 5-14)
│   │   └── sim6dof.py             # 6-DOF RK4 simulator       (Doc 4, Eqs 1-4)
│   ├── flatness/transform.py      # the flatness transform    (Doc 4, Eqs 20-56)
│   ├── trajgen/                   # Phase 2
│   │   ├── minsnap.py             # minimum-snap piecewise polynomials
│   │   ├── feasibility.py         # actuator demand via the flatness transform
│   │   ├── timeopt.py             # time allocation + time-optimal scaling
│   │   └── maneuvers.py           # aerobatic waypoint/yaw catalog
│   └── control/                   # Phase 3
│       ├── indi.py                # INDI tracking controller  (Doc 3, Eqs 37-48)
│       └── closedloop.py          # closed-loop runner + reference builder
├── notebooks/phase1_walkthrough.ipynb
├── notebooks/phase2_walkthrough.ipynb
├── notebooks/phase3_walkthrough.ipynb
├── tests/test_phase1.py
├── tests/test_phase2.py
├── tests/test_phase3.py
└── plots/                         # figures written by the notebooks
```

## Conventions

* **Frames:** world is NED; body is `bx` forward (chord), `by` along span,
  `bz` down. A rotation maps body→world: `v_world = R @ v_body`.
* **Quaternions:** Hamilton, scalar-first `[w, x, y, z]`.
* **Euler:** ZXY (yaw `psi` about z, roll `phi` about x, pitch `theta` about y).
  `psi = phi = theta = 0` is wings-level forward flight toward north; hover is
  `theta ~ +90 deg`. ZXY keeps the singularity at `+-90 deg` roll (avoided) rather
  than `+-90 deg` pitch (visited in hover).
* **Forces** are summed in the zero-lift axis system; **moments** in the body frame.

## How the flatness transform is implemented

The algebraic maps are coded exactly from the paper:
attitude + collective thrust (Eqs 20-26) and control allocation from the required
moment (Eqs 51-56). Angular velocity and acceleration (Eqs 27-50 in the paper) are
obtained by differentiating the exact algebraic attitude map along the trajectory —
mathematically equivalent, far less transcription risk, and exactly how the
quantities are consumed for offline trajectory generation. (For an embedded online
implementation you would code the closed forms for speed.)

Two approximations are inherited from the paper, on purpose:
* **A1** the attitude+thrust solve neglects the direct, non-minimum-phase flap
  force; the flaps are then set by the moment requirement;
* **A2** the allocation neglects the flap contribution to *yaw* moment
  (`~ sin(alpha0)`, stated negligible in Eq 52).

Both are closed by the INDI controller in Phase 3. `tests/test_phase1.py` verifies
the exact parts to machine precision and *quantifies* A1/A2 rather than hiding them.

## Validation (from `tests/test_phase1.py`)

```
T0 hover    angular accel == 0                         2.3e-14   PASS
            vdot == neglected flap-force term (A1)      1.6e-15   PASS
T1 circle   force/attitude reconstruction (Eqs 20-26)  7.2e-15   PASS
            roll+pitch moment reconstruction           1.4e-16   PASS
T2          numeric Omega carries the attitude         7.8e-12   PASS
```

## Phase 2 — trajectory generation

The pipeline is `maneuver -> min-snap -> feasibility -> time-optimal`:

```python
from tailsitter.config import load_config
from tailsitter.trajgen import (min_snap_trajectory, allocate_times,
                                TailsitterFeasibility, time_optimal_scale, maneuvers)

cfg = load_config()
m   = maneuvers.loop(radius=3.0, speed=7.0)          # waypoints + yaw + entry speed
k0  = allocate_times(m["waypoints"], m["nominal_speed"])
opt = time_optimal_scale(m["waypoints"], k0, yaw=m["yaw"], cfg=cfg,
                         v0=m["v0"], v1=m["v1"])       # fastest feasible timing
print(opt["alpha"], opt["feas"]["duration"], opt["feas"]["feasible"])
```

* **`minsnap.py`** — Richter-Bry / Mellinger-Kumar minimum-snap, the same
  formulation as MIT-AERA `mfboTrajectory` (`BaseTrajFunc`/`MinSnapTrajectory`),
  reimplemented self-contained: positions fixed at waypoints, interior derivatives
  free, minimise `int snap^2`; yaw is a separate jerk-minimising (degree-5) spline.
  Segments are time-normalised so the QP stays well conditioned.
* **`feasibility.py`** — the drop-in replacement for the quadrotor model: push the
  flat output through the Phase-1 flatness transform and read off rotor speeds and
  flap angles. Feasible iff they stay within limits everywhere — a pure algebraic
  check, no integration.
* **`timeopt.py`** — initial times from segment geometry, then a 1-D search for the
  smallest time multiplier that is feasible everywhere (the fastest version of the
  figure). Minimum-snap is covariant under time scaling, so this reproduces the
  exact time-scaled trajectory.
* **`maneuvers.py`** — loop, knife-edge, climbing turn, Immelmann, Split-S,
  differential-thrust turn, racing gates. Vertical figures keep the wing axis
  horizontal (yaw 0, pure pitch); horizontal turns use coordinated yaw.

## Validation (from `tests/test_phase2.py`)

```
T0 min-snap   waypoints interpolated                     7.6e-14   PASS
              C3 continuous at interior knots            <1e-3      PASS
T1 feasibility flying the same shape slower lowers demand           PASS
T2 time-opt   fastest feasible sits on an actuator limit (util>0.9) PASS
              e.g. loop alpha*=1.94 dur 5.20s -> flap-limited
T3 catalog    all 8 maneuvers generate + evaluate                   PASS
```

The loop is **flap-limited**: at the time-optimal timing the flap saturates
(~ -1 rad) during the aggressive pitch-down on the back of the loop, exactly where
the pitch rate peaks (q ~ -4.2 rad/s). See `plots/p2_feasibility_boundary.png` and
`plots/p2_reference_trajectory.png`.

## Phase 3 — closed-loop INDI tracking

Fly a Phase-2 reference through the plant:

```python
from tailsitter.config import load_config
from tailsitter.trajgen import min_snap_trajectory, allocate_times, maneuvers
from tailsitter.control import TailsitterINDI, fly

cfg = load_config()
m   = maneuvers.loop(radius=3.0, speed=6.0)
k   = allocate_times(m["waypoints"], m["nominal_speed"]) * 2.2   # comfortable margin
tp, ty = min_snap_trajectory(m["waypoints"], k, yaw=m["yaw"], v0=m["v0"]/2.2, v1=m["v1"]/2.2)

log = fly(cfg, TailsitterINDI(cfg), tp, ty)
print(log["rmse"], "m RMSE")            # ~0.08 m on the loop
```

The controller (`control/indi.py`) is the Doc-3 cascade: PD position/velocity ->
INDI linear acceleration (-> attitude + thrust via the flatness map) -> PD
attitude/rate -> INDI angular acceleration (-> moment -> actuators via the
allocation). Each INDI stage compares the **measured** acceleration with the model
prediction at the last applied input and commands the increment that cancels the
difference — the unmodelled force/moment — so it needs only *local* model accuracy.
The nonlinear inversion runs *through* the Phase-1 flatness transform. In simulation
the "IMU" is read from the plant; pass a perturbed `model_cfg` to `TailsitterINDI`
to give the controller a deliberately wrong model. Five gains live in the YAML
`controller` block.

## Validation (from `tests/test_phase3.py`)

```
T0 tracking     loop, matched model            RMSE  7.9 cm            PASS
T1 mismatch     +25% mass / +40% inertia        RMSE 17.5 cm (bounded)  PASS
T2 disturbance  2.2 N steady wind (~1/3 weight)  RMSE 27.5 cm            PASS
T3 feedback     open-loop feedforward diverges   175x worse than INDI   PASS
```

INDI absorbs exactly the A1/A2 flatness approximations from Phase 1: they are local
model errors, and the increments cancel them. The open-loop reference, by contrast,
diverges by metres — feedback is what makes the aerobatic reference flyable. See
`plots/p3_tracking.png` and `plots/p3_robustness.png`.

## Parameter caveat (read this before trusting absolute numbers)

`config/tailsitter.yaml` separates **published** Table II values from
geometry-based **estimates** (flagged `# ESTIMATE`): the inertia tensor, moment
arms (`lTy, lDy, lDx`), `alpha0`, `alphaT`, and actuator limits/time constants are
not published in the paper. The dynamics and flatness transform are *structurally*
exact; the *numeric* match to the paper's figures is bounded by these estimates
until you replace them with measured or flight-identified values. The hover flap
trim (~16 deg) and the A1 gap shrink notably with better `cmuT`/`lDx`/arm values.

## Roadmap

* **Phase 1 (done)** — dynamics, 6-DOF sim, differential-flatness transform.
* **Phase 2 (done)** — minimum-snap generation, tailsitter feasibility, and
  time-optimal scaling, with an aerobatic maneuver catalog.
* **Phase 3 (done)** — the tailsitter INDI tracking controller (Doc 3) flying the
  generated trajectories closed-loop, closing the A1/A2 flatness approximations
  with feedback.

Possible extensions: identify the estimated parameters from flight logs, add
actuator/servo dynamics and sensor noise to the plant, gain-schedule or auto-tune
the controller, and port the reference generation to run onboard.
