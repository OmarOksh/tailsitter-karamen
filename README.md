# tailsitter-aero

Replication of the trajectory-generation pipeline from

> E. Tal, G. Ryou, S. Karaman, *"Aerobatic Trajectory Generation for a VTOL
> Fixed-Wing Aircraft Using Differential Flatness,"* IEEE T-RO 39(6), 2023.

It pairs the `phi`-theory tailsitter aerodynamic model (Lustosa et al. 2019) with
the paper's differential-flatness transform, so any smooth flat output
`sigma(t) = [x(t), psi(t)]` (position + yaw) maps to the full state and the rotor /
flap inputs — and trajectory feasibility becomes a cheap open-loop check.

**Status: Phase 1 complete and validated.** (Phase 2 = trajectory generation,
Phase 3 = INDI tracking controller.)

---

## Install

```bash
pip install -e .          # add --break-system-packages on Debian/Ubuntu system Python
```

Requires numpy, scipy, pyyaml, matplotlib (installed automatically). The notebook
additionally needs `jupyter` / `nbconvert`.

## Run

Open the walkthrough — it drives every piece step by step and renders the plots:

```bash
jupyter notebook notebooks/phase1_walkthrough.ipynb
```

Run the validation suite directly:

```bash
python tests/test_phase1.py
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
│   ├── trajgen/                   # Phase 2 (min-snap + feasibility)  [stub]
│   └── control/                   # Phase 3 (INDI controller)         [stub]
├── notebooks/phase1_walkthrough.ipynb
├── tests/test_phase1.py
└── plots/                         # figures written by the notebook
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

## Parameter caveat (read this before trusting absolute numbers)

`config/tailsitter.yaml` separates **published** Table II values from
geometry-based **estimates** (flagged `# ESTIMATE`): the inertia tensor, moment
arms (`lTy, lDy, lDx`), `alpha0`, `alphaT`, and actuator limits/time constants are
not published in the paper. The dynamics and flatness transform are *structurally*
exact; the *numeric* match to the paper's figures is bounded by these estimates
until you replace them with measured or flight-identified values. The hover flap
trim (~16 deg) and the A1 gap shrink notably with better `cmuT`/`lDx`/arm values.

## Roadmap

* **Phase 2** — reuse MIT-AERA `mfboTrajectory` minimum-snap core
  (`BaseTrajFunc`/`MinSnapTrajectory`, vehicle-agnostic), replace their quadrotor
  feasibility model with `FlatTransform.reference`, add maneuver waypoint
  definitions (loop, knife-edge, Immelmann, Split-S, ...) and time-optimal scaling.
* **Phase 3** — the tailsitter INDI tracking controller (Doc 3) for closed-loop
  flight of the generated trajectories.
```
