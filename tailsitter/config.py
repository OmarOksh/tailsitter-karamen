"""
Configuration loading.

A single YAML (config/tailsitter.yaml) holds every tunable quantity: vehicle
physical/aero/propulsion parameters, simulation settings, optimization settings
(Phase 2) and controller gains (Phase 3). Load it with `load_config()` and pass
the resulting `Config` object everywhere.

Parameters flagged `# ESTIMATE` in the YAML are geometry-based guesses because
the source paper (Table II) does not publish them (inertia tensor, moment arms,
alpha0, alphaT, actuator time constants). Replace them with measured/identified
values for a tight numeric match to the paper's figures.
"""
from dataclasses import dataclass, field
import os
import numpy as np
import yaml


@dataclass
class Vehicle:
    # --- mass / geometry
    mass: float
    span: float
    aspect_ratio: float
    prop_diameter: float
    inertia: np.ndarray            # 3x3 tensor J (body frame)
    lTy: float                     # CG->motor distance along by
    lDy: float                     # CG->flap-center distance along by
    lDx: float                     # by-axis -> flap aero-center distance along bx
    alpha0: float                  # zero-lift angle of attack [rad]
    alphaT: float                  # thrust line tilt [rad]
    # --- phi-theory aerodynamic coefficients (absorb air density)
    cLV: float
    cDV: float
    cLT: float
    cDT: float
    cLV_flap: float                # c^delta_LV
    cLT_flap: float                # c^delta_LT
    cmuT: float                    # thrust-induced pitch moment coeff
    # --- propulsion
    cT: float                      # thrust coeff   Ti = cT * wi^2
    cmu: float                     # torque coeff   mu_i = -(-1)^i cmu wi^2
    w_max: float                   # max rotor speed [rad/s]
    w_min: float
    flap_max: float                # max flap deflection [rad]
    flap_min: float

    @property
    def chord(self):
        return self.span / self.aspect_ratio


@dataclass
class SimCfg:
    g: float = 9.81
    dt: float = 5.0e-4             # integrator step (2 kHz, matches onboard rate)
    integrator: str = "rk4"


@dataclass
class OptCfg:                       # Phase 2
    yaw_weight: float = 1.0
    n_sample: int = 40
    time_scale_init: float = 1.0


@dataclass
class CtrlCfg:                      # Phase 3 (placeholders)
    Kx: float = 4.0
    Kv: float = 4.0
    Ka: float = 1.0
    Kq: float = 80.0
    Komega: float = 20.0
    lpf_cutoff_hz: float = 15.0
    hpf_cutoff_hz: float = 1.0


@dataclass
class Config:
    vehicle: Vehicle
    sim: SimCfg = field(default_factory=SimCfg)
    opt: OptCfg = field(default_factory=OptCfg)
    ctrl: CtrlCfg = field(default_factory=CtrlCfg)


_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "config", "tailsitter.yaml")


def load_config(path: str = None) -> Config:
    path = path or os.path.normpath(_DEFAULT)
    with open(path) as f:
        d = yaml.safe_load(f)
    v = d["vehicle"]
    inertia = np.array(v["inertia"], dtype=float)
    if inertia.shape == (3,):                      # diagonal shorthand
        inertia = np.diag(inertia)
    veh = Vehicle(
        mass=v["mass"], span=v["span"], aspect_ratio=v["aspect_ratio"],
        prop_diameter=v["prop_diameter"], inertia=inertia,
        lTy=v["lTy"], lDy=v["lDy"], lDx=v["lDx"],
        alpha0=v["alpha0"], alphaT=v["alphaT"],
        cLV=v["cLV"], cDV=v["cDV"], cLT=v["cLT"], cDT=v["cDT"],
        cLV_flap=v["cLV_flap"], cLT_flap=v["cLT_flap"], cmuT=v["cmuT"],
        cT=v["cT"], cmu=v["cmu"],
        w_max=v["w_max"], w_min=v["w_min"],
        flap_max=v["flap_max"], flap_min=v["flap_min"],
    )
    s = d.get("sim", {})
    o = d.get("optimization", {})
    c = d.get("controller", {})
    return Config(
        vehicle=veh,
        sim=SimCfg(**s),
        opt=OptCfg(**o),
        ctrl=CtrlCfg(**c),
    )
