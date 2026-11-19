"""Peer-review control baselines, all sharing the convoy simulator so they are
scored on identical metrics.

* IDM / CACC  -- analytic car-following law (collision-aware heuristic).
* MPC         -- decentralized receding-horizon optimisation (constant-velocity
                 predecessor assumption), actuator-bounded.
* Unconstrained NN -- the D-PCNC policy architecture WITHOUT the hard projection
                 layer (actuator clamp only). Ablation isolating the projection.
* Soft-penalty PINN -- same policy trained with a soft safety penalty instead of a
                 hard projection (actuator clamp only at deploy). Ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.convoy_sim import SimParams, simulate, mu_bounds
from ..physics.hard_projection_control import ControllerConfig, ConvoyController


@dataclass
class IDMParams:
    s_min: float
    tau_safe: float
    a_max: float
    b: float = 2.5           # comfortable deceleration
    v0: float = 30.0         # desired free-flow speed (m/s)
    delta: float = 4.0


class IDMController:
    """Intelligent Driver Model with CACC-style predecessor feedforward."""

    def __init__(self, p: SimParams, idm: IDMParams):
        self.p = p
        self.idm = idm

    def _accel(self, obs):
        idm = self.idm
        v = max(obs.vf, 0.0)
        s = max(obs.gap, 0.1)
        dv = v - obs.vl                      # approaching rate (+ = closing)
        s_star = idm.s_min + max(0.0, v * self.p.tau_safe
                                 + v * dv / (2.0 * np.sqrt(idm.a_max * idm.b)))
        a = idm.a_max * (1.0 - (v / idm.v0) ** idm.delta - (s_star / s) ** 2)
        a += 0.3 * obs.al                    # CACC predecessor-accel feedforward
        a_min_s, a_max_s = mu_bounds(self.p, obs.mu)
        return float(np.clip(a, a_min_s, a_max_s))

    def rollout(self, s, mu=0.9, telemetry=None, energy=None):
        return simulate(self._accel, s, self.p, mu=mu, telemetry=telemetry, energy=energy)


class MPCController:
    """Decentralized receding-horizon controller.

    Chooses the current acceleration that minimises an H-step predicted cost
    (gap-error, speed-tracking, control effort) assuming the predecessor holds its
    current velocity, subject to actuator bounds. Single decision variable per step
    (held over the horizon), solved by bounded scalar minimisation.
    """

    def __init__(self, p: SimParams, horizon=10, q_gap=1.0, q_speed=0.5, r_u=0.05):
        self.p = p
        self.H = horizon
        self.q_gap, self.q_speed, self.r_u = q_gap, q_speed, r_u

    def _cost(self, u, obs):
        p = self.p
        dt = p.dt
        gap, vf = obs.gap, obs.vf
        vl = obs.vl
        pos_rel = gap
        c = 0.0
        for _ in range(self.H):
            vf_n = max(0.0, vf + u * dt)
            pos_rel = pos_rel + (vl - vf) * dt - 0.5 * u * dt * dt
            safe = p.s_min + p.tau_safe * vf_n
            c += self.q_gap * (pos_rel - safe) ** 2 + self.q_speed * (vf_n - vl) ** 2
            vf = vf_n
        c += self.r_u * u * u
        return c

    def _accel(self, obs):
        from scipy.optimize import minimize_scalar
        a_min_s, a_max_s = mu_bounds(self.p, obs.mu)
        res = minimize_scalar(self._cost, args=(obs,), bounds=(a_min_s, a_max_s),
                              method="bounded")
        return float(np.clip(res.x, a_min_s, a_max_s))

    def rollout(self, s, mu=0.9, telemetry=None, energy=None):
        return simulate(self._accel, s, self.p, mu=mu, telemetry=telemetry, energy=energy)


# --------------------------------------------------------------------------- #
def build_controllers(mcfg: dict, energy_m_eff: float):
    """Instantiate all controllers from the model config. Neural controllers are
    returned untrained; call ``.fit(...)`` on them."""
    ph = mcfg["physics"]
    p = SimParams(dt=ph["dt"], s_min=ph["s_min"], tau_safe=ph["tau_safe"],
                  a_max=ph["a_max_nominal"], a_min=ph["a_min_nominal"])

    def cc(mode):
        cfg = ControllerConfig(
            dt=ph["dt"], s_min=ph["s_min"], tau_safe=ph["tau_safe"],
            a_max=ph["a_max_nominal"], a_min=ph["a_min_nominal"],
            dP_max=ph["dP_max"], m_eff=energy_m_eff,
            hidden=mcfg["controller"]["hidden"], layers=mcfg["controller"]["layers"],
            mode=mode, soft_penalty_weight=mcfg["controller"]["soft_penalty_weight"])
        return ConvoyController(cfg)

    return {
        "IDM/CACC": IDMController(p, IDMParams(ph["s_min"], ph["tau_safe"], ph["a_max_nominal"])),
        "MPC": MPCController(p, mcfg["mpc"]["horizon"], mcfg["mpc"]["q_gap"],
                             mcfg["mpc"]["q_speed"], mcfg["mpc"]["r_u"]),
        "Unconstrained-NN": cc("unconstrained"),
        "Soft-PINN": cc("soft_pinn"),
        "D-PCNC": cc("dpcnc"),
    }
