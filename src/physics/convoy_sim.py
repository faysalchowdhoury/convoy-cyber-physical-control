"""Shared closed-loop convoy simulator and metric computation.

The leader replays its REAL recorded trajectory; every follower is driven by a
pluggable ``step_fn`` (neural, IDM/CACC, or MPC). Keeping one simulator guarantees
all controllers are scored on identical metrics.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np


@dataclass
class SimParams:
    dt: float
    s_min: float
    tau_safe: float
    a_max: float
    a_min: float


@dataclass
class Obs:
    t: int
    j: int
    gap: float
    vf: float
    vl: float          # REPORTED predecessor speed (V2V; may be spoofed)
    al: float          # REPORTED predecessor accel (V2V; may be spoofed)
    a_prev: float
    mu: float
    gap_rate: float = 0.0   # MEASURED d(gap)/dt (onboard radar; trusted)
    al_meas: float = 0.0    # MEASURED predecessor accel (onboard radar; trusted)


def mu_bounds(p: SimParams, mu: float):
    scale = mu / 0.9
    return p.a_min * scale, p.a_max * scale


def simulate(step_fn, s: dict, p: SimParams, mu: float = 0.9,
             telemetry=None, energy=None) -> dict:
    T, N = s["T"], s["N"]
    dt = p.dt
    pos = np.zeros((T, N)); vel = np.zeros((T, N)); acc = np.zeros((T, N))
    pos[0] = s["pos"][0]; vel[0] = s["vel"][0]
    v_lead_src = telemetry[0] if telemetry is not None else s["vel"]
    a_lead_src = telemetry[1] if telemetry is not None else s["acc"]

    collisions = 0
    inf_times: list[float] = []
    for t in range(T - 1):
        pos[t + 1, 0] = s["pos"][t + 1, 0]
        vel[t + 1, 0] = s["vel"][t + 1, 0]
        acc[t, 0] = s["acc"][t, 0]
        for j in range(1, N):
            gap = pos[t, j - 1] - pos[t, j] - s["length"][j - 1]
            vf = vel[t, j]
            vl = float(v_lead_src[t, j - 1])          # reported (spoofable)
            al = float(a_lead_src[t, j - 1])          # reported (spoofable)
            # measured / trusted onboard radar: relative velocity and predecessor accel
            gap_rate = float(vel[t, j - 1] - vel[t, j])
            al_meas = float((vel[t, j - 1] - vel[t - 1, j - 1]) / dt) if t > 0 else 0.0
            obs = Obs(t=t, j=j, gap=gap, vf=vf, vl=vl, al=al,
                      a_prev=acc[t - 1, j] if t > 0 else 0.0, mu=mu,
                      gap_rate=gap_rate, al_meas=al_meas)
            t0 = _time.perf_counter()
            u = step_fn(obs)
            inf_times.append((_time.perf_counter() - t0) * 1e3)
            u = float(u)
            acc[t, j] = u
            vel[t + 1, j] = max(0.0, vf + u * dt)
            pos[t + 1, j] = pos[t, j] + vel[t, j] * dt + 0.5 * u * dt * dt
            if pos[t + 1, j - 1] - pos[t + 1, j] - s["length"][j - 1] <= 0.0:
                collisions += 1
    return compute_metrics(pos, vel, acc, s, p, energy, collisions, inf_times)


def compute_metrics(pos, vel, acc, s, p: SimParams, energy, collisions, inf_times):
    T, N = vel.shape
    gap_arr = np.full((T, N), np.nan)
    for j in range(1, N):
        gap_arr[:, j] = pos[:, j - 1] - pos[:, j] - s["length"][j - 1]
    safe = p.s_min + p.tau_safe * vel
    headway_viol = np.mean((gap_arr[:, 1:] < safe[:, 1:]) & np.isfinite(gap_arr[:, 1:]))
    spacing_err = np.zeros((T, N))
    for j in range(1, N):
        spacing_err[:, j] = gap_arr[:, j] - safe[:, j]
    from .residuals import string_stability_ratio
    speed_rmse = float(np.sqrt(np.nanmean((vel[:, 1:] - s["vel"][:, 1:]) ** 2)))
    energy_kwh_km = _energy_cost(pos, vel, acc, s["length"], p.dt, energy) if energy else float("nan")
    return {
        "pos": pos, "vel": vel, "acc": acc, "gap": gap_arr,
        "collisions": int(collisions),
        "headway_violation": float(headway_viol),
        "string_stability": float(string_stability_ratio(spacing_err)),
        "speed_rmse": speed_rmse,
        "energy_kwh_km": energy_kwh_km,
        "latency_ms": float(np.mean(inf_times)) if inf_times else float("nan"),
    }


def _energy_cost(pos, vel, acc, length, dt, energy) -> float:
    T, N = vel.shape
    costs = []
    for j in range(1, N):
        P = energy.power(vel[:, j], acc[:, j])
        e_j = np.sum(np.clip(P, 0, None)) * dt / 3.6e6
        dist_km = max(1e-6, (pos[-1, j] - pos[0, j]) / 1000.0)
        costs.append(e_j / dist_km)
    return float(np.mean(costs))
