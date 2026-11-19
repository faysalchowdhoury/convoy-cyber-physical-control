"""Contested V2V communication channel applied at evaluation time (eval-only).

Produces predecessor-telemetry arrays [T, N] for the convoy simulator under:
  * packet drop (Bernoulli loss -> follower holds the last received value / stale)
  * latency (telemetry delayed by a fixed number of steps)
  * V2V spoofing (speed bias injected after an onset)
The clean arrays are the real recorded speeds/accelerations of the string.
"""
from __future__ import annotations

import numpy as np


def degrade_telemetry(s: dict, drop: float, latency: int, rng) -> tuple[np.ndarray, np.ndarray]:
    v = s["vel"].copy(); a = s["acc"].copy()
    T, N = v.shape
    if latency > 0:
        v = np.vstack([np.repeat(v[:1], latency, axis=0), v[:-latency]])
        a = np.vstack([np.repeat(a[:1], latency, axis=0), a[:-latency]])
    if drop > 0:
        for j in range(N):
            last_v, last_a = v[0, j], a[0, j]
            for t in range(T):
                if rng.random() < drop:
                    v[t, j] = last_v; a[t, j] = last_a   # stale hold
                else:
                    last_v, last_a = v[t, j], a[t, j]
    return v, a


def attack_telemetry_v2v(s: dict, magnitude: float, onset: int,
                         kind: str = "bias") -> tuple[np.ndarray, np.ndarray]:
    v = s["vel"].copy(); a = s["acc"].copy()
    if kind == "bias":
        v[onset:, :] += magnitude
    elif kind == "ramp":
        T = v.shape[0]
        ramp = np.linspace(0, magnitude, T - onset)[:, None]
        v[onset:, :] += ramp
    return v, a
