import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.physics.residuals import string_stability_ratio, kinematic_residual, safe_gap


def test_attenuating_string_ratio_below_one():
    T, N = 100, 5
    err = np.zeros((T, N))
    base = np.sin(np.linspace(0, 6, T))
    for j in range(1, N):
        err[:, j] = base * (0.6 ** (j - 1))  # disturbance decays down the string
    assert string_stability_ratio(err) < 1.0


def test_amplifying_string_ratio_above_one():
    T, N = 100, 5
    err = np.zeros((T, N))
    base = np.sin(np.linspace(0, 6, T))
    for j in range(1, N):
        err[:, j] = base * (1.5 ** (j - 1))  # disturbance grows -> unstable
    assert string_stability_ratio(err) > 1.0


def test_kinematic_residual_zero_when_consistent():
    dt = 0.1
    v_t = np.array([10.0, 12.0]); a_t = np.array([1.0, -2.0])
    v_tp1 = v_t + a_t * dt
    assert np.allclose(kinematic_residual(v_t, v_tp1, a_t, dt), 0.0)


def test_safe_gap_monotone_in_speed():
    g = safe_gap(np.array([0.0, 10.0, 20.0]), s_min=2.0, tau_safe=1.2)
    assert g[0] < g[1] < g[2] and g[0] == 2.0
