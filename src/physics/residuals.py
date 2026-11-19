"""Physical-consistency residuals used by the controller's projection feasibility
checks and by the cross-layer intrusion detector.

All residuals are expressed in SI units and are zero when the reported signals are
physically self-consistent.
"""
from __future__ import annotations

import numpy as np


def kinematic_residual(v_t: np.ndarray, v_tp1: np.ndarray, a_t: np.ndarray,
                       dt: float) -> np.ndarray:
    """|v(t+dt) - (v(t) + a(t)*dt)|.

    Nonzero when reported speed change disagrees with reported acceleration -- the
    classic signature of a V2V speed/accel spoof or a CAN speed-injection.
    """
    return np.abs(v_tp1 - (v_t + a_t * dt))


def spacing_residual(gap_t: np.ndarray, gap_tp1: np.ndarray,
                     v_lead: np.ndarray, v_follow: np.ndarray, dt: float) -> np.ndarray:
    """|gap(t+dt) - (gap(t) + (v_lead - v_follow)*dt)|.

    Checks that the reported predecessor speed is consistent with the measured
    closing/opening of the bumper-to-bumper gap. Catches predecessor-speed spoofs
    that leave the ego's own kinematics untouched.
    """
    return np.abs(gap_tp1 - (gap_t + (v_lead - v_follow) * dt))


def powertrain_residual(a_meas: np.ndarray, v: np.ndarray, load_frac: np.ndarray,
                        a_from_load: "callable") -> np.ndarray:
    """|a_meas - a_hat(load, v)| where a_hat is a train-fit map from engine load and
    speed to achievable acceleration. Catches CAN torque/load injection that is not
    reflected in the vehicle's measured motion."""
    return np.abs(a_meas - a_from_load(load_frac, v))


def string_stability_ratio(spacing_err: np.ndarray) -> float:
    """L2 string-stability ratio: ||e_tail||_2 / ||e_first_follower||_2.

    spacing_err: array [T, N] of spacing errors (member 0 is the leader, has no
    predecessor). A value <= 1 means disturbances attenuate down the string.
    """
    if spacing_err.shape[1] < 3:
        return float("nan")
    e_first = np.linalg.norm(spacing_err[:, 1])
    e_last = np.linalg.norm(spacing_err[:, -1])
    return float(e_last / (e_first + 1e-9))


def safe_gap(v_follow: np.ndarray, s_min: float, tau_safe: float) -> np.ndarray:
    """Constant-time-headway safe spacing policy s_min + tau_safe * v."""
    return s_min + tau_safe * v_follow
