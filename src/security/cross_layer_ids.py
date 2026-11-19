"""Cross-layer V2V + in-vehicle CAN intrusion / spoofing detector, with a
physics-propagated graceful-degradation fallback.

The detector enforces *physical consistency across layers* rather than learning an
attack signature, so it generalises across attack types:

  r_ego   ego CAN speed vs ego measured motion     |v_bus(t+dt) - (v_bus + a_meas*dt)|
  r_pt    ego powertrain: measured accel vs load    |a_meas - a_hat(load, v)|
  r_v2v   predecessor V2V speed vs measured gap      |gap(t+dt) - (gap + (v_lead_rep - v)*dt)|
  r_self  predecessor V2V self-consistency           |v_lead_rep(t+dt) - (v_lead_rep + a_lead_rep*dt)|

Each residual is scaled by a statistic fit on CLEAN TRAINING data only; the step
anomaly score is the max of the scaled residuals, and the integrity score is its
complement. A window is flagged when the score exceeds a clean-train quantile
threshold. When flagged, ``sanitize_telemetry`` discards the untrusted predecessor
V2V feed and propagates the last trusted state from the (trusted) measured gap --
keeping the convoy string-stable without V2V trust.

Attack models (eval-only, injected with labels):
  * V2V bias / ramp spoof on predecessor speed & accel
  * V2V replay of an earlier telemetry window
  * CAN speed-injection and CAN load/torque-injection on the ego bus
Real labeled CAN frames (HCRL drop-in) are used instead when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Onboard powertrain map, calibrated on real VED
# --------------------------------------------------------------------------- #
@dataclass
class OnboardModel:
    """a_hat = w . [1, load_frac, v, load_frac*v], fit by least squares on real VED
    OBD (engine load, speed) vs speed-derived acceleration."""
    w: np.ndarray

    def predict(self, load_frac, v):
        load_frac = np.asarray(load_frac, float)
        v = np.asarray(v, float)
        X = np.stack([np.ones_like(v), load_frac, v, load_frac * v], axis=-1)
        return X @ self.w


def fit_onboard_model(ved_df) -> OnboardModel:
    feats, targ = [], []
    for _, trip in ved_df.groupby(["VehId", "Trip"]):
        trip = trip.sort_values("Timestamp(ms)")
        t = trip["Timestamp(ms)"].to_numpy(float) / 1000.0
        v = trip["speed_ms"].to_numpy(float)
        if len(v) < 10 or "Absolute Load[%]" not in trip.columns:
            continue
        load = trip["Absolute Load[%]"].to_numpy(float) / 100.0
        dt = np.diff(t); dt[dt <= 0] = np.nan
        a = np.diff(v) / dt
        load_m, v_m = load[1:], v[1:]
        m = np.isfinite(a) & np.isfinite(load_m) & (np.abs(a) < 6)
        if not m.any():
            continue
        feats.append(np.stack([np.ones(m.sum()), load_m[m], v_m[m], load_m[m] * v_m[m]], axis=1))
        targ.append(a[m])
    X = np.concatenate(feats); y = np.concatenate(targ)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return OnboardModel(w=w)


# --------------------------------------------------------------------------- #
# Attack injection (eval-only, labeled)
# --------------------------------------------------------------------------- #
def inject_v2v_bias(rec: dict, magnitude: float, onset: int) -> dict:
    r = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
    r["v_lead_rep"][onset:] += magnitude
    r["a_lead_rep"][onset:] += 0.0  # bias on speed; accel left plausible -> stealthier
    r["label"] = 1; r["onset"] = onset
    return r


def inject_v2v_replay(rec: dict, window: int, onset: int) -> dict:
    r = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
    T = len(r["v_lead_rep"])
    src0 = max(0, onset - window)
    seg = r["v_lead_rep"][src0:onset]
    seg_a = r["a_lead_rep"][src0:onset]
    if len(seg) > 0:
        reps = int(np.ceil((T - onset) / len(seg)))
        r["v_lead_rep"][onset:] = np.tile(seg, reps)[: T - onset]
        r["a_lead_rep"][onset:] = np.tile(seg_a, reps)[: T - onset]
    r["label"] = 1; r["onset"] = onset
    return r


def inject_can_bias(rec: dict, signal: str, magnitude: float, onset: int) -> dict:
    r = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
    if signal == "speed":
        r["v_bus"][onset:] += magnitude            # CAN speed injection
    elif signal == "load":
        r["load_frac"][onset:] = np.clip(r["load_frac"][onset:] + magnitude, 0, 2.0)
    r["label"] = 1; r["onset"] = onset
    return r


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class GuardReport:
    scores: np.ndarray
    flags: np.ndarray
    integrity: np.ndarray
    threshold: float
    detected: bool
    delay: int                    # steps from onset to first flag (-1 if missed)


@dataclass
class CrossLayerIDS:
    onboard: OnboardModel
    dt: float
    scales: dict = field(default_factory=dict)
    threshold: float = float("inf")

    # -- per-step residuals -- #
    def _residuals(self, rec: dict) -> np.ndarray:
        dt = self.dt
        v = rec["v_follow"]; a = rec["a_meas"]; gap = rec["gap"]
        vbus = rec["v_bus"]; load = rec["load_frac"]
        vlr = rec["v_lead_rep"]; alr = rec["a_lead_rep"]
        T = len(v)
        r_ego = np.zeros(T); r_pt = np.zeros(T); r_v2v = np.zeros(T); r_self = np.zeros(T)
        r_ego[:-1] = np.abs(vbus[1:] - (vbus[:-1] + a[:-1] * dt))
        r_pt[:] = np.abs(a - self.onboard.predict(load, vbus))
        r_v2v[:-1] = np.abs(gap[1:] - (gap[:-1] + (vlr[:-1] - v[:-1]) * dt))
        r_self[:-1] = np.abs(vlr[1:] - (vlr[:-1] + alr[:-1] * dt))
        return np.stack([r_ego, r_pt, r_v2v, r_self], axis=1)

    # -- fit scales + threshold on clean training records -- #
    def fit(self, clean_recs: list[dict], q: float = 0.99):
        allr = [self._residuals(r) for r in clean_recs]
        R = np.concatenate(allr, axis=0)
        self.scales = {i: float(np.median(np.abs(R[:, i])) + 1e-6) for i in range(R.shape[1])}
        # Calibrate the threshold on clean WINDOW-max scores (not per-step): the
        # detector classifies at window granularity, so a per-step quantile would
        # inflate the window false-positive rate to ~1-(q)^T.
        win_scores = np.array([self._score_steps(r).max() for r in clean_recs])
        self.threshold = float(np.quantile(win_scores, q))
        return self

    def _score_steps(self, rec: dict) -> np.ndarray:
        R = self._residuals(rec)
        scaled = np.stack([R[:, i] / self.scales.get(i, 1.0) for i in range(R.shape[1])], axis=1)
        return scaled.max(axis=1)

    def run(self, rec: dict) -> GuardReport:
        scores = self._score_steps(rec)
        flags = scores > self.threshold
        integrity = 1.0 / (1.0 + scores)
        detected = bool(flags.any())
        delay = -1
        if detected and rec.get("label", 0) == 1:
            onset = rec.get("onset", 0)
            # a step injection is detectable at its leading edge (onset-1); count
            # detection delay from there, clamped to >= 0.
            start = max(0, onset - 1)
            after = np.where(flags[start:])[0]
            if len(after):
                delay = max(0, int(start + after[0]) - onset)
        return GuardReport(scores=scores, flags=flags, integrity=integrity,
                           threshold=self.threshold, detected=detected, delay=delay)


def ved_load_pool(ved_df, min_len: int = 100) -> list[np.ndarray]:
    """Real VED engine-load streams (fraction), one per trip with enough samples."""
    pool = []
    for _, trip in ved_df.groupby(["VehId", "Trip"]):
        if "Absolute Load[%]" not in trip.columns:
            continue
        load = trip.sort_values("Timestamp(ms)")["Absolute Load[%]"].to_numpy(float) / 100.0
        load = load[np.isfinite(load)]
        if len(load) >= min_len:
            pool.append(load)
    return pool


def _resample_to(x: np.ndarray, T: int) -> np.ndarray:
    if len(x) == 0:
        return np.full(T, 0.5)
    reps = int(np.ceil(T / len(x)))
    return np.tile(x, reps)[:T]


def build_ids_records(strings, idxs, load_pool, dt, rng) -> list[dict]:
    """One clean record per follower, pairing real NGSIM kinematics with a real
    VED engine-load stream (the disclosed constructed cross-modal pairing)."""
    recs = []
    for i in idxs:
        s = strings[i]
        for j in range(1, s["N"]):
            T = s["T"]
            gap = s["gap"][:, j].copy()
            gap = np.where(np.isfinite(gap), gap, np.nanmedian(gap[np.isfinite(gap)]) if np.isfinite(gap).any() else 10.0)
            vfollow = s["vel"][:, j].copy()
            load = _resample_to(load_pool[rng.integers(len(load_pool))], T)
            recs.append({
                "v_follow": vfollow,
                "a_meas": s["acc"][:, j].copy(),
                "gap": gap,
                "v_bus": vfollow.copy(),
                "load_frac": load,
                "v_lead_rep": s["vel"][:, j - 1].copy(),
                "a_lead_rep": s["acc"][:, j - 1].copy(),
                "label": 0, "onset": 0,
                "string": i, "member": j,
            })
    return recs


def sanitize_telemetry(v_lead_rep, a_lead_rep, gap, v_follow, flags, dt):
    """Physics-propagated fallback: where ``flags`` is set, discard the V2V feed and
    estimate the predecessor speed from the trusted measured gap derivative; hold
    predecessor accel at zero (conservative constant-velocity propagation)."""
    T = len(v_lead_rep)
    v_out = v_lead_rep.copy(); a_out = a_lead_rep.copy()
    gap_rate = np.zeros(T)
    gap_rate[:-1] = (gap[1:] - gap[:-1]) / dt
    for t in range(T):
        if flags[t]:
            v_out[t] = v_follow[t] + gap_rate[t]   # closing-rate consistent speed
            a_out[t] = 0.0
    return v_out, a_out
