"""Cross-layer IDS must separate injected attacks from clean telemetry."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.security.cross_layer_ids import (
    CrossLayerIDS, OnboardModel, inject_v2v_bias, inject_can_bias, sanitize_telemetry)
from src.eval.metrics import anomaly_metrics


def _clean_record(T=200, seed=0):
    rng = np.random.default_rng(seed)
    dt = 0.1
    a = 0.3 * np.sin(np.linspace(0, 8, T)) + rng.normal(0, 0.05, T)
    v = np.cumsum(a) * dt + 15.0
    v = np.clip(v, 0, None)
    v_lead = v + 1.0
    a_lead = np.gradient(v_lead, dt)
    gap = np.full(T, 20.0)
    gap[1:] = gap[:-1] + (v_lead[:-1] - v[:-1]) * dt
    load = np.clip(0.4 + 0.1 * np.sin(np.linspace(0, 5, T)), 0, 1)
    return {"v_follow": v, "a_meas": a, "gap": gap, "v_bus": v.copy(),
            "load_frac": load, "v_lead_rep": v_lead, "a_lead_rep": a_lead,
            "label": 0, "onset": 0}


def _ids():
    onboard = OnboardModel(w=np.array([0.0, 0.0, 0.0, 0.0]))  # a_hat ~ 0 baseline
    ids = CrossLayerIDS(onboard=onboard, dt=0.1)
    clean = [_clean_record(seed=s) for s in range(20)]
    ids.fit(clean, q=0.99)
    return ids


def test_v2v_bias_raises_score():
    ids = _ids()
    base = _clean_record(seed=100)
    clean_score = ids.run(base).scores.max()
    atk = inject_v2v_bias(base, magnitude=5.0, onset=100)
    atk_score = ids.run(atk).scores.max()
    assert atk_score > clean_score


def test_can_speed_injection_detected():
    ids = _ids()
    base = _clean_record(seed=101)
    atk = inject_can_bias(base, "speed", magnitude=6.0, onset=100)
    rep = ids.run(atk)
    assert rep.detected and rep.delay >= 0


def test_detection_auroc_above_half():
    ids = _ids()
    neg, pos, delays = [], [], []
    for s in range(30):
        base = _clean_record(seed=500 + s)
        neg.append(ids.run(base).scores.max())
        atk = inject_v2v_bias(_clean_record(seed=500 + s), magnitude=5.0, onset=100)
        rep = ids.run(atk)
        pos.append(rep.scores.max()); delays.append(rep.delay)
    scores = np.array(neg + pos)
    labels = np.array([0] * len(neg) + [1] * len(pos))
    m = anomaly_metrics(scores, labels, ids.threshold, delays)
    assert m["auroc"] > 0.8


def test_sanitize_replaces_only_flagged():
    T = 50
    v_lead = np.full(T, 20.0); a_lead = np.zeros(T)
    gap = np.full(T, 15.0); v = np.full(T, 18.0)
    flags = np.zeros(T, bool); flags[20:] = True
    vo, ao = sanitize_telemetry(v_lead, a_lead, gap, v, flags, 0.1)
    assert np.allclose(vo[:20], v_lead[:20])     # unflagged untouched
    assert not np.allclose(vo[20:], v_lead[20:]) # flagged replaced
