"""Aggregation of control metrics and detection metrics."""
from __future__ import annotations

import numpy as np


def aggregate_control(results: list[dict]) -> dict:
    """Mean over strings of the per-string control metrics."""
    def m(key):
        vals = [r[key] for r in results if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    total_collisions = sum(r["collisions"] for r in results)
    return {
        "spacing_rmse_m": m("spacing_rmse_m"),
        "speed_rmse_ms": m("speed_rmse"),
        "string_stability": m("string_stability"),
        "headway_violation_pct": 100.0 * m("headway_violation"),
        "collisions": int(total_collisions),
        "energy_kwh_km": m("energy_kwh_km"),
        "latency_ms": m("latency_ms"),
    }


def spacing_rmse(result: dict, s: dict, s_min: float, tau_safe: float) -> float:
    """RMSE of spacing error (gap - safe gap) over followers."""
    vel = result["vel"]; gap = result["gap"]
    safe = s_min + tau_safe * vel
    err = gap[:, 1:] - safe[:, 1:]
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def anomaly_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float,
                    delays: list[int]) -> dict:
    """AUROC, F1, FPR at the fit threshold, and mean detection delay (steps)."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    auroc = _auroc(scores, labels)
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    valid_delays = [d for d in delays if d >= 0]
    return {
        "auroc": float(auroc), "f1": float(f1), "precision": float(prec),
        "recall": float(rec), "fpr": float(fpr),
        "detection_delay_steps": float(np.mean(valid_delays)) if valid_delays else float("nan"),
        "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
    }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
