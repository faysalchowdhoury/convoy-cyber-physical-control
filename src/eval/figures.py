"""Publication figures: 300 DPI, Times New Roman, colorblind-safe (Okabe-Ito).
Each figure is written as both .png and .pdf."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = {
    "D-PCNC": "#E69F00", "MPC": "#0072B2", "Unconstrained-NN": "#009E73",
    "Soft-PINN": "#CC79A7", "IDM/CACC": "#56B4E9",
}
MARK = {"D-PCNC": "o", "MPC": "s", "Unconstrained-NN": "^", "Soft-PINN": "D", "IDM/CACC": "v"}


def _setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def _save(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def fig1_string_stability_and_safety(per_pos: dict, out_dir):
    """per_pos[regime] = array [N] mean |spacing error| by convoy position (1..N)."""
    _setup()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    styles = {"clean": "-", "drop30": "--", "drop70": ":"}
    for regime, arr in per_pos.items():
        xs = np.arange(1, len(arr) + 1)
        ax.plot(xs, arr, styles.get(regime, "-"), marker="o", color="#0072B2"
                if regime == "clean" else ("#E69F00" if regime == "drop30" else "#D55E00"),
                label=regime)
    ax.set_xlabel("Convoy position (vehicle index)")
    ax.set_ylabel("Mean spacing error magnitude (m)")
    ax.set_title("D-PCNC disturbance propagation along the convoy")
    ax.legend(title="Comms regime")
    _save(fig, out_dir, "fig1_string_stability_and_safety")


def fig2_hard_projection_ablation(points: dict, out_dir):
    """points[method] = (energy_kwh_km, headway_violation_pct)."""
    _setup()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method, (energy, viol) in points.items():
        ax.scatter(energy, viol, s=90, color=OKABE_ITO.get(method, "#555555"),
                   marker=MARK.get(method, "o"), edgecolor="black", linewidth=0.6,
                   label=method, zorder=3)
    ax.set_xlabel("Traction energy (kWh/km)")
    ax.set_ylabel("Safety / headway violation rate (%)")
    ax.set_title("Safety vs energy: hard projection eliminates violations")
    ax.legend()
    _save(fig, out_dir, "fig2_hard_projection_ablation")


def fig3_cross_layer_cyber_resilience(trace: dict, out_dir):
    """trace keys: t, onset, gap_guarded, gap_unguarded, safe_gap, score, threshold."""
    _setup()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    t = trace["t"]
    ax1.plot(t, trace["gap_unguarded"], color="#D55E00", label="no guard (gap)")
    ax1.plot(t, trace["gap_guarded"], color="#009E73", label="with guard (gap)")
    ax1.plot(t, trace["safe_gap"], color="black", ls="--", lw=1, label="safe gap")
    ax1.axvline(trace["onset"], color="gray", ls=":", lw=1)
    ax1.annotate("attack onset", (trace["onset"], ax1.get_ylim()[1]),
                 textcoords="offset points", xytext=(4, -12), fontsize=9, color="gray")
    ax1.set_ylabel("Bumper gap (m)")
    ax1.set_title("Coordinated V2V+CAN spoof: guard keeps the gap safe")
    ax1.legend(loc="upper right", fontsize=9)
    ax2.plot(t, trace["score"], color="#0072B2", label="integrity anomaly score")
    ax2.axhline(trace["threshold"], color="red", ls="--", lw=1, label="threshold")
    ax2.axvline(trace["onset"], color="gray", ls=":", lw=1)
    ax2.set_xlabel("Time step (0.1 s)")
    ax2.set_ylabel("Anomaly score")
    ax2.legend(loc="upper left", fontsize=9)
    _save(fig, out_dir, "fig3_cross_layer_cyber_resilience")
