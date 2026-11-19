"""End-to-end experiment: ingest real data -> build platoons -> train/eval all
controllers -> cross-layer detection -> Tables 1-3 + Figures 1-3.

Reproducible on CPU, no API keys. Seeds pinned across Python/NumPy/PyTorch.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import Manifest, fetch_ngsim, fetch_ved  # noqa: E402
from src.data.platoon_graph import build_platoons  # noqa: E402
from src.models.baselines import build_controllers  # noqa: E402
from src.physics.hard_projection_control import ConvoyController  # noqa: E402
from src.security.channel import degrade_telemetry, attack_telemetry_v2v  # noqa: E402
from src.security.cross_layer_ids import (  # noqa: E402
    fit_onboard_model, CrossLayerIDS, ved_load_pool, build_ids_records,
    inject_v2v_bias, inject_v2v_replay, inject_can_bias, sanitize_telemetry)
from src.eval.metrics import aggregate_control, spacing_rmse, anomaly_metrics  # noqa: E402
from src.eval.tables import write_table, write_metrics_json  # noqa: E402
from src.eval import figures as F  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_cfgs():
    with open(ROOT / "configs/data.yaml") as f:
        data = yaml.safe_load(f)
    with open(ROOT / "configs/model.yaml") as f:
        model = yaml.safe_load(f)
    with open(ROOT / "configs/experiment.yaml") as f:
        exp = yaml.safe_load(f)
    return data, model, exp


def run_over(ctrl, platoons, idxs, mu, telem_map, energy, s_min, tau_safe):
    """Roll out a controller over strings; attach spacing RMSE; return result list."""
    out = []
    for i in idxs:
        s = platoons.strings[i]
        telem = telem_map.get(i) if telem_map else None
        r = ctrl.rollout(s, mu=mu, telemetry=telem, energy=energy)
        r["spacing_rmse_m"] = spacing_rmse(r, s, s_min, tau_safe)
        out.append(r)
    return out


def main():
    t_start = time.time()
    data, model, exp = load_cfgs()
    seed = data.get("seed", 7)
    seed_all(seed)
    ph = model["physics"]
    friction = model["friction_regimes"]
    comm = exp["comm_regimes"]

    # ---- ingest real data ---- #
    manifest = Manifest()
    print("[1/6] ingesting real NGSIM I-80 + VED ...", flush=True)
    ng = fetch_ngsim(data, manifest)
    ved = fetch_ved(data, manifest)

    # ---- platoons + energy + onboard models ---- #
    print("[2/6] reconstructing real leader-follower strings ...", flush=True)
    platoons = build_platoons(ng, ved, data, ph, manifest)
    print("       strings:", platoons.meta["sizes"],
          "| VED mean power (W):", platoons.meta["ved_mean_power_w"],
          "| model mean power (W):", platoons.meta["model_mean_power_w"], flush=True)
    onboard = fit_onboard_model(ved)
    manifest.save(ROOT / data["manifest"])

    tr, va, te = platoons.splits["train"], platoons.splits["val"], platoons.splits["test"]
    if not tr or not te:
        raise RuntimeError(f"insufficient split sizes {platoons.meta['sizes']}; "
                           "increase ngsim.max_rows or loosen platoon window")
    te_eval = te[:40]  # bound rollout compute

    # ---- controllers (train the neural ones on train split) ---- #
    print("[3/6] training controllers ...", flush=True)
    ctrls = build_controllers(model, platoons.energy.mass)
    for name, c in ctrls.items():
        if isinstance(c, ConvoyController):
            c.fit(platoons.strings, tr, epochs=model["controller"]["epochs"],
                  lr=model["controller"]["lr"], batch=model["controller"]["batch"],
                  wd=model["controller"]["weight_decay"], seed=seed)

    rng = np.random.default_rng(seed)

    def telem_for(regime_name, mu_name):
        c = comm[regime_name]
        m = {}
        for i in te_eval:
            m[i] = degrade_telemetry(platoons.strings[i], c["drop"], c["latency_steps"], rng)
        return m

    # ---- Table 1: control & energy across regimes ---- #
    print("[4/6] Table 1 (control across comms x friction) ...", flush=True)
    regimes = [("clean", "high"), ("drop30", "medium"), ("drop70", "low")]
    rows = []
    cache_clean_high = {}
    for name, c in ctrls.items():
        for reg, mu_name in regimes:
            tmap = telem_for(reg, mu_name)
            res = run_over(c, platoons, te_eval, friction[mu_name], tmap,
                           platoons.energy, ph["s_min"], ph["tau_safe"])
            agg = aggregate_control(res)
            if (reg, mu_name) == ("clean", "high"):
                cache_clean_high[name] = (agg, res)
            rows.append({"method": name, "comms": reg, "friction": mu_name, **agg})
    t1 = pd.DataFrame(rows)
    write_table(t1, ROOT / "results/table1_control_energy.csv", index=False)

    # ---- Table 2: ablation at nominal (clean, high mu) ---- #
    print("[5/6] Table 2 (hard-projection ablation) ...", flush=True)
    abl_methods = ["IDM/CACC", "Soft-PINN", "Unconstrained-NN", "D-PCNC"]
    t2_rows = []
    for name in abl_methods:
        agg, _ = cache_clean_high[name]
        t2_rows.append({"method": name,
                        "headway_violation_pct": agg["headway_violation_pct"],
                        "collisions": agg["collisions"],
                        "string_stability": agg["string_stability"],
                        "energy_kwh_km": agg["energy_kwh_km"],
                        "latency_ms": agg["latency_ms"]})
    t2 = pd.DataFrame(t2_rows)
    write_table(t2, ROOT / "results/table2_ablation.csv", index=False)

    # ---- Table 3: cross-layer detection + convoy resilience ---- #
    print("[6/6] Table 3 (cross-layer detection) ...", flush=True)
    load_pool = ved_load_pool(ved)
    ids = CrossLayerIDS(onboard=onboard, dt=ph["dt"])
    clean_train = build_ids_records(platoons.strings, tr, load_pool, ph["dt"],
                                    np.random.default_rng(seed))
    ids.fit(clean_train, q=exp["eval"]["guard_threshold_quantile"])

    clean_test = build_ids_records(platoons.strings, te, load_pool, ph["dt"],
                                   np.random.default_rng(seed + 1))
    neg_scores = [ids.run(r).scores.max() for r in clean_test]

    attack_specs = {
        "V2V bias spoof": lambda r, o: inject_v2v_bias(r, 4.0, o),
        "V2V replay": lambda r, o: inject_v2v_replay(r, 15, o),
        "CAN speed inject": lambda r, o: inject_can_bias(r, "speed", 6.0, o),
        "CAN load inject": lambda r, o: inject_can_bias(r, "load", 0.6, o),
    }
    arng = np.random.default_rng(seed + 2)
    t3_rows = []
    for atk_name, fn in attack_specs.items():
        pos_scores, delays = [], []
        for r in clean_test:
            onset = int(arng.integers(len(r["v_follow"]) // 4, len(r["v_follow"]) // 2))
            rep = ids.run(fn(r, onset))
            pos_scores.append(rep.scores.max()); delays.append(rep.delay)
        scores = np.array(neg_scores + pos_scores)
        labels = np.array([0] * len(neg_scores) + [1] * len(pos_scores))
        m = anomaly_metrics(scores, labels, ids.threshold, delays)
        m["detection_latency_ms"] = (m["detection_delay_steps"] * ph["dt"] * 1000
                                     if np.isfinite(m["detection_delay_steps"]) else float("nan"))
        t3_rows.append({"attack": atk_name, "auroc": m["auroc"], "f1": m["f1"],
                        "detection_latency_ms": m["detection_latency_ms"],
                        "fpr": m["fpr"]})

    # convoy collision/instability under attack: guarded vs unguarded (D-PCNC)
    dpcnc = ctrls["D-PCNC"]
    guarded_col = unguarded_col = 0
    guarded_viol, unguarded_viol = [], []
    trace = None
    grng = np.random.default_rng(seed + 3)
    for i in te_eval:
        s = platoons.strings[i]
        onset = s["T"] // 3
        v_atk, a_atk = attack_telemetry_v2v(s, exp["attack_regimes"]["v2v_spoof"]["magnitude"], onset)
        # unguarded
        r_u = dpcnc.rollout(s, mu=friction["high"], telemetry=(v_atk, a_atk), energy=platoons.energy)
        unguarded_col += r_u["collisions"]; unguarded_viol.append(r_u["headway_violation"])
        # guarded: per-follower IDS flags -> sanitize each predecessor column
        v_san, a_san = v_atk.copy(), a_atk.copy()
        flags_col = np.zeros(s["T"], bool)
        for j in range(1, s["N"]):
            rec = {"v_follow": s["vel"][:, j], "a_meas": s["acc"][:, j],
                   "gap": np.nan_to_num(s["gap"][:, j], nan=10.0),
                   "v_bus": s["vel"][:, j].copy(),
                   "load_frac": np.full(s["T"], 0.5),
                   "v_lead_rep": v_atk[:, j - 1], "a_lead_rep": a_atk[:, j - 1]}
            rep = ids.run(rec)
            vs, as_ = sanitize_telemetry(v_atk[:, j - 1], a_atk[:, j - 1],
                                         rec["gap"], s["vel"][:, j], rep.flags, ph["dt"])
            v_san[:, j - 1] = vs; a_san[:, j - 1] = as_
            if j == 1:
                flags_col = rep.flags
        r_g = dpcnc.rollout(s, mu=friction["high"], telemetry=(v_san, a_san), energy=platoons.energy)
        guarded_col += r_g["collisions"]; guarded_viol.append(r_g["headway_violation"])
        if trace is None and s["N"] >= 3:
            safe = ph["s_min"] + ph["tau_safe"] * r_g["vel"][:, 1]
            trace = {"t": np.arange(s["T"]), "onset": onset,
                     "gap_guarded": r_g["gap"][:, 1], "gap_unguarded": r_u["gap"][:, 1],
                     "safe_gap": safe, "score": ids.run({
                         "v_follow": s["vel"][:, 1], "a_meas": s["acc"][:, 1],
                         "gap": np.nan_to_num(s["gap"][:, 1], nan=10.0),
                         "v_bus": s["vel"][:, 1].copy(), "load_frac": np.full(s["T"], 0.5),
                         "v_lead_rep": v_atk[:, 0], "a_lead_rep": a_atk[:, 0]}).scores,
                     "threshold": ids.threshold}

    t3 = pd.DataFrame(t3_rows)
    write_table(t3, ROOT / "results/table3_cross_layer_detection.csv", index=False)
    resilience = {
        "attack": "V2V speed spoof (+4 m/s)",
        "unguarded_collisions": int(unguarded_col),
        "guarded_collisions": int(guarded_col),
        "unguarded_headway_violation_pct": 100 * float(np.mean(unguarded_viol)),
        "guarded_headway_violation_pct": 100 * float(np.mean(guarded_viol)),
        "strings_evaluated": len(te_eval),
    }

    # ---- Figures ---- #
    # fig1: mean |spacing error| by convoy position, D-PCNC, across comms regimes
    per_pos = {}
    for reg, mu_name in regimes:
        tmap = telem_for(reg, mu_name)
        res = run_over(ctrls["D-PCNC"], platoons, te_eval, friction[mu_name], tmap,
                       platoons.energy, ph["s_min"], ph["tau_safe"])
        maxN = max(r["gap"].shape[1] for r in res)
        acc = np.zeros(maxN); cnt = np.zeros(maxN)
        for r in res:
            safe = ph["s_min"] + ph["tau_safe"] * r["vel"]
            err = np.abs(r["gap"] - safe)
            for j in range(1, r["gap"].shape[1]):
                col = err[:, j]; col = col[np.isfinite(col)]
                if col.size:
                    acc[j] += col.mean(); cnt[j] += 1
        per_pos[reg] = (acc[1:] / np.maximum(cnt[1:], 1))
    F.fig1_string_stability_and_safety(per_pos, ROOT / "results/figures")

    # fig2: safety-violation vs energy pareto (ablation + MPC)
    points = {}
    for name in ["IDM/CACC", "MPC", "Soft-PINN", "Unconstrained-NN", "D-PCNC"]:
        agg, _ = cache_clean_high[name]
        points[name] = (agg["energy_kwh_km"], agg["headway_violation_pct"])
    F.fig2_hard_projection_ablation(points, ROOT / "results/figures")

    # fig3: coordinated attack trace
    if trace is not None:
        F.fig3_cross_layer_cyber_resilience(trace, ROOT / "results/figures")

    # ---- metrics.json + environment.json ---- #
    all_metrics = {
        "data": platoons.meta,
        "energy_model": {"type": "physics_literature_coefficients",
                         "mass": platoons.energy.mass, "c_rr": platoons.energy.c_rr,
                         "c_aero": platoons.energy.c_aero,
                         "ved_mean_power_w": platoons.energy.ved_mean_power_w,
                         "model_mean_power_w": platoons.energy.model_mean_power_w},
        "table1_control_energy": t1.to_dict(orient="records"),
        "table2_ablation": t2.to_dict(orient="records"),
        "table3_detection": t3.to_dict(orient="records"),
        "convoy_resilience_under_attack": resilience,
        "runtime_s": round(time.time() - t_start, 1),
    }
    write_metrics_json(all_metrics, ROOT / "results/metrics.json")
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
           "seed": seed}
    write_metrics_json(env, ROOT / "results/environment.json")

    print("\n==== SUMMARY ====")
    print(t1.to_string(index=False))
    print("\nAblation (clean, high mu):")
    print(t2.to_string(index=False))
    print("\nCross-layer detection:")
    print(t3.to_string(index=False))
    print("\nConvoy under V2V spoof:", json.dumps(resilience, indent=2))
    print(f"\nDone in {all_metrics['runtime_s']}s. Results in results/.")


if __name__ == "__main__":
    main()
