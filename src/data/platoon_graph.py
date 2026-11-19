"""Reconstruct real leader-follower vehicle strings from NGSIM, build control
features, calibrate an energy model on real VED telemetry, and produce strictly
vehicle-disjoint, chronological train/val/test splits.

A "string" is a chain of vehicles ``[leader, f1, f2, ...]`` in one lane for which
NGSIM's native ``preceding`` linkage holds, unchanged, over a continuous window and
every member is co-present for the whole window. Spacing, relative velocity,
acceleration and jerk are computed per member per time step.

Anti-leakage: each NGSIM vehicle is assigned to exactly one split by chronological
order of first appearance, and a string is kept only if *all* its members fall in
the same split -- guaranteeing zero vehicle overlap across train/val/test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


G = 9.81


@dataclass
class EnergyModel:
    """Standard longitudinal-dynamics traction-power model (watts):

        F = m*a + m*g*c_rr + c_aero*v^2     (tractive force)
        P = max(F, 0) * v                   (propulsion only; regen/coast ignored)

    Coefficients are literature values (``configs/model.yaml``), not fit to VED --
    an instantaneous VED power fit does not converge (R^2 ~ 0 at 1 Hz). VED is used
    only to sanity-check the model's power magnitude (see ``ved_power_check``).
    """
    mass: float
    c_rr: float
    c_aero: float
    ved_mean_power_w: float = float("nan")      # measured, for magnitude sanity-check
    model_mean_power_w: float = float("nan")    # model at VED speed/accel

    @classmethod
    def from_physics(cls, phys: dict) -> "EnergyModel":
        return cls(mass=phys["veh_mass"], c_rr=phys["c_rr"], c_aero=phys["c_aero"])

    def power(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float); a = np.asarray(a, float)
        F = self.mass * a + self.mass * G * self.c_rr + self.c_aero * v ** 2
        return np.clip(F, 0.0, None) * v


@dataclass
class PlatoonData:
    strings: list[dict] = field(default_factory=list)
    splits: dict[str, list[int]] = field(default_factory=dict)
    energy: EnergyModel | None = None
    dt: float = 0.1
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Energy model from real VED
# --------------------------------------------------------------------------- #
GASOLINE_W_PER_LPH = 9500.0  # gasoline LHV proxy: 1 L/hr ~ 9.5 kW chemical power


def ved_power_check(ved_df, energy: EnergyModel) -> EnergyModel:
    """Fill the energy model's VED-vs-model mean-power sanity check (real VED fuel
    power vs the physics model evaluated at the same speed/accel). Does not fit."""
    import numpy as np

    def smooth(x, k=5):
        if len(x) < k:
            return x
        kern = np.ones(k) / k
        return np.convolve(x, kern, mode="same")

    meas, modeled = [], []
    for _, trip in ved_df.groupby(["VehId", "Trip"]):
        trip = trip.sort_values("Timestamp(ms)")
        t = trip["Timestamp(ms)"].to_numpy(float) / 1000.0
        v = smooth(trip["speed_ms"].to_numpy(float))
        if len(v) < 15:
            continue
        dt = np.diff(t); dt[dt <= 0] = np.nan
        a = smooth(np.diff(v) / dt)
        v_mid = v[1:]
        # real on-road power: fuel (ICE) if present, else HV battery (HEV/EV)
        P = np.full(len(v), np.nan)
        if "Fuel Rate[L/hr]" in trip.columns:
            fr = trip["Fuel Rate[L/hr]"].to_numpy(float) * GASOLINE_W_PER_LPH
            P = np.where(np.isfinite(fr) & (fr > 0), fr, P)
        if np.isnan(P).all() and {"HV Battery Voltage[V]", "HV Battery Current[A]"}.issubset(trip.columns):
            pb = trip["HV Battery Voltage[V]"].to_numpy(float) * trip["HV Battery Current[A]"].to_numpy(float)
            P = np.where(np.isfinite(pb), np.clip(pb, 0, None), P)
        P_mid = P[1:]
        m = np.isfinite(a) & np.isfinite(P_mid) & (P_mid > 0) & (v_mid >= 0) & (np.abs(a) < 5)
        if m.sum() < 10:
            continue
        meas.append(P_mid[m])
        modeled.append(energy.power(v_mid[m], a[m]))
    if meas:
        energy.ved_mean_power_w = float(np.mean(np.concatenate(meas)))
        energy.model_mean_power_w = float(np.mean(np.concatenate(modeled)))
    return energy


# --------------------------------------------------------------------------- #
# Platoon reconstruction
# --------------------------------------------------------------------------- #
def _vehicle_frame_index(df):
    """Map (vehicle_id, frame_id) -> row dict for fast chain checks."""
    idx: dict[int, dict] = {}
    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("frame_id")
        idx[int(vid)] = {
            "frames": g["frame_id"].to_numpy(np.int64),
            "preceding": g["preceding"].to_numpy(np.int64),
            "lane": g["lane_id"].to_numpy(np.int64),
            "y": g["local_y"].to_numpy(float),
            "v": g["v_vel"].to_numpy(float),
            "a": g["v_acc"].to_numpy(float),
            "gap": g["space_headway"].to_numpy(float),
            "length": float(np.nanmedian(g["v_length"].to_numpy(float))),
            "f0": int(g["frame_id"].min()),
        }
    return idx


def extract_strings(df, pcfg: dict) -> list[dict]:
    idx = _vehicle_frame_index(df)
    # per-vehicle frame->position lookup
    pos_in = {vid: {int(f): k for k, f in enumerate(d["frames"])}
              for vid, d in idx.items()}
    strings: list[dict] = []
    used: set[int] = set()  # each vehicle belongs to at most one accepted string

    for tail in sorted(idx):
        if tail in used:
            continue
        d = idx[tail]
        # reference frame = tail's first frame; build predecessor chain there
        f_ref = d["f0"]
        chain = [tail]
        cur = tail
        for _ in range(pcfg["max_len"] - 1):
            k = pos_in[cur].get(f_ref)
            if k is None:
                break
            pre = int(idx[cur]["preceding"][k])
            if pre <= 0 or pre not in idx or pre in chain:
                break
            if pre in used:
                break
            chain.append(pre)
            cur = pre
        if len(chain) < pcfg["min_len"]:
            continue
        if any(v in used for v in chain):
            continue
        chain = chain[::-1]  # [leader, ..., tail]

        # longest window where all members present, links stable, same lane
        members = chain
        # candidate frames: those present for the tail
        frames = d["frames"]
        good = []
        for f in frames:
            f = int(f)
            ok = True
            lane0 = None
            for j, vid in enumerate(members):
                k = pos_in[vid].get(f)
                if k is None:
                    ok = False
                    break
                ln = int(idx[vid]["lane"][k])
                if lane0 is None:
                    lane0 = ln
                elif ln != lane0:
                    ok = False
                    break
                if j > 0:  # follower's preceding must equal the member ahead
                    if int(idx[vid]["preceding"][k]) != members[j - 1]:
                        ok = False
                        break
            good.append(ok)
        good = np.asarray(good, bool)
        if not good.any():
            continue
        # longest contiguous True run
        best_s = best_len = 0
        s = None
        for i, gflag in enumerate(good):
            if gflag and s is None:
                s = i
            if (not gflag or i == len(good) - 1) and s is not None:
                e = i if not gflag else i + 1
                if e - s > best_len:
                    best_len, best_s = e - s, s
                s = None
        if best_len < pcfg["min_window"]:
            continue
        w = min(best_len, pcfg["max_window"])
        f_start = int(frames[best_s])
        for v in members:
            used.add(v)

        lane_val = int(idx[members[0]]["lane"][pos_in[members[0]][f_start]])
        N, T = len(members), w
        pos = np.zeros((T, N)); vel = np.zeros((T, N))
        acc = np.zeros((T, N)); gap = np.full((T, N), np.nan)
        lengths = np.array([idx[v]["length"] for v in members])
        for j, vid in enumerate(members):
            for ti in range(T):
                f = f_start + ti
                k = pos_in[vid].get(f)
                pos[ti, j] = idx[vid]["y"][k]
                vel[ti, j] = idx[vid]["v"][k]
                acc[ti, j] = idx[vid]["a"][k]
                if j > 0:
                    gap[ti, j] = idx[vid]["gap"][k] - lengths[j - 1]  # bumper-to-bumper
        strings.append({
            "members": [int(v) for v in members], "lane": lane_val,
            "f_start": f_start, "pos": pos, "vel": vel, "acc": acc,
            "gap": gap, "length": lengths, "N": N, "T": T,
        })
        if len(strings) >= pcfg["max_strings"]:
            break
    return strings


def make_splits(strings: list[dict], scfg: dict, seed: int = 7) -> dict[str, list[int]]:
    """Vehicle-disjoint, chronological splits.

    Strings are ordered by start time and swept train -> val -> test by cumulative
    time. A string is placed in its time-bucket only if none of its members is
    already committed to a different split; otherwise it is dropped. This guarantees
    zero vehicle overlap across splits while keeping the ordering chronological.
    """
    order = sorted(range(len(strings)), key=lambda i: (strings[i]["f_start"], i))
    n = len(order)
    n_tr = int(round(scfg["train"] * n))
    n_va = int(round(scfg["val"] * n))
    bucket_of_pos = (["train"] * n_tr + ["val"] * n_va
                     + ["test"] * (n - n_tr - n_va))

    committed: dict[int, str] = {}
    splits = {"train": [], "val": [], "test": []}
    for pos, i in enumerate(order):
        target = bucket_of_pos[pos]
        members = strings[i]["members"]
        conflict = any(committed.get(v, target) != target for v in members)
        if conflict:
            continue
        for v in members:
            committed[v] = target
        splits[target].append(i)
    return splits


def build_platoons(df, ved_df, cfg: dict, phys: dict, manifest=None) -> PlatoonData:
    pcfg = cfg["platoon"]
    strings = extract_strings(df, pcfg)
    if not strings:
        raise RuntimeError("no stable leader-follower strings extracted from NGSIM")
    splits = make_splits(strings, cfg["split"], seed=cfg.get("seed", 7))
    energy = ved_power_check(ved_df, EnergyModel.from_physics(phys))
    pd_obj = PlatoonData(strings=strings, splits=splits, energy=energy,
                         dt=cfg["ngsim"]["dt"],
                         meta={"n_strings": len(strings),
                               "sizes": {k: len(v) for k, v in splits.items()},
                               "energy_model": "physics (literature coefficients)",
                               "ved_mean_power_w": round(energy.ved_mean_power_w, 1),
                               "model_mean_power_w": round(energy.model_mean_power_w, 1)})
    if manifest is not None:
        manifest.add("platoons", n_strings=len(strings),
                     split_sizes=pd_obj.meta["sizes"],
                     ved_mean_power_w=round(energy.ved_mean_power_w, 1),
                     model_mean_power_w=round(energy.model_mean_power_w, 1),
                     note="Real NGSIM strings (vehicle-disjoint); energy = physics "
                          "model with literature coefficients, VED magnitude-checked.")
    return pd_obj
