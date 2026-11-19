# Convoy Cyber-Physical Control

**Physics-Constrained Decentralized Multi-Vehicle Control and In-Vehicle/V2X
Cyber-Resilience for Autonomous Ground Convoys**

A reproducible research codebase for trustworthy longitudinal control of autonomous
ground convoys under contested communications and cyber-physical attack. It
contributes two mechanisms and evaluates them against classical and neural
baselines on **real public vehicle data**:

1. **D-PCNC** — a Decentralized Physics-Constrained Neural Controller with a
   *differentiable hard physical projection layer* that enforces collision-free
   headway, actuator/traction limits, and powertrain power-ramp limits in closed
   form at each step. The per-step spacing bound is guaranteed by construction
   (proved in `tests/`); in closed-loop multi-vehicle replay it cuts collisions by
   ~99% vs the unconstrained policy (4 vs 588) — but the cloned policy tracks
   spacing poorly, which we report as a diagnostic, not a win (see *Results*).
2. **Cross-Layer IDS** — a V2V + in-vehicle-CAN intrusion detector that enforces
   *physical consistency across layers* and switches the controller to a
   physics-propagated sensor-only fallback when telemetry integrity drops. It
   detects speed/kinematic spoofs immediately (AUROC 1.0 / 0.94) but, as reported,
   fails on self-consistent CAN load injection (AUROC 0.50).

Every result is produced by `experiments/run_all.py` on CPU with no API keys.

> **Scope & honesty.** The *input data* is real and public. Two things are
> constructed and disclosed, not hidden:
> (a) the multi-vehicle **control is a replay-simulated plant** — followers are
> simulated tracking each string's *real recorded leader trajectory* (standard
> platoon-evaluation practice; it is not a real closed-loop convoy); and
> (b) no single public dataset carries trajectory + CAN + energy from one vehicle,
> so the three real datasets are **combined by role** (a constructed cross-modal
> vehicle). V2V attacks are injected-labeled on real telemetry; CAN attacks are
> injected-labeled on real VED OBD streams (a real-labeled HCRL drop-in is
> supported but not required). See *Honest limitations*.

## Real data (keyless, reproducible, provenance-logged)

All sources are public and cached under `data/raw/`; provenance (URLs, counts,
sampling rates, split boundaries) is written to `data/manifest.json`.

| Role | Source | Detail |
|---|---|---|
| Trajectory / platoon | USDOT **NGSIM** I-80 (ITS DataHub Socrata `8ect-6jqj`) | real **10 Hz** trajectories with native `preceding`/`following` linkage and `space_headway` → genuine leader-follower strings |
| Powertrain / energy + in-vehicle bus | **VED** (Vehicle Energy Dataset, U. Michigan; `gsoh/VED`) | real on-road OBD: vehicle speed, engine RPM, MAF, absolute load, fuel rate, HV battery — calibrates the energy model and the onboard-consistency check |
| CAN intrusion (optional) | **HCRL Car-Hacking / OTIDS** (IEEE DataPort) | real labeled CAN-injection frames; drop CSVs into `data/raw/can/` to use them instead of injected labels |

NGSIM native units (feet) are converted to SI on ingest. Strings are
**vehicle-disjoint** across train/val/test and ordered chronologically; all
scalers, kinematic bounds and the detector threshold are fit on the training split
only.

## Methods

* **D-PCNC** (`src/physics/hard_projection_control.py`): a per-follower MLP policy
  (local state + one-hop predecessor telemetry) whose raw command is projected onto
  the intersection of three feasibility intervals by nested `clamp` — an exact,
  differentiable, sub-millisecond analytic projection (no QP solver). Safety is
  prioritised when the intervals conflict.
* **Cross-Layer IDS** (`src/security/cross_layer_ids.py`): four physical-consistency
  residuals (ego kinematics, ego powertrain/load, predecessor-vs-gap, predecessor
  self-consistency), max-scored against a clean-train threshold, with a
  physics-propagated fallback (`sanitize_telemetry`).
* **Baselines** (`src/models/baselines.py`): IDM/CACC (analytic), decentralized
  receding-horizon **MPC**, **Unconstrained-NN** (D-PCNC policy without the
  projection — ablation), **Soft-PINN** (soft safety penalty instead of hard
  projection — ablation).

## Results

<!-- RESULTS_START -->
All numbers below are from an actual run (`results/metrics.json`) on the real
19-string, vehicle-disjoint **test** split. They are reported as measured,
including the results that do **not** favor the proposed method.

### What works

**Hard projection nearly eliminates collisions (the main control result).** On the
identical policy architecture, removing the projection is catastrophic; adding it
(D-PCNC) or even a soft penalty (Soft-PINN) collapses collisions by ~99%:

| Controller (clean, μ=0.9) | Collisions | Headway-violation % | Latency (ms/step) |
|---|--:|--:|--:|
| Unconstrained-NN (no projection) | **588** | 43.7 | 0.11 |
| Soft-PINN (soft penalty) | 7 | 20.1 | 0.11 |
| IDM/CACC | 6 | 36.7 | 0.007 |
| MPC | 6 | 44.4 | 0.18 |
| **D-PCNC (hard projection)** | **4** | **10.2** | 0.23 |

All controllers run at **sub-millisecond** edge latency, projection included.

**The cross-layer IDS detects kinematic/speed spoofs immediately** at a ~6.5%
false-positive operating point (threshold fit on clean train only):

| Attack | AUROC | F1 | Detection latency |
|---|--:|--:|--:|
| CAN speed injection | **1.000** | 0.97 | **0 ms** |
| V2V bias spoof | **0.938** | 0.97 | **0 ms** |
| V2V replay | 0.665 | 0.11 | 1033 ms |
| CAN load injection | 0.500 | 0.11 | — |

### What does not (reported honestly)

* **D-PCNC is not a good tracker.** Its hard projection + behavior-cloned policy is
  over-conservative: spacing RMSE **20.6 m** and speed RMSE 2.87 m/s are the *worst*
  of all methods (MPC 1.84 m / IDM 2.46 m). It buys safety by falling far behind.
  This is a diagnostic result, not a win: it motivates closed-loop (non-cloning)
  training and a less conservative one-step bound.
* **The per-step safety guarantee does not fully transfer to closed loop.** The
  projection provably satisfies the spacing bound under the *assumed* one-hop
  predecessor motion (see `tests/`), but with simultaneous multi-vehicle updates
  and a real leader braking harder than reported, D-PCNC still records 4 residual
  collision-steps — a ~99% reduction vs the unconstrained NN, not elimination.
* **The IDS fails on CAN load/torque injection (AUROC 0.50)** and is weak on replay
  (0.665): consistency checks catch signals that break kinematics, not a plausibly
  self-consistent load bias. A learned powertrain model is needed there.
* **The telemetry guard does not reduce collisions** under V2V spoofing (4 guarded
  vs 4 unguarded): D-PCNC's projection already uses the *measured* gap for safety,
  so spoofed predecessor speed cannot cause a collision it would not otherwise have.
  The guard's value here is detection/situational awareness, not collision
  prevention.
* **Energy barely discriminates controllers** (0.10–0.17 kWh/km): they all track the
  same real leader. The energy figure is a physics estimate (literature
  coefficients); a VED magnitude check gives real on-road power 6.5 kW vs a 2.8 kW
  modeled tractive mean (same order; VED also includes idle/accessory/driveline).

Full numbers: `results/table{1,2,3}_*.csv`, `results/metrics.json`;
figures in `results/figures/`.
<!-- RESULTS_END -->

## Quickstart

```bash
pip install -r requirements.txt
python experiments/run_all.py     # ingest (cached) + train + Tables 1-3 + Figures 1-3
pytest                            # unit tests
```

Runs on CPU, no keys. Seeds pinned across Python/NumPy/PyTorch;
`results/environment.json` records versions.

## Layout

```
configs/{data,model,experiment}.yaml   real-data + physics/controller + attack configs
data/{raw,processed,manifest.json}      cached pulls + provenance
src/data/ingest.py                      NGSIM (Socrata) + VED (7z) ingestion, SI conversion
src/data/platoon_graph.py               real leader-follower strings, energy model, leak-free splits
src/physics/hard_projection_control.py  D-PCNC + differentiable hard projection (contribution 1)
src/physics/convoy_sim.py               shared closed-loop simulator + metrics
src/physics/residuals.py                kinematic/spacing/string-stability residuals
src/security/cross_layer_ids.py         V2V+CAN cross-layer IDS + fallback (contribution 2)
src/security/channel.py                 contested comms (drop/latency/spoof), eval-only
src/models/baselines.py                 IDM/CACC, MPC, Unconstrained-NN, Soft-PINN
src/eval/{metrics,tables,figures}.py    metrics, CSV/JSON, 300-DPI Times New Roman figures
experiments/run_all.py                  end-to-end: Tables 1-3 + Figures 1-3
tests/                                   projection feasibility, string stability, splits, IDS
```

## Honest limitations / future work

* **Replay-simulated plant.** Followers are simulated tracking real leader
  trajectories; this is not a real closed-loop convoy with real actuation. Results
  should be confirmed in higher-fidelity simulation and on hardware.
* **Constructed cross-modal vehicle.** Trajectory (NGSIM), energy/OBD (VED) and CAN
  come from different vehicles/datasets combined by role. The onboard-consistency
  model is *fit on real VED OBD data*; the energy model uses *literature physics
  coefficients* (an instantaneous VED power fit did not converge, R²≈0 at 1 Hz) and
  is only magnitude-checked against VED. Both are applied to NGSIM kinematics.
* **Injected V2V/CAN labels.** No public labeled V2V-attack set exists; V2V attacks
  (and, by default, CAN attacks) are injected with ground-truth labels. The real
  HCRL labeled-CAN drop-in strengthens the CAN evaluation when available.
* **Friction regimes** are a parameterized sensitivity sweep, not measured off-road
  friction.

## License

MIT. NGSIM (USDOT), VED (U. Michigan) and HCRL are public data sources, credited
above.
