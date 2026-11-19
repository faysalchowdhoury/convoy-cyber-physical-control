# Convoy Cyber-Physical Control

**Physics-Constrained Decentralized Multi-Vehicle Control and In-Vehicle/V2X
Cyber-Resilience for Autonomous Ground Convoys**

A reproducible research codebase for **layered cyber-physical safety** of autonomous
ground convoys under contested communications and attack. The thesis: *detect the
attacks you can, and guarantee safety against the ones you cannot.* Two mechanisms,
evaluated against classical and neural baselines on **real public vehicle data**:

1. **D-PCNC** — a Decentralized Physics-Constrained Neural Controller with a
   *differentiable hard projection layer* whose safety bound uses only **trusted
   onboard sensing** (measured gap and relative velocity), never the reported V2V
   predecessor speed. A learned policy, trained end-to-end through the
   differentiable projected rollout (`fit_closed_loop`), proposes commands; the
   projection makes them safe. **Value: robustness to V2V spoofing** — under a
   +4 m/s spoof it roughly halves headway violations (76% → 37%) vs a policy that
   trusts V2V. In *benign* conditions the projection *costs* ~2.5 m of tracking
   accuracy, which we report plainly rather than hide.
2. **Cross-Layer IDS** — a V2V + in-vehicle-CAN detector enforcing *physical
   consistency across layers*, with a CUSUM sequential stage and a measured-gap
   fallback. It detects aggressive speed/kinematic spoofs immediately (AUROC 1.00 /
   0.94) and degrades gracefully with attack magnitude, but — reported honestly — a
   slow-ramp stealthy attacker evades it and self-consistent CAN load injection is
   undetectable (AUROC 0.50).

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
  differentiable, sub-millisecond analytic projection (no QP solver). The **safety
  bound uses only measured gap and relative velocity** (onboard radar), never the
  reported V2V speed, which is what makes it spoof-robust. The policy is trained
  **end-to-end through the differentiable projected rollout** (`fit_closed_loop`,
  truncated BPTT), not by one-step behaviour cloning.
* **Cross-Layer IDS** (`src/security/cross_layer_ids.py`): four physical-consistency
  residuals (ego kinematics, ego powertrain/load, predecessor-vs-gap, predecessor
  self-consistency), combined with a **sequential CUSUM** stage, scored against a
  clean-train window-max threshold, with a measured-gap fallback
  (`sanitize_telemetry`).
* **Baselines** (`src/models/baselines.py`): IDM/CACC (analytic), decentralized
  receding-horizon **MPC**, **Unconstrained-NN** (same policy + same closed-loop
  training, no projection — isolates the projection), **Soft-PINN** (soft safety
  penalty instead of hard projection). All neural variants share training, so the
  ablation holds everything but the feasibility treatment constant.

## Results

<!-- RESULTS_START -->
All numbers are from an actual run (`results/metrics.json`) on the real 19-string,
vehicle-disjoint **test** split, with closed-loop training and bootstrap 95% CIs.
Reported as measured — the thesis is **layered cyber-physical safety**, and the
numbers support that thesis, not an unconditional "our controller is best" claim.

### 1. The projection's value is attack-robust safety, not nominal tracking

Under a **+4 m/s V2V speed spoof**, a nominally-trained policy that trusts the V2V
feed is badly fooled; the radar-based projection (which uses the *measured* gap and
relative velocity, never the reported predecessor speed) roughly halves headway
violations:

| Under V2V spoof (test) | Headway-violation % | Collisions |
|---|--:|--:|
| No projection (trusts V2V) | 76.2 | 6 |
| **D-PCNC (radar projection)** | **36.8** | **4** |
| D-PCNC + telemetry guard | 36.6 | 4 |

This is the paper's core result: **the hard projection buys cyber-physical
robustness** — a ~52% violation reduction under spoofing — because its safety
guarantee depends only on trusted onboard sensing.

### 2. In benign conditions the projection *costs* performance (reported plainly)

With all neural controllers trained identically in closed loop, the projection is
**not** a nominal-performance win — it trades tracking for its safety guarantee:

| Controller (clean, μ=0.9) | Spacing RMSE (m) | Headway-viol % | Collisions | Latency (ms) |
|---|--:|--:|--:|--:|
| Unconstrained-NN (no projection) | **1.77** | 42.1 | 6 | 0.11 |
| MPC | 1.84 | 44.4 | 6 | 0.20 |
| Soft-PINN | 1.81 | 24.3 | 6 | 0.12 |
| IDM/CACC | 2.46 | 36.7 | 6 | 0.009 |
| **D-PCNC (hard projection)** | 4.55 [1.3, 10.1] | 31.2 | 4 | 0.23 |

Once the policy is trained properly in closed loop, a plain NN already tracks best
and is as safe as the baselines in benign conditions; D-PCNC's projection adds
~2.5 m of spacing error and the most energy. All controllers run **sub-millisecond**.
(Behaviour cloning alone gave D-PCNC a 20.6 m error; closed-loop training cut it to
4.55 m — see `fit_closed_loop`.)

### 3. Cross-layer detection: strong on aggressive attacks, graceful decay, honest failures

At a ~6.5% false-positive operating point (threshold fit on clean train only),
AUROC with bootstrap 95% CI:

| Attack | AUROC [95% CI] | F1 | Detection latency |
|---|--:|--:|--:|
| CAN speed injection | **1.00** [1.00, 1.00] | 0.97 | **0 ms** |
| V2V bias spoof | **0.94** [0.88, 0.99] | 0.97 | **0 ms** |
| V2V replay | 0.67 [0.57, 0.76] | 0.11 | 1033 ms |
| CAN load injection | 0.50 [0.40, 0.60] | 0.11 | — |

The **adaptive-attacker sweep** (`fig4`, `table3b`) shows detection AUROC climbing
0.50 → 1.00 with step-spoof magnitude, but a **slow-ramp stealthy attacker largely
evades it** (AUROC ≤ 0.86, recall at chance even at 8 m/s). Two honest failures:
CAN **load injection is undetectable here** (CI includes 0.50) — a direct
consequence of the constructed cross-modal vehicle (load and kinematics come from
different vehicles, so there is no real consistency to break); and the detector
is a magnitude/stealth-limited device, not a universal one.

### 4. The telemetry guard adds little on top of the projection

Projection+guard (36.6%, 4 collisions) ≈ projection alone (36.8%, 4): once the
safety filter uses the measured gap, detecting the spoof does not further prevent
collisions. The guard's value is **situational awareness**, not safety.

Full numbers: `results/table{1,2,3,3b}_*.csv`, `results/metrics.json`;
figures in `results/figures/`.
<!-- RESULTS_END -->

## Quickstart

```bash
pip install -r requirements.txt
python experiments/run_all.py     # ingest (cached) + closed-loop train + Tables 1-3b + Figures 1-4
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
experiments/run_all.py                  end-to-end: Tables 1-3b + Figures 1-4 + bootstrap CIs
tests/                                   projection feasibility, string stability, splits, IDS
```

## Honest limitations / future work

* **The projection costs nominal performance.** It earns its keep only under attack
  (contribution 1); in benign operation a plain closed-loop-trained NN tracks better
  and is equally safe. A tighter / predictive safety filter that closes the benign
  gap is the natural next step.
* **Replay-simulated plant.** Followers are simulated tracking real leader
  trajectories; this is not a real closed-loop convoy with real actuation. Results
  should be confirmed in higher-fidelity simulation and on hardware.
* **Constructed cross-modal vehicle.** Trajectory (NGSIM), energy/OBD (VED) and CAN
  come from different vehicles/datasets combined by role. The onboard-consistency
  model is *fit on real VED OBD data*; the energy model uses *literature physics
  coefficients* (an instantaneous VED power fit did not converge, R²≈0 at 1 Hz) and
  is only magnitude-checked against VED. This pairing is *why* CAN load injection is
  undetectable here — a real single-platform dataset is needed to test it properly.
* **Detection is magnitude- and stealth-limited.** Aggressive step spoofs are caught
  at AUROC→1.0; a slow-ramp adaptive attacker evades the detector, and replay is
  weak. Injected (not real) V2V/CAN labels are used — no public labeled V2V-attack
  set exists; the optional HCRL drop-in supplies real CAN labels when available.
* **Friction regimes** are a parameterized sensitivity sweep, not measured off-road
  friction.

## License

MIT. NGSIM (USDOT), VED (U. Michigan) and HCRL are public data sources, credited
above.
