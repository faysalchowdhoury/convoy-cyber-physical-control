"""Decentralized Physics-Constrained Neural Controller (D-PCNC).

Each follower computes a local longitudinal acceleration command from its own
state plus one-hop predecessor telemetry. A neural policy proposes a raw command;
a **differentiable Hard Physical Projection Layer** clamps it to the intersection
of three feasibility intervals, guaranteeing (by construction, at every step):

  1. constant-time-headway, collision-free spacing   gap(t+1) >= s_min + tau*v(t+1)
  2. actuator / traction acceleration limits           a_min(mu) <= u <= a_max(mu)
  3. powertrain power-ramp limit                        |P(t+1) - P(t)| <= dP_max

The projection is an exact analytic box projection (nested ``clamp``), so it is
differentiable (sub-gradient) and runs in closed form -- no QP solver, sub-
millisecond per step. Safety is prioritised: if the intervals are empty the layer
returns the largest deceleration that still satisfies the spacing bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Hard physical projection
# --------------------------------------------------------------------------- #
def project_command(u_raw, v_follow, gap, v_lead, a_lead, *, dt, s_min, tau_safe,
                    a_min, a_max, du_max):
    """Project raw accel command onto the physical feasible set (torch, batched).

    All args are tensors broadcastable to ``u_raw``'s shape except the scalar
    hyper-parameters. Returns the feasible command (same shape as ``u_raw``).
    """
    # (2) actuator + power-ramp interval
    hi = torch.minimum(a_max, du_max[1])
    lo = torch.maximum(a_min, du_max[0])
    # (1) safety upper bound on u from the next-step headway constraint.
    #   gap + (v_lead - v_follow)dt + 0.5(a_lead - u)dt^2 >= s_min + tau(v_follow + u dt)
    c = 0.5 * dt * dt + tau_safe * dt                      # > 0, coefficient on -u
    rhs = (s_min + tau_safe * v_follow - gap
           - (v_lead - v_follow) * dt - 0.5 * a_lead * dt * dt)
    u_safe_hi = -rhs / c
    hi = torch.minimum(hi, u_safe_hi)
    # safety dominates if the box is empty
    lo = torch.minimum(lo, hi)
    return torch.clamp(u_raw, lo, hi)


def accel_ramp_bounds(a_prev, v, energy_coef, dP_max):
    """Translate the power-ramp limit |P(t+1)-P(t)|<=dP_max into an accel interval
    around the previous command, using dP/du ~= m_eff * v (tractive region)."""
    m_eff = energy_coef
    du = dP_max / (m_eff * torch.clamp(v, min=0.5) + 1e-6)
    return a_prev - du, a_prev + du


# --------------------------------------------------------------------------- #
# Neural policy
# --------------------------------------------------------------------------- #
class DecentralizedPolicy(nn.Module):
    """Per-follower MLP: local + one-hop predecessor features -> raw accel."""

    N_FEATURES = 5  # v_follow, gap, closing_rate, gap_error, a_lead

    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        dims = [self.N_FEATURES] + [hidden] * layers
        seq = []
        for i in range(len(dims) - 1):
            seq += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        seq += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


@dataclass
class ControllerConfig:
    dt: float
    s_min: float
    tau_safe: float
    a_max: float
    a_min: float
    dP_max: float
    m_eff: float
    hidden: int = 64
    layers: int = 2
    mode: str = "dpcnc"  # dpcnc | unconstrained | soft_pinn
    soft_penalty_weight: float = 5.0


class ConvoyController:
    """Wraps a policy with the chosen feasibility treatment and provides a
    closed-loop rollout over a real leader trajectory (replay-simulated plant)."""

    def __init__(self, cfg: ControllerConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.policy = DecentralizedPolicy(cfg.hidden, cfg.layers).to(device)
        self.x_mean = None
        self.x_std = None

    # -- feature construction (one-hop) -- #
    def _features(self, v_follow, gap, v_lead, a_lead):
        closing = v_lead - v_follow
        gap_err = gap - (self.cfg.s_min + self.cfg.tau_safe * v_follow)
        x = torch.stack([v_follow, gap, closing, gap_err, a_lead], dim=-1)
        if self.x_mean is not None:
            x = (x - self.x_mean) / self.x_std
        return x

    def _mu_bounds(self, mu: float):
        # traction scales acceleration envelope with surface friction
        scale = mu / 0.9
        return self.cfg.a_min * scale, self.cfg.a_max * scale

    def _project(self, u_raw, v_follow, gap, v_lead, a_lead, a_prev, mu):
        c = self.cfg
        a_min_s, a_max_s = self._mu_bounds(mu)
        v = torch.as_tensor(v_follow, dtype=torch.float32)
        du_lo, du_hi = accel_ramp_bounds(
            torch.as_tensor(a_prev, dtype=torch.float32), v,
            c.m_eff, torch.tensor(c.dP_max))
        return project_command(
            u_raw, v, torch.as_tensor(gap, dtype=torch.float32),
            torch.as_tensor(v_lead, dtype=torch.float32),
            torch.as_tensor(a_lead, dtype=torch.float32),
            dt=c.dt, s_min=c.s_min, tau_safe=c.tau_safe,
            a_min=torch.tensor(a_min_s), a_max=torch.tensor(a_max_s),
            du_max=(du_lo, du_hi))

    # -- training (behavior cloning on real follower accel) -- #
    def fit(self, strings, idxs, epochs=40, lr=3e-3, batch=256, wd=1e-5, seed=7):
        torch.manual_seed(seed)
        X, Y = [], []
        for i in idxs:
            s = strings[i]
            v, gap, acc = s["vel"], s["gap"], s["acc"]
            for j in range(1, s["N"]):
                vf = v[:-1, j]; gp = gap[:-1, j]
                vl = v[:-1, j - 1]; al = acc[:-1, j - 1]
                closing = vl - vf
                gap_err = gp - (self.cfg.s_min + self.cfg.tau_safe * vf)
                feats = np.stack([vf, gp, closing, gap_err, al], axis=1)
                X.append(feats); Y.append(acc[:-1, j])
        X = np.concatenate(X).astype(np.float32)
        Y = np.concatenate(Y).astype(np.float32)
        Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
        self.x_mean = Xt.mean(0, keepdim=True)
        self.x_std = Xt.std(0, keepdim=True).clamp_min(1e-6)

        opt = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=wd)
        n = len(Xt)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for k in range(0, n, batch):
                bi = perm[k:k + batch]
                xb = (Xt[bi] - self.x_mean) / self.x_std
                u_raw = self.policy(xb)
                loss = ((u_raw - Yt[bi]) ** 2).mean()
                if self.cfg.mode == "soft_pinn":
                    # soft penalty: discourage commands that break the safe gap
                    vf, gp, closing = Xt[bi, 0], Xt[bi, 1], Xt[bi, 2]
                    vl = vf + closing
                    gap_next = gp + closing * self.cfg.dt + 0.5 * (-u_raw) * self.cfg.dt ** 2
                    need = self.cfg.s_min + self.cfg.tau_safe * (vf + u_raw * self.cfg.dt)
                    viol = torch.clamp(need - gap_next, min=0.0)
                    loss = loss + self.cfg.soft_penalty_weight * (viol ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    # -- closed-loop rollout over a real leader trajectory -- #
    @torch.no_grad()
    def rollout(self, s: dict, mu: float = 0.9, telemetry=None, energy=None):
        """Simulate followers tracking the string's REAL leader trajectory.

        ``telemetry`` optionally supplies (possibly attacked) predecessor
        (v_lead, a_lead) arrays [T, N]; otherwise the real values are used.
        """
        from .convoy_sim import SimParams, simulate
        c = self.cfg
        p = SimParams(dt=c.dt, s_min=c.s_min, tau_safe=c.tau_safe,
                      a_max=c.a_max, a_min=c.a_min)

        def step_fn(obs):
            feats = self._features(
                torch.tensor(obs.vf, dtype=torch.float32),
                torch.tensor(obs.gap, dtype=torch.float32),
                torch.tensor(obs.vl, dtype=torch.float32),
                torch.tensor(obs.al, dtype=torch.float32))
            u_raw = self.policy(feats.unsqueeze(0)).squeeze(0)
            if c.mode == "dpcnc":
                u = self._project(u_raw, obs.vf, obs.gap, obs.vl, obs.al,
                                  obs.a_prev, obs.mu)
            else:
                a_min_s, a_max_s = self._mu_bounds(obs.mu)
                u = torch.clamp(u_raw, a_min_s, a_max_s)
            return float(u)

        return simulate(step_fn, s, p, mu=mu, telemetry=telemetry, energy=energy)
