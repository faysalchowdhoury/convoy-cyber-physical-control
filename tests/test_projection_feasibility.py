"""The hard projection layer must guarantee the collision-free spacing bound at
every step, for arbitrary raw commands and states."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.physics.hard_projection_control import project_command, accel_ramp_bounds


def test_safety_bound_always_satisfied():
    rng = np.random.default_rng(0)
    dt, s_min, tau = 0.1, 2.0, 1.2
    for _ in range(2000):
        v = float(rng.uniform(0, 30))
        gap = float(rng.uniform(1.0, 60.0))
        v_lead = float(rng.uniform(0, 30))
        a_lead = float(rng.uniform(-5, 3))
        u_raw = torch.tensor(float(rng.uniform(-20, 20)))  # arbitrary, possibly unsafe
        du_lo, du_hi = accel_ramp_bounds(torch.tensor(0.0), torch.tensor(v), 1500.0, torch.tensor(60000.0))
        u = project_command(
            u_raw, torch.tensor(v), torch.tensor(gap), torch.tensor(v_lead),
            torch.tensor(a_lead), dt=dt, s_min=s_min, tau_safe=tau,
            a_min=torch.tensor(-5.0), a_max=torch.tensor(2.5), du_max=(du_lo, du_hi))
        u = float(u)
        # next-step predicted gap and safe gap
        gap_next = gap + (v_lead - v) * dt + 0.5 * (a_lead - u) * dt * dt
        v_next = v + u * dt
        safe = s_min + tau * v_next
        # guarantee holds to float32 numerical precision (sub-mm)
        assert gap_next >= safe - 1e-3, (gap_next, safe, u)


def test_projection_passes_feasible_commands():
    dt, s_min, tau = 0.1, 2.0, 1.2
    # large gap, slow follower: a mild command is feasible and must be returned ~unchanged
    u_raw = torch.tensor(0.5)
    du_lo, du_hi = accel_ramp_bounds(torch.tensor(0.5), torch.tensor(10.0), 1500.0, torch.tensor(60000.0))
    u = project_command(u_raw, torch.tensor(10.0), torch.tensor(50.0), torch.tensor(12.0),
                        torch.tensor(0.0), dt=dt, s_min=s_min, tau_safe=tau,
                        a_min=torch.tensor(-5.0), a_max=torch.tensor(2.5), du_max=(du_lo, du_hi))
    assert abs(float(u) - 0.5) < 1e-6


def test_projection_differentiable():
    dt, s_min, tau = 0.1, 2.0, 1.2
    u_raw = torch.tensor(1.0, requires_grad=True)
    du_lo, du_hi = accel_ramp_bounds(torch.tensor(0.0), torch.tensor(10.0), 1500.0, torch.tensor(60000.0))
    u = project_command(u_raw, torch.tensor(10.0), torch.tensor(40.0), torch.tensor(10.0),
                        torch.tensor(0.0), dt=dt, s_min=s_min, tau_safe=tau,
                        a_min=torch.tensor(-5.0), a_max=torch.tensor(2.5), du_max=(du_lo, du_hi))
    u.backward()
    assert u_raw.grad is not None
