"""Phase 5 gate: autograd gradients agree with finite differences (float64) and stay finite.

Inputs span many orders of magnitude (mixing ratios ~1e-9..1e-2), so gradients are
checked in well-scaled variables: temperature offsets [K] and log mixing ratios.
"""

import numpy as np
import pytest
import torch

from pysoc.core import Atmosphere
from pysoc.isca import ISCA_WELL_MIXED_DEFAULTS
from reference_runs import lw_model, sw_model
from tools.refcases import make_cases

D = torch.float64


def small_case():
    atm = make_cases(n_profile=2, n_layer=6, seed=5)
    atm.coszen[:] = [0.7, 0.3]
    return atm


def T(a):
    return torch.tensor(np.asarray(a), dtype=D)


def make_fn(region, atm, numerics):
    model = (lw_model() if region == "lw" else sw_model()).double()
    model.numerics = numerics
    gen = torch.Generator().manual_seed(0)
    w_up = torch.randn(2, 7, dtype=D, generator=gen)
    w_div = torch.randn(2, 6, dtype=D, generator=gen)
    fixed = {k: T(v) for k, v in ISCA_WELL_MIXED_DEFAULTS.items()}

    def f(dt, dt_level, log_h2o, log_co2, log_o3, dt_surf, surface):
        gas = dict(fixed)
        gas.update({1: T(atm.h2o) * torch.exp(log_h2o), 2: T(atm.co2) * torch.exp(log_co2),
                    3: T(atm.o3) * torch.exp(log_o3)})
        a = Atmosphere(p=T(atm.p_layer), t=T(atm.t_layer) + dt, t_level=T(atm.t_level) + dt_level,
                       d_mass=T(atm.d_mass), density=T(atm.density), gas_mmr=gas)
        if region == "lw":
            out = model(a, T(atm.t_surf) + dt_surf, surface)
        else:
            out = model(a, T(atm.coszen) + dt_surf * 1e-2, T(atm.solar_irrad), surface)
        return (out["flux_up"] * w_up).sum() + (out["flux_divergence"] * w_div).sum()

    x0 = (torch.zeros(2, 6, dtype=D), torch.zeros(2, 7, dtype=D), torch.zeros(2, 6, dtype=D),
          torch.zeros(2, 6, dtype=D), torch.zeros(2, 6, dtype=D), torch.zeros(2, dtype=D),
          T([0.95, 0.9]) if region == "lw" else T([0.2, 0.6]))
    return model, f, x0


@pytest.mark.parametrize("numerics", ["reference", "stable"])
@pytest.mark.parametrize("region", ["lw", "sw"])
def test_gradients_match_finite_differences(region, numerics):
    atm = small_case()
    model, f, x0 = make_fn(region, atm, numerics)
    try:
        inputs = tuple(x.clone().requires_grad_(True) for x in x0)
        grads = torch.autograd.grad(f(*inputs), inputs, allow_unused=True)
        for i, g in enumerate(grads):
            g = torch.zeros_like(x0[i]) if g is None else g
            num = torch.zeros_like(g)
            # The Fortran-exact expressions carry thin-layer round-off noise that dominates central
            # differences for small steps (error ~ 1/h); the stable formulation is ~1000x smoother.
            # Surface properties (input 6) need small steps: SOCRATES weights the minor-gas equivalent
            # extinction by |flux|, which has kinks where a differential flux changes sign.
            h = 1e-5 if (numerics == "stable" or i == 6) else 1e-3
            for j in range(g.numel()):
                args = [x.clone() for x in x0]
                args[i].view(-1)[j] += h
                fp = f(*args)
                args[i].view(-1)[j] -= 2 * h
                fm = f(*args)
                num.view(-1)[j] = (fp - fm) / (2 * h)
            err = (g - num).abs()
            scale = num.abs().max()
            assert torch.all(err <= 1e-5 * num.abs() + 1e-6 * scale + 1e-8), (i, err.max().item(), scale.item())
    finally:
        model.numerics = "auto"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("region", ["lw", "sw"])
def test_gradients_finite_at_edge_cases(region, dtype):
    """Night, grazing sun (coszen = 1e-6), zero humidity: gradients must stay finite."""
    atm = make_cases(n_profile=4, n_layer=10, seed=6)
    atm.coszen[:] = [-0.5, 0.0, 1e-6, 1.0]
    atm.h2o[0] = 0.0
    model = (lw_model() if region == "lw" else sw_model()).to(dtype)
    F = lambda a: torch.tensor(a, dtype=dtype)  # noqa: E731
    t = F(atm.t_layer).requires_grad_(True)
    h2o = F(atm.h2o).requires_grad_(True)
    gas = {k: F(v) for k, v in ISCA_WELL_MIXED_DEFAULTS.items()}
    gas.update({1: h2o, 2: F(atm.co2), 3: F(atm.o3)})
    a = Atmosphere(p=F(atm.p_layer), t=t, t_level=F(atm.t_level), d_mass=F(atm.d_mass), density=F(atm.density),
                   gas_mmr=gas)
    try:
        if region == "lw":
            out = model(a, F(atm.t_surf), 1.0)
        else:
            out = model(a, F(atm.coszen), F(atm.solar_irrad), F(atm.albedo))
        (out["flux_divergence"].sum() + out["flux_up"].sum()).backward()
    finally:
        model.double()
    assert torch.isfinite(t.grad).all() and torch.isfinite(h2o.grad).all()
