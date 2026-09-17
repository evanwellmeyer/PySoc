"""Phase 5 gate: low precision / other devices track the float64 reference; dtype moves are lossless."""

import numpy as np
import pytest
import torch

from pysoc.core import SocratesSW
from pysoc.mathutil import expm1
from reference_runs import lw_model, run_python, standard_cases, sw_model
from tools.refcases import SP_SW_GA7, make_cases

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
          (["cuda"] if torch.cuda.is_available() else [])
SECONDS_PER_DAY = 86400.0


def _run(region, atm, dtype, device, numerics):
    model = lw_model() if region == "lw" else sw_model()
    old = model.numerics
    model.numerics = numerics
    try:
        out = run_python(region, atm, dtype=dtype, device=device, return_intermediates=False)
    finally:
        model.numerics = old
    return {k: v.detach().cpu().double().numpy() for k, v in out.items()}


@pytest.fixture(scope="module", params=["standard", "L60"])
def atm(request):
    return standard_cases() if request.param == "standard" else make_cases(n_profile=32, n_layer=60, seed=11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("region", ["lw", "sw"])
def test_float32_tracks_float64(atm, region, device):
    ref = _run(region, atm, torch.float64, "cpu", "reference")
    out = _run(region, atm, torch.float32, device, "auto")
    for k in ("flux_up", "flux_down"):
        assert np.all(np.isfinite(out[k]))
        assert np.abs(out[k] - ref[k]).max() < 1e-2, k  # W/m2
    dhr = np.abs(out["flux_divergence"] - ref["flux_divergence"]) / atm.heat_capacity * SECONDS_PER_DAY
    assert dhr.max() < 1e-3  # K/day


@pytest.mark.parametrize("region", ["lw", "sw"])
def test_stable_float64_matches_reference(atm, region):
    ref = _run(region, atm, torch.float64, "cpu", "reference")
    out = _run(region, atm, torch.float64, "cpu", "stable")
    for k in ("flux_up", "flux_down", "flux_divergence"):
        assert np.abs(out[k] - ref[k]).max() < 1e-7, k


def test_divergence_consistent_with_fluxes(atm):
    for region in ("lw", "sw"):
        out = _run(region, atm, torch.float64, "cpu", "reference")
        div = (out["flux_down"][:, :-1] - out["flux_down"][:, 1:]) + (out["flux_up"][:, 1:] - out["flux_up"][:, :-1])
        np.testing.assert_allclose(out["flux_divergence"], div, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expm1_accuracy(device, dtype):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS has no float64")
    x = torch.cat([-torch.logspace(-12, 2, 200, dtype=torch.float64), torch.logspace(-12, 1.5, 200, dtype=torch.float64)])
    got = expm1(x.to(device=device, dtype=dtype)).cpu().double()
    rel = ((got - torch.expm1(x)) / torch.expm1(x)).abs().max().item()
    assert rel < 50 * torch.finfo(dtype).eps


def test_dtype_round_trip_is_lossless():
    model = SocratesSW(SP_SW_GA7)
    before = {n: b.clone() for n, b in model.named_buffers()}
    model.to(torch.float32)
    if torch.backends.mps.is_available():
        model.to("mps")
        model.to("cpu")
    model.to(torch.float64)
    for n, b in model.named_buffers():
        assert b.dtype == before[n].dtype and torch.equal(b, before[n]), n


@pytest.mark.parametrize("device", DEVICES)
def test_stable_solar_kernel_near_singularity(device):
    """lambda at, and exactly delta-relative from, sec_0: finite and close to the float64 reference."""
    import pysoc.twostream as ts

    r = np.random.default_rng(1)
    n = 50000
    om = torch.tensor(np.minimum(r.uniform(0, 1, n) ** 0.3, 1 - 1e-6))
    g = torch.tensor(r.uniform(0, 0.9, n))
    tau = torch.tensor(10 ** r.uniform(-8, 2, n))
    s, d = ts.two_coeff_basic(om, g, ts.IP_PIFM80)
    lam = torch.sqrt(s * d)
    om, g, tau, lam = om[lam > 1], g[lam > 1], tau[lam > 1], lam[lam > 1]
    k = lam.numel()
    delta = float(np.cbrt(np.finfo(np.float32).eps))
    choice = torch.as_tensor(r.integers(0, 4, k))
    sec = torch.stack([lam, lam / (1 + delta), lam / (1 - delta), lam * (1 + 1e-5)])[choice, torch.arange(k)]
    # At lambda == sec_0 exactly the Fortran expressions return round-off garbage (e.g. -1/32), so the
    # reference is the float64 stable kernel; it is itself checked against the Fortran expressions
    # wherever those are well conditioned.
    fortran = ts.two_coeff_solar(om[:, None], g[:, None], tau[:, None], sec, ts.IP_PIFM80)
    ref = ts.two_coeff_solar_stable(om[:, None], g[:, None], tau[:, None], sec, ts.IP_PIFM80)
    well = (torch.abs(lam / sec - 1) > 1e-3)[:, None]
    for a, b in zip(ref, fortran):
        assert (a - b)[well].abs().max() < 1e-9
    out = ts.two_coeff_solar_stable(om[:, None].float().to(device), g[:, None].float().to(device),
                                    tau[:, None].float().to(device), sec.float().to(device), ts.IP_PIFM80)
    for a, b in zip(out, ref):
        a = a.cpu().double()
        assert torch.isfinite(a).all()
        assert (a - b).abs().max() < 1e-4


def test_large_batch_finite():
    """Regression: an 8192-column batch once produced NaN in two float32 SW columns."""
    atm = make_cases(n_profile=8192, n_layer=40, seed=0)
    for region in ("lw", "sw"):
        out = _run(region, atm, torch.float32, "cpu", "auto")
        for key in ("flux_up", "flux_down", "flux_divergence"):
            assert np.all(np.isfinite(out[key])), (region, key)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_compiled_mps_matches_reference():
    """compile=True on MPS (self-validating kernels) stays within the float32 error budget."""
    import warnings

    atm = make_cases(n_profile=64, n_layer=40, seed=3)
    for region in ("lw", "sw"):
        model = lw_model() if region == "lw" else sw_model()
        ref = _run(region, atm, torch.float64, "cpu", "reference")
        model.compile = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = _run(region, atm, torch.float32, "mps", "auto")
        finally:
            model.compile = False
        for k in ("flux_up", "flux_down"):
            assert np.abs(out[k] - ref[k]).max() < 1e-2, (region, k)
