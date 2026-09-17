"""Phase 7 gate: liquid clouds (Isca simple cloud path) match the Fortran driver with do_clouds."""

import numpy as np
import pytest
import torch

from conftest import require_exe
from pysoc.clouds import LiquidCloud
from reference_runs import atmosphere, heating_rate, lw_model, sw_model
from tools.refcases import IP_INFRA_RED, IP_SOLAR, SP_LW_GA7, SP_SW_GA7, make_cases, run_reference


def cloudy_case(seed=0, n_profile=24, n_layer=40, overcast=False):
    atm = make_cases(n_profile=n_profile, n_layer=n_layer, seed=seed)
    r = np.random.default_rng(seed + 100)
    shape = atm.cld_frac.shape
    frac = np.where(r.random(shape) < 0.35, r.uniform(0.0, 1.0, shape), 0.0)
    frac[:, :8] = 0.0  # no cloud in the stratosphere
    frac[0] = 0.0  # a clear column
    frac[1, 20:30] = 1.0  # overcast deck
    frac[2, 25] = 5e-5  # below min_cloud_fraction
    frac[3, 10:35] = 0.999999
    if overcast:
        frac[:, 15:30] = 1.0
    mmr = frac * 10.0 ** r.uniform(-6, -3.3, shape)
    reff = r.uniform(1e-6, 6e-5, shape)  # extends beyond the valid range 1.5e-6..5e-5
    atm.cld_frac[:] = frac
    atm.mmr_cl[:] = mmr
    atm.reff[:] = reff
    return atm


def run_python_cloudy(region, atm, dtype=torch.float64, device="cpu", numerics="auto"):
    a = atmosphere(atm, dtype, device)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
    cloud = LiquidCloud(fraction=T(atm.cld_frac), mmr=T(atm.mmr_cl), reff=T(atm.reff))
    model = (lw_model() if region == "lw" else sw_model()).to(device=device, dtype=dtype)
    model.numerics = numerics
    try:
        if region == "lw":
            out = model(a, T(atm.t_surf), T(atm.emissivity), cloud=cloud)
        else:
            out = model(a, T(atm.coszen), T(atm.solar_irrad), T(atm.albedo), cloud=cloud)
    finally:
        model.numerics = "auto"
        model.cpu().double()
    return {k: v.detach().cpu().double().numpy() for k, v in out.items()}


@pytest.mark.parametrize("overcast", [False, True], ids=["broken", "overcast"])
@pytest.mark.parametrize("region", ["lw", "sw"])
def test_cloudy_fluxes_match_fortran(region, overcast):
    require_exe("soc_ref")
    atm = cloudy_case(overcast=overcast)
    sp, isolir = (SP_LW_GA7, IP_INFRA_RED) if region == "lw" else (SP_SW_GA7, IP_SOLAR)
    ref = run_reference(sp, isolir, atm, do_clouds=True)
    py = run_python_cloudy(region, atm)
    atol = 2e-8 if region == "lw" else 1e-9
    pairs = [("flux_up", "flux_up"), ("flux_down", "flux_down"), ("flux_up_clear", "flux_up_clear"),
             ("flux_down_clear", "flux_down_clear"), ("flux_up_clear_band", "flux_up_clear_band"),
             ("flux_down_clear_band", "flux_down_clear_band")]
    if region == "sw":
        pairs += [("flux_direct", "flux_direct"), ("flux_direct_clear", "flux_direct_clear")]
    for ours, theirs in pairs:
        np.testing.assert_allclose(py[ours], ref[theirs], rtol=1e-10, atol=atol, err_msg=ours)
    np.testing.assert_allclose(py["tot_cloud_cover"], ref["tot_cloud_cover"], rtol=1e-12, atol=1e-14)
    hr = py["flux_divergence"] / atm.heat_capacity
    tol = 4 * atol / atm.heat_capacity + 1e-8 * np.abs(ref["heating_rate"])
    assert np.all(np.abs(hr - ref["heating_rate"]) <= tol)
    # clouds must matter
    assert np.abs(py["flux_up"] - py["flux_up_clear"]).max() > 1.0


@pytest.mark.parametrize("region", ["lw", "sw"])
def test_cloudy_float32_tracks_float64(region):
    atm = cloudy_case(seed=3)
    ref = run_python_cloudy(region, atm, numerics="reference")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        out = run_python_cloudy(region, atm, dtype=torch.float32, device=dev)
        for k in ("flux_up", "flux_down"):
            assert np.all(np.isfinite(out[k]))
            assert np.abs(out[k] - ref[k]).max() < 1e-2, (dev, k)
        dhr = np.abs(out["flux_divergence"] - ref["flux_divergence"]) / atm.heat_capacity * 86400
        assert dhr.max() < 1e-3, dev


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("region", ["lw", "sw"])
def test_cloudy_gradients_finite_and_consistent(region, dtype):
    """Gradients w.r.t. cloud water and temperature are finite (overcast, clear and sub-threshold
    layers), and in float64 agree with central differences in log(cloud water)."""
    atm = cloudy_case(seed=7, n_profile=6, n_layer=40)
    model = (lw_model() if region == "lw" else sw_model()).to(dtype)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype)  # noqa: E731
    base = atmosphere(atm, dtype)
    weights = torch.randn(6, 41, dtype=dtype, generator=torch.Generator().manual_seed(1))

    def f(log_scale, dt):
        a = atmosphere(atm, dtype)
        a.t = base.t + dt
        cloud = LiquidCloud(fraction=T(atm.cld_frac), mmr=T(atm.mmr_cl) * torch.exp(log_scale), reff=T(atm.reff))
        if region == "lw":
            out = model(a, T(atm.t_surf), T(atm.emissivity), cloud=cloud)
        else:
            out = model(a, T(atm.coszen), T(atm.solar_irrad), T(atm.albedo), cloud=cloud)
        return (out["flux_up"] * weights).sum() + out["flux_divergence"].sum()

    try:
        x = torch.zeros(6, 40, dtype=dtype, requires_grad=True)
        dt = torch.zeros(6, 40, dtype=dtype, requires_grad=True)
        gx, gt = torch.autograd.grad(f(x, dt), (x, dt))
        assert torch.isfinite(gx).all() and torch.isfinite(gt).all()
        if dtype == torch.float64:
            idx = [(i, j) for i, j in zip(*np.nonzero(atm.cld_frac > 1e-4))][:12]
            for i, j in idx:
                h = 1e-4
                e = torch.zeros(6, 40, dtype=dtype)
                e[i, j] = h
                num = (f(e, torch.zeros_like(e)) - f(-e, torch.zeros_like(e))) / (2 * h)
                assert abs(gx[i, j] - num) <= 1e-5 * abs(num) + 1e-6, (i, j, gx[i, j].item(), num.item())
    finally:
        model.double()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cloudy_compiled_mps():
    import warnings

    atm = cloudy_case(seed=11, n_profile=48)
    for region in ("lw", "sw"):
        ref = run_python_cloudy(region, atm, numerics="reference")
        model = lw_model() if region == "lw" else sw_model()
        model.compile = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = run_python_cloudy(region, atm, dtype=torch.float32, device="mps")
        finally:
            model.compile = False
        for k in ("flux_up", "flux_down", "flux_up_clear"):
            assert np.abs(out[k] - ref[k]).max() < 1e-2, (region, k)
