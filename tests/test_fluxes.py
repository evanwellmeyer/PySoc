"""Phase 4 gate: fluxes and heating rates of the PyTorch model match the Fortran driver (float64, CPU)."""

import numpy as np
import pytest

from conftest import require_exe
from reference_runs import heating_rate, run_fortran, run_python, standard_cases
from tools.refcases import make_cases

CASES = {
    "standard": dict(n_profile=24, n_layer=40, seed=0),
    "coarse": dict(n_profile=16, n_layer=10, seed=1),
    "fine": dict(n_profile=8, n_layer=90, seed=2),
    "dry_and_cold": dict(n_profile=16, n_layer=40, seed=3, profiles=("saw", "sas"), co2_ppmv_range=(10, 30)),
    "hot_high_co2": dict(n_profile=16, n_layer=40, seed=4, profiles=("tro",), co2_ppmv_range=(3000, 5000)),
}


def case(name):
    kw = CASES[name]
    if name == "standard":
        return standard_cases()
    atm = make_cases(**kw)
    if name == "dry_and_cold":
        atm.h2o[:] *= 1e-3
        atm.t_layer[:] -= 25.0
        atm.t_level[:] -= 25.0
        atm.t_surf[:] -= 25.0
    if name == "hot_high_co2":
        atm.t_layer[:] += 20.0
        atm.t_level[:] += 20.0
        atm.t_surf[:] += 20.0
        atm.emissivity = 0.9
    return atm


def compare(region, atm, rtol, atol_flux, atol_hr):
    ref, _ = run_fortran(region, atm, dump=False)
    py = run_python(region, atm, return_intermediates=False)
    names = [("flux_up", "flux_up"), ("flux_down", "flux_down")]
    band_names = [("flux_up_band", "flux_up_clear_band"), ("flux_down_band", "flux_down_clear_band")]
    if region == "sw":
        names.append(("flux_direct", "flux_direct"))
        band_names.append(("flux_direct_band", "flux_direct_clear_band"))
    for ours, theirs in names:
        np.testing.assert_allclose(py[ours].numpy(), ref[theirs], rtol=rtol, atol=atol_flux, err_msg=ours)
        np.testing.assert_allclose(py[ours].numpy(), ref[theirs + "_clear"], rtol=rtol, atol=atol_flux)
    for ours, theirs in band_names:
        np.testing.assert_allclose(py[ours].numpy(), ref[theirs], rtol=rtol, atol=atol_flux, err_msg=ours)
    hr = heating_rate(py["flux_up"].numpy(), py["flux_down"].numpy(), atm.heat_capacity)
    # a flux tolerance translates into a heating-rate tolerance of 4*atol_flux/heat_capacity
    tol = np.maximum(atol_hr, 4 * atol_flux / atm.heat_capacity) + 1e-8 * np.abs(ref["heating_rate"])
    excess = np.abs(hr - ref["heating_rate"]) - tol
    assert np.all(excess <= 0), ("heating rate", np.unravel_index(excess.argmax(), excess.shape), excess.max())
    return ref, py


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("region", ["lw", "sw"])
def test_fluxes_match_fortran(region, name):
    require_exe("soc_ref")
    atm = case(name)
    # SW: 1e-9 W/m2 absolute, 1e-10 relative.  LW: the thin-layer source terms cancel
    # catastrophically, so a one-ulp change in exp() alone moves Fortran's own fluxes by ~1e-7 W/m2;
    # we require agreement 5x tighter than that (2e-8 W/m2).  Heating rates to 1e-12 K/s (~1e-7 K/day).
    atol_flux = 1e-9 if region == "sw" else 2e-8
    compare(region, atm, rtol=1e-10, atol_flux=atol_flux, atol_hr=1e-12)


def extreme_case(n_profile=16, n_layer=50, seed=7):
    """Pressures below the k-table range, temperatures outside it, grazing sun, albedo 0 and 1."""
    from tools.refcases import CP_AIR, GRAV, RDGAS, AtmInputs

    r = np.random.default_rng(seed)
    p_half = np.exp(np.linspace(np.log(1e-2), np.log(1.05e5), n_layer + 1))[None, :].repeat(n_profile, 0)
    p_half[:, 0] = 0.0
    p_full = np.sqrt(np.maximum(p_half[:, :-1], 1e-3) * p_half[:, 1:])
    t = r.uniform(100.0, 400.0, (n_profile, n_layer))
    t_level = np.concatenate([t[:, :1], 0.5 * (t[:, 1:] + t[:, :-1]), t[:, -1:]], 1)
    h2o = 10.0 ** r.uniform(-8, -1.5, (n_profile, n_layer))
    o3 = 10.0 ** r.uniform(-9, -4.5, (n_profile, n_layer))
    co2 = np.full((n_profile, n_layer), 6e-4) * r.uniform(0.01, 10, (n_profile, 1))
    d_mass = (p_half[:, 1:] - p_half[:, :-1]) / GRAV
    coszen = np.concatenate([[1e-4, 1e-10, 0.0, -1.0], r.uniform(0.0, 1.0, n_profile - 4)])
    albedo = np.concatenate([[0.0, 1.0, 1.0, 0.0], r.uniform(0, 1, n_profile - 4)])
    zeros = np.zeros((n_profile, n_layer))
    return AtmInputs(p_layer=p_full, t_layer=t, t_level=t_level, d_mass=d_mass, density=p_full / (RDGAS * t),
                     h2o=h2o, o3=o3, co2=co2, t_surf=r.uniform(150, 350, n_profile), coszen=coszen,
                     solar_irrad=np.full(n_profile, 1361.0), albedo=albedo, emissivity=0.8,
                     heat_capacity=d_mass * CP_AIR, cld_frac=zeros, reff=zeros.copy(), mmr_cl=zeros.copy())


@pytest.mark.parametrize("region", ["lw", "sw"])
def test_extreme_inputs(region):
    require_exe("soc_ref")
    atm = extreme_case()
    # LW: with 100-400 K random profiles a one-ulp change in exp() moves the Fortran fluxes by up to
    # 8e-5 W/m2; require 1e-5 W/m2.
    ref, py = compare(region, atm, rtol=1e-10, atol_flux=1e-9 if region == "sw" else 1e-5, atol_hr=1e-10)
    for v in py.values():
        if hasattr(v, "numpy"):
            assert np.all(np.isfinite(v.numpy()))
