"""Phase 6 gate: IscaSocrates (Isca-level preprocessing + LW -> T update -> SW) matches the Fortran reference."""

import os
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from conftest import require_exe
from pysoc.isca import IscaSocrates, IscaSocratesConfig, interp_temp
from tools.refcases import HARNESS, SP_LW_GA7, SP_SW_GA7, _f, format_namelist, isca_interp_temp, make_cases, read_dump


def isca_fields(n_profile=24, n_layer=40, seed=0):
    atm = make_cases(n_profile=n_profile, n_layer=n_layer, seed=seed)
    ex = atm.extra
    return dict(temp=atm.t_layer, q=ex["q"], p_full=atm.p_layer, p_half=ex["p_half"], z_full=ex["z_full"],
                z_half=ex["z_half"], t_surf=atm.t_surf, albedo=atm.albedo, coszen=atm.coszen, ozone=atm.o3)


def run_fortran_isca(f, rrsun, delta_t, namelist=None, clouds=None):
    with tempfile.TemporaryDirectory() as td:
        fin, fout, fnml = (os.path.join(td, n) for n in ("in.bin", "out.bin", "nml"))
        n_profile, n_layer = f["temp"].shape
        with open(fin, "wb") as fh:
            fh.write(np.array([n_profile, n_layer], dtype="<i4").tobytes())
            for k in ("temp", "q", "p_full", "p_half", "z_full", "z_half", "t_surf", "albedo", "coszen", "ozone"):
                fh.write(_f(f[k]))
            fh.write(np.array([rrsun, delta_t], dtype="<f8").tobytes())
            fh.write(np.array([0 if clouds is None else 1], dtype="<i4").tobytes())
            if clouds is not None:
                for k in ("cf_rad", "reff_rad", "qcl_rad"):
                    fh.write(_f(clouds[k]))
        args = [str(HARNESS / "soc_isca"), str(SP_LW_GA7), str(SP_SW_GA7), fin, fout]
        if namelist:
            open(fnml, "w").write(format_namelist(namelist))
            args.append(fnml)
        subprocess.run(args, check=True, capture_output=True)
        return read_dump(fout)


@pytest.fixture(scope="module")
def model():
    return IscaSocrates(SP_LW_GA7, SP_SW_GA7)


def test_interp_temp_matches_numpy_transcription():
    f = isca_fields()
    ours = interp_temp(*(torch.tensor(f[k]) for k in ("z_full", "z_half", "temp"))).numpy()
    np.testing.assert_allclose(ours, isca_interp_temp(f["z_full"], f["z_half"], f["temp"]), rtol=1e-14)


@pytest.mark.parametrize("settings", [
    dict(rrsun=1.0, delta_t=0.0, nml=None),
    dict(rrsun=1.03, delta_t=1800.0, nml=dict(stellar_constant=1370.0, co2_ppmv=560.0, input_planet_emissivity=0.95)),
    dict(rrsun=0.97, delta_t=3600.0, nml=dict(account_for_effect_of_water=False, account_for_effect_of_ozone=False,
                                              co2_ppmv=280.0, input_co2_mmr=True)),
], ids=["defaults", "namelist", "dry_no_ozone"])
def test_isca_interface_matches_fortran(model, settings):
    require_exe("soc_isca")
    f = isca_fields()
    nml = settings["nml"] or {}
    ref = run_fortran_isca(f, settings["rrsun"], settings["delta_t"], settings["nml"])
    cfg = IscaSocratesConfig(**{k: v for k, v in nml.items()})
    model.config = cfg
    T = lambda a: torch.tensor(np.asarray(a))  # noqa: E731
    out = model(T(f["temp"]), T(f["q"]), T(f["p_full"]), T(f["p_half"]), T(f["z_full"]), T(f["z_half"]),
                T(f["t_surf"]), T(f["albedo"]), T(f["coszen"]), settings["delta_t"], rrsun=settings["rrsun"],
                ozone=T(f["ozone"]))
    heat_capacity = (f["p_half"][:, 1:] - f["p_half"][:, :-1]) / 9.80 * (287.04 / (2.0 / 7.0))
    np.testing.assert_allclose(out["co2"].numpy(), ref["co2"], rtol=1e-15)
    np.testing.assert_allclose(out["t_half"].numpy(), ref["t_half_lw"], rtol=1e-14)
    # Without water vapour and ozone the shortwave is nearly pure Rayleigh scattering (omega clipped to
    # 1 - 32*eps), where a one-ulp change in exp() moves the Fortran's own fluxes by ~1e-5 W/m2.
    sw_atol = 1e-6 if not nml.get("account_for_effect_of_water", True) else 1e-9
    for ours, theirs, atol in (("flux_lw_up", "flux_lw_up", 2e-8), ("flux_lw_down", "flux_lw_down", 2e-8),
                               ("flux_sw_up", "flux_sw_up", sw_atol), ("flux_sw_down", "flux_sw_down", sw_atol),
                               ("soc_spectral_olr", "spectral_olr", 2e-8)):
        np.testing.assert_allclose(out[ours].numpy(), ref[theirs], rtol=1e-10, atol=atol, err_msg=ours)
    for ours, theirs, atol_flux in (("tdt_lw", "tdt_lw", 2e-8), ("tdt_sw", "tdt_sw", sw_atol)):
        tol = 4 * atol_flux / heat_capacity + 1e-8 * np.abs(ref[theirs])
        assert np.all(np.abs(out[ours].numpy() - ref[theirs]) <= tol), ours
    model.config = IscaSocratesConfig()


def test_leading_dimensions_and_diagnostics(model):
    """(lon, lat, level) inputs keep their shape; diagnostics are consistent with the fluxes."""
    f = isca_fields(n_profile=12)
    shape = (3, 4)
    T = lambda a, n=None: torch.tensor(np.asarray(a)).reshape(shape + ((n,) if n else ()))  # noqa: E731
    L = f["temp"].shape[1]
    out = model(T(f["temp"], L), T(f["q"], L), T(f["p_full"], L), T(f["p_half"], L + 1), T(f["z_full"], L),
                T(f["z_half"], L + 1), T(f["t_surf"]), T(f["albedo"]), T(f["coszen"]), 900.0, ozone=T(f["ozone"], L))
    assert out["tdt_rad"].shape == shape + (L,)
    assert out["soc_olr"].shape == shape
    assert out["soc_spectral_olr"].shape == shape + (9,)
    torch.testing.assert_close(out["soc_olr"], out["flux_lw_up"][..., 0])
    torch.testing.assert_close(out["soc_spectral_olr"].sum(-1), out["soc_olr"])
    torch.testing.assert_close(out["soc_toa_sw"], out["flux_sw_down"][..., 0] - out["flux_sw_up"][..., 0])


def test_isca_interface_with_clouds_matches_fortran(model):
    require_exe("soc_isca")
    f = isca_fields(seed=4)
    r = np.random.default_rng(9)
    shape = f["temp"].shape
    cf = np.where(r.random(shape) < 0.4, r.uniform(0, 1, shape), 0.0)
    cf[:, :10] = 0.0
    clouds = dict(cf_rad=cf, reff_rad=1e-6 * r.uniform(8, 20, shape), qcl_rad=cf * 10.0 ** r.uniform(-6, -3.5, shape))
    ref = run_fortran_isca(f, 1.0, 1800.0, None, clouds)
    model.config = IscaSocratesConfig()
    T = lambda a: torch.tensor(np.asarray(a))  # noqa: E731
    out = model(T(f["temp"]), T(f["q"]), T(f["p_full"]), T(f["p_half"]), T(f["z_full"]), T(f["z_half"]),
                T(f["t_surf"]), T(f["albedo"]), T(f["coszen"]), 1800.0, ozone=T(f["ozone"]),
                cf_rad=T(clouds["cf_rad"]), reff_rad=T(clouds["reff_rad"]), qcl_rad=T(clouds["qcl_rad"]))
    heat_capacity = (f["p_half"][:, 1:] - f["p_half"][:, :-1]) / 9.80 * (287.04 / (2.0 / 7.0))
    for name, atol in (("flux_lw_up", 2e-8), ("flux_lw_down", 2e-8), ("flux_sw_up", 1e-9), ("flux_sw_down", 1e-9),
                       ("flux_lw_up_clear", 2e-8), ("flux_lw_down_clear", 2e-8), ("flux_sw_up_clear", 1e-9),
                       ("flux_sw_down_clear", 1e-9)):
        np.testing.assert_allclose(out[name].numpy(), ref[name], rtol=1e-10, atol=atol, err_msg=name)
    for ours, theirs, atol_flux in (("tdt_lw", "tdt_lw", 2e-8), ("tdt_sw", "tdt_sw", 1e-9)):
        tol = 4 * atol_flux / heat_capacity + 1e-8 * np.abs(ref[theirs])
        assert np.all(np.abs(out[ours].numpy() - ref[theirs]) <= tol), ours
    np.testing.assert_allclose(out["soc_tot_cloud_cover"].numpy(), ref["tot_cloud_cover_lw"], rtol=1e-12)
    assert np.abs(out["soc_olr"].numpy() - out["soc_olr_clr"].numpy()).max() > 1.0
