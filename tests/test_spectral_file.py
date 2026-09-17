"""Phase 1 gate: the Python spectral-file reader matches SOCRATES read_spectrum."""

import numpy as np
import pytest

from conftest import require_exe
from pysoc.spectral_file import IP_SCALE_LOOKUP, N_SCALE_VARIABLE, read_spectral_file
from tools.refcases import SP_LW_GA7, SP_SW_GA7, dump_spectrum


@pytest.fixture(scope="module", params=[SP_LW_GA7, SP_SW_GA7], ids=["lw_ga7", "sw_ga7"])
def pair(request):
    require_exe("dump_spectrum")
    return read_spectral_file(request.param), dump_spectrum(request.param)


def eq(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def close(a, b, rtol=0.0):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=0)


def test_basic(pair):
    sp, f = pair
    assert sp.n_band == f["n_band"][0]
    assert sp.n_absorb == f["n_absorb"][0]
    present = {i for i, v in enumerate(f["l_present"]) if v}
    assert present == sp.blocks_present
    eq(sp.type_absorb, f["type_absorb"][: sp.n_absorb])
    close(sp.wavelength_short, f["wavelength_short"])
    close(sp.wavelength_long, f["wavelength_long"])
    eq(sp.n_band_exclude, f["n_band_exclude"])
    for b in range(sp.n_band):
        eq(np.array(sp.index_exclude[b]) + 1, f["index_exclude"][: sp.n_band_exclude[b], b])
    if "solar_flux_band" in f:
        close(sp.solar_flux_band, f["solar_flux_band"])
    if "rayleigh_coeff" in f:
        assert sp.i_rayleigh_scheme == f["i_rayleigh_scheme"][0]
        close(sp.rayleigh_coeff, f["rayleigh_coeff"])


def test_gas(pair):
    sp, f = pair
    eq(sp.n_band_absorb, f["n_band_absorb"])
    eq(sp.i_overlap, f["i_overlap"])
    eq(sp.index_sb, f["index_sb"][: sp.n_absorb])
    n_checked = 0
    for b in range(sp.n_band):
        idx = f["index_absorb"][: sp.n_band_absorb[b], b] - 1
        eq(sp.index_absorb[b], idx)
        for g in idx:
            gt = sp.gas[(b, g)]
            assert gt.n_k == f["i_band_k"][b, g]
            assert gt.i_scale_k == f["i_scale_k"][b, g]
            assert gt.i_scale_fnc == f["i_scale_fnc"][b, g]
            close(gt.k, f["k"][: gt.n_k, b, g])
            close(gt.w, f["w"][: gt.n_k, b, g])
            eq(gt.i_scat, f["i_scat"][: gt.n_k, b, g])
            nsv = N_SCALE_VARIABLE[gt.i_scale_fnc]
            close(gt.scale, f["scale"][:nsv, : gt.n_k, b, g].T)
            if gt.i_scale_fnc == IP_SCALE_LOOKUP:
                assert gt.num_ref_p == f["num_ref_p"][g, b]
                assert gt.num_ref_t == f["num_ref_t"][g, b]
                close(gt.k_lookup, np.transpose(f["k_lookup"][:, :, : gt.n_k, g, b], (2, 1, 0)))
            else:
                close(gt.p_ref, f["p_ref"][g, b])
                close(gt.t_ref, f["t_ref"][g, b])
            n_checked += 1
    assert n_checked == sum(sp.n_band_absorb)
    if "p_lookup" in f:
        close(sp.p_lookup, f["p_lookup"], rtol=1e-15)
        close(sp.t_lookup, f["t_lookup"].T)


def test_planck(pair):
    sp, f = pair
    if "thermal_coeff" not in f:
        assert 6 not in sp.blocks_present
        return
    assert sp.n_deg_fit == f["n_deg_fit"][0]
    close(sp.t_ref_planck, f["t_ref_planck"][0])
    close(sp.thermal_coeff, f["thermal_coeff"].T)


def test_continuum(pair):
    sp, f = pair
    eq(sp.n_band_continuum, f["n_band_continuum"])
    for b in range(sp.n_band):
        n = sp.n_band_continuum[b]
        eq(sp.index_continuum[b], f["index_continuum"][b, :n])
        for c in sp.index_continuum[b]:
            d = sp.continuum[(b, c)]
            nsv = N_SCALE_VARIABLE[d["i_scale_fnc"]]
            assert d["i_scale_fnc"] == f["i_scale_fnc_cont"][b, c - 1]
            close(d["k"], f["k_cont"][b, c - 1])
            close(d["scale"], f["scale_cont"][:nsv, b, c - 1])
            close(d["p_ref"], f["p_ref_cont"][c - 1, b])
            close(d["t_ref"], f["t_ref_cont"][c - 1, b])
    types = list(sp.type_absorb)
    assert sp.index_water == types.index(1)


@pytest.mark.parametrize("kind", ["drop", "ice"])
def test_cloud_params(pair, kind):
    sp, f = pair
    store = getattr(sp, kind)
    l_type = f[f"{kind}_l_type"]
    assert {i + 1 for i, v in enumerate(l_type) if v} == set(store)
    for t, cp in store.items():
        assert cp.i_parm == f[f"{kind}_i_parm"][t - 1]
        assert cp.n_phf == f[f"{kind}_n_phf"][t - 1]
        close(cp.min_dim, f[f"{kind}_parm_min_dim"][t - 1])
        close(cp.max_dim, f[f"{kind}_parm_max_dim"][t - 1])
        npar = cp.parm_list.shape[1]
        close(cp.parm_list, f[f"{kind}_parm_list"][:npar, :, t - 1].T)
