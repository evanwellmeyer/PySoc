"""Phase 3 gate: gas/grey optics, Planck terms and equivalent extinction match SOCRATES intermediates."""

import numpy as np
import pytest
import torch

from conftest import require_exe
from reference_runs import run_fortran, run_python, standard_cases

RTOL = 1e-12


@pytest.fixture(scope="module", params=["lw", "sw"])
def runs(request):
    require_exe("soc_ref_dump")
    atm = standard_cases()
    ref, dump = run_fortran(request.param, atm)
    py = run_python(request.param, atm)
    return request.param, atm, dump, py["intermediates"]


def npy(x):
    return x.detach().cpu().numpy()


def close(name, ours, ref, rtol=RTOL, atol=0.0):
    np.testing.assert_allclose(npy(ours) if torch.is_tensor(ours) else ours, ref, rtol=rtol, atol=atol, err_msg=name)


def n_band(dump):
    return max(int(k.split("_b")[-1].split("_")[0]) for k in dump if "_b" in k)


def test_lookup_weights(runs):
    _, _, dump, im = runs
    jp, jt, jtt, fac00, fac01, fac10, fac11 = im["lookup_weights"]
    np.testing.assert_array_equal(npy(jp) + 1, dump["jp"])
    np.testing.assert_array_equal(npy(jt) + 1, dump["jt"])
    np.testing.assert_array_equal(npy(jtt) + 1, dump["jtt"])
    for name, v in (("fac00", fac00), ("fac01", fac01), ("fac10", fac10), ("fac11", fac11)):
        close(name, v, dump[name], atol=1e-15)


def test_k_abs(runs):
    _, _, dump, im = runs
    for b, band in enumerate(im["k_abs"]):
        ref = dump[f"k_abs_layer_b{b + 1}"]  # (P, L, nd_k, n_abs)
        np.testing.assert_array_equal(dump[f"index_abs_b{b + 1}"], np.arange(1, len(band) + 1))
        for j, k in enumerate(band):
            n_k = k.shape[0]
            assert dump[f"n_abs_esft_b{b + 1}"][j] == n_k
            close(f"k_abs b{b + 1} gas{j}", k, np.moveaxis(ref[:, :, :n_k, j], 2, 0), atol=1e-300)


def test_grey(runs):
    region, _, dump, im = runs
    for b in range(n_band(dump)):
        close(f"k_grey_tot b{b + 1}", im["k_grey_tot"][b], dump[f"k_grey_tot_clr_b{b + 1}"], atol=1e-300)
        np.testing.assert_array_equal(dump[f"phase_fnc_clr_b{b + 1}"], 0.0)
        np.testing.assert_array_equal(dump[f"forward_scatter_clr_b{b + 1}"], 0.0)
        if region == "sw":
            close(f"k_ext_scat b{b + 1}", im["k_ext_scat"][b], dump[f"k_ext_scat_clr_b{b + 1}"])


def test_planck(runs):
    region, _, dump, im = runs
    if region != "lw":
        pytest.skip("longwave only")
    for b in range(n_band(dump)):
        close(f"planck_flux b{b + 1}", im["planck_flux"][b], dump[f"planck_flux_b{b + 1}"])
        close(f"planck_ground b{b + 1}", im["planck_ground"][b], dump[f"planck_flux_ground_b{b + 1}"])
        close(f"planck_diff b{b + 1}", im["planck_diff"][b], dump[f"planck_diff_b{b + 1}"], rtol=1e-10, atol=1e-12)
        close(f"planck_diff_2 b{b + 1}", im["planck_diff_2"][b], dump[f"planck_diff_2_b{b + 1}"], rtol=1e-9,
              atol=1e-11)


def test_equivalent_extinction(runs):
    region, _, dump, im = runs
    for b in range(n_band(dump)):
        close(f"k_eqv b{b + 1}", im["k_eqv"][b], dump[f"k_eqv_b{b + 1}"], rtol=1e-10, atol=1e-300)
        close(f"k_grey_final b{b + 1}", im["k_grey_final"][b], dump[f"k_grey_tot_final_b{b + 1}"], rtol=1e-10,
              atol=1e-300)
        if region == "sw":
            close(f"adjust_solar_ke b{b + 1}", im["adjust_solar_ke"][b], dump[f"adjust_solar_ke_b{b + 1}"],
                  rtol=1e-10)


def test_gpoint_fluxes(runs):
    region, atm, dump, im = runs
    up, down = npy(im["gpoint_flux_up"]), npy(im["gpoint_flux_down"])
    g = 0
    for b in range(n_band(dump)):
        k = 1
        while f"flux_total_b{b + 1}_k{k}" in dump:
            ft = dump[f"flux_total_b{b + 1}_k{k}"]
            scale = np.abs(ft).max() + 1.0
            close(f"up b{b + 1} k{k}", up[g], ft[:, 0::2], rtol=1e-9, atol=1e-9 * scale)
            close(f"down b{b + 1} k{k}", down[g], ft[:, 1::2], rtol=1e-9, atol=1e-9 * scale)
            if region == "sw":
                close(f"direct b{b + 1} k{k}", im["gpoint_flux_direct"][g], dump[f"flux_direct_b{b + 1}_k{k}"],
                      rtol=1e-9, atol=1e-9 * scale)
            g += 1
            k += 1
    assert g == up.shape[0]
