"""Idealised columns (pysoc.column), spectral-file lookup and the conveniences the notebooks rely on."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from pysoc.column import (dobson_units, isca_sigma_levels, liquid_cloud, make_column,
                          saturation_specific_humidity)
from pysoc.isca import GRAV, RDGAS, IscaSocrates, ozone_mmr_from_vmr
from pysoc.spectra import GA7_SHA256, ga7_spectral_files


@lru_cache(maxsize=1)
def model():
    return IscaSocrates(*ga7_spectral_files())


def run(col, **kw):
    return model()(**col, albedo=kw.pop("albedo", 0.3), coszen=kw.pop("coszen", 0.5), delta_t=0.0, **kw)


def test_spectral_files_match_pinned_hashes():
    lw, sw = ga7_spectral_files()
    for path in (lw, lw + "_k", sw, sw + "_k"):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == GA7_SHA256[Path(path).name]


def test_isca_sigma_levels():
    s = isca_sigma_levels(40)
    assert s.shape == (41,) and s[0] == 0.0 and s[-1] == 1.0
    assert bool((torch.diff(s) > 0).all())
    # Isca's formula: exp(-scale_heights * (surf_res*zeta + (1-surf_res)*zeta**exponent)), zeta = 1 - k/N
    zeta = 1.0 - 1.0 / 40
    assert float(s[1]) == pytest.approx(np.exp(-11.0 * (0.2 * zeta + 0.8 * zeta**7)), rel=1e-14)


def test_column_shapes_and_profiles():
    col = make_column(t_surf=290.0, lapse_rate=6.0, ozone_du=280.0)
    L = 40
    assert col["temp"].shape == (L,) and col["p_half"].shape == (L + 1,) and col["t_surf"].shape == ()
    # hydrostatic: dz between full levels matches R T / g d(ln p) to second order
    z, p, t = col["z_full"], col["p_full"], col["temp"]
    dz = z[:-1] - z[1:]
    expected = RDGAS / GRAV * 0.5 * (t[:-1] + t[1:]) * torch.log(p[1:] / p[:-1])
    assert torch.allclose(dz[1:], expected[1:], rtol=0.02)
    # tropospheric lapse rate as requested
    lapse = (t[-2] - t[-1]) / (z[-2] - z[-1]) * -1e3
    assert float(lapse) == pytest.approx(6.0, rel=1e-6)
    # ozone column and water vapour limits
    vmr = col["ozone"] / ozone_mmr_from_vmr(1.0)
    assert float(dobson_units(vmr, col["p_half"])) == pytest.approx(280.0, rel=1e-12)
    q = col["q"]
    assert bool(((q <= saturation_specific_humidity(t, p) * (1 + 1e-12)) | (q == 3e-6)).all())
    assert bool((q[:-1] <= q[1:] + 1e-18).all()) and float(q.min()) >= 3e-6


def test_column_batching_and_gradients():
    ts = torch.tensor([280.0, 290.0, 300.0], dtype=torch.float64, requires_grad=True)
    cols = make_column(t_surf=ts, co2_ppmv=torch.tensor([280.0, 560.0, 280.0]))
    assert cols["temp"].shape == (3, 40) and cols["co2"].shape == (3, 40)
    single = make_column(t_surf=290.0, co2_ppmv=560.0)
    for k in single:
        assert torch.allclose(cols[k][1], single[k], rtol=1e-13, atol=0.0), k
    out = run(cols)
    assert out["soc_olr"].shape == (3,)
    (g,) = torch.autograd.grad(out["soc_olr"].sum(), ts)
    assert bool(torch.isfinite(g).all()) and bool((g > 0).all())


def test_liquid_cloud_water_path():
    col = make_column()
    cloud = liquid_cloud(col, 700e2, 900e2, lwp=80.0, fraction=0.5, reff=12.0)
    inside = cloud["cf_rad"] > 0
    assert bool(inside.any()) and bool(((col["p_full"][inside] >= 700e2) & (col["p_full"][inside] <= 900e2)).all())
    d_mass = torch.diff(col["p_half"]) / GRAV
    mmr = cloud["qcl_rad"] / (1.0 - cloud["qcl_rad"])
    in_cloud_path = float((mmr / 0.5 * d_mass)[inside].sum())
    assert in_cloud_path == pytest.approx(0.080, rel=1e-12)
    assert torch.allclose(cloud["reff_rad"][inside], torch.tensor(12e-6, dtype=torch.float64))
    with pytest.raises(ValueError):
        liquid_cloud(col, 971e2, 972e2)
    out = run(col, **cloud)
    assert float(out["soc_tot_cloud_cover"]) == pytest.approx(0.5)


def test_isca_interface_conveniences():
    batch = make_column(t_surf=torch.tensor([288.0]))  # one column with a leading dimension
    floats = run(batch, albedo=0.25, coszen=0.6)
    tensors = run(batch, albedo=torch.tensor([0.25], dtype=torch.float64),
                  coszen=torch.tensor([0.6], dtype=torch.float64))
    assert float(floats["soc_olr"][0]) == float(tensors["soc_olr"][0])
    assert float(floats["soc_toa_sw"][0]) == float(tensors["soc_toa_sw"][0])
    col = make_column()
    scalar = run(col, albedo=0.25, coszen=0.6)
    assert float(scalar["soc_olr"]) == float(tensors["soc_olr"][0])
    assert scalar["flux_lw_up_band"].shape == (41, 9) and scalar["flux_sw_down_band"].shape == (41, 6)
    assert torch.allclose(scalar["flux_lw_up_band"].sum(-1), scalar["flux_lw_up"], rtol=1e-12)
    inter = run(col, return_intermediates=True)
    assert inter["lw_intermediates"]["tau"].shape == (81, 1, 40)
    assert inter["sw_intermediates"]["tau"].shape == (41, 1, 40)


def test_non_finite_temperature_gives_nan_not_error():
    col = make_column()
    temp = col["temp"].clone()
    temp[5] = float("nan")
    out = run(dict(col, temp=temp))
    assert bool(torch.isnan(out["soc_olr"]))
