"""Phase 2 gate: two-stream kernels match the Fortran routines on random and edge-case inputs."""

import numpy as np
import pytest
import torch

from conftest import require_exe
from pysoc import twostream as ts
from tools.refcases import run_unit, split_flux_total

NP, NL = 64, 30
RTOL = 1e-12


@pytest.fixture(autouse=True)
def _exe():
    require_exe("soc_unit")


def rng(seed):
    return np.random.default_rng(seed)


def T(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def check(name, ours, ref, rtol=RTOL, atol=1e-300):
    ours = ours.detach().cpu().numpy() if torch.is_tensor(ours) else np.asarray(ours)
    np.testing.assert_allclose(ours, ref, rtol=rtol, atol=atol, err_msg=name)


def optical_depths(r, shape):
    """Log-uniform optical depths from 1e-12 to 1e4 with exact zeros mixed in."""
    tau = 10.0 ** r.uniform(-12, 4, shape)
    tau[r.random(shape) < 0.05] = 0.0
    return tau


def albedos(r, shape):
    w = r.uniform(0, 1, shape)
    w[r.random(shape) < 0.1] = 1.0 - 1e-12
    w[r.random(shape) < 0.1] = 0.0
    return w


@pytest.mark.parametrize("method", [ts.IP_SCATTER_FULL, ts.IP_NO_SCATTER_ABS, ts.IP_NO_SCATTER_EXT])
def test_single_scattering(method):
    r = rng(1)
    kg = 10.0 ** r.uniform(-8, 2, (NP, NL))
    ks = kg * r.uniform(0, 1, (NP, NL))
    ks[0] = kg[0]  # omega -> 1 clipping
    kgas = 10.0 ** r.uniform(-10, 3, (NP, NL))
    kgas[1] = 0.0
    kg[1] = 0.0
    ks[1] = 0.0  # 0/0 -> TINY handling
    dm = r.uniform(1, 1e4, (NP, NL))
    ref = run_unit("single_scattering", method, NP, NL, [kg, ks, kgas, dm])
    tau, omega = ts.single_scattering(T(kg), T(ks), T(kgas), T(dm), method)
    check("tau", tau, ref["tau"])
    check("omega", omega, ref["omega"])


def test_rescale_tau_omega():
    r = rng(2)
    tau = optical_depths(r, (NP, NL))
    om = albedos(r, (NP, NL)) * (1 - 1e-9)
    f = r.uniform(0, 0.9, (NP, NL))
    ref = run_unit("rescale_tau_omega", 0, NP, NL, [tau, om, f])
    t2, o2 = ts.rescale_tau_omega(T(tau), T(om), T(f))
    check("tau", t2, ref["tau"])
    check("omega", o2, ref["omega"])


EPS = np.finfo(float).eps


def solar_roundoff_bounds(om, g, tau, sec0, scheme):
    """Condition-aware bounds on round-off differences between two IEEE implementations.

    Each bound is (a few) x eps times the magnitudes that cancel in the Fortran
    expressions, propagated through the ill-conditioned factors 1/(1-e^2 gamma^2)
    and 1/(lambda - sec_0).
    """
    s, d = ts.two_coeff_basic(T(om), T(g), scheme)
    lam = torch.sqrt(s * d)
    s, d, lam, gu, gd = ts.solar_coefficient_basic(T(om), T(g), T(sec0), s, d, lam, scheme)
    s, d, lam, gu, gd = (x.numpy() for x in (s, d, lam, gu, gd))
    sec = sec0[:, None]
    e = np.exp(-lam * tau)
    gam = (s - lam) / (s + lam)
    tmp_inv = 1.0 / (1.0 - e * e * gam * gam)
    trans = e * (1 - gam**2) * tmp_inv
    reflect = gam * (1 - e * e) * tmp_inv
    tr0 = np.exp(-tau * sec)
    err_e2 = 8 * EPS
    err_reflect = 16 * EPS * np.abs(reflect) + np.abs(gam) * tmp_inv * err_e2 * (1 + tmp_inv)
    err_trans = 16 * EPS * np.abs(trans) + np.abs(e * (1 - gam**2)) * tmp_inv**2 * err_e2
    cond = np.abs(lam) / np.maximum(np.abs(lam - sec), 1e-300)
    err_gu = np.abs(gu) * 16 * EPS * (1 + cond)
    err_gd = np.abs(gd) * 16 * EPS * (1 + cond)
    err_su = (16 * EPS * (np.abs(gu) + np.abs(reflect * (1 + gd)) + np.abs(gu * trans * tr0))
              + err_gu * (1 + trans) + err_reflect * np.abs(1 + gd) + np.abs(reflect) * err_gd
              + np.abs(gu) * (err_trans + 4 * EPS))
    err_sd = (16 * EPS * (np.abs(tr0 * (1 + gd)) + np.abs(tr0 * gu * reflect) + np.abs((1 + gd) * trans))
              + err_gd * (1 + trans) + err_gu * np.abs(reflect) + np.abs(gu) * err_reflect
              + np.abs(1 + gd) * err_trans + 4 * EPS)
    return dict(trans=err_trans, reflect=err_reflect, trans_0=8 * EPS * tr0, source_up=err_su, source_down=err_sd)


@pytest.mark.parametrize("scheme", [ts.IP_PIFM80, ts.IP_EDDINGTON, ts.IP_ELSASSER, ts.IP_DISCRETE_ORD,
                                    ts.IP_PIFM85, ts.IP_HEMI_MEAN])
def test_two_coeff_solar(scheme):
    r = rng(3 + scheme)
    g = r.uniform(-0.2, 0.95, (NP, NL))
    om = np.minimum(albedos(r, (NP, NL)), 1 - 32 * EPS)
    tau = optical_depths(r, (NP, NL))
    sec0 = 1.0 / r.uniform(1e-3, 1.0, NP)
    # Put the first layer of a few columns exactly on the singularity lambda == sec_0
    # (only possible when lambda >= 1, i.e. for fairly absorbing layers).
    om[:4, 0] = r.uniform(0.0, 0.3, 4)
    s, d = ts.two_coeff_basic(T(om), T(g), scheme)
    lam = torch.sqrt(s * d).numpy()
    sing = lam[:4, 0] >= 1.0
    sec0[:4][sing] = lam[:4, 0][sing]
    assert sing.any()
    ref = run_unit("two_coeff_solar", scheme, NP, NL, [g, om, tau, sec0])
    out = ts.two_coeff_solar(T(om), T(g), T(tau), T(sec0), scheme)
    bounds = solar_roundoff_bounds(om, g, tau, sec0, scheme)
    for name, o in zip(["trans", "reflect", "trans_0", "source_up", "source_down"], out):
        o = o.numpy()
        assert np.all(np.isfinite(o)), name
        excess = np.abs(o - ref[name]) - (1e-13 * np.abs(ref[name]) + bounds[name])
        assert np.all(excess <= 0), (name, np.unravel_index(np.argmax(excess), excess.shape), excess.max())


@pytest.mark.parametrize("quad", [0, 1])
def test_two_coeff_fast_lw(quad):
    r = rng(5)
    tau = optical_depths(r, (NP, NL))
    tau[2] = np.logspace(-9, -2, NL)  # around the thin/thick switch
    ref = run_unit("two_coeff_fast_lw", quad, NP, NL, [tau])
    trans, s1, s2 = ts.two_coeff_fast_lw(T(tau), bool(quad))
    check("trans", trans, ref["trans"], rtol=1e-14)
    # 1 - trans (and 1 + trans - 2 s1) cancel for thin layers: allow the round-off of exp()
    # amplified by 1/(1.66 tau + sqrt(eps)) (squared for the quadratic term).
    denom = 1.66 * tau + np.sqrt(EPS)
    assert np.all(np.abs(s1.numpy() - ref["source_1"]) <= 1e-14 * np.abs(ref["source_1"]) + 8 * EPS / denom)
    if quad:
        assert np.all(np.abs(s2.numpy() - ref["source_2"]) <= 1e-14 * np.abs(ref["source_2"]) + 32 * EPS / denom**2)


@pytest.mark.parametrize("quad", [0, 1])
def test_ir_source(quad):
    r = rng(6)
    a, b, c, d = (r.normal(size=(NP, NL)) for _ in range(4))
    ref = run_unit("ir_source", quad, NP, NL, [a, b, c, d])
    s_down, s_up = ts.ir_source(T(a), T(b), T(c), T(d), bool(quad))
    check("s_down", s_down, ref["s_down"])
    check("s_up", s_up, ref["s_up"])


@pytest.mark.parametrize("scale", [0, 1])
def test_solar_source(scale):
    r = rng(7)
    finc = r.uniform(0, 1400, NP)
    tr0 = r.uniform(0, 1, (NP, NL))
    su, sd = r.normal(size=(NP, NL)), r.normal(size=(NP, NL))
    adj = r.uniform(0.5, 1, (NP, NL))
    ref = run_unit("solar_source", scale, NP, NL, [finc, tr0, su, sd, adj])
    fd, s_down, s_up = ts.solar_source(T(finc), T(tr0), T(su), T(sd), T(adj) if scale else None)
    check("flux_direct", fd, ref["flux_direct"])
    check("s_down", s_down, ref["s_down"])
    check("s_up", s_up, ref["s_up"])


def test_solver_homogen_direct():
    r = rng(8)
    om = albedos(r, (NP, NL)) * (1 - 1e-9)
    g = r.uniform(0, 0.9, (NP, NL))
    tau = optical_depths(r, (NP, NL))
    sec0 = 1.0 / r.uniform(0.05, 1.0, NP)
    trans, reflect, tr0, su, sd = ts.two_coeff_solar(T(om), T(g), T(tau), T(sec0), ts.IP_PIFM80)
    fd, s_down, s_up = ts.solar_source(T(r.uniform(0, 1400, NP)), tr0, su, sd, T(r.uniform(0.8, 1, (NP, NL))))
    alb = r.uniform(0, 1, NP)
    finc = r.uniform(0, 10, NP)
    sg = alb * fd[:, -1].numpy() * 0.0 + r.uniform(0, 100, NP)
    args = [trans, reflect, s_down, s_up]
    ref = run_unit("solver_homogen_direct", 0, NP, NL, [a.numpy() for a in args] + [alb, finc, sg])
    up, down = ts.solver_homogen_direct(*args, T(alb), T(finc), T(sg))
    rup, rdown = split_flux_total(ref["flux_total"])
    check("up", up, rup, atol=1e-10)
    check("down", down, rdown, atol=1e-10)


def test_solver_no_scat():
    r = rng(9)
    tau = optical_depths(r, (NP, NL))
    trans, s1, s2 = ts.two_coeff_fast_lw(T(tau))
    dpl = T(r.normal(0, 5, (NP, NL)))
    dpl2 = T(r.normal(0, 1, (NP, NL)))
    s_down, s_up = ts.ir_source(s1, s2, dpl, dpl2)
    alb, finc, dps = r.uniform(0, 0.2, NP), r.normal(0, 50, NP), r.normal(0, 30, NP)
    ref = run_unit("solver_no_scat", 0, NP, NL, [trans.numpy(), s_down.numpy(), s_up.numpy(), alb, finc, dps])
    up, down = ts.solver_no_scat(trans, s_down, s_up, T(alb), T(finc), T(dps))
    rup, rdown = split_flux_total(ref["flux_total"])
    check("up", up, rup, atol=1e-10)
    check("down", down, rdown, atol=1e-10)


def test_monochromatic_gas_flux_ir():
    r = rng(10)
    tau = optical_depths(r, (NP, NL))
    dpl = r.normal(0, 5, (NP, NL))
    finc, dps, alb = r.normal(0, 50, NP), r.normal(0, 30, NP), r.uniform(0, 0.2, NP)
    ref = run_unit("monochromatic_gas_flux_ir", 0, NP, NL, [tau, dpl, finc, dps, alb])
    up, down = ts.monochromatic_gas_flux_ir(T(tau), T(finc), T(dpl), T(dps), T(alb))
    rup, rdown = split_flux_total(ref["flux_total"])
    # Layer sources carry the thin-layer round-off of (1 - trans); bound its accumulation.
    bound = (8 * EPS * np.abs(dpl) / (1.66 * tau + np.sqrt(EPS))).sum(-1, keepdims=True) * 2
    for o, rf in ((up, rup), (down, rdown)):
        assert np.all(np.abs(o.numpy() - rf) <= 1e-12 * np.abs(rf) + bound + 1e-12)
