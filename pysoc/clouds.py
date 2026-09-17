"""Cloudy two-region solve of SOCRATES as used by Isca's simple cloud schemes.

Isca passes liquid water only (``liq_frac = cloud fraction``, ``ice = 0``), with
``i_cloud_representation = ip_cloud_ice_water``, maximum-random overlap
(``ip_cloud_mix_max``) and ``solver_mix_direct_hogan``.  Ports of

* Isca ``socrates_set_cld.F90`` (liquid part),
* ``overlap_coupled.F90`` (max-random, two regions),
* ``opt_prop_pade_2_mod.F90`` (droplet parametrisation 5),
* ``trans_source_coeff.F90`` (infrared branch), ``two_coeff_cloud.F90``,
* ``mixed_solar_source.F90`` and ``solver_mix_direct_hogan.F90``.

SOCRATES picks ``n_cloud_top`` (the highest layer with cloud in *any* column of
the batch).  Results do not depend on it beyond round-off, so every layer is
treated as potentially cloudy here (``n_cloud_top = 1``), which keeps the batch
dimension independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import twostream as ts

MIN_CLOUD_FRACTION = 1.0e-4  # socrates_set_cld.F90
EPS_R8 = float(torch.finfo(torch.float64).eps)  # EPSILON(1.0) with -fdefault-real-8


@dataclass
class LiquidCloud:
    """Liquid cloud as passed by Isca: fraction, grid-box mean mass mixing ratio, effective radius [m]."""

    fraction: torch.Tensor
    mmr: torch.Tensor
    reff: torch.Tensor


def set_cloud(cloud: LiquidCloud, min_dim: float, max_dim: float):
    """Isca ``set_cld`` for liquid-only clouds.

    Returns ``w_cloud`` (cloud cover of each layer), in-cloud mass mixing ratio and the
    clamped effective radius, all (P, L).
    """
    frac = cloud.fraction
    in_cloud_mmr = cloud.mmr / torch.clamp(frac, min=EPS_R8)
    dim = torch.clamp(cloud.reff, min=min_dim, max=max_dim)
    w = torch.where(frac > MIN_CLOUD_FRACTION, frac, torch.zeros_like(frac))
    w = torch.clamp(w, max=1.0)  # SOCRATES stops if w > 1 + MIN_CLOUD_FRACTION, and clips otherwise
    return w, in_cloud_mmr, dim


def overlap_max_random(w_cloud: torch.Tensor):
    """Max-random overlap coefficients between adjacent layers (``overlap_coupled``, two regions).

    Returns a dict of (P, L+1) tensors for interfaces i = 0..L (0 = above layer 1, L = ground)
    with keys ``dn_ff, dn_fc, dn_cf, dn_cc, up_ff, up_fc, up_cf, up_cc`` (SOCRATES index
    order 1..8), plus ``w_free`` (P, L) and ``tot_cloud_cover`` (P,).
    """
    P, L = w_cloud.shape
    dtype = w_cloud.dtype
    tol = 1.0e2 * torch.finfo(dtype).eps
    w_free = 1.0 - w_cloud
    zero = torch.zeros(P, dtype=dtype, device=w_cloud.device)
    one = torch.ones_like(zero)
    cover = torch.where(w_free[:, 0] > tol, w_free[:, 0], zero)
    up_f, up_c = one, zero  # area_upper for i = n_cloud_top - 1 = 0
    names = ("dn_ff", "dn_fc", "dn_cf", "dn_cc", "up_ff", "up_fc", "up_cf", "up_cc")
    out = {n: [] for n in names}
    for i in range(L + 1):
        if i < L:
            lo_f, lo_c = w_free[:, i], w_cloud[:, i]  # layer i+1 (0-based index i)
        else:
            lo_f, lo_c = one, zero
        ov_ff = torch.minimum(lo_f, up_f)  # corr_factor = 1 (max overlap)
        ov_cc = torch.minimum(lo_c, up_c)
        if 1 <= i < L:
            cover = torch.where(lo_f > tol, cover * lo_f / (1.0 - ov_cc), zero)
        rnd_tot = 1.0 - ov_ff - ov_cc
        rnd_up_f, rnd_up_c = up_f - ov_ff, up_c - ov_cc
        rnd_lo_f, rnd_lo_c = lo_f - ov_ff, lo_c - ov_cc
        pos = rnd_tot > tol
        safe_tot = torch.where(pos, rnd_tot, one)
        o11 = torch.where(pos, ov_ff + rnd_up_f * rnd_lo_f / safe_tot, ov_ff)
        o12 = torch.where(pos, 0.0 + rnd_up_f * rnd_lo_c / safe_tot, zero)
        o21 = torch.where(pos, 0.0 + rnd_up_c * rnd_lo_f / safe_tot, zero)
        o22 = torch.where(pos, ov_cc + rnd_up_c * rnd_lo_c / safe_tot, ov_cc)

        def ratio(num, den, default):
            ok = den > tol
            return torch.where(ok, num / torch.where(ok, den, one), default)

        # downward: index n*(j-1)+k = area_overlap(k, j) / area_upper(k)
        out["dn_ff"].append(ratio(o11, up_f, one))      # k=1, j=1
        out["dn_fc"].append(ratio(o21, up_c, zero))     # k=2, j=1
        out["dn_cf"].append(ratio(o12, up_f, zero))     # k=1, j=2
        out["dn_cc"].append(ratio(o22, up_c, one))      # k=2, j=2
        # upward: index n*(n+j-1)+k = area_overlap(j, k) / area_lower(k)
        out["up_ff"].append(ratio(o11, lo_f, one))      # k=1, j=1
        out["up_fc"].append(ratio(o12, lo_c, zero))     # k=2, j=1
        out["up_cf"].append(ratio(o21, lo_f, zero))     # k=1, j=2
        out["up_cc"].append(ratio(o22, lo_c, one))      # k=2, j=2
        if i < L:
            up_f, up_c = lo_f, lo_c
    res = {n: torch.stack(v, -1) for n, v in out.items()}
    res["w_free"] = w_free
    res["tot_cloud_cover"] = 1.0 - cover
    return res


def opt_prop_pade_2(param: torch.Tensor, mmr: torch.Tensor, reff: torch.Tensor, return_absorption=False):
    """Droplet optical properties (parametrisation 5).

    ``param`` (..., 16) per band, broadcast against (P, L).  Returns mass extinction and
    scattering (per kg of air), scattering-weighted asymmetry and forward scattering.
    """
    p = [param[..., j, None, None] for j in range(16)]
    r = reff
    k_ext = mmr * (p[0] + r * (p[1] + r * p[2])) / (1.0 + r * (p[3] + r * (p[4] + r * p[5])))
    coalbedo = (p[6] + r * (p[7] + r * p[8])) / (1.0 + r * (p[9] + r * p[10]))
    k_scat = k_ext * (1.0 - coalbedo)
    g = (p[11] + r * (p[12] + r * p[13])) / (1.0 + r * (p[14] + r * p[15]))
    phase = k_scat * g
    forward = phase * g
    if return_absorption:
        return k_ext, k_scat, phase, forward, k_ext * coalbedo
    return k_ext, k_scat, phase, forward


# ---------------------------------------------------------------------------
# Two-stream coefficients for the infrared (general form, trans_source_coeff.F90)
# ---------------------------------------------------------------------------
def two_coeff_ir(omega, asymmetry, tau, i_2stream=ts.IP_ELSASSER, l_ir_source_quad=True):
    """two_coeff.F90 + trans_source_coeff.F90 for the infrared region.

    Returns ``trans, reflect, source_1, source_2``.
    """
    s, d = ts.two_coeff_basic(omega, asymmetry, i_2stream)
    lam = torch.sqrt(s * d)
    sq_eps_r = ts._eps(tau) ** 0.5
    tol = sq_eps_r ** 0.5
    exponential = torch.exp(-lam * tau)
    exponential2 = exponential * exponential
    gamma = (s - lam) / (s + lam)
    gamma2 = gamma * gamma
    tmp_inv = 1.0 / (1.0 - exponential2 * gamma2)
    trans = exponential * (1.0 - gamma2) * tmp_inv
    reflect = gamma * (1.0 - exponential2) * tmp_inv
    source_1 = (1.0 - trans + reflect + sq_eps_r) / (sq_eps_r + tau * s)
    if not l_ir_source_quad:
        return trans, reflect, source_1, None
    thick = -2.0 * (1.0 - trans - reflect + sq_eps_r) / (d * tau + sq_eps_r)
    thin = -2.0 + d * tau
    tmp = torch.where(tau > tol, thick, thin)
    source_2 = -(1.0 + reflect + trans + tmp) / (s * tau + sq_eps_r)
    return trans, reflect, source_1, source_2


def two_coeff_ir_nonscattering_stable(tau, l_ir_source_quad=True):
    """Stable equivalent of :func:`two_coeff_ir` for omega = 0 (Isca's longwave).

    With omega = 0 the Elsasser coefficients are s = d = lambda = 1.66 and gamma = 0, and the
    general infrared expressions reduce to those of ``two_coeff_fast_lw``.
    """
    trans, source_1, source_2 = ts.two_coeff_fast_lw_stable(tau, l_ir_source_quad)
    return trans, torch.zeros_like(trans), source_1, source_2


# ---------------------------------------------------------------------------
# Mixed (clear + cloudy region) direct solar source
# ---------------------------------------------------------------------------
def mixed_solar_source(flux_inc_direct, adjust_solar_ke, trans_0_free, su_free, sd_free,
                       g_ff, g_fc, g_cf, g_cc, trans_0_cloud, su_cloud, sd_cloud, adjust_m1=None):
    """``mixed_solar_source.F90`` with ``n_cloud_top = 1`` and ``l_scale_solar``.

    Overlap arguments use the names of the Fortran dummy arguments (mix_column passes
    dn_ff, dn_cf, dn_fc, dn_cc into g_ff, g_fc, g_cf, g_cc).  Returns ``flux_direct``
    (..., L+1), ``flux_direct_ground_cloud`` and the four layer sources.
    ``adjust_m1`` (= adjust_solar_ke - 1) selects the cancellation-free source form.
    """
    L = trans_0_free.shape[-1]
    base_free = flux_inc_direct
    base_cloud = torch.zeros_like(flux_inc_direct)
    flux = [flux_inc_direct]
    s_up_f, s_dn_f, s_up_c, s_dn_c = [], [], [], []
    for i in range(L):
        top_cloud = g_cc[..., i] * base_cloud + g_fc[..., i] * base_free
        top_free = g_ff[..., i] * base_free + g_cf[..., i] * base_cloud
        adj = adjust_solar_ke[..., i]
        base_free = top_free * trans_0_free[..., i] * adj
        base_cloud = top_cloud * trans_0_cloud[..., i] * adj
        s_up_f.append(su_free[..., i] * top_free)
        s_up_c.append(su_cloud[..., i] * top_cloud)
        if adjust_m1 is None:
            s_dn_f.append((sd_free[..., i] - trans_0_free[..., i]) * top_free + base_free)
            s_dn_c.append((sd_cloud[..., i] - trans_0_cloud[..., i]) * top_cloud + base_cloud)
        else:
            am1 = adjust_m1[..., i]
            s_dn_f.append((sd_free[..., i] + trans_0_free[..., i] * am1) * top_free)
            s_dn_c.append((sd_cloud[..., i] + trans_0_cloud[..., i] * am1) * top_cloud)
        flux.append(base_free + base_cloud)
    st = lambda v: torch.stack(v, -1)  # noqa: E731
    return st(flux), base_cloud, st(s_up_f), st(s_dn_f), st(s_up_c), st(s_dn_c)


def solver_mix_direct_hogan(t, r, s_down, s_up, t_cloud, r_cloud, s_down_cloud, s_up_cloud,
                            v11, v21, v12, v22, u11, u12, u21, u22,
                            flux_inc_down, source_ground_free, source_ground_cloud, albedo_surface_diff):
    """``solver_mix_direct_hogan.F90`` with ``n_cloud_top = 1``.

    Layer arrays (..., L); overlap coefficients (..., L+1) for interfaces 0..L.
    Returns ``flux_up, flux_down`` (..., L+1).
    """
    L = t.shape[-1]
    alpha11 = albedo_surface_diff
    alpha22 = albedo_surface_diff
    g1 = source_ground_free
    g2 = source_ground_cloud
    b11, gm11, h1, b22, gm22, h2 = ([None] * L for _ in range(6))
    for i in range(L - 1, -1, -1):  # Fortran layer i+1; interface below it is i+1
        theta11 = alpha11 * v11[..., i + 1] + alpha22 * v21[..., i + 1]
        theta22 = alpha11 * v12[..., i + 1] + alpha22 * v22[..., i + 1]
        b11[i] = 1.0 / (1.0 - theta11 * r[..., i])
        gm11[i] = theta11 * t[..., i]
        h1[i] = g1 + theta11 * s_down[..., i]
        b22[i] = 1.0 / (1.0 - theta22 * r_cloud[..., i])
        gm22[i] = theta22 * t_cloud[..., i]
        h2[i] = g2 + theta22 * s_down_cloud[..., i]
        lambda1 = s_up[..., i] + h1[i] * t[..., i] * b11[i]
        lambda2 = s_up_cloud[..., i] + h2[i] * t_cloud[..., i] * b22[i]
        alpha11 = r[..., i] + theta11 * t[..., i] * t[..., i] * b11[i]
        g1 = u11[..., i] * lambda1 + u12[..., i] * lambda2
        alpha22 = r_cloud[..., i] + theta22 * t_cloud[..., i] * t_cloud[..., i] * b22[i]
        g2 = u21[..., i] * lambda1 + u22[..., i] * lambda2
    up = [g1 + flux_inc_down * (v11[..., 0] * alpha11 + v21[..., 0] * alpha22)]
    down = [flux_inc_down]
    fd1 = v11[..., 0] * flux_inc_down
    fd2 = v21[..., 0] * flux_inc_down
    for i in range(L):
        if i > 0:
            fd1, fd2 = v11[..., i] * fd1 + v12[..., i] * fd2, v21[..., i] * fd1 + v22[..., i] * fd2
        fu1 = (gm11[i] * fd1 + h1[i]) * b11[i]
        fu2 = (gm22[i] * fd2 + h2[i]) * b22[i]
        fd1 = t[..., i] * fd1 + r[..., i] * fu1 + s_down[..., i]
        fd2 = t_cloud[..., i] * fd2 + r_cloud[..., i] * fu2 + s_down_cloud[..., i]
        up.append(fu1 + fu2)
        down.append(fd1 + fd2)
    return torch.stack(up, -1), torch.stack(down, -1)
