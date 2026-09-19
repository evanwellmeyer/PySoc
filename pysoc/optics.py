"""Gas absorption, continuum, Rayleigh and Planck terms of SOCRATES in PyTorch.

Ports of ``inter_pt_lookup``, the k-table interpolation in ``radiance_calc``,
``scale_absorb``, ``rescale_continuum`` and ``planck_flux_band``.  Profile/layer
tensors have shape ``(..., n_layer)``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .spectral_file import (
    IP_SCALE_DBL_POW_LAW,
    IP_SCALE_DBL_POW_QUAD,
    IP_SCALE_DOPPLER_QUAD,
    IP_SCALE_FNC_NULL,
    IP_SCALE_POWER_LAW,
    IP_SCALE_POWER_QUAD,
    IP_SCALE_WENYI,
)

# rad_ccf.F90 (default values; Isca does not call set_socrates_constants)
MOL_WEIGHT_AIR = 28.966e-03
REPSILON = 18.0153 / 28.966
IP_SELF_CONTINUUM = 1
IP_FRN_CONTINUUM = 2


def _eps(t: torch.Tensor) -> float:
    return torch.finfo(t.dtype).eps


# ---------------------------------------------------------------------------
# p/T lookup tables (inter_pt_lookup.F90, without self-broadening)
# ---------------------------------------------------------------------------
def lookup_weights(p: torch.Tensor, t: torch.Tensor, p_lookup: torch.Tensor, t_lookup: torch.Tensor):
    """Bilinear (ln p, T) interpolation indices and weights.

    ``p_lookup`` (n_pre,) holds ln(p); ``t_lookup`` is (n_pre, n_tmp).
    Returns ``jp, jt, jtt`` (0-based long tensors) and ``fac00, fac01, fac10, fac11``.
    """
    eps = _eps(p)
    n_pre, n_tmp = t_lookup.shape
    p_min = p_lookup[0] + torch.clamp(torch.abs(p_lookup[0]) * eps, min=eps)
    p_max = p_lookup[-1] - torch.clamp(torch.abs(p_lookup[-1]) * eps, min=eps)
    p_layer = torch.minimum(torch.maximum(torch.log(p), p_min), p_max)
    # the clamps only matter for non-finite inputs, which then give NaN instead of an index error
    jp = torch.clamp(torch.searchsorted(p_lookup, p_layer.detach().contiguous(), right=True) - 1, 0, n_pre - 2)
    fp = (p_layer - p_lookup[jp]) / (p_lookup[jp + 1] - p_lookup[jp])

    def t_index(row):
        tl = t_lookup[row]  # (..., n_tmp)
        t_lay = torch.minimum(torch.maximum(t, tl[..., 0] * (1.0 + eps)), tl[..., -1] * (1.0 - eps))
        j = torch.searchsorted(tl.detach().contiguous(), t_lay.detach().unsqueeze(-1).contiguous(), right=True)[..., 0] - 1
        j = torch.clamp(j, 0, n_tmp - 2)
        lo = torch.gather(tl, -1, j.unsqueeze(-1))[..., 0]
        hi = torch.gather(tl, -1, (j + 1).unsqueeze(-1))[..., 0]
        return j, (t_lay - lo) / (hi - lo)

    jt, ft = t_index(jp)
    jtt, ftt = t_index(jp + 1)
    compfp = 1.0 - fp
    fac00 = compfp * (1.0 - ft)
    fac10 = fp * (1.0 - ftt)
    fac01 = compfp * ft
    fac11 = fp * ftt
    return jp, jt, jtt, fac00, fac01, fac10, fac11


def interp_k_lookup(k_lookup: torch.Tensor, weights) -> torch.Tensor:
    """Interpolate k-tables ``(..., n_k, n_pre, n_tmp)`` to layers: returns ``(..., n_k, *layer_shape)``.

    Mirrors ``k_esft_layer = MAX(0, fac00*K(jt,jp) + fac10*K(jtt,jp+1) + fac01*K(jt+1,jp) + fac11*K(jtt+1,jp+1))``.
    """
    jp, jt, jtt, fac00, fac01, fac10, fac11 = weights
    n_tmp = k_lookup.shape[-1]
    flat = k_lookup.reshape(*k_lookup.shape[:-2], -1)  # (..., n_k, n_pre*n_tmp)
    lead = flat.shape[:-1]
    layer_shape = jp.shape

    def take(ip, it):
        idx = (ip * n_tmp + it).reshape(-1)
        out = flat.index_select(-1, idx)
        return out.reshape(*lead, *layer_shape)

    k = (fac00 * take(jp, jt) + fac10 * take(jp + 1, jtt)
         + fac01 * take(jp, jt + 1) + fac11 * take(jp + 1, jtt + 1))
    return torch.clamp(k, min=0.0)


# ---------------------------------------------------------------------------
# Scaling of absorber amounts (scale_absorb.F90)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def wenyi_tables_numpy():
    """Hard-wired k-scaling tables of scale_wenyi.F90 (dumped from the Fortran)."""
    d = np.load(Path(__file__).parent / "data" / "scale_wenyi.npz")
    return {k: d[k] for k in d.files}


def _wenyi(p, t, iex, i_band_1based, tables=None):
    """ip_scale_wenyi: hard-wired CO2 (band 4) and O3 (band 6) tables of the GA LW files.

    ``tables``: dict of tensors (plg, ttb, tto, gk250b, gk4, gk6) on p's device/dtype.
    """
    if tables is None:
        tables = {k: torch.as_tensor(v, dtype=p.dtype, device=p.device) for k, v in wenyi_tables_numpy().items()}
    plg = tables["plg"]
    cgp = torch.clamp(torch.log(p / 100.0), min=-5.5)
    jp = torch.clamp(torch.trunc((5.5 + cgp.detach()) * 2.0).long() + 1, 1, 26)
    jp1 = torch.clamp(jp + 1, max=26)
    if i_band_1based == 4:
        gk, tt, nt, off = tables["gk4"], tables["ttb"], 10, 7.0
    elif i_band_1based == 6:
        gk, tt, nt, off = tables["gk6"], tables["tto"], 9, 5.0
    else:
        # scale_absorb leaves gas_frac_rescaled undefined for other bands
        raise NotImplementedError("Wenyi scaling is only defined for bands 4 and 6")
    jt = torch.clamp(torch.trunc((t.detach() - 240.0) / 20.0 + off).long(), 1, nt)
    jt1 = torch.clamp(jt + 1, max=nt)
    g = gk[:, :, iex - 1]  # (nt, 26)

    def G(a, b):
        return g[a - 1, b - 1]

    dp = (cgp - plg[jp - 1]) * 2.0
    gkpb = G(jt, jp) + (G(jt, jp1) - G(jt, jp)) * dp
    gkpc = G(jt1, jp) + (G(jt1, jp1) - G(jt1, jp)) * dp
    out = gkpb + (gkpc - gkpb) * (t - tt[jt - 1]) / 20.0
    return out / tables["gk250b"][iex - 1] if i_band_1based == 4 else out


def scale_absorb(p, t, i_fnc, p_ref, t_ref, scale, iex=1, i_band_1based=0, doppler_offset=None,
                 wenyi_tables=None):
    """Scaling of the absorber amount for one k-term (``gas_frac_rescaled``)."""
    offset = 0.0 if doppler_offset is None else doppler_offset
    if i_fnc == IP_SCALE_POWER_LAW:
        pwk = torch.pow((p + offset) * (1.0 / (p_ref + offset)), scale[0])
        twk = torch.pow(t * (1.0 / t_ref), scale[1])
        return pwk * twk
    if i_fnc == IP_SCALE_POWER_QUAD:
        pwk = torch.pow((p + offset) * (1.0 / (p_ref + offset)), scale[0])
        tmp = t * (1.0 / t_ref) - 1.0
        return pwk * (1.0 + tmp * scale[1] + scale[2] * tmp * tmp)
    if i_fnc == IP_SCALE_DOPPLER_QUAD:
        pwk = torch.pow((p + scale[1]) / (p_ref + scale[1]), scale[0])
        tmp = t * (1.0 / t_ref) - 1.0
        return pwk * (1.0 + tmp * scale[2] + scale[3] * tmp * tmp)
    if i_fnc == IP_SCALE_DBL_POW_LAW:
        hi = torch.exp(scale[2] * torch.log(p / scale[4]) + scale[3] * torch.log(t / scale[5]))
        lo = torch.exp(scale[0] * torch.log(p / scale[4]) + scale[1] * torch.log(t / scale[5]))
        return torch.where(p > scale[4], hi, lo)
    if i_fnc == IP_SCALE_DBL_POW_QUAD:
        tt = t / scale[7] - 1.0
        hi = torch.exp(scale[3] * torch.log(p / scale[6])) * (1.0 + scale[4] * tt + scale[5] * tt**2)
        lo = torch.exp(scale[0] * torch.log(p / scale[6])) * (1.0 + scale[1] * tt + scale[2] * tt**2)
        return torch.where(p > scale[6], hi, lo)
    if i_fnc == IP_SCALE_WENYI:
        return _wenyi(p, t, iex, i_band_1based, wenyi_tables)
    if i_fnc == IP_SCALE_FNC_NULL:
        # scale_absorb returns without defining gas_frac_rescaled: undefined behaviour in SOCRATES
        raise NotImplementedError("k-scaling with a null scaling function is undefined in SOCRATES")
    raise NotImplementedError(f"scaling function {i_fnc}")


# ---------------------------------------------------------------------------
# Water vapour continuum (rescale_continuum.F90)
# ---------------------------------------------------------------------------
def rescale_continuum(p, t, density, water_frac, i_continuum, i_fnc, p_ref, t_ref, scale, l_mixing_ratio=True):
    pwk = torch.pow(p / p_ref, scale[0])
    if i_fnc == IP_SCALE_POWER_LAW:
        amount = pwk * torch.pow(t / t_ref, scale[1])
    elif i_fnc == IP_SCALE_POWER_QUAD:
        x = t / t_ref - 1.0
        amount = pwk * (1.0 + scale[1] * x + scale[2] * x**2)
    else:
        raise NotImplementedError(f"continuum scaling function {i_fnc}")
    if i_continuum == IP_SELF_CONTINUUM:
        molar_density_water = density * water_frac / (REPSILON * MOL_WEIGHT_AIR)
        amount = amount * molar_density_water * water_frac
    elif i_continuum == IP_FRN_CONTINUUM:
        if l_mixing_ratio:
            amount = amount * density * water_frac / MOL_WEIGHT_AIR
        else:
            amount = amount * density * water_frac * (1.0 - water_frac) / MOL_WEIGHT_AIR
    else:
        raise NotImplementedError(f"continuum type {i_continuum}")
    return amount


# ---------------------------------------------------------------------------
# Planck function (planck_flux_band_mod.F90, polynomial form)
# ---------------------------------------------------------------------------
def planck_flux_band(thermal_coeff: torch.Tensor, t_ref_planck: float, temperature: torch.Tensor) -> torch.Tensor:
    """Band-integrated Planck flux. ``thermal_coeff`` is ``(..., n_deg_fit+1)`` broadcast against ``temperature``."""
    t_ratio = temperature / t_ref_planck
    n = thermal_coeff.shape[-1] - 1
    b = thermal_coeff[..., n]
    for j in range(n - 1, -1, -1):
        b = b * t_ratio + thermal_coeff[..., j]
    return b
