"""Idealised atmospheric columns, laid out as Isca passes them to SOCRATES.

:func:`make_column` builds temperature, humidity, ozone and CO2 on Isca's stretched sigma
levels, ready for :class:`pysoc.isca.IscaSocrates`::

    col = make_column(t_surf=288.0, co2_ppmv=280.0)
    out = model(**col, albedo=0.3, coszen=0.5, delta_t=0.0)

Scalar parameters give one column of shape ``(L,)``; tensor parameters add leading
dimensions (``make_column(t_surf=torch.linspace(280, 300, 5))`` gives five columns).
Everything is differentiable.  Index 0 is the top of the atmosphere, as in Isca.

The profiles are simple textbook choices, not observations:

* temperature falls at ``lapse_rate`` from ``t_surf`` until it meets the US Standard
  Atmosphere (1976) stratosphere and mesosphere, which sets the tropopause;
* relative humidity follows Manabe & Wetherald (1967), ``rh_surface * (sigma - 0.02) / 0.98``,
  with the specific humidity never increasing with height (a cold trap) and at least ``q_min``;
* ozone has a fixed shape peaking near 15 hPa, scaled to a total column of ``ozone_du``
  Dobson units;
* CO2 is well mixed.
"""

from __future__ import annotations

import torch

from .isca import GRAV, RDGAS, co2_mmr_from_ppmv, ozone_mmr_from_vmr

EPS_WATER = 0.622  # ratio of the molar masses of water and dry air
AVOGADRO = 6.02214076e23
DOBSON_UNIT = 2.6867e20  # molecules per m2
MOLAR_MASS_AIR = 8.314 / RDGAS  # kg/mol, consistent with Isca's constants


def isca_sigma_levels(n_layer=40, surf_res=0.2, scale_heights=11.0, exponent=7.0, dtype=torch.float64):
    """Half-level sigma values of Isca's ``uneven_sigma`` coordinate (top first, top = 0)."""
    zeta = 1.0 - torch.arange(n_layer + 1, dtype=dtype) / n_layer
    z = surf_res * zeta + (1.0 - surf_res) * zeta ** exponent
    sigma = torch.exp(-z * scale_heights)
    sigma[0] = 0.0
    return sigma


def full_level_pressure(p_half):
    """Simmons & Burridge (1981) full-level pressure, as in Isca; a zero top level is allowed."""
    p1, p2 = p_half[..., :-1], p_half[..., 1:]
    top = p1 <= 0.0
    p1_safe = torch.where(top, torch.ones_like(p1), p1)
    ln_pf = torch.where(top, torch.log(p2) - 1.0,
                        (p2 * torch.log(p2) - p1_safe * torch.log(p1_safe)) / (p2 - p1_safe) - 1.0)
    return torch.exp(ln_pf)


def hydrostatic_heights(temp, p_half, p_full):
    """Heights (m) of full and half levels above the surface, integrating the dry hydrostatic equation."""
    L = temp.shape[-1]
    ln_ph = torch.log(torch.clamp(p_half, min=torch.finfo(p_half.dtype).tiny))
    dz_layer = RDGAS * temp[..., 1:] / GRAV * (ln_ph[..., 2:] - ln_ph[..., 1:-1])  # layers below the top
    # z_half from the surface up (index L is the surface)
    z_below = torch.flip(torch.cumsum(torch.flip(dz_layer, [-1]), -1), [-1])  # heights of half levels 1..L-1
    z_half_lower = torch.cat([z_below, torch.zeros_like(temp[..., :1])], -1)  # half levels 1..L
    z_full = z_half_lower + RDGAS * temp / GRAV * (ln_ph[..., 1:] - torch.log(p_full))
    z_top = 2.0 * z_full[..., :1] - z_half_lower[..., :1]  # the top half level (p = 0) is unbounded; unused
    return z_full, torch.cat([z_top, z_half_lower], -1)


def _us_standard_upper(z):
    """US Standard Atmosphere (1976) temperature above 11 km, extended down isothermally."""
    return (216.65 + 1.0e-3 * torch.clamp(z - 20.0e3, 0.0, 12.0e3) + 2.8e-3 * torch.clamp(z - 32.0e3, 0.0, 15.0e3)
            - 2.8e-3 * torch.clamp(z - 51.0e3, 0.0, 20.0e3) - 2.0e-3 * torch.clamp(z - 71.0e3, min=0.0))


def saturation_vapour_pressure(temp):
    """Saturation vapour pressure over liquid water (Pa), Bolton (1980)."""
    tc = temp - 273.15
    return 611.2 * torch.exp(17.67 * tc / (tc + 243.5))


def saturation_specific_humidity(temp, p):
    es = saturation_vapour_pressure(temp)
    return EPS_WATER * es / torch.maximum(p - (1.0 - EPS_WATER) * es, es)


def specific_humidity(temp, p_full, p_surf, rh_surface=0.8, q_min=3.0e-6):
    """Specific humidity (kg/kg) for the Manabe & Wetherald relative-humidity profile, with a cold trap."""
    p_surf = torch.as_tensor(p_surf, dtype=temp.dtype, device=temp.device)
    rh_surface = torch.as_tensor(rh_surface, dtype=temp.dtype, device=temp.device)
    sigma = p_full / p_surf.unsqueeze(-1) if p_surf.dim() else p_full / p_surf
    rh = (rh_surface.unsqueeze(-1) if rh_surface.dim() else rh_surface) * torch.clamp((sigma - 0.02) / 0.98, min=0.0)
    q = rh * saturation_specific_humidity(temp, p_full)
    q = torch.flip(torch.cummin(torch.flip(q, [-1]), -1).values, [-1])  # never increases with height
    return torch.clamp(q, min=q_min)


def ozone_shape(p_full, p_peak=1.5e3, width_above=2.0, width_below=1.0, troposphere=5.0e-3):
    """Unnormalised ozone volume mixing ratio: a skewed Gaussian in log-pressure plus a small floor."""
    x = torch.log(p_full / p_peak)
    width = torch.where(x < 0.0, torch.full_like(x, width_above), torch.full_like(x, width_below))
    return torch.exp(-(x / width) ** 2) + troposphere


def dobson_units(vmr, p_half):
    """Total column (Dobson units) of a gas with volume mixing ratio ``vmr`` on layers bounded by ``p_half``."""
    d_mass = (p_half[..., 1:] - p_half[..., :-1]) / GRAV
    return (vmr * d_mass).sum(-1) / MOLAR_MASS_AIR * AVOGADRO / DOBSON_UNIT


def make_column(t_surf=288.0, lapse_rate=6.5, rh_surface=0.8, co2_ppmv=280.0, ozone_du=300.0, n_layer=40,
                p_surf=1.0e5, q_min=3.0e-6, dtype=torch.float64, device="cpu"):
    """An idealised column, as a dict of the :class:`pysoc.isca.IscaSocrates` inputs it defines.

    ``t_surf`` in K, ``lapse_rate`` in K/km, ``rh_surface`` as a fraction, ``co2_ppmv`` in ppmv,
    ``ozone_du`` in Dobson units, ``p_surf`` in Pa.  Returns ``temp``, ``q``, ``p_full``, ``p_half``,
    ``z_full``, ``z_half``, ``t_surf``, ``ozone`` (mass mixing ratio) and ``co2`` (mass mixing ratio).
    """
    params = [torch.as_tensor(v, dtype=dtype, device=device)
              for v in (t_surf, lapse_rate, rh_surface, co2_ppmv, ozone_du, p_surf)]
    lead = torch.broadcast_shapes(*(v.shape for v in params))
    t_surf, lapse_rate, rh_surface, co2_ppmv, ozone_du, p_surf = (torch.broadcast_to(v, lead) for v in params)
    col = lambda v: v.unsqueeze(-1)  # noqa: E731  (lead,) -> (lead, 1)

    p_half = col(p_surf) * isca_sigma_levels(n_layer, dtype=dtype).to(device)
    p_full = full_level_pressure(p_half)
    # temperature and heights depend on each other: iterate (converges in a few passes)
    z_full = RDGAS * 250.0 / GRAV * torch.log(col(p_surf) / p_full)
    for _ in range(4):
        temp = torch.maximum(col(t_surf) - 1.0e-3 * col(lapse_rate) * z_full, _us_standard_upper(z_full))
        z_full, z_half = hydrostatic_heights(temp, p_half, p_full)
    q = specific_humidity(temp, p_full, p_surf, rh_surface, q_min)
    shape = ozone_shape(p_full)
    vmr = shape * col(ozone_du / dobson_units(shape, p_half))
    ones = torch.ones_like(temp)
    return dict(temp=temp, q=q, p_full=p_full, p_half=p_half, z_full=z_full, z_half=z_half, t_surf=t_surf,
                ozone=ozone_mmr_from_vmr(vmr), co2=co2_mmr_from_ppmv(col(co2_ppmv)) * ones)


def liquid_cloud(column, p_top, p_bottom, lwp=100.0, fraction=1.0, reff=10.0):
    """Isca simple-cloud inputs for a uniform liquid cloud between two pressures (Pa).

    ``lwp`` is the in-cloud liquid water path (g/m2), ``fraction`` the cloud fraction of each layer,
    ``reff`` the droplet effective radius (microns).  Returns ``cf_rad``, ``reff_rad`` (m) and
    ``qcl_rad`` (kg/kg, grid-box mean), to pass to :class:`pysoc.isca.IscaSocrates` with the column.
    """
    p_full, p_half = column["p_full"], column["p_half"]
    like = lambda v: torch.as_tensor(v, dtype=p_full.dtype, device=p_full.device)  # noqa: E731
    in_cloud = (p_full >= like(p_top)) & (p_full <= like(p_bottom))
    d_mass = (p_half[..., 1:] - p_half[..., :-1]) / GRAV
    cloud_mass = torch.where(in_cloud, d_mass, torch.zeros_like(d_mass)).sum(-1, keepdim=True)
    if bool((cloud_mass <= 0.0).any()):
        raise ValueError("no layer has its mid-point between p_top and p_bottom; widen the cloud")
    fraction = torch.broadcast_to(like(fraction), p_full.shape[:-1]).unsqueeze(-1)
    mmr = fraction * 1.0e-3 * like(lwp) / cloud_mass  # grid-box mean mass mixing ratio
    zeros = torch.zeros_like(p_full)
    return dict(cf_rad=torch.where(in_cloud, fraction * torch.ones_like(p_full), zeros),
                reff_rad=torch.where(in_cloud, 1.0e-6 * like(reff) * torch.ones_like(p_full), zeros),
                qcl_rad=torch.where(in_cloud, mmr / (1.0 + mmr), zeros))
