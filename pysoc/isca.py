"""Isca's interface to SOCRATES (socrates_interface.F90 / run_socrates), in PyTorch.

:class:`IscaSocrates` takes Isca model fields (full/half-level pressures and heights,
temperature, specific humidity, surface temperature and albedo, cosine of the solar
zenith angle) and returns the radiative heating rates and the diagnostics Isca writes
(``soc_*``).  It reproduces, for clear skies:

* ``q -> q/(1-q)`` water vapour mass mixing ratio, CO2 from ``co2_ppmv``;
* half-level temperatures by linear interpolation in height (``interp_temp``);
* layer mass, density and heat capacity with Isca's constants;
* the longwave call, the update ``T <- T + tdt_lw * delta_t``, then the shortwave call;
* optionally the simple-cloud fields (``cf_rad``, ``reff_rad`` in metres as passed to
  ``run_socrates`` -- idealized_moist_phys multiplies the micron values by 1e-6 --, and
  ``qcl_rad``, converted to a mass mixing ratio ``qcl/(1-qcl)``).

Astronomy (``diurnal_solar``), radiation time-stepping and ozone/CO2 file interpolation are
not part of this module: pass ``coszen``, ``rrsun``, ``ozone`` and ``co2`` in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .clouds import LiquidCloud
from .core import Atmosphere, SocratesLW, SocratesSW

# gas_list_pcf.F90 identifiers
IP_H2O, IP_CO2, IP_O3, IP_N2O, IP_CO, IP_CH4, IP_O2 = 1, 2, 3, 4, 5, 6, 7
IP_SO2, IP_N2, IP_CFC11, IP_CFC12, IP_CFC113, IP_HCFC22, IP_HFC134A = 9, 13, 14, 15, 16, 17, 19
IP_OCS = 25
GAS_NAMES = {IP_H2O: "H2O", IP_CO2: "CO2", IP_O3: "O3", IP_N2O: "N2O", IP_CO: "CO", IP_CH4: "CH4", IP_O2: "O2",
             IP_SO2: "SO2", IP_N2: "N2", IP_CFC11: "CFC-11", IP_CFC12: "CFC-12", IP_CFC113: "CFC-113",
             IP_HCFC22: "HCFC-22", IP_HFC134A: "HFC-134a", IP_OCS: "OCS"}

# socrates_config_mod.f90 defaults for the well-mixed gases (kg/kg).  In Isca these
# are applied by set_atm whenever the gas is present in the spectral file,
# regardless of the inc_* namelist switches.
ISCA_WELL_MIXED_DEFAULTS = {
    IP_CO: 0.0,
    IP_N2O: 4.945e-07,
    IP_N2: 0.0,
    IP_CH4: 1.006e-06,
    IP_O2: 0.2314,
    IP_SO2: 0.0,
    IP_CFC11: 1.110e-09,
    IP_CFC12: 2.187e-09,
    IP_CFC113: 4.826e-10,
    IP_HCFC22: 6.866e-10,
    IP_HFC134A: 2.536e-10,
}

# Isca constants_mod
GRAV = 9.80
RDGAS = 287.04
KAPPA = 2.0 / 7.0
CP_AIR = RDGAS / KAPPA
WTMCO2 = 44.00995
WTMOZONE = 47.99820
GAS_CONSTANT = 8.314
MOLAR_MASS_DRY_AIR_G = 1000.0 * GAS_CONSTANT / RDGAS  # as written in run_socrates


@dataclass
class IscaSocratesConfig:
    """Subset of ``socrates_rad_nml`` relevant to a radiation call."""

    stellar_constant: float = 1368.22
    input_planet_emissivity: float = 1.0
    co2_ppmv: float = 300.0
    input_co2_mmr: bool = False
    account_for_effect_of_water: bool = True
    account_for_effect_of_ozone: bool = True
    well_mixed: dict = field(default_factory=lambda: dict(ISCA_WELL_MIXED_DEFAULTS))
    # Isca's inc_h2o, inc_co2, ... switches have no effect (compress_spectrum is never called).
    # Set this to a set of gas ids to *actually* exclude those gases.
    exclude_gases: frozenset = frozenset()
    use_pressure_interp_for_half_levels: bool = False


def co2_mmr_from_ppmv(co2_ppmv, input_co2_mmr=False):
    if input_co2_mmr:
        return co2_ppmv * 1.0e-6
    return co2_ppmv * 1.0e-6 * WTMCO2 / (1000.0 * GAS_CONSTANT / RDGAS)


def ozone_mmr_from_vmr(ozone_vmr):
    return ozone_vmr * WTMOZONE / (1000.0 * GAS_CONSTANT / RDGAS)


def interp_temp(z_full: torch.Tensor, z_half: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """Half-level temperature by linear interpolation in height (Isca ``interp_temp``).

    Shapes (..., L), (..., L+1), (..., L) -> (..., L+1); index 0 is the top.
    """
    dzk2 = 1.0 / (z_full[..., :-1] - z_full[..., 1:])
    dzk = (z_half[..., 1:-1] - z_full[..., 1:]) * dzk2
    dzk1 = (z_full[..., :-1] - z_half[..., 1:-1]) * dzk2
    interior = temp[..., 1:] * dzk1 + temp[..., :-1] * dzk
    top = 0.5 * (3 * temp[..., 0] - temp[..., 1])
    bottom = temp[..., -2] + (z_half[..., -1] - z_full[..., -2]) * (temp[..., -1] - temp[..., -2]) / (
        z_full[..., -1] - z_full[..., -2])
    return torch.cat([top.unsqueeze(-1), interior, bottom.unsqueeze(-1)], -1)


class IscaSocrates(nn.Module):
    """Clear-sky SOCRATES radiation as called by Isca's ``run_socrates``."""

    def __init__(self, lw_spectral_file, sw_spectral_file, config: IscaSocratesConfig | None = None,
                 numerics="auto", compile=False):
        super().__init__()
        self.config = config or IscaSocratesConfig()
        self.lw = SocratesLW(lw_spectral_file, numerics=numerics, compile=compile)
        self.sw = SocratesSW(sw_spectral_file, numerics=numerics, compile=compile)

    def _gases(self, h2o, o3, co2, like):
        cfg = self.config
        gas = {k: like.new_tensor(v) for k, v in cfg.well_mixed.items()}
        gas[IP_H2O] = h2o
        gas[IP_O3] = o3
        gas[IP_CO2] = co2
        for g in cfg.exclude_gases:
            gas[g] = torch.zeros_like(like)
        return gas

    def _column(self, temp, p_full, p_half, z_full, z_half, gas):
        if self.config.use_pressure_interp_for_half_levels:
            # the Fortran branch indexes temperature at level 0 (out of bounds); not reproducible
            raise NotImplementedError("use_pressure_interp_for_half_levels is not supported")
        t_level = interp_temp(z_full, z_half, temp)
        d_mass = (p_half[..., 1:] - p_half[..., :-1]) / GRAV
        density = p_full / (RDGAS * temp)
        atm = Atmosphere(p=p_full, t=temp, t_level=t_level, d_mass=d_mass, density=density, gas_mmr=gas)
        return atm, d_mass * CP_AIR

    def forward(self, temp, q, p_full, p_half, z_full, z_half, t_surf, albedo, coszen, delta_t,
                rrsun=1.0, ozone=None, co2=None, cf_rad=None, reff_rad=None, qcl_rad=None,
                return_intermediates=False):
        """All fields (..., L) or (..., L+1) for half levels, surface fields (...).

        ``t_surf``, ``albedo``, ``coszen``, ``ozone`` and ``co2`` may also be anything that broadcasts
        to those shapes (e.g. a Python float).

        ``ozone`` and ``co2`` are mass mixing ratios; ``co2=None`` uses ``config.co2_ppmv``.
        Returns a dict of tensors with the leading shape restored; heating rates in K/s,
        fluxes in W/m2 (``*_up``/``*_down`` on half levels, index 0 at the top; ``*_band`` fluxes
        have a trailing band dimension).
        ``return_intermediates=True`` adds ``lw_intermediates`` / ``sw_intermediates``, the
        :class:`SocratesLW` / :class:`SocratesSW` intermediates with the columns flattened.
        """
        cfg = self.config
        lead = temp.shape[:-1]
        L = temp.shape[-1]
        flat = lambda x, n: x.reshape(-1, n)  # noqa: E731
        temp, q, p_full, z_full = (flat(x, L) for x in (temp, q, p_full, z_full))
        p_half, z_half = flat(p_half, L + 1), flat(z_half, L + 1)
        as_lead = lambda x: torch.broadcast_to(torch.as_tensor(x, dtype=temp.dtype, device=temp.device), lead)  # noqa: E731
        t_surf, albedo, coszen = (as_lead(x).reshape(-1) for x in (t_surf, albedo, coszen))
        P = temp.shape[0]

        h2o = q / (1.0 - q) if cfg.account_for_effect_of_water else torch.zeros_like(q)
        if ozone is None or not cfg.account_for_effect_of_ozone:
            o3 = torch.zeros_like(q)
        else:
            o3 = torch.broadcast_to(torch.as_tensor(ozone, dtype=q.dtype, device=q.device), lead + (L,)).reshape(P, L)
        if co2 is None:
            co2 = torch.full_like(q, co2_mmr_from_ppmv(cfg.co2_ppmv, cfg.input_co2_mmr))
        else:
            co2 = torch.broadcast_to(torch.as_tensor(co2, dtype=q.dtype, device=q.device), lead + (L,)).reshape(P, L)
        gas = self._gases(h2o, o3, co2, q)
        cloud = None
        if cf_rad is not None:
            if reff_rad is None or qcl_rad is None:
                raise ValueError("cf_rad, reff_rad and qcl_rad must be given together")
            qcl = flat(qcl_rad, L)
            cloud = LiquidCloud(fraction=flat(cf_rad, L), mmr=qcl / (1.0 - qcl), reff=flat(reff_rad, L))
        rrsun = torch.as_tensor(rrsun, dtype=temp.dtype, device=temp.device)
        solar_irrad = torch.broadcast_to(cfg.stellar_constant * rrsun, lead).reshape(P)

        atm_lw, heat_capacity = self._column(temp, p_full, p_half, z_full, z_half, gas)
        lw = self.lw(atm_lw, t_surf, cfg.input_planet_emissivity, cloud=cloud,
                     return_intermediates=return_intermediates)
        tdt_lw = lw["flux_divergence"] / heat_capacity

        temp_sw = temp + tdt_lw * delta_t
        atm_sw, heat_capacity_sw = self._column(temp_sw, p_full, p_half, z_full, z_half, gas)
        sw = self.sw(atm_sw, coszen, solar_irrad, albedo, cloud=cloud, return_intermediates=return_intermediates)
        tdt_sw = sw["flux_divergence"] / heat_capacity_sw

        out = dict(
            tdt_lw=tdt_lw, tdt_sw=tdt_sw, tdt_rad=tdt_lw + tdt_sw,
            flux_lw_up=lw["flux_up"], flux_lw_down=lw["flux_down"],
            flux_sw_up=sw["flux_up"], flux_sw_down=sw["flux_down"], flux_sw_direct=sw["flux_direct"],
            t_half=atm_lw.t_level, co2=co2, ozone=o3,
            flux_lw_up_band=lw["flux_up_band"], flux_lw_down_band=lw["flux_down_band"],
            flux_sw_up_band=sw["flux_up_band"], flux_sw_down_band=sw["flux_down_band"],
        )
        # Isca diagnostics (socrates_interface.F90 / run_socrates)
        out["soc_flux_lw"] = lw["flux_up"] - lw["flux_down"]
        out["soc_flux_sw"] = sw["flux_up"] - sw["flux_down"]
        out["soc_surf_flux_lw"] = lw["flux_up"][:, -1] - lw["flux_down"][:, -1]
        out["soc_surf_flux_lw_down"] = lw["flux_down"][:, -1]
        out["soc_surf_flux_sw"] = sw["flux_down"][:, -1] - sw["flux_up"][:, -1]
        out["soc_surf_flux_sw_down"] = sw["flux_down"][:, -1]
        out["soc_olr"] = lw["flux_up"][:, 0]
        out["soc_toa_sw"] = sw["flux_down"][:, 0] - sw["flux_up"][:, 0]
        out["soc_toa_sw_down"] = sw["flux_down"][:, 0]
        out["soc_toa_sw_up"] = sw["flux_up"][:, 0]
        out["soc_spectral_olr"] = lw["flux_up_clear_band"][:, 0, :]  # socrates_calc: flux_up_clear_band
        out["flux_lw_up_clear"], out["flux_lw_down_clear"] = lw["flux_up_clear"], lw["flux_down_clear"]
        out["flux_sw_up_clear"], out["flux_sw_down_clear"] = sw["flux_up_clear"], sw["flux_down_clear"]
        out["tdt_lw_clear"] = lw["flux_divergence_clear"] / heat_capacity
        out["tdt_sw_clear"] = sw["flux_divergence_clear"] / heat_capacity_sw
        out["soc_olr_clr"] = lw["flux_up_clear"][:, 0]
        out["soc_toa_sw_clr"] = sw["flux_down_clear"][:, 0] - sw["flux_up_clear"][:, 0]
        out["soc_toa_sw_up_clr"] = sw["flux_up_clear"][:, 0]
        out["soc_flux_lw_clr"] = lw["flux_up_clear"] - lw["flux_down_clear"]
        out["soc_flux_sw_clr"] = sw["flux_up_clear"] - sw["flux_down_clear"]
        out["soc_surf_flux_lw_clr"] = lw["flux_up_clear"][:, -1] - lw["flux_down_clear"][:, -1]
        out["soc_surf_flux_lw_down_clr"] = lw["flux_down_clear"][:, -1]
        out["soc_surf_flux_sw_clr"] = sw["flux_down_clear"][:, -1] - sw["flux_up_clear"][:, -1]
        out["soc_surf_flux_sw_down_clr"] = sw["flux_down_clear"][:, -1]
        if cloud is not None:
            out["soc_tot_cloud_cover"] = lw["tot_cloud_cover"]

        def restore(v):
            if v.shape[0] != P:
                return v
            return v.reshape(lead + v.shape[1:])

        out = {k: restore(v) for k, v in out.items()}
        if return_intermediates:
            out["lw_intermediates"], out["sw_intermediates"] = lw["intermediates"], sw["intermediates"]
        return out
