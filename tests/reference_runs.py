"""Shared reference runs (Fortran driver with intermediate dumps) and the matching PyTorch runs."""

import os
import tempfile
from functools import lru_cache

import numpy as np
import torch

from pysoc.core import Atmosphere, SocratesLW, SocratesSW
from pysoc.isca import ISCA_WELL_MIXED_DEFAULTS
from tools.refcases import (HARNESS, IP_INFRA_RED, IP_SOLAR, SP_LW_GA7, SP_SW_GA7, make_cases, read_dump,
                            run_reference)


def atmosphere(atm, dtype=torch.float64, device="cpu"):
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
    gas = {k: T(v) for k, v in ISCA_WELL_MIXED_DEFAULTS.items()}
    gas.update({1: T(atm.h2o), 2: T(atm.co2), 3: T(atm.o3)})
    return Atmosphere(p=T(atm.p_layer), t=T(atm.t_layer), t_level=T(atm.t_level), d_mass=T(atm.d_mass),
                      density=T(atm.density), gas_mmr=gas)


def run_fortran(region, atm, dump=True):
    sp, isolir = (SP_LW_GA7, IP_INFRA_RED) if region == "lw" else (SP_SW_GA7, IP_SOLAR)
    if not dump:
        return run_reference(sp, isolir, atm), None
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dump.bin")
        os.environ["SOC_DUMP"] = path
        try:
            out = run_reference(sp, isolir, atm, exe=HARNESS / "soc_ref_dump")
        finally:
            del os.environ["SOC_DUMP"]
        return out, read_dump(path)


@lru_cache(maxsize=None)
def lw_model():
    return SocratesLW(SP_LW_GA7)


@lru_cache(maxsize=None)
def sw_model():
    return SocratesSW(SP_SW_GA7)


def run_python(region, atm, dtype=torch.float64, device="cpu", return_intermediates=True):
    a = atmosphere(atm, dtype, device)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
    if region == "lw":
        model = lw_model().to(device=device, dtype=dtype)
        return model(a, T(atm.t_surf), T(atm.emissivity), return_intermediates=return_intermediates)
    model = sw_model().to(device=device, dtype=dtype)
    return model(a, T(atm.coszen), T(atm.solar_irrad), T(atm.albedo), return_intermediates=return_intermediates)


def heating_rate(flux_up, flux_down, heat_capacity):
    return (flux_down[..., :-1] - flux_down[..., 1:] + flux_up[..., 1:] - flux_up[..., :-1]) / heat_capacity


@lru_cache(maxsize=None)
def standard_cases(n_profile=24, n_layer=40, seed=0):
    return make_cases(n_profile=n_profile, n_layer=n_layer, seed=seed)
