"""Reference test cases and I/O for the Fortran SOCRATES reference driver.

The driver (reference/harness/soc_ref) runs the SOCRATES core exactly as Isca's
socrates_calc does.  This module builds Isca-like columns (numpy, float64),
writes them in the driver's binary format, runs it and reads the results back.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOCRATES = ROOT / "reference" / "socrates"
HARNESS = ROOT / "reference" / "harness"
SPECTRA = SOCRATES / "data" / "spectra"
SP_LW_GA7 = SPECTRA / "ga7" / "sp_lw_ga7"
SP_SW_GA7 = SPECTRA / "ga7" / "sp_sw_ga7"
MCC = SOCRATES / "data" / "mcc_profiles" / "one_km"

# Isca constants (src/shared/constants/constants.F90)
GRAV = 9.80
RDGAS = 287.04
KAPPA = 2.0 / 7.0
CP_AIR = RDGAS / KAPPA
RVGAS = 461.50
WTMCO2 = 44.00995
WTMOZONE = 47.99820
GAS_CONSTANT = 8.314

IP_SOLAR = 1
IP_INFRA_RED = 2


@dataclass
class AtmInputs:
    """Inputs to Isca's socrates_calc, shapes (n_profile, n_layer[+1])."""

    p_layer: np.ndarray
    t_layer: np.ndarray
    t_level: np.ndarray  # (np, nl+1), index 0 = TOA
    d_mass: np.ndarray
    density: np.ndarray
    h2o: np.ndarray
    o3: np.ndarray
    co2: np.ndarray
    t_surf: np.ndarray
    coszen: np.ndarray
    solar_irrad: np.ndarray
    albedo: np.ndarray
    emissivity: float
    heat_capacity: np.ndarray
    cld_frac: np.ndarray
    reff: np.ndarray
    mmr_cl: np.ndarray
    # Not passed to Fortran; kept for the Isca-level interface tests.
    extra: dict = field(default_factory=dict)

    @property
    def n_profile(self) -> int:
        return self.p_layer.shape[0]

    @property
    def n_layer(self) -> int:
        return self.p_layer.shape[1]

    def subset(self, idx) -> "AtmInputs":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v = v[idx]
            kw[f.name] = v
        return AtmInputs(**kw)


# ----------------------------------------------------------------------------
# McClatchey profiles
# ----------------------------------------------------------------------------
def _read_cdl_var(path: Path, name: str) -> np.ndarray:
    text = path.read_text()
    data = text.split("data:", 1)[1]
    m = re.search(rf"\b{name}\s*=\s*([^;]*);", data)
    if m is None:
        raise KeyError(f"{name} not in {path}")
    return np.array([float(x) for x in m.group(1).replace("\n", " ").split(",")])


def mcc_profile(name: str = "mls"):
    """Return (p [Pa], T [K], q [kg/kg], o3 mmr, t_surf, p_surf) for a McClatchey atmosphere."""
    p = _read_cdl_var(MCC / f"{name}.t", "plev")
    t = _read_cdl_var(MCC / f"{name}.t", "t")
    q = _read_cdl_var(MCC / f"{name}.q", "q")
    o3 = _read_cdl_var(MCC / f"{name}.o3", "o3")
    tstar = _read_cdl_var(MCC / f"{name}.tstar", "tstar")[0]
    pstar = _read_cdl_var(MCC / f"{name}.pstar", "pstar")[0]
    return p, t, q, o3, tstar, pstar


# ----------------------------------------------------------------------------
# Isca-like column construction
# ----------------------------------------------------------------------------
def isca_half_levels(ps: np.ndarray, n_layer: int, surf_res: float = 0.5, exponent: float = 2.5):
    """Isca 'uneven_sigma'-like half-level pressures, top = 0 Pa. Returns (np, nl+1)."""
    zeta = np.linspace(0.0, 1.0, n_layer + 1)
    sigma = surf_res * zeta + (1.0 - surf_res) * zeta**exponent
    return ps[:, None] * sigma[None, :]


def isca_full_levels(p_half: np.ndarray) -> np.ndarray:
    """Simmons & Burridge full levels as in Isca press_and_geopot (top level factor -1)."""
    nl = p_half.shape[1] - 1
    ln_ph = np.log(np.where(p_half > 0, p_half, 1.0))
    ln_pf = np.empty((p_half.shape[0], nl))
    for k in range(1, nl):
        alpha = 1.0 - p_half[:, k] * (ln_ph[:, k + 1] - ln_ph[:, k]) / (p_half[:, k + 1] - p_half[:, k])
        ln_pf[:, k] = ln_ph[:, k + 1] - alpha
    ln_pf[:, 0] = ln_ph[:, 1] - 1.0
    return np.exp(ln_pf)


def isca_heights(p_half, p_full, t, q):
    """Hydrostatic z_half/z_full (virtual temperature), z_half(surface)=0."""
    tv = t * (1.0 + (RVGAS / RDGAS - 1.0) * q)
    nl = t.shape[1]
    z_half = np.zeros_like(p_half)
    z_full = np.zeros_like(t)
    ln_ph = np.log(np.where(p_half > 0, p_half, 1.0))
    ln_pf = np.log(p_full)
    for k in range(nl - 1, -1, -1):
        z_full[:, k] = z_half[:, k + 1] + RDGAS * tv[:, k] * (ln_ph[:, k + 1] - ln_pf[:, k]) / GRAV
        if k > 0:
            z_half[:, k] = z_half[:, k + 1] + RDGAS * tv[:, k] * (ln_ph[:, k + 1] - ln_ph[:, k]) / GRAV
        else:
            z_half[:, k] = z_full[:, k] + (z_full[:, k] - z_half[:, k + 1])
    return z_half, z_full


def isca_interp_temp(z_full, z_half, t):
    """Port of Isca socrates_interface.F90 interp_temp (0-based, t_half index 0 = TOA)."""
    npf, kend = t.shape
    t_half = np.empty((npf, kend + 1))
    for k in range(1, kend):
        dzk2 = 1.0 / (z_full[:, k - 1] - z_full[:, k])
        dzk = (z_half[:, k] - z_full[:, k]) * dzk2
        dzk1 = (z_full[:, k - 1] - z_half[:, k]) * dzk2
        t_half[:, k] = t[:, k] * dzk1 + t[:, k - 1] * dzk
    t_half[:, 0] = 0.5 * (3 * t[:, 0] - t[:, 1])
    t_half[:, kend] = t[:, kend - 2] + (z_half[:, kend] - z_full[:, kend - 2]) * (t[:, kend - 1] - t[:, kend - 2]) / (
        z_full[:, kend - 1] - z_full[:, kend - 2]
    )
    return t_half


def make_cases(
    n_profile: int = 16,
    n_layer: int = 40,
    seed: int = 0,
    profiles=("mls", "mlw", "sas", "saw", "tro"),
    co2_ppmv_range=(150.0, 1200.0),
    perturb: bool = True,
) -> AtmInputs:
    """Build a batch of Isca-like columns from McClatchey atmospheres with random perturbations."""
    rng = np.random.default_rng(seed)
    ps = np.empty(n_profile)
    t = np.empty((n_profile, n_layer))
    q = np.empty((n_profile, n_layer))
    o3 = np.empty((n_profile, n_layer))
    tsurf = np.empty(n_profile)
    ps[:] = 1.0e5 + (rng.uniform(-5e3, 3e3, n_profile) if perturb else 0.0)
    p_half = isca_half_levels(ps, n_layer)
    p_full = isca_full_levels(p_half)
    for i in range(n_profile):
        name = profiles[i % len(profiles)]
        pm, tm, qm, o3m, tstar, _ = mcc_profile(name)
        lp = np.log(p_full[i])
        t[i] = np.interp(lp, np.log(pm), tm)
        q[i] = np.exp(np.interp(lp, np.log(pm), np.log(qm)))
        o3[i] = np.exp(np.interp(lp, np.log(pm), np.log(o3m)))
        tsurf[i] = tstar
        if perturb:
            t[i] += rng.normal(0, 3.0) + rng.normal(0, 1.0, n_layer)
            q[i] *= np.exp(rng.normal(0, 0.3))
            o3[i] *= np.exp(rng.normal(0, 0.2))
            tsurf[i] += rng.normal(0, 3.0)
    h2o = q / (1.0 - q)
    co2_ppmv = rng.uniform(*co2_ppmv_range, n_profile) if perturb else np.full(n_profile, 300.0)
    co2 = (co2_ppmv * 1e-6 * WTMCO2 / (1000.0 * GAS_CONSTANT / RDGAS))[:, None] * np.ones((1, n_layer))
    z_half, z_full = isca_heights(p_half, p_full, t, q)
    t_level = isca_interp_temp(z_full, z_half, t)
    d_mass = (p_half[:, 1:] - p_half[:, :-1]) / GRAV
    density = p_full / (RDGAS * t)
    heat_capacity = d_mass * CP_AIR
    if perturb:
        coszen = rng.uniform(-0.3, 1.0, n_profile)
        coszen[0] = 1.0
        if n_profile > 1:
            coszen[1] = -0.5  # night column
        albedo = rng.uniform(0.0, 0.8, n_profile)
    else:
        coszen = np.full(n_profile, 0.5)
        albedo = np.full(n_profile, 0.3)
    solar_irrad = np.full(n_profile, 1370.0)
    zeros = np.zeros((n_profile, n_layer))
    return AtmInputs(
        p_layer=p_full,
        t_layer=t,
        t_level=t_level,
        d_mass=d_mass,
        density=density,
        h2o=h2o,
        o3=o3,
        co2=co2,
        t_surf=tsurf,
        coszen=coszen,
        solar_irrad=solar_irrad,
        albedo=albedo,
        emissivity=1.0,
        heat_capacity=heat_capacity,
        cld_frac=zeros.copy(),
        reff=zeros.copy(),
        mmr_cl=zeros.copy(),
        extra=dict(p_half=p_half, z_half=z_half, z_full=z_full, q=q, ps=ps),
    )


# ----------------------------------------------------------------------------
# Binary I/O and running the driver
# ----------------------------------------------------------------------------
def _f(a):
    return np.asfortranarray(np.asarray(a, dtype="<f8")).ravel(order="F").tobytes()


def write_input(path, isolir: int, atm: AtmInputs, do_clouds: bool = False):
    with open(path, "wb") as fh:
        fh.write(np.array([isolir, atm.n_profile, atm.n_layer, int(do_clouds)], dtype="<i4").tobytes())
        for name in ("p_layer", "t_layer", "t_level", "d_mass", "density", "h2o", "o3", "co2",
                     "t_surf", "coszen", "solar_irrad", "albedo"):
            fh.write(_f(getattr(atm, name)))
        fh.write(np.array([atm.emissivity], dtype="<f8").tobytes())
        for name in ("heat_capacity", "cld_frac", "reff", "mmr_cl"):
            fh.write(_f(getattr(atm, name)))


def read_output(path) -> dict:
    raw = open(path, "rb").read()
    npf, nl, nb = np.frombuffer(raw[:12], dtype="<i4")
    data = np.frombuffer(raw[12:], dtype="<f8")
    out, pos = {}, 0

    def take(name, shape):
        nonlocal pos
        n = int(np.prod(shape))
        out[name] = data[pos:pos + n].reshape(shape, order="F").copy()
        pos += n

    for name in ("flux_direct", "flux_down", "flux_up", "flux_direct_clear", "flux_down_clear", "flux_up_clear"):
        take(name, (npf, nl + 1))
    take("heating_rate", (npf, nl))
    for name in ("flux_direct_clear_band", "flux_down_clear_band", "flux_up_clear_band"):
        take(name, (npf, nl + 1, nb))
    take("tot_cloud_cover", (npf,))
    assert pos == data.size, (pos, data.size)
    return out


def format_namelist(values: dict | None) -> str:
    lines = ["&socrates_rad_nml"]
    for k, v in (values or {}).items():
        if isinstance(v, bool):
            s = ".true." if v else ".false."
        elif isinstance(v, str):
            s = f"'{v}'"
        else:
            s = repr(float(v)) if isinstance(v, float) else str(v)
        lines.append(f"  {k} = {s}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def run_reference(spectral_file, isolir: int, atm: AtmInputs, do_clouds: bool = False,
                  namelist: dict | None = None, exe: Path | None = None, capture_trace: bool = False):
    exe = Path(exe or HARNESS / "soc_ref")
    with tempfile.TemporaryDirectory() as td:
        fin, fout, fnml = os.path.join(td, "in.bin"), os.path.join(td, "out.bin"), os.path.join(td, "nml")
        write_input(fin, isolir, atm, do_clouds)
        args = [str(exe), str(spectral_file), fin, fout]
        if namelist is not None:
            Path(fnml).write_text(format_namelist(namelist))
            args.append(fnml)
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"soc_ref failed ({res.returncode}):\n{res.stdout}\n{res.stderr[-4000:]}")
        out = read_output(fout)
        if capture_trace:
            out["trace"] = res.stderr
        return out


# ----------------------------------------------------------------------------
# Record dumps written by reference/harness programs
# ----------------------------------------------------------------------------
def read_dump(path) -> dict:
    """Read a record dump (char(40) name, int32 kind, int32 rank, int32 dims, data)."""
    raw = open(path, "rb").read()
    out, pos = {}, 0
    while pos < len(raw):
        name = raw[pos:pos + 40].decode().strip()
        pos += 40
        kind, rank = np.frombuffer(raw[pos:pos + 8], dtype="<i4")
        pos += 8
        dims = tuple(int(d) for d in np.frombuffer(raw[pos:pos + 4 * rank], dtype="<i4"))
        pos += 4 * rank
        n = int(np.prod(dims))
        dt, size = ("<i4", 4) if kind == 1 else ("<f8", 8)
        out[name] = np.frombuffer(raw[pos:pos + size * n], dtype=dt).reshape(dims, order="F").copy()
        pos += size * n
    return out


def dump_spectrum(spectral_file) -> dict:
    with tempfile.TemporaryDirectory() as td:
        fout = os.path.join(td, "dump.bin")
        res = subprocess.run([str(HARNESS / "dump_spectrum"), str(spectral_file), fout],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr)
        return read_dump(fout)


def run_unit(mode: str, iopt: int, n_profile: int, n_layer: int, arrays) -> dict:
    """Run one kernel through reference/harness/soc_unit (see soc_unit.F90 for argument order)."""
    with tempfile.TemporaryDirectory() as td:
        fin, fout = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(fin, "wb") as fh:
            fh.write(mode.ljust(32).encode())
            fh.write(np.array([n_profile, n_layer, iopt], dtype="<i4").tobytes())
            for a in arrays:
                fh.write(_f(a))
        res = subprocess.run([str(HARNESS / "soc_unit"), fin, fout], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"soc_unit {mode} failed: {res.stderr}")
        return read_dump(fout)


def split_flux_total(ft: np.ndarray):
    """SOCRATES flux_total (..., 2*nl+2) -> (up, down) each (..., nl+1)."""
    return ft[..., 0::2], ft[..., 1::2]
