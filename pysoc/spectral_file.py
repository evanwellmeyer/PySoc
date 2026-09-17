"""Reader for SOCRATES text spectral files (and their ``_k`` extensions).

Mirrors ``radiance_core/read_spectrum.F90`` for the blocks used by Isca's
configuration of SOCRATES.  Arrays are numpy and use 0-based indices:
``index_absorb[band]`` lists 0-based gas indices, etc.  Values that SOCRATES
stores 1-based as *types* (gas type ids, parametrisation ids, scaling function
ids) are kept as in the file.

Unsupported blocks that do not influence Isca's calculation (aerosols, AOD,
spectral variability) are skipped; blocks that would change the physics but
are not implemented raise ``NotImplementedError`` when used by the solver.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

# rad_pcf.F90: number of parameters of each scaling function (index = function id)
N_SCALE_VARIABLE = (0, 2, 3, 4, 0, 0, 6, 8, 10, 0, 0)

IP_SCALE_FNC_NULL = 0
IP_SCALE_POWER_LAW = 1
IP_SCALE_POWER_QUAD = 2
IP_SCALE_DOPPLER_QUAD = 3
IP_SCALE_WENYI = 4
IP_SCALE_SES2 = 5
IP_SCALE_DBL_POW_LAW = 6
IP_SCALE_DBL_POW_QUAD = 7
IP_SCALE_DBL_DOP_QUAD = 8
IP_SCALE_LOOKUP = 9
IP_SCALE_T_LOOKUP = 10

IP_SCALE_NULL = 0
IP_SCALE_BAND = 1
IP_SCALE_TERM = 2

IP_RAYLEIGH_TOTAL = 1
IP_RAYLEIGH_CUSTOM = 2


def _fortran_float(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    # Fortran allows exponents without 'E' (e.g. 1.0-100) and 'D' exponents.
    s = s.replace("D", "E").replace("d", "e")
    m = re.match(r"^([+-]?\d*\.?\d*)([+-]\d+)$", s)
    if m and "e" not in s.lower():
        s = f"{m.group(1)}E{m.group(2)}"
    return float(s)


def _fortran_int(s: str) -> int:
    s = s.strip()
    return int(s) if s else 0


def _fixed(line: str, start: int, width: int) -> str:
    return line[start:start + width] if len(line) > start else ""


@dataclass
class GasTerm:
    """k-distribution data for one gas in one band (block 5)."""

    n_k: int
    i_scale_k: int
    i_scale_fnc: int
    k: np.ndarray  # (n_k,)
    w: np.ndarray  # (n_k,)
    i_scat: np.ndarray  # (n_k,) int
    scale: np.ndarray  # (n_k, n_scale_variable)
    p_ref: float = 0.0
    t_ref: float = 0.0
    num_ref_p: int = 0
    num_ref_t: int = 0
    k_lookup: np.ndarray | None = None  # (n_k, n_pre, n_tmp)


@dataclass
class CloudParam:
    """Droplet (block 10) or ice (block 12) parametrisation."""

    i_parm: int
    n_phf: int
    min_dim: float
    max_dim: float
    parm_list: np.ndarray  # (n_band, n_parameter)


@dataclass
class SpectralFile:
    path: str
    blocks_present: set = field(default_factory=set)
    n_band: int = 0
    n_absorb: int = 0
    type_absorb: np.ndarray = None  # (n_absorb,) gas type ids (gas_list_pcf)
    n_aerosol: int = 0
    # block 1
    wavelength_short: np.ndarray = None
    wavelength_long: np.ndarray = None
    # block 2
    solar_flux_band: np.ndarray = None
    weight_blue: np.ndarray = None
    # block 3
    i_rayleigh_scheme: int = 0
    rayleigh_coeff: np.ndarray = None
    # block 4
    n_band_absorb: np.ndarray = None  # (n_band,)
    index_absorb: list = None  # list over bands of 0-based gas indices
    i_overlap: np.ndarray = None
    # block 5
    index_sb: np.ndarray = None
    gas: dict = field(default_factory=dict)  # (band, gas) -> GasTerm, 0-based
    p_lookup: np.ndarray = None  # (n_pre,) natural log of pressure [Pa]
    t_lookup: np.ndarray = None  # (n_pre, n_tmp)
    # block 6
    l_planck_tbl: bool = False
    n_deg_fit: int = 0
    t_ref_planck: float = 0.0
    thermal_coeff: np.ndarray = None  # (n_band, n_deg_fit+1)
    # blocks 8/9
    n_band_continuum: np.ndarray = None
    index_continuum: list = None  # per band, continuum type ids (1 self, 2 foreign, ...)
    index_water: int = -1  # 0-based gas index of water vapour
    continuum: dict = field(default_factory=dict)  # (band, cont_type) -> dict
    # blocks 10/12
    drop: dict = field(default_factory=dict)  # drop type -> CloudParam
    ice: dict = field(default_factory=dict)  # ice type -> CloudParam
    # block 14
    n_band_exclude: np.ndarray = None
    index_exclude: list = None  # per band, 0-based excluded band indices

    @property
    def max_k_terms(self) -> int:
        return max((g.n_k for g in self.gas.values()), default=0)


def read_spectral_file(path: str | os.PathLike) -> SpectralFile:
    path = str(path)
    with open(path) as fh:
        lines = fh.read().splitlines()
    sp = SpectralFile(path=path)

    kpath = path + "_k"
    klines = None
    if os.path.exists(kpath):
        with open(kpath) as fh:
            klines = fh.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("*BLOCK"):
            m = re.match(r"\*BLOCK: TYPE =\s*(\d+): SUBTYPE =\s*(\d+): VERSION =\s*(\d+)", line, re.I)
            if m is None:
                raise ValueError(f"bad block header in {path}: {line!r}")
            btype, bsub, bver = (int(x) for x in m.groups())
            j = i + 1
            while not lines[j].startswith("*END"):
                j += 1
            body = lines[i + 1:j]
            _read_block(sp, btype, bsub, bver, body, klines)
            sp.blocks_present.add(btype)
            i = j + 1
        else:
            i += 1

    if sp.n_band_exclude is None:
        sp.n_band_exclude = np.zeros(sp.n_band, dtype=int)
        sp.index_exclude = [[] for _ in range(sp.n_band)]
    # read_spectrum: index of water is the gas whose type is ip_h2o
    if sp.type_absorb is not None:
        for g, t in enumerate(sp.type_absorb):
            if t == 1:
                sp.index_water = g
    return sp


# ---------------------------------------------------------------------------
def _read_block(sp: SpectralFile, btype, bsub, bver, body, klines):
    key = (btype, bsub, bver)
    if key == (0, 0, 2):
        _block_0_0_2(sp, body)
    elif key == (1, 0, 0):
        _block_1(sp, body)
    elif key in ((2, 0, 0), (2, 0, 1)):
        _block_2(sp, body, bver)
    elif key == (3, 0, 0):
        _block_3(sp, body)
    elif key in ((4, 0, 0), (4, 0, 1)):
        _block_4(sp, body, bver)
    elif key == (5, 0, 1):
        _block_5_0_1(sp, body, klines)
    elif key == (6, 0, 1):
        _block_6_0_1(sp, body)
    elif key == (8, 0, 0):
        _block_8(sp, body)
    elif key == (9, 0, 0):
        _block_9(sp, body)
    elif key == (10, 0, 2):
        _block_cloud(sp.drop, sp.n_band, body, read_star=False)
    elif key == (12, 0, 2):
        _block_cloud(sp.ice, sp.n_band, body, read_star=True)
    elif key == (14, 0, 0):
        _block_14(sp, body)
    elif btype in (11, 15, 17):
        # aerosols, aerosol optical depths, spectral variability: not used by Isca
        pass
    else:
        raise NotImplementedError(f"spectral block {btype}.{bsub}.{bver} is not supported ({sp.path})")


def _block_0_0_2(sp, body):
    it = iter(range(len(body)))
    for n in it:
        line = body[n]
        desc, _, val = line.rpartition("=") if "=" in line else (line.strip(), "", "")
        desc = desc.strip() if "=" in line else line.strip()
        if desc in ("Number of spectral bands", "nd_band"):
            sp.n_band = int(val)
        elif desc in ("Total number of gaseous absorbers", "nd_species"):
            sp.n_absorb = int(val)
        elif desc in ("Total number of aerosols", "nd_aerosol_species"):
            sp.n_aerosol = int(val)
        elif desc == "List of indexing numbers and absorbers.":
            types = []
            for g in range(sp.n_absorb):
                types.append(int(body[n + 2 + g][12:17]))
            sp.type_absorb = np.array(types, dtype=int)
        elif desc in ("Total number of generalised continua", "nd_cont"):
            if int(val) > 0:
                raise NotImplementedError("generalised continua are not supported")
        elif desc in ("Total number of photolysis pathways", "nd_pathway"):
            if int(val) > 0:
                raise NotImplementedError("photolysis pathways are not supported")


def _block_1(sp, body):
    ws, wl = [], []
    for b in range(sp.n_band):
        toks = body[3 + b].split()
        ws.append(_fortran_float(toks[1]))
        wl.append(_fortran_float(toks[2]))
    sp.wavelength_short = np.array(ws)
    sp.wavelength_long = np.array(wl)


def _block_2(sp, body, bver):
    flux, blue = [], []
    for b in range(sp.n_band):
        toks = body[2 + b].split()
        flux.append(_fortran_float(toks[1]))
        if bver == 1:
            blue.append(_fortran_float(toks[2]))
    sp.solar_flux_band = np.array(flux)
    if bver == 1:
        sp.weight_blue = np.array(blue)


def _block_3(sp, body):
    sp.i_rayleigh_scheme = IP_RAYLEIGH_TOTAL
    sp.rayleigh_coeff = np.array([_fortran_float(body[3 + b].split()[1]) for b in range(sp.n_band)])


def _block_4(sp, body, bver):
    n = 5
    sp.n_band_absorb = np.zeros(sp.n_band, dtype=int)
    sp.i_overlap = np.zeros(sp.n_band, dtype=int)
    sp.index_absorb = []
    for b in range(sp.n_band):
        toks = body[n].split()
        n += 1
        sp.n_band_absorb[b] = int(toks[1])
        if bver == 1:
            sp.i_overlap[b] = int(toks[2])
        idx = []
        while len(idx) < sp.n_band_absorb[b]:
            idx += [int(t) for t in body[n].split()]
            n += 1
        sp.index_absorb.append([g - 1 for g in idx])


def _read_fixed_e(line: str, width: int, count: int, offset: int = 0) -> list:
    return [_fortran_float(_fixed(line, offset + width * c, width)) for c in range(count)]


def _block_5_0_1(sp, body, klines):
    n = 0
    sp.index_sb = np.zeros(sp.n_absorb, dtype=int)
    while True:
        line = body[n].strip()
        n += 1
        if line == "Self-broadened indexing numbers of all absorbers.":
            vals = []
            while len(vals) < sp.n_absorb:
                vals += [int(t) for t in body[n + 1 + (len(vals) // 6)].split()]
            sp.index_sb = np.array(vals[: sp.n_absorb])
            n += 1 + (sp.n_absorb + 5) // 6
        elif line.startswith("Band        Gas, Number of k-terms,"):
            break
    if np.any(sp.index_sb > 0):
        raise NotImplementedError("self-broadened k-tables are not supported")
    n += 2  # skip two header lines
    order = []
    l_lookup = False
    for b in range(sp.n_band):
        for _ in range(sp.n_band_absorb[b]):
            toks = body[n].split()
            n += 1
            ib, ig, nk, iscale, ifnc = (int(t) for t in toks[:5])
            if ifnc not in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
                raise ValueError(f"illegal scaling function {ifnc}")
            gt = GasTerm(n_k=nk, i_scale_k=iscale, i_scale_fnc=ifnc,
                         k=np.zeros(nk), w=np.zeros(nk), i_scat=np.zeros(nk, dtype=int),
                         scale=np.zeros((nk, N_SCALE_VARIABLE[ifnc])))
            if ifnc == IP_SCALE_LOOKUP:
                t2 = body[n].split()
                gt.num_ref_p, gt.num_ref_t = int(t2[0]), int(t2[1])
                l_lookup = True
                n += 1
            elif ifnc == IP_SCALE_T_LOOKUP:
                raise NotImplementedError("temperature-only k-tables are not supported")
            else:
                ln = body[n]
                gt.p_ref = _fortran_float(_fixed(ln, 6, 16))
                gt.t_ref = _fortran_float(_fixed(ln, 28, 16))
                n += 1
            nsv = N_SCALE_VARIABLE[ifnc]
            for kk in range(nk):
                # '(2(3x, 1pe16.9),i3,:,(t42, 1pe16.9,3x,1pe16.9))'
                ln = body[n]
                n += 1
                gt.k[kk] = _fortran_float(_fixed(ln, 3, 16))
                gt.w[kk] = _fortran_float(_fixed(ln, 22, 16))
                gt.i_scat[kk] = _fortran_int(_fixed(ln, 38, 3))
                sv = []
                while len(sv) < nsv:
                    sv.append(_fortran_float(_fixed(ln, 41, 16)))
                    if len(sv) < nsv:
                        sv.append(_fortran_float(_fixed(ln, 60, 16)))
                    if len(sv) < nsv:
                        ln = body[n]
                        n += 1
                gt.scale[kk, :] = sv[:nsv]
            sp.gas[(ib - 1, ig - 1)] = gt
            order.append((ib - 1, ig - 1))

    if l_lookup:
        if klines is None:
            raise FileNotFoundError(f"k-table file {sp.path}_k is required")
        _read_k_tables(sp, klines, order)


def _read_k_tables(sp, klines, order):
    start = next(i for i, l in enumerate(klines) if l.startswith("*BLOCK") and l[8:15] == "k-table")
    m = re.search(r"(\d+)\s+pressures,\s+(\d+)\s+temperatures", klines[start + 2])
    n_pre, n_tmp = int(m.group(1)), int(m.group(2))
    n = start + 3
    p = np.zeros(n_pre)
    t = np.zeros((n_pre, n_tmp))
    for ip in range(n_pre):
        vals = _read_fixed_e(klines[n], 13, 1 + n_tmp)
        p[ip] = np.log(vals[0])
        t[ip] = vals[1:]
        n += 1
    sp.p_lookup, sp.t_lookup = p, t
    for (b, g) in order:
        gt = sp.gas[(b, g)]
        if gt.i_scale_fnc != IP_SCALE_LOOKUP:
            continue
        if gt.num_ref_p != n_pre or gt.num_ref_t != n_tmp:
            raise ValueError("P/T lookup table size is not consistent")
        # skip blank line + "Band: ..., gas: ..., k-terms: ..." header
        hdr = klines[n + 1]
        mh = re.search(r"Band:\s*(\d+), gas:\s*(\d+), k-terms:\s*(\d+)", hdr)
        if mh is None or (int(mh.group(1)) - 1, int(mh.group(2)) - 1) != (b, g):
            raise ValueError(f"unexpected k-table header {hdr!r} (expected band {b+1} gas {g+1})")
        n += 2
        kl = np.zeros((gt.n_k, n_pre, n_tmp))
        for kk in range(gt.n_k):
            for ip in range(n_pre):
                kl[kk, ip] = _read_fixed_e(klines[n], 13, n_tmp)
                n += 1
        gt.k_lookup = kl


def _block_6_0_1(sp, body):
    kind = body[1][14:24]
    if kind.startswith("table"):
        raise NotImplementedError("tabulated Planck functions are not supported")
    sp.l_planck_tbl = False
    sp.n_deg_fit = int(body[2][24:29])
    sp.t_ref_planck = _fortran_float(body[2][55:71])
    sp.thermal_coeff = np.zeros((sp.n_band, sp.n_deg_fit + 1))
    n = 5
    for b in range(sp.n_band):
        vals = []
        first = True
        while len(vals) < sp.n_deg_fit + 1:
            ln = body[n]
            n += 1
            if first:
                first = False
            vals += [_fortran_float(t) for t in ln[12:].split()]
        sp.thermal_coeff[b] = vals[: sp.n_deg_fit + 1]


def _block_8(sp, body):
    n = 5
    sp.n_band_continuum = np.zeros(sp.n_band, dtype=int)
    sp.index_continuum = []
    for b in range(sp.n_band):
        toks = body[n].split()
        n += 1
        sp.n_band_continuum[b] = int(toks[1])
        idx = []
        while len(idx) < sp.n_band_continuum[b]:
            idx += [int(t) for t in body[n].split()]
            n += 1
        sp.index_continuum.append(idx)
    m = re.search(r"Index of water =\s*(\d+)", body[n + 1])
    # read_spectrum later resets index_water from the gas types; keep for reference
    sp._cont_index_water_block8 = int(m.group(1)) if m else 0


def _block_9(sp, body):
    n = 3
    for b in range(sp.n_band):
        for _ in range(sp.n_band_continuum[b]):
            toks = body[n].split()
            n += 1
            ib, ic, ifnc = int(toks[0]), int(toks[1]), int(toks[2])
            pt = body[n].split()
            n += 1
            nsv = N_SCALE_VARIABLE[ifnc]
            vals = [_fortran_float(t) for t in body[n].split()]
            n += 1
            while len(vals) < 1 + nsv:
                vals += [_fortran_float(t) for t in body[n].split()]
                n += 1
            # Stored by continuum *type* as read_spectrum does; radiance_calc
            # looks entries up by position within the band (see optics).
            sp.continuum[(ib - 1, ic)] = dict(
                i_scale_fnc=ifnc, p_ref=_fortran_float(pt[0]), t_ref=_fortran_float(pt[1]),
                k=vals[0], scale=np.array(vals[1:1 + nsv]))


def _block_cloud(store, n_band, body, read_star):
    i_type = int(body[1].split("=")[1])
    m = re.search(r"=\s*(\d+):\s*[Nn]umber of parameters =\s*(\d+)", body[2])
    i_parm, n_par = int(m.group(1)), int(m.group(2))
    n_phf = int(body[3].split("=")[1])
    rng = body[4].split("=")[1].split("--")
    parm = np.zeros((n_band, n_par))
    n = 5
    for b in range(n_band):
        n += 1  # "Band = ... Fitting parameters:"
        vals = []
        while len(vals) < n_par:
            vals += [_fortran_float(t) for t in body[n].split()]
            n += 1
        parm[b] = vals[:n_par]
    store[i_type] = CloudParam(i_parm=i_parm, n_phf=n_phf, min_dim=_fortran_float(rng[0]),
                               max_dim=_fortran_float(rng[1]), parm_list=parm)


def _block_14(sp, body):
    n = 3
    sp.n_band_exclude = np.zeros(sp.n_band, dtype=int)
    sp.index_exclude = []
    for b in range(sp.n_band):
        toks = body[n].split()
        n += 1
        sp.n_band_exclude[b] = int(toks[1])
        idx = []
        while len(idx) < sp.n_band_exclude[b]:
            idx += [int(t) for t in body[n].split()]
            n += 1
        sp.index_exclude.append([x - 1 for x in idx])
