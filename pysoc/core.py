"""SOCRATES core (``radiance_calc`` with ``solve_band_k_eqv_scl``) for Isca's clear-sky configuration.

The Fortran loops band -> k-term -> column.  Here every (band, k-term) pair of
the major gas in each band is a "g-point" and all g-points are solved at once,
batched along a leading dimension.  Minor-gas k-terms needed for the
equivalent-extinction scaling are batched the same way.

Shapes: ``P`` profiles, ``L`` layers (index 0 at the top), ``B`` bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import clouds as cl
from . import twostream as ts
from .mathutil import expm1
from .optics import (interp_k_lookup, lookup_weights, planck_flux_band, rescale_continuum, scale_absorb,
                     wenyi_tables_numpy)
from .spectral_file import (
    IP_RAYLEIGH_TOTAL,
    IP_SCALE_BAND,
    IP_SCALE_LOOKUP,
    IP_SCALE_NULL,
    IP_SCALE_SES2,
    IP_SCALE_T_LOOKUP,
    IP_SCALE_TERM,
    IP_SCALE_WENYI,
    SpectralFile,
    read_spectral_file,
)

IP_SOLAR = 1
IP_INFRA_RED = 2
IP_OVERLAP_K_EQV_SCL = 4
IP_OVERLAP_HYBRID = 0


@dataclass
class Atmosphere:
    """Column state in SOCRATES units (all tensors on the same device/dtype).

    p, t, d_mass, density: (P, L); t_level: (P, L+1);
    gas_mmr: gas type id (gas_list_pcf, e.g. 1=H2O, 2=CO2, 3=O3) -> mass mixing
    ratio broadcastable to (P, L).  Gases in the spectral file but missing here
    get zero, as in Isca's set_atm ``CASE DEFAULT``.
    """

    p: torch.Tensor
    t: torch.Tensor
    t_level: torch.Tensor
    d_mass: torch.Tensor
    density: torch.Tensor
    gas_mmr: dict = field(default_factory=dict)


class Spectrum(nn.Module):
    """Spectral file data as tensors (buffers move with ``.to(device, dtype)``)."""

    def __init__(self, sp: SpectralFile | str, dtype=torch.float64):
        super().__init__()
        if not isinstance(sp, SpectralFile):
            sp = read_spectral_file(sp)
        self.sp = sp
        self.n_band = sp.n_band
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype)  # noqa: E731

        if sp.p_lookup is not None:
            self.register_buffer("p_lookup", t(sp.p_lookup))
            self.register_buffer("t_lookup", t(sp.t_lookup))
        else:
            self.p_lookup = None
        if sp.thermal_coeff is not None:
            self.register_buffer("thermal_coeff", t(sp.thermal_coeff))
        if sp.solar_flux_band is not None:
            self.register_buffer("solar_flux_band", t(sp.solar_flux_band))
        if sp.rayleigh_coeff is not None:
            self.register_buffer("rayleigh_coeff", t(sp.rayleigh_coeff))

        # Per (band, gas) k-term data.
        self.gas_entries = []  # list over bands of lists of dict(meta)
        for b in range(sp.n_band):
            if sp.i_overlap[b] not in (IP_OVERLAP_HYBRID, IP_OVERLAP_K_EQV_SCL):
                pass  # Isca forces k_eqv_scl unless a gas uses SES2 (checked below)
            entries = []
            for g in sp.index_absorb[b]:
                gt = sp.gas[(b, g)]
                if gt.i_scale_fnc == IP_SCALE_SES2:
                    raise NotImplementedError("SES2 scaling is not supported")
                if gt.i_scale_fnc == IP_SCALE_T_LOOKUP:
                    raise NotImplementedError("temperature-only k-tables are not supported")
                name = f"b{b}_g{g}"
                self.register_buffer(f"{name}_k", t(gt.k))
                self.register_buffer(f"{name}_w", t(gt.w))
                if gt.k_lookup is not None:
                    self.register_buffer(f"{name}_lookup", t(gt.k_lookup))
                entries.append(dict(gas=g, type=int(sp.type_absorb[g]), n_k=gt.n_k, i_scale_k=gt.i_scale_k,
                                    i_scale_fnc=gt.i_scale_fnc, p_ref=gt.p_ref, t_ref=gt.t_ref,
                                    scale=[list(map(float, row)) for row in gt.scale], name=name))
            self.gas_entries.append(entries)
            if not entries:
                raise NotImplementedError(f"band {b + 1} has no gaseous absorbers (solve_band_without_gas)")

        # Continua, looked up by position in the band as radiance_calc does.
        self.continua = []
        for b in range(sp.n_band):
            lst = []
            if sp.n_band_continuum is not None:
                for pos, ctype in enumerate(sp.index_continuum[b]):
                    if ctype != pos + 1:
                        raise NotImplementedError("continua must be listed in type order (see radiance_calc)")
                    c = sp.continuum[(b, ctype)]
                    if c["i_scale_fnc"] == IP_SCALE_SES2:
                        raise NotImplementedError("SES2 continuum is not supported")
                    lst.append(dict(type=ctype, k=float(c["k"]), i_scale_fnc=c["i_scale_fnc"], p_ref=c["p_ref"],
                                    t_ref=c["t_ref"], scale=list(map(float, c["scale"]))))
            self.continua.append(lst)

        # Droplet parametrisation used by Isca (read_control: i_st_water = 5)
        self.drop_type = 5
        if self.drop_type in sp.drop:
            dp = sp.drop[self.drop_type]
            if dp.i_parm != 5:
                raise NotImplementedError(f"droplet parametrisation {dp.i_parm} (only Pade 2 / 5 is ported)")
            self.register_buffer("drop_param", t(dp.parm_list))
            self.drop_min_dim, self.drop_max_dim = float(dp.min_dim), float(dp.max_dim)
        else:
            self.drop_param = None

        self.uses_wenyi = any(e["i_scale_fnc"] == IP_SCALE_WENYI for entries in self.gas_entries for e in entries)
        if self.uses_wenyi:
            for k, v in wenyi_tables_numpy().items():
                self.register_buffer(f"wenyi_{k}", t(v))

        # g-points: major gas k-terms of every band
        gp_band, gp_k = [], []
        for b, entries in enumerate(self.gas_entries):
            for k in range(entries[0]["n_k"]):
                gp_band.append(b)
                gp_k.append(k)
        self.register_buffer("gp_band", torch.tensor(gp_band, dtype=torch.long))
        self.n_gpoint = len(gp_band)
        self.register_buffer("gp_weight", torch.cat([getattr(self, f"{e[0]['name']}_w") for e in self.gas_entries]))

        # Minor-gas k-terms (all bands) and their grouping by (band, gas).
        mt_band, mt_group, grp_band, grp_len = [], [], [], []
        for b, entries in enumerate(self.gas_entries):
            for e in entries[1:]:
                gid = len(grp_band)
                grp_band.append(b)
                grp_len.append(e["n_k"])
                mt_band += [b] * e["n_k"]
                mt_group += [gid] * e["n_k"]
        self.n_minor_term, self.n_minor_group = len(mt_band), len(grp_band)
        self.register_buffer("mt_band", torch.tensor(mt_band, dtype=torch.long))
        self.register_buffer("mt_group", torch.tensor(mt_group, dtype=torch.long))
        self.register_buffer("grp_band", torch.tensor(grp_band, dtype=torch.long))
        k_max = max(grp_len, default=1)
        pad = np.full((len(grp_len), k_max), len(mt_band), dtype=np.int64)
        pos = 0
        for gidx, n in enumerate(grp_len):
            pad[gidx, :n] = np.arange(pos, pos + n)
            pos += n
        self.register_buffer("grp_pad_index", torch.as_tensor(pad))
        self.register_buffer("mt_weight", torch.cat([getattr(self, f"{e['name']}_w")
                                                     for entries in self.gas_entries for e in entries[1:]])
                             if mt_band else torch.zeros(0, dtype=dtype))
        # Slot of each minor group within its band (for the sequential product over gases).
        slot = []
        count = {}
        for b in grp_band:
            slot.append(count.get(b, 0))
            count[b] = count.get(b, 0) + 1
        self.register_buffer("grp_slot", torch.tensor(slot, dtype=torch.long))
        self.max_minor_per_band = max(count.values(), default=0)

        # Exact float64 copies of the floating buffers: every .to()/.float()/.double() rebuilds
        # from these, so a float32 round trip never degrades the spectral data.
        self._pristine = {n: b.detach().to(torch.float64).clone() for n, b in self._buffers.items()
                          if b is not None and b.is_floating_point()}
        if dtype != torch.float64:
            for n in self._pristine:
                self._buffers[n] = self._pristine[n].to(dtype)

    def cloud_optics(self, atm: "Atmosphere", cloud, k_grey_tot, k_ext_scat, k_grey_abs=None):
        """Clear + liquid-cloud grey optics of the cloudy region (grey_opt_prop, rescale_phase_fnc).

        Returns w_cloud (P, L), cloudy mask, and for the cloudy region (B, P, L): k_grey_tot,
        k_ext_scat, rescaled asymmetry and forward-scattering fraction.
        """
        if self.drop_param is None:
            raise ValueError("spectral file has no droplet parametrisation for Isca's cloud type")
        w, mmr, dim = cl.set_cloud(cloud, self.drop_min_dim, self.drop_max_dim)
        cloudy = w > 0.0
        k_ext_c, k_scat_c, phase_c, fwd_c, k_abs_c = cl.opt_prop_pade_2(self.drop_param, mmr, dim,
                                                                      return_absorption=True)
        zero = torch.zeros_like(k_grey_tot)
        kgt = torch.where(cloudy, k_grey_tot + k_ext_c, k_grey_tot)
        ks = torch.where(cloudy, k_ext_scat + k_scat_c, k_ext_scat)
        phase = torch.where(cloudy, 0.0 + phase_c, zero)  # clear-sky phase function is zero (Rayleigh)
        fwd = torch.where(cloudy, 0.0 + fwd_c, zero)
        norm = ks > torch.finfo(ks.dtype).tiny
        inv = 1.0 / torch.where(norm, ks, torch.ones_like(ks))
        fwd = torch.where(norm, fwd * inv, fwd)
        phase = torch.where(norm, phase * inv, phase)
        phase = (phase - fwd) / (1.0 - fwd)  # rescale_phase_fnc
        if k_grey_abs is not None:
            return w, cloudy, kgt, ks, phase, fwd, torch.where(cloudy, k_grey_abs + k_abs_c, k_grey_abs)
        return w, cloudy, kgt, ks, phase, fwd

    def _apply(self, fn, recurse=True):
        if not self._pristine:
            return super()._apply(fn, recurse)
        first = next(iter(self._pristine))
        current = self._buffers[first]

        def probe(dt, device):
            try:
                return fn(torch.zeros(1, dtype=dt, device=device))
            except (RuntimeError, TypeError):
                return None

        p_dev = probe(torch.float32, current.device)  # where does fn send tensors?
        p64 = probe(torch.float64, "cpu")  # does fn change the dtype?
        if p_dev is not None:
            if p64 is not None and p64.dtype != torch.float64:
                target = p64.dtype
            elif p_dev.dtype != torch.float32:
                target = p_dev.dtype
            else:
                target = current.dtype
            for n, t in self._pristine.items():
                self._buffers[n] = t.to(dtype=target).to(device=p_dev.device)
        return super()._apply(fn, recurse)

    # ------------------------------------------------------------------
    def gas_k_abs(self, atm: Atmosphere, weights=None):
        """k_abs_layer for every gas in every band: list over bands of lists of (n_k, P, L) tensors."""
        wenyi = ({k: getattr(self, f"wenyi_{k}") for k in ("plg", "ttb", "tto", "gk250b", "gk4", "gk6")}
                 if self.uses_wenyi else None)
        out = []
        for b, entries in enumerate(self.gas_entries):
            band_out = []
            for e in entries:
                mix = atm.gas_mmr.get(e["type"])
                if mix is None:
                    mix = torch.zeros_like(atm.p)
                mix = torch.broadcast_to(mix, atm.p.shape)
                k = getattr(self, f"{e['name']}_k")
                if e["i_scale_k"] == IP_SCALE_TERM and e["i_scale_fnc"] == IP_SCALE_LOOKUP:
                    kl = interp_k_lookup(getattr(self, f"{e['name']}_lookup"), weights)
                    k_abs = torch.clamp(kl * mix, min=0.0)
                elif e["i_scale_k"] == IP_SCALE_TERM:
                    terms = []
                    for iex in range(e["n_k"]):
                        gfr = scale_absorb(atm.p, atm.t, e["i_scale_fnc"], e["p_ref"], e["t_ref"], e["scale"][iex],
                                           iex=iex + 1, i_band_1based=b + 1, wenyi_tables=wenyi)
                        terms.append(k[iex] * torch.clamp(gfr * mix, min=0.0))
                    k_abs = torch.stack(terms, 0)
                elif e["i_scale_k"] == IP_SCALE_BAND:
                    gfr = scale_absorb(atm.p, atm.t, e["i_scale_fnc"], e["p_ref"], e["t_ref"], e["scale"][0],
                                       iex=1, i_band_1based=b + 1, wenyi_tables=wenyi)
                    gfr = torch.clamp(gfr * mix, min=0.0)
                    k_abs = k[:, None, None] * gfr
                elif e["i_scale_k"] == IP_SCALE_NULL:
                    k_abs = k[:, None, None] * mix
                else:
                    raise NotImplementedError(f"k scaling type {e['i_scale_k']}")
                band_out.append(k_abs)
            out.append(band_out)
        return out

    def grey_optics(self, atm: Atmosphere, l_rayleigh: bool, return_absorption=False):
        """k_grey_tot and k_ext_scat per band: (B, P, L) each (clear sky, no aerosol).

        ``return_absorption`` adds the grey absorption (continuum) computed separately."""
        water = atm.gas_mmr.get(1)
        water = torch.zeros_like(atm.p) if water is None else torch.broadcast_to(water, atm.p.shape)
        k_grey, k_scat, k_abs = [], [], []
        for b in range(self.n_band):
            if l_rayleigh:
                if self.sp.i_rayleigh_scheme != IP_RAYLEIGH_TOTAL:
                    raise NotImplementedError("per-gas Rayleigh coefficients are not supported")
                ks = torch.broadcast_to(self.rayleigh_coeff[b], atm.p.shape)
            else:
                ks = torch.zeros_like(atm.p)
            kg = torch.zeros_like(atm.p)
            for c in self.continua[b]:
                amount = rescale_continuum(atm.p, atm.t, atm.density, water, c["type"], c["i_scale_fnc"],
                                           c["p_ref"], c["t_ref"], c["scale"])
                kg = kg + c["k"] * amount
            k_grey.append(kg + ks)
            k_scat.append(ks)
            k_abs.append(kg)
        if return_absorption:
            return torch.stack(k_grey, 0), torch.stack(k_scat, 0), torch.stack(k_abs, 0)
        return torch.stack(k_grey, 0), torch.stack(k_scat, 0)

    def split_major_minor(self, k_abs, atm: Atmosphere):
        """Concatenate major-gas k-terms (G, P, L) and minor-gas k-terms (T, P, L)."""
        major = torch.cat([band[0] for band in k_abs], 0)
        minors = [k for band in k_abs for k in band[1:]]
        minor = torch.cat(minors, 0) if minors else atm.p.new_zeros((0,) + atm.p.shape)
        return major, minor

    def minor_k_min(self, minor: torch.Tensor):
        """Minimum over the k-terms of each minor gas: (n_minor_group, P, L)."""
        huge = torch.finfo(minor.dtype).max
        ext = torch.cat([minor, torch.full_like(minor[:1], huge)], 0)
        idx = self.grp_pad_index
        padded = ext.index_select(0, idx.reshape(-1)).reshape(idx.shape + minor.shape[1:])
        return padded.amin(1)


_COMPILED = {}


class _CheckedCompiled:
    """``torch.compile``d kernel that validates itself against eager execution.

    The inductor backends (notably the experimental MPS one) occasionally miscompile
    kernels.  On the first call for each input signature both versions run; the compiled
    one is used from then on only if it reproduces eager (same NaN pattern, relative
    difference below ``rtol``), otherwise that signature permanently falls back to eager
    with a warning.
    """

    def __init__(self, fn, device_type, rtol=1e-5):
        self.fn = fn
        options = {"max_fusion_size": 8} if device_type == "mps" else None
        self.compiled = torch.compile(fn, dynamic=False, options=options)
        self.rtol = rtol
        self.status = {}

    @staticmethod
    def _signature(args, kwargs):
        def sig(a):
            return (tuple(a.shape), a.dtype, a.device.type) if torch.is_tensor(a) else a
        return tuple(sig(a) for a in args) + tuple((k, sig(v)) for k, v in sorted(kwargs.items()))

    def __call__(self, *args, **kwargs):
        key = self._signature(args, kwargs)
        state = self.status.get(key)
        if state is True:
            return self.compiled(*args, **kwargs)
        if state is False:
            return self.fn(*args, **kwargs)
        import warnings

        eager = self.fn(*args, **kwargs)
        try:
            comp = self.compiled(*args, **kwargs)
        except Exception as exc:  # compilation failure: fall back
            warnings.warn(f"torch.compile of {self.fn.__name__} failed ({type(exc).__name__}); using eager")
            self.status[key] = False
            return eager
        ok = True
        with torch.no_grad():
            for e, c in zip(eager if isinstance(eager, tuple) else (eager,), comp if isinstance(comp, tuple) else (comp,)):
                if e is None:
                    continue
                if not torch.equal(torch.isnan(e), torch.isnan(c)):
                    ok = False
                    break
                fin = torch.isfinite(e)
                diff = torch.where(fin, (e - c).abs(), torch.zeros_like(e)).max()
                scale = torch.where(fin, e.abs(), torch.zeros_like(e)).max()
                if bool(diff > self.rtol * scale + torch.finfo(e.dtype).tiny):
                    ok = False
                    break
        self.status[key] = ok
        if not ok:
            warnings.warn(f"torch.compile of {self.fn.__name__} disagrees with eager on {args[0].device.type} "
                          f"(backend bug); using eager for this input signature")
        return eager


def _kernel(fn, enabled: bool, device_type: str = "cpu"):
    """``fn`` or its cached, self-validating compiled version (see :class:`_CheckedCompiled`).

    Kernels are compiled one by one, and on MPS with a small inductor fusion size:
    larger fused kernels exceed Metal's limit of 31 buffers per GPU kernel.
    """
    if not enabled:
        return fn
    key = (fn, device_type)
    if key not in _COMPILED:
        _COMPILED[key] = _CheckedCompiled(fn, device_type)
    return _COMPILED[key]


def _use_stable(numerics: str, dtype) -> bool:
    if numerics == "reference":
        return False
    if numerics == "stable":
        return True
    if numerics == "auto":
        return torch.finfo(dtype).eps > 1e-10
    raise ValueError(f"numerics must be 'auto', 'reference' or 'stable', not {numerics!r}")


def _layer_divergence(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Net flux convergence of each layer (W/m2, positive = heating) from level fluxes (..., L+1)."""
    return (down[..., :-1] - down[..., 1:]) + (up[..., 1:] - up[..., :-1])


def _per_band(x: torch.Tensor, n_band: int) -> torch.Tensor:
    """Surface property (P,) or (P, B) -> (B, P)."""
    if x.dim() == 1:
        return x.unsqueeze(0).expand(n_band, -1)
    return x.transpose(0, 1)


class SocratesLW(nn.Module):
    """Longwave SOCRATES as configured by Isca (Elsasser two-stream, no scattering, quadratic source)."""

    def __init__(self, spectral_file, dtype=torch.float64, numerics="auto", compile=False):
        super().__init__()
        self.spectrum = Spectrum(spectral_file, dtype=dtype)
        self.numerics = numerics
        self.compile = compile
        if self.spectrum.sp.thermal_coeff is None:
            raise ValueError("longwave spectral file has no thermal source block")

    def forward(self, atm: Atmosphere, t_ground: torch.Tensor, emissivity=1.0, cloud=None,
                return_intermediates=False):
        """Longwave fluxes; ``cloud`` is an optional :class:`pysoc.clouds.LiquidCloud`."""
        spec = self.spectrum
        P, L = atm.p.shape
        B = spec.n_band
        stable = _use_stable(self.numerics, atm.p.dtype)
        weights = lookup_weights(atm.p, atm.t, spec.p_lookup, spec.t_lookup) if spec.p_lookup is not None else None
        k_abs = spec.gas_k_abs(atm, weights)
        k_grey_tot, _ = spec.grey_optics(atm, l_rayleigh=False)  # (B, P, L)
        emissivity = torch.as_tensor(emissivity, dtype=atm.p.dtype, device=atm.p.device)
        if emissivity.dim() <= 1:
            emissivity = torch.broadcast_to(emissivity, (P,))
        diffuse_albedo = _per_band(1.0 - emissivity, B)  # (B, P); set_bound: rho_alb = 1 - emissivity

        # Planck terms per band (diff_planck_source)
        coeff = spec.thermal_coeff[:, None, None, :]  # (B, 1, 1, n+1)
        t_ref = spec.sp.t_ref_planck
        planck_flux = planck_flux_band(coeff, t_ref, atm.t_level.unsqueeze(0))  # (B, P, L+1)
        planck_diff = planck_flux[..., 1:] - planck_flux[..., :-1]
        planck_layer = planck_flux_band(coeff, t_ref, atm.t.unsqueeze(0))
        planck_diff_2 = 2.0 * (planck_flux[..., 1:] + planck_flux[..., :-1] - 2.0 * planck_layer)
        planck_ground = planck_flux_band(spec.thermal_coeff[:, None, :], t_ref, t_ground.unsqueeze(0))  # (B, P)
        flux_inc_down = -planck_flux[..., 0]  # (B, P)
        d_planck_flux_surface = planck_ground - planck_flux[..., L]

        major, minor = spec.split_major_minor(k_abs, atm)

        # Equivalent extinction of the minor gases (monochromatic_gas_flux on every minor k-term).
        k_eqv = torch.zeros_like(k_grey_tot)
        if spec.n_minor_term:
            mb = spec.mt_band
            dev = atm.p.device.type
            trans_m, src_m = _kernel(ts.gas_flux_sources_ir, self.compile, dev)(
                minor * atm.d_mass, planck_diff[mb], ts.DIFFUSIVITY_FACTOR_MINOR, stable)
            up, down = _kernel(ts.gas_flux_recurrence_ir, self.compile, dev)(
                trans_m, src_m, flux_inc_down[mb], d_planck_flux_surface[mb], diffuse_albedo[mb])
            w = spec.mt_weight[:, None, None]
            # fluxes incident on each layer: downward at its top, upward at its bottom
            f_top = torch.abs(down[..., :-1])
            f_bot = torch.abs(up[..., 1:])
            kw = minor * w
            G = spec.n_minor_group
            zeros = minor.new_zeros((G,) + minor.shape[1:])
            grp = spec.mt_group
            sum_k_flux = zeros.index_add(0, grp, kw * f_top) + zeros.index_add(0, grp, kw * f_bot)
            sum_flux = zeros.index_add(0, grp, w * f_top) + zeros.index_add(0, grp, w * f_bot)
            k_min = spec.minor_k_min(minor)
            pos = sum_flux > torch.finfo(minor.dtype).tiny ** 0.5
            k_eqv_grp = torch.where(pos, sum_k_flux / torch.where(pos, sum_flux, torch.ones_like(sum_flux)), k_min)
            k_eqv = k_eqv.index_add(0, spec.grp_band, k_eqv_grp)

        k_grey_final = k_grey_tot + k_eqv  # (+ k_grey = 0)
        gb = spec.gp_band
        K = lambda f: _kernel(f, self.compile, atm.p.device.type)  # noqa: E731
        tau = (k_grey_final[gb] + major) * atm.d_mass
        alb_g, finc_g, dps_g = diffuse_albedo[gb], flux_inc_down[gb], d_planck_flux_surface[gb]
        cover = None
        if cloud is None:
            trans, s1, s2 = K(ts.two_coeff_fast_lw_stable if stable else ts.two_coeff_fast_lw)(tau, True)
            s_down, s_up = K(ts.ir_source)(s1, s2, planck_diff[gb], planck_diff_2[gb], True)
            up, down = K(ts.solver_no_scat)(trans, s_down, s_up, alb_g, finc_g, dps_g)
            up_clr, down_clr = up, down
        else:
            # no Rayleigh scattering in the longwave; cloud extinction is treated as absorption (omega = 0)
            w_cloud, cloudy, k_grey_tot_c, _, _, _ = spec.cloud_optics(atm, cloud, k_grey_tot,
                                                                     torch.zeros_like(k_grey_tot))
            ov = cl.overlap_max_random(w_cloud)
            cover = ov["tot_cloud_cover"]
            tau_c = ((k_grey_tot_c + k_eqv)[gb] + major) * atm.d_mass
            zeros = torch.zeros_like(tau)
            if stable:
                coeff_f = K(cl.two_coeff_ir_nonscattering_stable)(tau, True)
                coeff_c = K(cl.two_coeff_ir_nonscattering_stable)(tau_c, True)
            else:
                coeff_f = K(cl.two_coeff_ir)(zeros, zeros, tau)
                coeff_c = K(cl.two_coeff_ir)(zeros, zeros, tau_c)
            trans, reflect, s1, s2 = coeff_f
            trans_c, reflect_c, s1_c, s2_c = (torch.where(cloudy, x, torch.zeros_like(x)) for x in coeff_c)
            s_down_clr, s_up_clr = K(ts.ir_source)(s1, s2, planck_diff[gb], planck_diff_2[gb], True)
            s_down_c, s_up_c = K(ts.ir_source)(s1_c, s2_c, planck_diff[gb], planck_diff_2[gb], True)
            w_free = ov["w_free"]
            up, down = K(cl.solver_mix_direct_hogan)(
                trans, reflect, w_free * s_down_clr, w_free * s_up_clr,
                trans_c, reflect_c, w_cloud * s_down_c, w_cloud * s_up_c,
                ov["dn_ff"], ov["dn_cf"], ov["dn_fc"], ov["dn_cc"], ov["up_ff"], ov["up_fc"], ov["up_cf"], ov["up_cc"],
                finc_g, ov["up_ff"][..., L] * (1.0 - alb_g) * dps_g, ov["up_cf"][..., L] * (1.0 - alb_g) * dps_g,
                alb_g)
            up_clr, down_clr = K(ts.solver_no_scat)(trans, s_down_clr, s_up_clr, alb_g, finc_g, dps_g)
        w = spec.gp_weight[:, None, None]
        zeros = up.new_zeros((B, P, L + 1))

        def band(x):
            return zeros.index_add(0, gb, w * x) + planck_flux

        flux_up_band, flux_down_band = band(up), band(down)
        flux_up_band_clr, flux_down_band_clr = band(up_clr), band(down_clr)
        # the Planck terms added to up and down cancel in the divergence
        out = dict(
            flux_up=flux_up_band.sum(0), flux_down=flux_down_band.sum(0),
            flux_divergence=(w * _layer_divergence(up, down)).sum(0),
            flux_up_clear=flux_up_band_clr.sum(0), flux_down_clear=flux_down_band_clr.sum(0),
            flux_divergence_clear=(w * _layer_divergence(up_clr, down_clr)).sum(0),
            flux_up_band=flux_up_band.permute(1, 2, 0), flux_down_band=flux_down_band.permute(1, 2, 0),
            flux_up_clear_band=flux_up_band_clr.permute(1, 2, 0),
            flux_down_clear_band=flux_down_band_clr.permute(1, 2, 0),
        )
        if cover is not None:
            out["tot_cloud_cover"] = cover
        if return_intermediates:
            out["intermediates"] = dict(
                lookup_weights=weights, k_abs=k_abs, k_grey_tot=k_grey_tot, k_eqv=k_eqv, k_grey_final=k_grey_final,
                planck_flux=planck_flux, planck_diff=planck_diff, planck_diff_2=planck_diff_2,
                planck_ground=planck_ground, tau=tau, gpoint_flux_up=up, gpoint_flux_down=down)
        return out


class SocratesSW(nn.Module):
    """Shortwave SOCRATES as configured by Isca (PIFM80, full scattering, Rayleigh, direct solver)."""

    def __init__(self, spectral_file, dtype=torch.float64, i_2stream=ts.IP_PIFM80, numerics="auto", compile=False):
        super().__init__()
        self.spectrum = Spectrum(spectral_file, dtype=dtype)
        self.i_2stream = i_2stream
        self.numerics = numerics
        self.compile = compile
        if self.spectrum.sp.solar_flux_band is None:
            raise ValueError("shortwave spectral file has no solar spectrum block")

    def forward(self, atm: Atmosphere, cos_zenith: torch.Tensor, solar_irrad: torch.Tensor,
                albedo_diffuse: torch.Tensor, albedo_direct: torch.Tensor | None = None, cloud=None,
                return_intermediates=False):
        """Shortwave fluxes; ``cloud`` is an optional :class:`pysoc.clouds.LiquidCloud`."""
        spec = self.spectrum
        P, L = atm.p.shape
        B = spec.n_band
        if albedo_direct is None:
            albedo_direct = albedo_diffuse
        stable = _use_stable(self.numerics, atm.p.dtype)
        weights = lookup_weights(atm.p, atm.t, spec.p_lookup, spec.t_lookup) if spec.p_lookup is not None else None
        k_abs = spec.gas_k_abs(atm, weights)
        k_grey_tot, k_ext_scat, k_grey_abs = spec.grey_optics(atm, l_rayleigh=True, return_absorption=True)

        # set_bound: night columns get zero irradiance and zen_0 = 1
        lit = cos_zenith > 0.0
        sec_0 = torch.where(lit, 1.0 / torch.where(lit, cos_zenith, torch.ones_like(cos_zenith)),
                            torch.ones_like(cos_zenith))
        irrad = torch.where(lit, solar_irrad, torch.zeros_like(solar_irrad))
        solar_irrad_band = irrad.unsqueeze(0) * spec.solar_flux_band[:, None]  # (B, P)
        alb_diff = _per_band(torch.broadcast_to(albedo_diffuse, (P,)) if albedo_diffuse.dim() <= 1 else albedo_diffuse, B)
        alb_dir = _per_band(torch.broadcast_to(albedo_direct, (P,)) if albedo_direct.dim() <= 1 else albedo_direct, B)

        major, minor = spec.split_major_minor(k_abs, atm)

        # Minor gases: equivalent extinction weighted by direct-beam transmission, and the
        # layer-by-layer correction of the direct beam (adjust_solar_ke).
        k_eqv, adjust, adjust_m1 = self._minor_gases(spec, atm, minor, k_grey_tot, sec_0, stable)

        k_grey_final = k_grey_tot + k_eqv
        gb = spec.gp_band
        K = lambda f: _kernel(f, self.compile, atm.p.device.type)  # noqa: E731
        sec_g = sec_0.unsqueeze(0).expand(len(gb), -1)
        asymmetry = torch.zeros_like(major)
        if stable:
            # carry the co-albedo separately: omega is limited to 1 - 32*EPSILON(float64)
            tau, omega, coalb = ts.single_scattering_stable(k_grey_final[gb], k_ext_scat[gb],
                                                            (k_grey_abs + k_eqv)[gb], major, atm.d_mass)
            tau, omega, coalb = ts.rescale_tau_omega_coalbedo(tau, omega, coalb, torch.zeros_like(tau))
            trans, reflect, trans_0, su, sd = K(ts.two_coeff_solar_stable)(
                omega, asymmetry, tau, sec_g, self.i_2stream, None, coalb)
        else:
            tau, omega = ts.single_scattering(k_grey_final[gb], k_ext_scat[gb], major, atm.d_mass,
                                              ts.IP_SCATTER_FULL)
            tau, omega = ts.rescale_tau_omega(tau, omega, torch.zeros_like(tau))
            trans, reflect, trans_0, su, sd = K(ts.two_coeff_solar)(omega, asymmetry, tau, sec_g, self.i_2stream)
        flux_inc = solar_irrad_band[gb] / sec_g
        if stable:
            flux_direct, s_down, s_up = K(ts.solar_source_stable)(flux_inc, trans_0, su, sd, adjust_m1[gb])
        else:
            flux_direct, s_down, s_up = K(ts.solar_source)(flux_inc, trans_0, su, sd, adjust[gb])
        alb_d, alb_s = alb_diff[gb], alb_dir[gb]
        source_ground = (alb_s - alb_d) * flux_direct[..., L]
        up, down = K(ts.solver_homogen_direct)(trans, reflect, s_down, s_up, alb_d, flux_inc, source_ground)
        up_clr, down_clr, direct_clr = up, down, flux_direct
        cover = None
        if cloud is not None:
            w_cloud, cloudy, kgt_c, ks_c, phase_c, fwd_c, kabs_c = spec.cloud_optics(atm, cloud, k_grey_tot,
                                                                                    k_ext_scat, k_grey_abs)
            ov = cl.overlap_max_random(w_cloud)
            cover = ov["tot_cloud_cover"]
            if stable:
                tau_c, omega_c, coalb_c = ts.single_scattering_stable((kgt_c + k_eqv)[gb], ks_c[gb],
                                                                      (kabs_c + k_eqv)[gb], major, atm.d_mass)
                tau_c, omega_c, coalb_c = ts.rescale_tau_omega_coalbedo(tau_c, omega_c, coalb_c, fwd_c[gb])
                coeff_c = K(ts.two_coeff_solar_stable)(omega_c, phase_c[gb], tau_c, sec_g, self.i_2stream,
                                                       None, coalb_c)
            else:
                tau_c, omega_c = ts.single_scattering((kgt_c + k_eqv)[gb], ks_c[gb], major, atm.d_mass,
                                                      ts.IP_SCATTER_FULL)
                tau_c, omega_c = ts.rescale_tau_omega(tau_c, omega_c, fwd_c[gb])
                coeff_c = K(ts.two_coeff_solar)(omega_c, phase_c[gb], tau_c, sec_g, self.i_2stream)
            trans_c, reflect_c, trans_0_c, su_c, sd_c = (torch.where(cloudy, x, torch.zeros_like(x)) for x in coeff_c)
            flux_direct, ground_cloud, s_up_f, s_dn_f, s_up_c, s_dn_c = K(cl.mixed_solar_source)(
                flux_inc, adjust[gb], trans_0, su, sd, ov["dn_ff"], ov["dn_cf"], ov["dn_fc"], ov["dn_cc"],
                trans_0_c, su_c, sd_c, adjust_m1[gb] if stable else None)
            sg_free = (alb_s - alb_d) * (flux_direct[..., L] - ground_cloud)
            sg_cloud = (alb_s - alb_d) * ground_cloud
            up, down = K(cl.solver_mix_direct_hogan)(
                trans, reflect, s_dn_f, s_up_f, trans_c, reflect_c, s_dn_c, s_up_c,
                ov["dn_ff"], ov["dn_cf"], ov["dn_fc"], ov["dn_cc"], ov["up_ff"], ov["up_fc"], ov["up_cf"], ov["up_cc"],
                flux_inc, sg_free, sg_cloud, alb_d)
        w = spec.gp_weight[:, None, None]
        zeros = up.new_zeros((B, P, L + 1))

        def band(x):
            return zeros.index_add(0, gb, w * x)

        fb = {n: band(x) for n, x in (("up", up), ("down", down), ("direct", flux_direct),
                                      ("up_clear", up_clr), ("down_clear", down_clr), ("direct_clear", direct_clr))}
        out = dict(
            flux_up=fb["up"].sum(0), flux_down=fb["down"].sum(0), flux_direct=fb["direct"].sum(0),
            flux_divergence=(w * _layer_divergence(up, down)).sum(0),
            flux_up_clear=fb["up_clear"].sum(0), flux_down_clear=fb["down_clear"].sum(0),
            flux_direct_clear=fb["direct_clear"].sum(0),
            flux_divergence_clear=(w * _layer_divergence(up_clr, down_clr)).sum(0),
        )
        for n, x in fb.items():
            name = f"flux_{n}_band" if not n.endswith("_clear") else f"flux_{n[:-6]}_clear_band"
            out[name] = x.permute(1, 2, 0)
        if cover is not None:
            out["tot_cloud_cover"] = cover
        if return_intermediates:
            out["intermediates"] = dict(
                lookup_weights=weights, k_abs=k_abs, k_grey_tot=k_grey_tot, k_ext_scat=k_ext_scat, k_eqv=k_eqv,
                k_grey_final=k_grey_final, adjust_solar_ke=adjust, tau=tau, omega=omega,
                gpoint_flux_up=up_clr, gpoint_flux_down=down_clr,
                gpoint_flux_direct=direct_clr)
        return out

    @staticmethod
    def _minor_gases(spec, atm, minor, k_grey_tot, sec_0, stable):
        """solve_band_k_eqv_scl (solar): k_eqv (B,P,L), adjust_solar_ke (B,P,L) and adjust_solar_ke - 1."""
        P, L = atm.p.shape
        k_eqv = torch.zeros_like(k_grey_tot)
        dtype = atm.p.dtype
        eps_ref = torch.finfo(torch.float64 if stable else dtype).eps
        temp_max = float(np.log(1.0 / eps_ref))
        if not spec.n_minor_term:
            adjust = torch.exp(torch.clamp(k_eqv * atm.d_mass * sec_0[:, None], max=temp_max))
            return k_eqv, adjust, expm1(torch.clamp(k_eqv * atm.d_mass * sec_0[:, None], max=temp_max))
        G = spec.n_minor_group
        grp = spec.mt_group
        w = spec.mt_weight[:, None]
        sec = sec_0.unsqueeze(0)
        prev = w.expand(-1, P)  # flux_direct_term(0) = esft weight
        fg_prev = minor.new_ones((G, P))  # flux_gas(0) = 1
        # SOCRATES tests "> 0"; transmissions below sqrt(TINY) (an underflowed beam carrying
        # < 1e-150 W/m2) are treated as zero so that the division has finite gradients.
        floor = torch.finfo(dtype).tiny ** 0.5
        levels, ratios = [], []
        for i in range(L):
            x = -minor[..., i] * atm.d_mass[:, i] * sec
            cur = prev * torch.exp(x)
            fg = minor.new_zeros((G, P)).index_add(0, grp, cur)
            pos = fg_prev > floor
            safe = torch.where(pos, fg_prev, torch.ones_like(fg_prev))
            if stable:
                # ratio - 1 = (sum_t prev_t*expm1(x_t) + (sum_t prev_t - fg_prev)) / fg_prev
                num = (minor.new_zeros((G, P)).index_add(0, grp, prev * expm1(x))
                       + (minor.new_zeros((G, P)).index_add(0, grp, prev) - fg_prev))
                ratios.append(torch.where(pos, num / safe, torch.zeros_like(num)))
            else:
                ratios.append(torch.where(pos, fg / safe, torch.ones_like(fg)))
            levels.append(cur)
            prev, fg_prev = cur, fg
        surf = levels[-1]  # (T, P)
        term_k = minor * surf.unsqueeze(-1)
        sum_k_flux = minor.new_zeros((G, P, L)).index_add(0, grp, term_k)
        sum_flux = minor.new_zeros((G, P)).index_add(0, grp, surf).unsqueeze(-1)
        k_min = spec.minor_k_min(minor)
        pos = sum_flux > floor
        k_eqv_grp = torch.where(pos, sum_k_flux / torch.where(pos, sum_flux, torch.ones_like(sum_flux)), k_min)
        k_eqv = k_eqv.index_add(0, spec.grp_band, k_eqv_grp)
        ratio = torch.stack(ratios, -1)  # (G, P, L)
        exponent = torch.clamp(k_eqv * atm.d_mass * sec_0[:, None], max=temp_max)
        if stable:
            log_ratio = torch.log1p(torch.clamp(ratio, min=-1.0 + torch.finfo(dtype).eps))
            log_adj = torch.zeros_like(k_grey_tot).index_add(0, spec.grp_band, log_ratio) + exponent
            adjust_m1 = expm1(log_adj)
            return k_eqv, 1.0 + adjust_m1, adjust_m1
        adjust = torch.ones_like(k_grey_tot)
        for slot in range(spec.max_minor_per_band):
            sel = spec.grp_slot == slot
            factor = torch.ones_like(adjust).index_copy(0, spec.grp_band[sel], ratio[sel])
            adjust = adjust * factor
        adjust = adjust * torch.exp(exponent)
        return k_eqv, adjust, adjust - 1.0
