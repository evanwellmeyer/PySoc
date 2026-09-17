"""Two-stream kernels of SOCRATES (radiance_core), vectorised in PyTorch.

Every function accepts arbitrary leading batch dimensions.  Layer quantities
have shape ``(..., n_layer)``; column quantities (surface values, incident
fluxes, secant of the zenith angle) have shape ``(...)``.  Fluxes are returned
as separate up/down tensors of shape ``(..., n_layer + 1)`` with index 0 at the
top of the atmosphere, instead of SOCRATES' interleaved ``flux_total`` array.

Tolerances follow the Fortran, which derives them from ``EPSILON``/``TINY`` of
the working precision; here they come from ``torch.finfo(dtype)``.
"""

from __future__ import annotations

import torch

from .mathutil import expm1

# rad_pcf.F90
IP_SCATTER_FULL = 1
IP_NO_SCATTER_ABS = 2
IP_NO_SCATTER_EXT = 3
IP_SCATTER_APPROX = 4
IP_SCATTER_HYBRID = 5

IP_EDDINGTON = 2
IP_DISCRETE_ORD = 4
IP_ELSASSER = 12
IP_PIFM85 = 6
IP_2S_TEST = 14
IP_HEMI_MEAN = 15
IP_PIFM80 = 16

ELSASSER_FACTOR = 1.66  # diffusivity_factor.F90
DIFFUSIVITY_FACTOR_MINOR = 1.66


def _eps(t: torch.Tensor) -> float:
    return torch.finfo(t.dtype).eps


def _tiny(t: torch.Tensor) -> float:
    return torch.finfo(t.dtype).tiny


# ---------------------------------------------------------------------------
# Single scattering properties (single_scattering.F90, rescale_tau_omega.F90)
# ---------------------------------------------------------------------------
def single_scattering(k_grey_tot, k_ext_scat, k_gas_abs, d_mass, i_scatter_method=IP_SCATTER_FULL):
    """Optical depth and single scattering albedo of each layer."""
    if i_scatter_method in (IP_SCATTER_FULL, IP_SCATTER_APPROX):
        k_total = k_grey_tot + k_gas_abs
        tau = k_total * d_mass
        omega = k_ext_scat / (k_total + _tiny(k_total))
        omega = torch.clamp(omega, max=1.0 - 32.0 * _eps(omega))
    elif i_scatter_method == IP_NO_SCATTER_ABS:
        tau = (k_grey_tot + k_gas_abs - k_ext_scat) * d_mass
        omega = torch.zeros_like(tau)
    elif i_scatter_method == IP_NO_SCATTER_EXT:
        tau = (k_grey_tot + k_gas_abs) * d_mass
        omega = torch.zeros_like(tau)
    else:
        raise NotImplementedError(f"scatter method {i_scatter_method}")
    return tau, omega


def rescale_tau_omega(tau, omega, forward_scatter):
    """Delta-rescaling of optical depth and single scattering albedo."""
    tau_new = tau * (1.0 - omega * forward_scatter)
    omega_new = omega * (1.0 - forward_scatter) / (1.0 - omega * forward_scatter)
    return tau_new, omega_new


# ---------------------------------------------------------------------------
# Two-stream coefficients
# ---------------------------------------------------------------------------
def two_coeff_basic(omega, asymmetry, i_2stream, coalbedo=None):
    """Sum and difference coefficients of the two-stream equations (two_coeff_basic.F90).

    ``coalbedo`` (1 - omega computed independently) replaces ``1.0 - omega`` in the difference
    coefficient; it keeps nearly conservative scattering accurate in low precision.
    """
    one_m_omega = (1.0 - omega) if coalbedo is None else coalbedo
    if i_2stream == IP_EDDINGTON:
        s = 1.5 * (1.0 - omega * asymmetry)
        d = 2.0 * one_m_omega
    elif i_2stream == IP_ELSASSER:
        s = ELSASSER_FACTOR - 1.5 * omega * asymmetry
        d = ELSASSER_FACTOR * one_m_omega
    elif i_2stream == IP_DISCRETE_ORD:
        root_3 = 1.7320508075688772
        s = root_3 * (1.0 - omega * asymmetry)
        d = root_3 * one_m_omega
    elif i_2stream == IP_PIFM85:
        s = 2.0 - 1.5 * omega * asymmetry
        d = 2.0 * one_m_omega
    elif i_2stream == IP_2S_TEST:
        s = 1.5 - 1.5 * omega * asymmetry
        d = 1.5 * one_m_omega
    elif i_2stream == IP_HEMI_MEAN:
        s = 2.0 * (1.0 - omega * asymmetry)
        d = 2.0 * one_m_omega
    elif i_2stream == IP_PIFM80:
        s = 2.0 - 1.5 * omega * asymmetry - 0.5 * omega
        d = 2.0 * one_m_omega
    else:
        raise NotImplementedError(f"two-stream scheme {i_2stream}")
    return s, d


def solar_coefficient_basic(omega, asymmetry, sec_0, s, d, lam, i_2stream):
    """Coefficients of the direct solar source (solar_coefficient_basic.F90, plane-parallel).

    Returns possibly perturbed ``(s, d, lam)`` and ``(gamma_up, gamma_down)``.
    """
    sec_0 = sec_0.unsqueeze(-1)
    tol_perturb = 32.0 * _eps(lam)
    perturb = torch.abs(lam - sec_0) < tol_perturb
    # (not torch.where with python scalars: that would produce the default dtype and lose the
    # 1 + 32*eps perturbation in float64)
    fac = 1.0 + tol_perturb * perturb.to(lam.dtype)
    s = fac * s
    d = fac * d
    lam = fac * lam
    if i_2stream in (IP_EDDINGTON, IP_ELSASSER, IP_PIFM85, IP_2S_TEST, IP_HEMI_MEAN, IP_PIFM80):
        ksi_0 = 1.5 * asymmetry / sec_0
    elif i_2stream == IP_DISCRETE_ORD:
        ksi_0 = 1.7320508075688772 * asymmetry / sec_0
    else:
        raise NotImplementedError(f"solar two-stream scheme {i_2stream}")
    factor = 0.5 * omega * sec_0 / ((lam - sec_0) * (lam + sec_0))
    gamma_up = factor * (s - sec_0 - ksi_0 * (d - sec_0))
    gamma_down = factor * (s + sec_0 + ksi_0 * (d + sec_0))
    return s, d, lam, gamma_up, gamma_down


def two_coeff_solar(omega, asymmetry, tau, sec_0, i_2stream):
    """two_coeff.F90 + trans_source_coeff.F90 for the solar region (plane-parallel).

    Returns ``trans, reflect, trans_0, source_up, source_down`` (all ``(..., n_layer)``).
    """
    s, d = two_coeff_basic(omega, asymmetry, i_2stream)
    lam = torch.sqrt(s * d)
    s, d, lam, gamma_up, gamma_down = solar_coefficient_basic(omega, asymmetry, sec_0, s, d, lam, i_2stream)
    exponential = torch.exp(-lam * tau)
    exponential2 = exponential * exponential
    gamma = (s - lam) / (s + lam)
    gamma2 = gamma * gamma
    tmp_inv = 1.0 / (1.0 - exponential2 * gamma2)
    trans = exponential * (1.0 - gamma2) * tmp_inv
    reflect = gamma * (1.0 - exponential2) * tmp_inv
    trans_0 = torch.exp(-tau * sec_0.unsqueeze(-1))
    source_up = (gamma_up - reflect * (1.0 + gamma_down)) - gamma_up * trans * trans_0
    source_down = trans_0 * (1.0 + gamma_down - gamma_up * reflect) - (1.0 + gamma_down) * trans
    return trans, reflect, trans_0, source_up, source_down


def two_coeff_fast_lw(tau, l_ir_source_quad=True):
    """Transmission and IR source coefficients for a non-scattering layer (two_coeff_fast_lw.F90)."""
    sq_eps_r = _eps(tau) ** 0.5
    tol = sq_eps_r ** 0.5
    trans = torch.exp(-1.66 * tau)
    denom = 1.66 * tau + sq_eps_r
    source_1 = (1.0 - trans + sq_eps_r) / denom
    if not l_ir_source_quad:
        return trans, source_1, None
    thick = -(1.0 + trans - 2.0 * source_1) / denom
    thin = -(1.0 + trans - 2.0 + 1.66 * tau) / denom
    source_2 = torch.where(tau > tol, thick, thin)
    return trans, source_1, source_2


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def ir_source(source_1, source_2, diff_planck, diff_planck_2=None, l_ir_source_quad=True):
    """Layer IR source terms (ir_source.F90). Returns ``s_down, s_up``."""
    if l_ir_source_quad:
        s_up = source_1 * diff_planck + source_2 * diff_planck_2
        s_down = -source_1 * diff_planck + source_2 * diff_planck_2
    else:
        s_up = source_1 * diff_planck
        s_down = -s_up
    return s_down, s_up


def solar_source(flux_inc_direct, trans_0, source_up, source_down, adjust_solar_ke=None):
    """Direct flux and diffuse solar sources (solar_source.F90, no orography).

    ``adjust_solar_ke=None`` corresponds to ``l_scale_solar = .FALSE.``.
    Returns ``flux_direct (..., n_layer+1), s_down, s_up``.
    """
    n_layer = trans_0.shape[-1]
    flux = [flux_inc_direct]
    s_up, s_down = [], []
    for i in range(n_layer):
        prev = flux[-1]
        if adjust_solar_ke is not None:
            cur = prev * trans_0[..., i] * adjust_solar_ke[..., i]
            s_down.append((source_down[..., i] - trans_0[..., i]) * prev + cur)
        else:
            cur = prev * trans_0[..., i]
            s_down.append(source_down[..., i] * prev)
        s_up.append(source_up[..., i] * prev)
        flux.append(cur)
    return torch.stack(flux, -1), torch.stack(s_down, -1), torch.stack(s_up, -1)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def solver_homogen_direct(trans, reflect, s_down, s_up, diffuse_albedo, flux_inc_down, source_ground):
    """Direct elimination for a homogeneous column (solver_homogen_direct.F90).

    Returns ``flux_up, flux_down`` of shape ``(..., n_layer+1)``.
    """
    n_layer = trans.shape[-1]
    alpha = diffuse_albedo
    s_up_prime = source_ground
    beta, gamma, h = [None] * n_layer, [None] * n_layer, [None] * n_layer
    for i in range(n_layer - 1, -1, -1):
        beta[i] = 1.0 / (1.0 - alpha * reflect[..., i])
        gamma[i] = alpha * trans[..., i]
        h[i] = s_up_prime + alpha * s_down[..., i]
        alpha = reflect[..., i] + beta[i] * gamma[i] * trans[..., i]
        s_up_prime = s_up[..., i] + beta[i] * trans[..., i] * h[i]
    down = [flux_inc_down]
    up = [alpha * flux_inc_down + s_up_prime]
    for i in range(n_layer):
        u = beta[i] * (h[i] + gamma[i] * down[-1])
        dn = s_down[..., i] + trans[..., i] * down[-1] + reflect[..., i] * u
        up.append(u)
        down.append(dn)
    return torch.stack(up, -1), torch.stack(down, -1)


def nonscat_recurrence(trans, s_down, s_up, diffuse_albedo, flux_inc_down, surface_source):
    """Two-sweep recurrence for a non-scattering column; upward surface flux = source + albedo * down."""
    n_layer = trans.shape[-1]
    down = [flux_inc_down]
    for i in range(n_layer):
        down.append(s_down[..., i] + trans[..., i] * down[-1])
    up = [None] * (n_layer + 1)
    up[n_layer] = surface_source + diffuse_albedo * down[-1]
    for i in range(n_layer - 1, -1, -1):
        up[i] = s_up[..., i] + trans[..., i] * up[i + 1]
    return torch.stack(up, -1), torch.stack(down, -1)


def solver_no_scat(trans, s_down, s_up, diffuse_albedo, flux_inc_down, d_planck_flux_surface):
    """Fluxes in a non-scattering column (solver_no_scat.F90). Returns ``flux_up, flux_down``."""
    return nonscat_recurrence(trans, s_down, s_up, diffuse_albedo, flux_inc_down,
                              (1.0 - diffuse_albedo) * d_planck_flux_surface)


def gas_flux_sources_ir(tau_gas, diff_planck, diffusivity_factor=DIFFUSIVITY_FACTOR_MINOR, stable=False):
    """Transmission and upward source of monochromatic_gas_flux.F90 (IR); the downward source is its negative.

    ``stable=True`` evaluates 1 - trans with expm1 and the float64 regularisation (for float32)."""
    x = diffusivity_factor * tau_gas
    trans = torch.exp(-x)
    if stable:
        source_up = (-expm1(-x) + SQ_EPS_64) * diff_planck / (x + SQ_EPS_64)
    else:
        sq_eps_r = _eps(tau_gas) ** 0.5
        source_up = (1.0 - trans + sq_eps_r) * diff_planck / (diffusivity_factor * tau_gas + sq_eps_r)
    return trans, source_up


def gas_flux_recurrence_ir(trans, source_up, flux_inc_down, d_planck_flux_surface, diffuse_albedo):
    """Recurrence of monochromatic_gas_flux.F90 (IR), same operation order as the Fortran."""
    n_layer = trans.shape[-1]
    down = [flux_inc_down]
    for i in range(n_layer):
        down.append(trans[..., i] * down[-1] + (-source_up[..., i]))
    up = [None] * (n_layer + 1)
    up[n_layer] = d_planck_flux_surface + diffuse_albedo * down[-1]
    for i in range(n_layer - 1, -1, -1):
        up[i] = trans[..., i] * up[i + 1] + source_up[..., i]
    return torch.stack(up, -1), torch.stack(down, -1)


def monochromatic_gas_flux_ir(tau_gas, flux_inc_down, diff_planck, d_planck_flux_surface, diffuse_albedo,
                              diffusivity_factor=DIFFUSIVITY_FACTOR_MINOR, stable=False):
    """Non-scattering IR fluxes for a single gas (monochromatic_gas_flux.F90, isolir = IR)."""
    trans, source_up = gas_flux_sources_ir(tau_gas, diff_planck, diffusivity_factor, stable)
    return gas_flux_recurrence_ir(trans, source_up, flux_inc_down, d_planck_flux_surface, diffuse_albedo)


# ---------------------------------------------------------------------------
# Numerically stable variants for low precision (float32 on MPS/GPU)
#
# The Fortran expressions subtract nearly equal numbers in optically thin layers
# (1 - exp(-x), T0 - T, 1 - T*T0, ...) and regularise with sqrt(EPSILON) of the
# working precision.  In float32 that costs ~0.1 W/m2.  The functions below
# evaluate the same quantities without cancellation and with the float64
# regularisation constants, so float32 results track the float64 reference.
# ---------------------------------------------------------------------------
EPS_64 = float(torch.finfo(torch.float64).eps)
SQ_EPS_64 = EPS_64 ** 0.5
TOL_FAST_LW_64 = SQ_EPS_64 ** 0.5


def _series(x, coeffs):
    """Horner evaluation of sum_n coeffs[n] * x**n."""
    out = torch.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        out = out * x + c
    return out


def _exp_m1_plus_x(x):
    """g(x) = exp(-x) - 1 + x, accurate for small x (x >= 0)."""
    import math
    n_terms = 16
    coeffs = [0.0, 0.0] + [(-1.0) ** n / math.factorial(n) for n in range(2, n_terms)]
    small = x < 1.0
    xs = torch.where(small, x, torch.zeros_like(x))
    return torch.where(small, _series(xs, coeffs), expm1(-x) + x)


def _h_fast_lw(x):
    """h(x) = (2 + x) exp(-x) - 2 + x = sum_{n>=3} (-1)^n (2-n) x^n / n!, accurate for small x."""
    import math
    n_terms = 24
    coeffs = [0.0, 0.0, 0.0] + [(-1.0) ** n * (2 - n) / math.factorial(n) for n in range(3, n_terms)]
    small = x < 2.0
    xs = torch.where(small, x, torch.zeros_like(x))
    return torch.where(small, _series(xs, coeffs), (2.0 + x) * torch.exp(-x) - 2.0 + x)


def two_coeff_fast_lw_stable(tau, l_ir_source_quad=True):
    """Cancellation-free equivalent of :func:`two_coeff_fast_lw` with float64 regularisation."""
    sq = SQ_EPS_64
    x = 1.66 * tau
    trans = torch.exp(-x)
    u = -expm1(-x)  # 1 - trans
    denom = x + sq
    source_1 = (u + sq) / denom
    if not l_ir_source_quad:
        return trans, source_1, None
    # thick: -(1 + T - 2 s1)/(x + sq) with 1 + T - 2 s1 = (2 h(x)/x ... ) rewritten as (h(x) - u*sq)/(x + sq)
    # derivation: (2 - u)(x + sq) - 2(u + sq) = 2(x - u) - u*x - u*sq = h(x) - u*sq
    thick = -(_h_fast_lw(x) - u * sq) / (denom * denom)
    # thin: -(T - 1 + x)/(x + sq)
    thin = -_exp_m1_plus_x(x) / denom
    source_2 = torch.where(tau > TOL_FAST_LW_64, thick, thin)
    return trans, source_1, source_2


def _solar_coeff_stable_core(omega, asymmetry, tau, m, s, d, lam, i_2stream):
    if i_2stream == IP_DISCRETE_ORD:
        ksi_0 = 1.7320508075688772 * asymmetry / m
    else:
        ksi_0 = 1.5 * asymmetry / m
    factor = 0.5 * omega * m / ((lam - m) * (lam + m))
    gamma_up = factor * (s - m - ksi_0 * (d - m))
    gamma_down = factor * (s + m + ksi_0 * (d + m))
    gamma = (s - lam) / (s + lam)
    one_m_g2 = 4.0 * s * lam / ((s + lam) * (s + lam))  # 1 - gamma^2
    g2 = gamma * gamma
    q = -expm1(-2.0 * lam * tau)  # 1 - E^2
    e = torch.exp(-lam * tau)
    e0 = torch.exp(-m * tau)
    denom = one_m_g2 + g2 * q  # 1 - E^2 gamma^2
    trans = e * one_m_g2 / denom
    reflect = gamma * q / denom
    one_m_tt0 = (one_m_g2 * (-expm1(-(lam + m) * tau)) + g2 * q) / denom  # 1 - T*T0
    # E0 - E without overflow: e0*(1 - exp(-(lam-m)tau)) if lam >= m, else -e*(1 - exp(-(m-lam)tau))
    a = (lam - m) * tau
    e0_m_e = torch.where(a >= 0.0, -e0 * expm1(-torch.clamp(a, min=0.0)),
                         e * expm1(-torch.clamp(-a, min=0.0)))
    t0_m_t = (one_m_g2 * e0_m_e + e0 * g2 * q) / denom  # T0 - T
    source_up = gamma_up * one_m_tt0 - reflect * (1.0 + gamma_down)
    source_down = (1.0 + gamma_down) * t0_m_t - gamma_up * reflect * e0
    return trans, reflect, e0, source_up, source_down


def two_coeff_solar_stable(omega, asymmetry, tau, sec_0, i_2stream, delta=None, coalbedo=None):
    """Cancellation-free equivalent of :func:`two_coeff_solar`.

    Returns ``trans, reflect, trans_0, source_up, source_down``.

    Near the removable singularity lambda = sec_0 SOCRATES scales (s, d, lambda) by
    1 + 32*EPSILON, which in float32 leaves a large cancellation.  Here, for
    |lambda - sec_0| < delta*sec_0 the coefficients are evaluated with (s, d, lambda)
    scaled so that lambda = sec_0*(1 - delta) and sec_0*(1 + delta) -- always exactly
    delta from the singularity -- and interpolated linearly in lambda.  The error is
    O(delta^2) + O(EPSILON/delta), minimised by delta = EPSILON**(1/3).
    """
    s, d = two_coeff_basic(omega, asymmetry, i_2stream, coalbedo)
    lam = torch.sqrt(s * d)
    m = sec_0.unsqueeze(-1)
    if delta is None:
        delta = _eps(lam) ** (1.0 / 3.0)
    near = torch.abs(lam - m) < delta * m
    f_plus = torch.where(near, m * (1.0 + delta) / lam, torch.ones_like(lam))
    f_minus = torch.where(near, m * (1.0 - delta) / lam, torch.ones_like(lam))
    theta = torch.where(near, (lam / m - (1.0 - delta)) / (2.0 * delta), torch.full_like(lam, 0.5))
    plus = _solar_coeff_stable_core(omega, asymmetry, tau, m, f_plus * s, f_plus * d, f_plus * lam, i_2stream)
    minus = _solar_coeff_stable_core(omega, asymmetry, tau, m, f_minus * s, f_minus * d, f_minus * lam, i_2stream)
    # away from the singularity plus == minus, so this returns them unchanged
    return tuple(b + theta * (a - b) for a, b in zip(plus, minus))


def solar_source_stable(flux_inc_direct, trans_0, source_up, source_down, adjust_solar_ke_m1):
    """:func:`solar_source` with ``adjust_solar_ke = 1 + adjust_solar_ke_m1``, avoiding the
    cancellation in ``(source_down - T0) F + T0 * adjust * F``."""
    n_layer = trans_0.shape[-1]
    flux = [flux_inc_direct]
    s_up, s_down = [], []
    for i in range(n_layer):
        prev = flux[-1]
        adj = 1.0 + adjust_solar_ke_m1[..., i]
        cur = prev * trans_0[..., i] * adj
        s_down.append((source_down[..., i] + trans_0[..., i] * adjust_solar_ke_m1[..., i]) * prev)
        s_up.append(source_up[..., i] * prev)
        flux.append(cur)
    return torch.stack(flux, -1), torch.stack(s_down, -1), torch.stack(s_up, -1)


def single_scattering_stable(k_grey_tot, k_ext_scat, k_grey_abs, k_gas_abs, d_mass):
    """:func:`single_scattering` (full scattering) that also returns the co-albedo 1 - omega.

    ``k_grey_abs`` is the grey absorption (``k_grey_tot - k_ext_scat``) computed without that
    subtraction.  SOCRATES limits omega to 1 - 32*EPSILON; the float64 limit is used here, carried
    by the co-albedo so that it survives in float32.
    """
    k_total = k_grey_tot + k_gas_abs
    tau = k_total * d_mass
    denom = k_total + _tiny(k_total)
    omega = torch.clamp(k_ext_scat / denom, max=1.0)
    coalbedo = torch.clamp((k_grey_abs + k_gas_abs) / denom, min=32.0 * EPS_64, max=1.0)
    return tau, omega, coalbedo


def rescale_tau_omega_coalbedo(tau, omega, coalbedo, forward_scatter):
    """Delta-rescaling that also rescales the co-albedo: 1 - omega' = (1 - omega)/(1 - omega f)."""
    one_m_wf = 1.0 - omega * forward_scatter
    return tau * one_m_wf, omega * (1.0 - forward_scatter) / one_m_wf, coalbedo / one_m_wf
