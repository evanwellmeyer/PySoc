# PySoc — SOCRATES (as used in Isca) in PyTorch

Goal: a PyTorch re-implementation of the SOCRATES two-stream radiative transfer
code, configured exactly as Isca's `socrates_interface` runs it, that runs on
CPU / CUDA / Apple MPS, is differentiable, and reproduces the Fortran to
round-off.

## Ground truth

* SOCRATES Fortran: `reference/socrates` (github.com/MetOffice/socrates, main @ 76a675b, BSD-3).
* Isca interface: `reference/isca/src/atmos_param/socrates/interface`.
* `reference/build/build_socrates.sh` builds `modules_core + radiance_core` with gfortran.
* `reference/harness/soc_ref` = Isca's `socrates_calc` (Isca's own `read_control`,
  `set_control`, `set_dimen`, `set_atm`, `set_bound`, `set_cld`, `set_aer`) driven by
  binary I/O. `tools/refcases.py` builds Isca-like columns (McClatchey atmospheres +
  random perturbations), runs the driver, and reads its output.
* `soc_ref_trace` is the same driver built with a DR_HOOK that logs every routine entry,
  used to pin down exactly which code paths Isca exercises.

### Isca configuration (from `read_control.F90`)

| | SW (`sp_sw_ga7`, 6 bands, 41 g-points) | LW (`sp_lw_ga7`, 9 bands, 81 g-points) |
|---|---|---|
| two-stream | PIFM80 | Elsasser (D = 1.66) |
| scattering | full, Rayleigh on | `ip_no_scatter_ext` (no scattering) |
| solver (clear) | `solver_homogen_direct` | `solver_no_scat` |
| source | direct beam | quadratic IR source (`l_ir_source_quad`) |
| gas overlap | `k_eqv_scl` (major gas resolved, minor gases → equivalent extinction) | same |
| k-terms | p/T lookup tables (`_k` files) + `scale_absorb` for others | same |
| continuum | water vapour self/foreign (`rescale_continuum`) | same |
| delta-rescaling | on | on |
| surface | grey albedo (diffuse = direct) | emissivity → albedo = 1 − ε |
| clouds | off unless `do_cloud_simple` (liquid only, max-random, `solver_mix_direct_hogan`) | same |

Finding: Isca never calls `compress_spectrum`, and the core never reads `l_h2o`, `l_co2`, …,
so the `inc_*` namelist switches do nothing. Every gas in the spectral file is active, at
the mixing ratios set in `set_atm` (CFCs, N2O and CH4 at the namelist defaults; OCS and
SO2 at zero). The port reproduces this by default and offers a real on/off option.

Other findings while porting:
* Isca builds with `-fdefault-real-8 -fdefault-double-8`, so real-typed namelist defaults
  such as `n2o_mix_ratio = 4.945e-07` are double precision. The reference is built the same way.
* LW band 4 (CO₂) uses `ip_scale_wenyi`, whose hard-coded tables in `scale_wenyi.F90` are dumped
  from the Fortran into `pysoc/data/scale_wenyi.npz`.
* With `l_grey_single = .FALSE.`, minor gases with a single k-term still go through
  `monochromatic_gas_flux` (LW: 85 minor-term solves = 56 multi-term + 29 single-term).
* Reference builds use `-ffp-contract=off`, so FMA contraction doesn't blur comparisons.
* In the shortwave, `i_direct_tau` is left unset (`imdi`), so no direct-beam delta-rescaling is applied.

### Call tree actually executed (clear sky, from the trace)

Both regions: `inter_pt_lookup`, `rescale_continuum`, `scale_absorb`, `grey_opt_prop`,
`rescale_phase_fnc`, `solve_band_k_eqv_scl`, `monochromatic_radiance`,
`monochromatic_radiance_tseq`, `single_scattering_all`, `single_scattering`,
`rescale_tau_omega`, `two_stream`, `column_solver`, `augment_radiance`,
`augment_channel`, `copy_clr_full`.
SW only: `two_coeff`, `two_coeff_basic`, `solar_coefficient_basic`,
`trans_source_coeff`, `solar_source`, `solver_homogen_direct`.
LW only: `diff_planck_source`, `monochromatic_gas_flux`, `two_coeff_fast_lw`,
`ir_source`, `solver_no_scat`, `adjust_ir_radiance`.

## Design

* **Package** `pysoc/`: pure functions for the kernels, plus `nn.Module` wrappers that keep
  spectral data as buffers, so `.to(device, dtype)` works.
* **Batching**: instead of Fortran's loops over band → k-term → profile, all g-points
  (band, k-term) are flattened into one leading dimension. Gas optics produce
  `tau[g, profile, layer]`, and one vectorised two-stream solve handles every g-point.
  Only the tridiagonal elimination loops, over layers. Minor-gas equivalent-extinction
  terms are padded to the largest k-term count and masked.
* **Precision**: every function is dtype-generic, and tolerances come from
  `torch.finfo(dtype)`, just as Fortran uses `EPSILON` / `TINY`. The Fortran comparison
  runs in float64 on CPU. MPS supports float32 only, so the float32 output is compared
  with float64 torch using physically meaningful tolerances.
* **Differentiability**: no in-place writes to tensors that need grads, and no data-dependent
  Python branching (use `torch.where` with safe denominators so neither branch makes
  NaN/inf gradients).
* **Explicit scope**: raise `NotImplementedError` for any spectral-file feature or option
  not on the Isca path (SES2, spherical harmonics, MCICA, aerosols, …) rather than
  silently ignoring it.

## Phases — each ends with a test gate against the Fortran

Status: phases 0–7 and 9 are done (108 tests, `pytest tests/`). Phase 8 (extras) is optional.

0. ✅ **Infrastructure**: reference build, Isca-mirroring driver, case generator, call trace
   (`reference/build_all.sh`).
1. ✅ **Spectral file reader** (`pysoc/spectral_file.py`). *Gate:* SOCRATES `read_spectrum` arrays
   dumped by `dump_spectrum` match exactly (12 tests, bitwise).
2. ✅ **Two-stream kernels** (`pysoc/twostream.py`). *Gate:* `soc_unit` runs each routine on random and
   edge-case inputs. 98–99% of values are bitwise identical; well-conditioned values agree to ≤ 4e-16;
   the rest agree within explicit round-off bounds.
3. ✅ **Gas and grey optics** (`pysoc/optics.py`, `pysoc/core.py`). *Gate:* the instrumented
   `soc_ref_dump` intermediates (lookup weights, k-terms of every gas, continuum/Rayleigh, Planck terms,
   k_eqv, adjust_solar_ke, fluxes for all 122 g-points) match.
4. ✅ **Band solve and assembly** (`SocratesLW`, `SocratesSW`). *Gate:* `soc_ref` totals, per-band fluxes and
   heating rates on standard, coarse (10 levels), fine (90 levels), dry/cold, hot/high-CO₂ and extreme inputs.
   SW agrees to 1e-10 relative / 1e-9 W m⁻². LW agrees to ≤ 2e-8 W m⁻², 5× tighter than the Fortran's own
   sensitivity to a one-ulp change in `exp` (explained in the tests).
5. ✅ **Devices, precision, gradients, speed.**
   * Float32 (MPS or CPU) uses cancellation-free kernels (`numerics="auto"` → `"stable"`):
     |ΔF| ≤ 2e-3 W m⁻² and |ΔHR| ≤ 1.3e-4 K/day against float64. The naive float32 port was off by
     0.1 W m⁻² and 0.065 K/day.
   * MPS's `torch.expm1` is inaccurate (it is evaluated as exp−1), so `pysoc/mathutil.py` supplies a series version.
   * Gradients are checked against finite differences and stay finite at night, with grazing sun and with zero humidity.
   * `compile=True`: per-kernel `torch.compile`, validated against eager on first use. The MPS inductor
     backend miscompiled one fused kernel, and the validator falls back to eager when that happens.
   * Speed: see "Benchmark" below. On an M3 Max, compiled MPS float32 matches about 3–6 Fortran cores
     and is 2–3× slower than 12-core Fortran.
6. ✅ **Isca interface** (`pysoc/isca.py`, `IscaSocrates`). *Gate:* `soc_isca` (Isca's preprocessing copied
   verbatim around two socrates_calc calls, with the T update in between) matches heating rates, fluxes,
   half-level T and spectral OLR for default, namelist-modified and dry/no-ozone settings.
7. ✅ **Clouds** (Isca simple cloud; `pysoc/clouds.py`, `cloud=LiquidCloud(...)` on the models, `cf_rad/reff_rad/qcl_rad`
   on `IscaSocrates`). Traced path: `set_cloud_pointer`, `set_cloud_geometry`, `overlap_coupled`, `opt_prop_water_cloud`
   (Padé-2 droplets), `mix_column`, `two_coeff_cloud`, `mixed_solar_source`, `solver_mix_direct_hogan`.
   *Gate:* `soc_ref` with `do_clouds` (broken and overcast skies, sub-threshold fractions, reff outside the valid range)
   matches all-sky and clear fluxes, clear band fluxes, heating rates and total cloud cover. SW agrees to 1e-9 W m⁻²
   and LW to 2e-8. `soc_isca` with clouds matches end to end. Gradients agree with finite differences; float32 on
   CPU/MPS is within 1.3e-3 W m⁻² and 1.6e-4 K/day.
   Findings:
   * Isca's liquid clouds use drop type 5 (Padé 2). Ice type 11 (Baran) is set up but always receives zero ice,
     so it contributes nothing and is not ported.
   * SOCRATES's `n_cloud_top` depends on the whole batch, but results change by round-off only (verified),
     so the port treats all layers as potentially cloudy.
   * With clouds on, the LW uses the general `trans_source_coeff` IR expressions instead of `two_coeff_fast_lw`.
     These are algebraically identical for ω = 0, and the reference path uses the general form.
   * `mix_column` passes the overlap coefficients to `mixed_solar_source` and the Hogan solver in swapped order
     (fc ↔ cf); the port follows it exactly.
   * float32: SOCRATES caps ω at 1 − 32·ε(working precision), which in float32 over-absorbs in clouds
     (0.045 W m⁻²). The stable path carries the co-albedo (1 − ω) separately, computed from absorption
     coefficients, with the float64 cap.
   * Isca's `idealized_moist_phys` converts `reff_rad` from microns to metres before calling SOCRATES.
8. **Extras** (optional): other spectral files (GA9, planetary), SES2, radiation time-stepping and astronomy helpers.
9. ✅ **Teaching notebooks** (`notebooks/`, Colab). Supporting code: `pysoc/column.py` (idealised columns on Isca's
   `uneven_sigma` grid, Manabe–Wetherald humidity, ozone scaled to a Dobson total, `liquid_cloud`), `pysoc/spectra.py`
   (GA7 files from the local checkout, or downloaded at the validated SOCRATES commit with SHA-256 checks),
   `pyproject.toml`. `IscaSocrates` now also accepts scalars for surface/gas inputs, returns per-band fluxes and,
   on request, the LW/SW intermediates (including per-k-term optical depths). Non-finite temperatures now give NaN
   instead of an index error in the p/T lookup. *Gate:* `tests/test_column.py`; both notebooks run end to end in a
   fresh Python 3.12 environment with PySoc installed from the repository and the spectral files downloaded.

## Benchmark (`tools/benchmark_parallel.py`, results in `tools/benchmark_parallel_results.json`)

8192 columns × 40 layers, Apple M3 Max (12 performance + 4 efficiency cores), mains power, High Power mode.
Only the radiation calculation is timed, with no file I/O or spectral-file reading. The Fortran
(`reference/harness/soc_bench`) calls SOCRATES in Isca's chunks of 16 columns; the columns are split across
N processes as Isca's MPI decomposition would split them, all released together, and the slowest process sets
the time. PySoc runs on MPS in float32 with inputs already on the device.

| case | Fortran, 1 core | Fortran, 12 cores | PySoc MPS, compiled | PySoc MPS, eager |
|---|---|---|---|---|
| clear LW  | 0.70 s | 0.063 s | 0.11 s  | 0.31 s |
| clear SW  | 0.45 s | 0.041 s | 0.091 s | 0.29 s |
| cloudy LW | 1.33 s | 0.12 s  | 0.29 s  | 0.61 s |
| cloudy SW | 0.79 s | 0.071 s | 0.24 s  | 0.62 s |

* Fortran scales almost linearly to the 12 performance cores (11×); the efficiency cores add little.
* Isca's chunking matters: one 8192-column call is up to 2× slower on multiple cores (cache effects).
* Earlier single-core figures in this project (0.99 s / 0.73 s clear, 3.3 s cloudy) were taken on battery with
  Low Power Mode on and, for clouds, included file I/O; they are superseded by this table.
