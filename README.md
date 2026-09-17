# PySoc

A PyTorch port of the SOCRATES radiative transfer code as Isca configures it: the GA7 spectral
files, the Elsasser (LW) and PIFM80 (SW) two-stream schemes, and k-equivalent-extinction gas overlap.
It runs on CPU, CUDA and Apple MPS, is differentiable, and reproduces the Fortran to round-off in float64.
See [PLAN.md](PLAN.md) for the design, the validation gates and the findings made while porting.

## Quick start

```python
import torch
from pysoc.isca import IscaSocrates, IscaSocratesConfig

model = IscaSocrates("reference/socrates/data/spectra/ga7/sp_lw_ga7",
                     "reference/socrates/data/spectra/ga7/sp_sw_ga7",
                     IscaSocratesConfig(stellar_constant=1370.0, co2_ppmv=300.0))
model = model.to("mps", torch.float32)        # or keep float64 on CPU for reference results

out = model(temp, q, p_full, p_half, z_full, z_half,  # (..., L) / (..., L+1), index 0 = top
            t_surf, albedo, coszen,                   # (...)
            delta_t=1800.0, ozone=o3_mmr)             # ozone as mass mixing ratio
out["tdt_rad"]      # K/s
out["soc_olr"]      # W/m2, plus the other Isca soc_* diagnostics (and *_clr clear-sky versions)

# with Isca's simple clouds (reff in metres, as passed to run_socrates; qcl specific humidity)
out = model(temp, q, p_full, p_half, z_full, z_half, t_surf, albedo, coszen, delta_t=1800.0,
            ozone=o3_mmr, cf_rad=cf, reff_rad=1e-6 * reff_microns, qcl_rad=qcl)
out["soc_tot_cloud_cover"]
```

The lower-level `pysoc.core.SocratesLW` / `SocratesSW` take SOCRATES-level inputs (`Atmosphere`:
layer p, T, level T, layer mass, density, gas mass mixing ratios).

Options on the models:
* `numerics="auto" | "reference" | "stable"`: `reference` evaluates the Fortran expressions exactly;
  `stable` evaluates them without cancellation, for float32. `auto` picks `stable` for float32.
  `stable` is also ~1000× smoother under finite differences, which helps gradient-based work.
* `compile=True`: `torch.compile` the heavy kernels (3–4× faster on MPS). Each kernel is checked
  against eager execution on first use.
* On CPU in float64, `torch.set_flush_denormal(True)` avoids a ~3× denormal-float slowdown in SW.

## Tests

The reference Fortran is built from `reference/socrates` (MetOffice/socrates) and Isca's interface:

```bash
reference/build_all.sh
```

```bash
pytest tests/
```

`tools/precision_report.py` and `tools/benchmark.py` report float32 accuracy and timings.

SOCRATES is © Crown Copyright Met Office (BSD-3-Clause), and the tables in `pysoc/data` come from it.
