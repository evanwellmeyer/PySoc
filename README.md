# PySoc

A PyTorch port of the SOCRATES radiative transfer code as Isca configures it: the GA7 spectral
files, the Elsasser (LW) and PIFM80 (SW) two-stream schemes, and k-equivalent-extinction gas overlap.
It runs on CPU, CUDA and Apple MPS, is differentiable, and reproduces the Fortran to round-off in float64.
See [PLAN.md](PLAN.md) for the design, the validation gates and the findings made while porting.

## Teaching notebooks

Two Colab notebooks for undergraduates are in [notebooks/](notebooks/README.md):

| | |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/PySoc/blob/main/notebooks/01_how_radiation_works.ipynb) | **How a climate model computes radiation**: an interactive tour of the scheme |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/PySoc/blob/main/notebooks/02_perturbation_experiments.ipynb) | **Climate experiments**: CO₂ forcing, feedbacks, clouds, and a radiative–convective equilibrium |

## Install

```bash
pip install git+https://github.com/evanwellmeyer/PySoc
```

## Quick start

```python
import torch
from pysoc.isca import IscaSocrates, IscaSocratesConfig
from pysoc.spectra import ga7_spectral_files

# the GA7 spectral files: from reference/socrates if present, otherwise downloaded from
# MetOffice/socrates at the validated commit (SHA-256 checked) into ~/.cache/pysoc
lw_file, sw_file = ga7_spectral_files()
model = IscaSocrates(lw_file, sw_file, IscaSocratesConfig(stellar_constant=1370.0, co2_ppmv=300.0))
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

`pysoc.column.make_column` builds idealised columns on Isca's vertical grid (a surface temperature and lapse
rate, a Manabe–Wetherald humidity profile, an ozone layer of a given Dobson-unit total, well-mixed CO₂). It
returns the inputs above as a dict, and tensor arguments give a batch of columns:

```python
from pysoc.column import make_column, liquid_cloud

col = make_column(t_surf=288.0, co2_ppmv=torch.tensor([280.0, 560.0]))    # two columns
out = model(**col, albedo=0.3, coszen=0.5, delta_t=0.0)
cloud = liquid_cloud(col, p_top=700e2, p_bottom=900e2, lwp=100.0)          # g/m2, in-cloud
out = model(**col, **cloud, albedo=0.3, coszen=0.5, delta_t=0.0)
```

The lower-level `pysoc.core.SocratesLW` / `SocratesSW` take SOCRATES-level inputs (`Atmosphere`:
layer p, T, level T, layer mass, density, gas mass mixing ratios).

Options on the models:
* `numerics="auto" | "reference" | "stable"`: `reference` evaluates the Fortran expressions exactly;
  `stable` evaluates them without cancellation, for float32. `auto` picks `stable` for float32.
  `stable` is also ~1000× smoother under finite differences, which helps gradient-based work.
* `compile=True`: `torch.compile` the heavy kernels (2–3× faster than eager on MPS). Each kernel is checked
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

`tools/precision_report.py` reports float32 accuracy; `tools/benchmark_parallel.py` compares PySoc on the GPU
with multi-process Fortran (see the Benchmark section of PLAN.md: on an M3 Max, the GPU matches about 3–6
Fortran cores).

SOCRATES is © Crown Copyright Met Office (BSD-3-Clause), and the tables in `pysoc/data` come from it.
