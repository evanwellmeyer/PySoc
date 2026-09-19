# Notebooks

Two notebooks for undergraduate classes, built on PySoc (SOCRATES as used in the Isca climate model).
They run in Google Colab with no local setup: the first cell installs PySoc from this repository
and downloads the SOCRATES spectral files. The first cell takes about a minute; after that, each cell
takes a few seconds or less.

| Notebook | Contents |
|---|---|
| [01_how_radiation_works.ipynb](01_how_radiation_works.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/PySoc/blob/main/notebooks/01_how_radiation_works.ipynb) | An interactive introduction: shortwave and longwave radiation, the model column (with a column explorer), fluxes and heating rates, the greenhouse effect band by band, k-terms and emission levels, scattering and the sun's angle, clouds. Students move sliders and tick boxes and describe what changes. |
| [02_perturbation_experiments.ipynb](02_perturbation_experiments.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/PySoc/blob/main/notebooks/02_perturbation_experiments.ipynb) | Eight perturbation activities: students change CO₂, the Sun, the column's temperature and humidity, a cloud, or the surface albedo, and describe how the column responds (energy budget, heating rates, net flux at each level, OLR by band). Topics: CO₂ forcing and its logarithmic dependence, the stratospheric-cooling fingerprint, Planck, water-vapour and lapse-rate feedbacks, gradient-based sensitivity kernels, clouds, surface albedo, and a challenge that builds a Manabe–Wetherald radiative–convective equilibrium model to estimate climate sensitivity (under a minute of computing). |

All calculations use clear or simply clouded single columns on CPU, in float64. Keep Colab's default CPU runtime:
a GPU runtime would be slower, because one or two columns are far too small to keep a GPU busy (on an Apple M3
Max a single-column call takes 13 ms on the CPU and 44 ms on the GPU; the two break even around 20 columns).

## Running locally

```bash
pip install -e ".[notebooks]" jupyter
jupyter lab notebooks/
```

Run from inside the repository, the notebooks import the local `pysoc` package and use the spectral files in
`reference/socrates` if they are there (otherwise the files are downloaded to `~/.cache/pysoc`).

## For instructors

* The code in every cell is hidden behind a `#@title` (Colab's form view); students click *Show code* to read it.
* The controls are ipywidgets sliders, tick boxes and menus that redraw the plots while they move (each redraw
  takes a few tenths of a second). The Activity 8 experiments take 5–15 s, so they wait for a **Run the model**
  button, and the cloud sweep in Activity 6 redraws when its slider is let go. Students write their descriptions
  in a *Your notes* cell under each activity.
* Prompts ask students to describe the column's response, with a short *Explain* question per activity. Hints
  are in collapsible `<details>` blocks; there are no answer keys. The notebooks are stored without outputs.
* Values the activities produce (clear sky, default settings): OLR ≈ 264 W/m²; doubled-CO₂ forcing ≈ 2.8 W/m²
  at the top of the atmosphere and ≈ 4.8 W/m² at 200 hPa; Planck response ≈ 4.0 W/m²/K, and ≈ 2.2 W/m²/K with
  fixed relative humidity; radiative–convective equilibrium surface temperature ≈ 289 K with convection and ≈ 317 K
  with radiation alone; climate sensitivity ≈ 2.3 K.
* Colab installs from the `main` branch, so changes to the notebooks or to `pysoc` reach students once they are
  pushed.
