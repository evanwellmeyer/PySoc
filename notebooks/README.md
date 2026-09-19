# Notebooks

Two notebooks for undergraduate classes, built on PySoc (SOCRATES as used in the Isca climate model).
They run in Google Colab with no local setup: the first cell installs PySoc from this repository
and downloads the SOCRATES spectral files. The first cell takes about a minute; after that, each cell
takes a few seconds or less.

| Notebook | Contents |
|---|---|
| [01_how_radiation_works.ipynb](01_how_radiation_works.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/PySoc/blob/main/notebooks/01_how_radiation_works.ipynb) | An interactive introduction: shortwave and longwave radiation, the model column, fluxes and heating rates, the greenhouse effect band by band, k-terms and emission levels, scattering and the sun's angle, clouds. Sliders and checkboxes throughout. |
| [02_perturbation_experiments.ipynb](02_perturbation_experiments.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/PySoc/blob/main/notebooks/02_perturbation_experiments.ipynb) | Eight activities with predictions and questions: CO₂ forcing, its logarithmic dependence, the stratospheric-cooling fingerprint, Planck and water-vapour feedbacks, gradient-based sensitivity kernels, high and low clouds, surface albedo, and a challenge that builds a Manabe–Wetherald radiative–convective equilibrium model to estimate climate sensitivity (under a minute of computing). |

All calculations use clear or simply clouded single columns on CPU, in float64. Neither notebook needs a GPU.

## Running locally

```bash
pip install -e ".[notebooks]" jupyter
jupyter lab notebooks/
```

Run from inside the repository, the notebooks import the local `pysoc` package and use the spectral files in
`reference/socrates` if they are there (otherwise the files are downloaded to `~/.cache/pysoc`).

## For instructors

* The notebooks are stored without outputs. Hints are in collapsible `<details>` blocks; there are no answer keys.
* Values the activities produce (clear sky, default column): OLR ≈ 264 W/m²; doubled-CO₂ forcing ≈ 2.6 W/m²
  at the top of the atmosphere and ≈ 4.8 W/m² at 200 hPa; Planck response ≈ 4.0 W/m²/K, and ≈ 2.2 W/m²/K with
  fixed relative humidity; radiative–convective equilibrium climate sensitivity ≈ 2.3 K.
* Colab installs from the `main` branch, so changes to the notebooks or to `pysoc` reach students once they are
  pushed.
