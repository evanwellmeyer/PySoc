"""Compare float32 (CPU/MPS, stable numerics) against the float64 reference model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import numpy as np
import torch

from reference_runs import run_python, standard_cases, lw_model, sw_model
from tools.refcases import make_cases

SECONDS_PER_DAY = 86400.0


def report(atm, label):
    rows = []
    for region in ("lw", "sw"):
        model = lw_model() if region == "lw" else sw_model()
        model.numerics = "auto"
        ref = {k: v.numpy() for k, v in run_python(region, atm, return_intermediates=False).items()}
        hr_ref = ref["flux_divergence"] / atm.heat_capacity * SECONDS_PER_DAY
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for dev in devices:
            for numerics in ("reference", "stable"):
                model.numerics = numerics
                o = run_python(region, atm, dtype=torch.float32, device=dev, return_intermediates=False)
                o = {k: v.detach().cpu().double().numpy() for k, v in o.items()}
                df = max(np.abs(o["flux_up"] - ref["flux_up"]).max(), np.abs(o["flux_down"] - ref["flux_down"]).max())
                hr = o["flux_divergence"] / atm.heat_capacity * SECONDS_PER_DAY
                dhr = np.abs(hr - hr_ref)
                rows.append((label, region, dev, numerics, df, dhr.max(), np.sqrt((dhr**2).mean())))
        model.numerics = "auto"
    for r in rows:
        print("%-10s %s %-4s float32 %-9s max|dF| %.2e W/m2   max|dHR| %.2e K/day   rms|dHR| %.2e K/day" % r)


if __name__ == "__main__":
    report(standard_cases(), "standard")
    report(make_cases(n_profile=64, n_layer=60, seed=11), "L60")
