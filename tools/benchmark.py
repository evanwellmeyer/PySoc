"""Timing of the PyTorch SOCRATES port against the Fortran reference driver."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import torch

from reference_runs import atmosphere, lw_model, sw_model
from tools.refcases import HARNESS, IP_INFRA_RED, IP_SOLAR, SP_LW_GA7, SP_SW_GA7, make_cases, write_input


def fortran_time(region, atm, repeats=2):
    sp, isolir = (SP_LW_GA7, IP_INFRA_RED) if region == "lw" else (SP_SW_GA7, IP_SOLAR)
    with tempfile.TemporaryDirectory() as td:
        fin, fout = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        write_input(fin, isolir, atm)
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            subprocess.run([str(HARNESS / "soc_ref"), str(sp), fin, fout], check=True)
            best = min(best, time.perf_counter() - t0)
        one = atm.subset(slice(0, 1))
        write_input(fin, isolir, one)
        overhead = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            subprocess.run([str(HARNESS / "soc_ref"), str(sp), fin, fout], check=True)
            overhead = min(overhead, time.perf_counter() - t0)
    return best - overhead


def torch_time(region, atm, dtype, device, repeats=3, numerics="auto", grad=False, compile=False):
    model = (lw_model() if region == "lw" else sw_model()).to(device=device, dtype=dtype)
    model.numerics = numerics
    model.compile = compile
    a = atmosphere(atm, dtype, device)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
    args = (T(atm.t_surf), T(atm.emissivity)) if region == "lw" else (T(atm.coszen), T(atm.solar_irrad), T(atm.albedo))

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def once():
        if grad:
            a.t.requires_grad_(True)
            out = model(a, *args)
            out["flux_divergence"].sum().backward()
            a.t.grad = None
        else:
            with torch.no_grad():
                model(a, *args)
        sync()

    once()  # warm-up
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        best = min(best, time.perf_counter() - t0)
    model.cpu().double()
    model.numerics = "auto"
    model.compile = False
    a.t.requires_grad_(False)
    return best


if __name__ == "__main__":
    n_profile = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    n_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    atm = make_cases(n_profile=n_profile, n_layer=n_layer, seed=0)
    print(f"{n_profile} columns x {n_layer} layers; torch {torch.__version__}, threads={torch.get_num_threads()}")
    for region in ("lw", "sw"):
        tf = fortran_time(region, atm)
        print(f"{region}: Fortran (1 core)            {tf:7.3f} s")
        configs = [("cpu", torch.float64), ("cpu", torch.float32)]
        if torch.backends.mps.is_available():
            configs.append(("mps", torch.float32))
        for dev, dt in configs:
            t = torch_time(region, atm, dt, dev)
            print(f"{region}: torch {dev} {str(dt):14s}      {t:7.3f} s   ({tf / t:5.1f}x Fortran)")
        if torch.backends.mps.is_available():
            t = torch_time(region, atm, torch.float32, "mps", compile=True)
            print(f"{region}: torch mps float32 compiled    {t:7.3f} s   ({tf / t:5.1f}x Fortran)")
            t = torch_time(region, atm, torch.float32, "mps", grad=True)
            print(f"{region}: torch mps float32 fwd+bwd     {t:7.3f} s")
