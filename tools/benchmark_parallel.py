"""Multi-process Fortran SOCRATES vs PySoc on MPS.

Fortran: the 8192 columns are split across N independent processes (as Isca's MPI
decomposition would), each calling SOCRATES in chunks of ``chunk_size`` columns
(Isca's socrates_interface, default 16).  All processes load their inputs and the
spectral files first; each timed round then starts them together and records the
slowest one (the wall time of the parallel step).  Only the radiation calculation
is timed: no file I/O, no spectral-file reading.

PySoc: the same columns on MPS in float32 (compiled and eager), inputs already on the device.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import torch

from pysoc.clouds import LiquidCloud
from reference_runs import atmosphere, lw_model, sw_model
from test_clouds import cloudy_case
from tools.refcases import HARNESS, IP_INFRA_RED, IP_SOLAR, SP_LW_GA7, SP_SW_GA7, make_cases, write_input

N_PROFILE, N_LAYER = 8192, 40
PROCESSES = [1, 2, 4, 8, 12, 16]
CHUNKS = [16, 8192]
ROUNDS = 3


def power_state():
    out = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
    low = [ln.split()[-1] for ln in out.splitlines() if ln.strip().startswith(("powermode", "lowpowermode"))]
    return {"source": batt.strip(), "low_power_mode": low[0] if low else "unknown"}


def fortran_parallel(region, atm, do_clouds, n_proc, chunk):
    sp, isolir = (SP_LW_GA7, IP_INFRA_RED) if region == "lw" else (SP_SW_GA7, IP_SOLAR)
    bounds = np.linspace(0, atm.n_profile, n_proc + 1).astype(int)
    with tempfile.TemporaryDirectory() as td:
        procs = []
        for i in range(n_proc):
            path = os.path.join(td, f"in{i}.bin")
            write_input(path, isolir, atm.subset(slice(bounds[i], bounds[i + 1])), do_clouds)
            p = subprocess.Popen([str(HARNESS / "soc_bench"), str(sp), path, str(chunk)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True, bufsize=1)
            procs.append(p)
        for p in procs:
            assert p.stdout.readline().strip() == "READY"
        rounds, checksum = [], None
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for p in procs:
                p.stdin.write("go\n")
                p.stdin.flush()
            times, sums = [], []
            for p in procs:
                _, t, c = p.stdout.readline().split()
                times.append(float(t))
                sums.append(float(c))
            rounds.append(dict(slowest=max(times), outside=time.perf_counter() - t0))
            checksum = sum(sums)
        for p in procs:
            p.stdin.write("quit\n")
            p.stdin.flush()
            p.wait()
    best = min(rounds, key=lambda r: r["slowest"])
    return best["slowest"], best["outside"], checksum


def pysoc_mps(region, atm, do_clouds, compile_):
    dev, dt = "mps", torch.float32
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)  # noqa: E731
    a = atmosphere(atm, dt, dev)
    cloud = LiquidCloud(T(atm.cld_frac), T(atm.mmr_cl), T(atm.reff)) if do_clouds else None
    model = (lw_model() if region == "lw" else sw_model()).to(device=dev, dtype=dt)
    model.compile = compile_
    args = (T(atm.t_surf), T(atm.emissivity)) if region == "lw" else (T(atm.coszen), T(atm.solar_irrad), T(atm.albedo))
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(2):
                model(a, *args, cloud=cloud)
                torch.mps.synchronize()
            best = np.inf
            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                model(a, *args, cloud=cloud)
                torch.mps.synchronize()
                best = min(best, time.perf_counter() - t0)
    finally:
        model.compile = False
        model.cpu().double()
    return best


def main():
    results = {"power": power_state(), "cpu": "Apple M3 Max, 12 performance + 4 efficiency cores",
               "columns": N_PROFILE, "layers": N_LAYER, "cases": {}}
    print(results["power"], flush=True)
    cases = {"clear": make_cases(n_profile=N_PROFILE, n_layer=N_LAYER, seed=0),
             "cloudy": cloudy_case(seed=0, n_profile=N_PROFILE, n_layer=N_LAYER)}
    for sky, atm in cases.items():
        for region in ("lw", "sw"):
            key = f"{sky}_{region}"
            do_clouds = sky == "cloudy"
            entry = {"fortran": {}, "pysoc": {}}
            ref_sum = None
            for chunk in CHUNKS:
                for n in PROCESSES:
                    t, outside, s = fortran_parallel(region, atm, do_clouds, n, chunk)
                    if ref_sum is None:
                        ref_sum = s
                    rel = abs(s - ref_sum) / abs(ref_sum)
                    assert rel < 1e-12, (key, chunk, n, rel)
                    entry["fortran"][f"chunk{chunk}_proc{n}"] = t
                    print(f"{key:10s} Fortran chunk={chunk:5d} processes={n:2d}: {t:7.3f} s "
                          f"(round incl. dispatch {outside:.3f} s)", flush=True)
            for comp in (True, False):
                t = pysoc_mps(region, atm, do_clouds, comp)
                entry["pysoc"]["mps_float32_" + ("compiled" if comp else "eager")] = t
                print(f"{key:10s} PySoc MPS float32 {'compiled' if comp else 'eager   '}: {t:7.3f} s", flush=True)
            results["cases"][key] = entry
    results["power_after"] = power_state()
    out = ROOT / "tools" / "benchmark_parallel_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
