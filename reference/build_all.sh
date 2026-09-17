#!/bin/bash
# Build every Fortran reference artefact used by the test-suite.
#   build/obj        SOCRATES core, compiled like Isca (-fdefault-real-8) without FMA contraction
#   build/obj_dump   same, with instrumented radiance_calc/solve_band_k_eqv_scl (SOC_DUMP=<file>)
#   build/obj_trace  same, with a DR_HOOK that logs routine entries (call-tree tracing)
#   harness/soc_ref, soc_ref_dump, soc_ref_trace, soc_unit, dump_spectrum, dump_wenyi
set -eu
R="$(cd "$(dirname "$0")" && pwd)"
FLAGS="-ffp-contract=off -fdefault-real-8 -fdefault-double-8"
cd "$R/build"
./build_socrates.sh obj $FLAGS | tail -1
OVERRIDE_DIR="$R/build/dump_override" ./build_socrates.sh obj_dump $FLAGS | tail -1
OVERRIDE_DIR="$R/build/trace_override" ./build_socrates.sh obj_trace $FLAGS | tail -1
"$R/harness/build_harness.sh" "$R/build/obj" "$R/harness/soc_ref"
"$R/harness/build_harness.sh" "$R/build/obj_dump" "$R/harness/soc_ref_dump"
"$R/harness/build_harness.sh" "$R/build/obj_trace" "$R/harness/soc_ref_trace"
"$R/harness/build_tools.sh" "$R/build/obj"
