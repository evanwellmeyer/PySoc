#!/bin/bash
# Build dump_spectrum and soc_unit against the reference SOCRATES library.
set -eu
H="$(cd "$(dirname "$0")" && pwd)"
LIBDIR="${1:-$H/../build/obj}"
OBJ="$H/obj_tools"
mkdir -p "$OBJ"
FFLAGS="-O2 -ffp-contract=off -ffree-line-length-none -J$OBJ -I$OBJ -I$LIBDIR"
cd "$OBJ"
gfortran $FFLAGS -c "$H/dump_io.F90"
for p in dump_spectrum soc_unit dump_wenyi; do
  gfortran $FFLAGS -c "$H/$p.F90"
  gfortran -o "$H/$p" $p.o dump_io.o "$LIBDIR/libsocrates_core.a"
  echo "built $H/$p"
done
# Isca-interface reference (needs the Isca interface objects built by build_harness.sh)
ISCA_OBJ="$H/obj_soc_ref"
gfortran -O2 -ffp-contract=off -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
  -J"$OBJ" -I"$OBJ" -I"$ISCA_OBJ" -I"$LIBDIR" -c "$H/soc_isca.F90"
gfortran -o "$H/soc_isca" soc_isca.o dump_io.o $(ls "$ISCA_OBJ"/*.o | grep -v "/soc_ref.o") "$LIBDIR/libsocrates_core.a"
echo "built $H/soc_isca"
# Timing driver for tools/benchmark_parallel.py (Isca-style chunked calls, no I/O in the timed region)
gfortran -O2 -ffp-contract=off -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
  -J"$OBJ" -I"$OBJ" -I"$ISCA_OBJ" -I"$LIBDIR" -c "$H/soc_bench.F90"
gfortran -o "$H/soc_bench" soc_bench.o $(ls "$ISCA_OBJ"/*.o | grep -v "/soc_ref.o") "$LIBDIR/libsocrates_core.a"
echo "built $H/soc_bench"
