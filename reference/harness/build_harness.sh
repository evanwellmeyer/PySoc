#!/bin/bash
# Build the reference driver.  Usage: build_harness.sh [obj_dir_of_socrates] [exe_name]
set -eu
H="$(cd "$(dirname "$0")" && pwd)"
LIBDIR="${1:-$H/../build/obj}"
EXE="${2:-$H/soc_ref}"
OBJ="$H/obj_$(basename "$EXE")"
mkdir -p "$OBJ"
FFLAGS="-O2 -ffp-contract=off -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -J$OBJ -I$OBJ -I$LIBDIR"
cd "$OBJ"
for f in soc_constants.f90 fms_mod_stub.F90 socrates_config_mod.f90 read_control.F90 set_control.F90 \
         set_dimen.F90 set_atm.F90 set_bound.F90 set_aer.F90 socrates_set_cld.F90; do
  gfortran $FFLAGS -c "$H/isca_iface/$f"
done
gfortran $FFLAGS -c "$H/soc_ref.F90"
gfortran -o "$EXE" *.o "$LIBDIR/libsocrates_core.a"
echo "built $EXE"
