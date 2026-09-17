#!/bin/bash
# Build the SOCRATES core (modules_core + radiance_core) into a static library.
# Usage: build_socrates.sh [outdir] [extra FFLAGS...]
# Iterative compile: files whose module dependencies are not yet built are retried.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../socrates/src"
OUT="${1:-$HERE/obj}"; shift || true
FFLAGS="-O2 -fPIC -ffree-line-length-none $*"
mkdir -p "$OUT"
FILES=$(ls "$SRC"/modules_core/*.[fF]90 "$SRC"/radiance_core/*.[fF]90)
# Allow overriding individual files (e.g. a tracing yomhook) via $OVERRIDE_DIR
if [ -n "${OVERRIDE_DIR:-}" ]; then
  for f in "$OVERRIDE_DIR"/*.[fF]90; do
    b=$(basename "$f")
    FILES=$(echo "$FILES" | grep -v "/$b\$"); FILES="$FILES
$f"
  done
fi
todo="$FILES"
for pass in $(seq 1 30); do
  failed=""
  for f in $todo; do
    o="$OUT/$(basename "${f%.*}").o"
    if gfortran $FFLAGS -J"$OUT" -I"$OUT" -c "$f" -o "$o" 2> "$OUT/err.tmp"; then :; else failed="$failed $f"; fi
  done
  n_before=$(echo $todo | wc -w); n_after=$(echo $failed | wc -w)
  echo "pass $pass: $((n_before-n_after)) compiled, $n_after remaining"
  [ "$n_after" -eq 0 ] && break
  if [ "$n_after" -eq "$n_before" ]; then
    echo "No progress; last errors:"; for f in $failed; do echo "== $f"; gfortran $FFLAGS -J"$OUT" -I"$OUT" -c "$f" -o /dev/null 2>&1 | head -5; done
    exit 1
  fi
  todo="$failed"
done
rm -f "$OUT/libsocrates_core.a"
ar rcs "$OUT/libsocrates_core.a" "$OUT"/*.o
echo "built $OUT/libsocrates_core.a"
