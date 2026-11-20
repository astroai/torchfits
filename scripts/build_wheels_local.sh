#!/usr/bin/env bash
# Build torchfits wheels locally and drop them into dist-local/. Docker-free
# smoke (linux_x86_64 / macos tags). Published manylinux / macosx wheels come
# from `bash scripts/cibuildwheel.sh` / `.github/workflows/build_wheels.yml`.
#
# The extension links no libtorch, so the build needs no PyTorch and there is
# no per-torch cell: CPython 3.12+ yields one cp312-abi3 wheel (build it once,
# on any 3.12+), 3.11 its own cp311 wheel. Each cell builds in a throwaway copy
# of the tree, so the committed repo state is never touched.
#
# Usage:
#   bash scripts/build_wheels_local.sh [--pythons 3.11,3.13] [--jobs 2] [out-dir]
#
# Flags:
#   --pythons    comma-separated CPython versions (default: 3.11,3.13)
#   --jobs       parallel build cells (default: 2)
#   --prerelease prerelease suffix (e.g. rc5) rendered with
#                `release_lane.py --apply --prerelease` (matches the rc tag
#                state the CI wheel build runs on)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="dist-local"
JOBS=2
PYTHONS="3.11,3.13"
PRERELEASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pythons) PYTHONS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --prerelease) PRERELEASE="$2"; shift 2 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) OUT_DIR="$1"; shift ;;
  esac
done

cd "$ROOT"
mkdir -p "$OUT_DIR"
# Absolute, so an absolute OUT_DIR is not glued onto $ROOT below.
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [[ ! -d extern/cfitsio ]]; then
  echo "== vendoring cfitsio (needed by every build) =="
  bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt
fi

PIXI_PY="$(pixi run python -c 'import sys; print(sys.executable)')"
PIXI_PREFIX="$(dirname "$(dirname "$PIXI_PY")")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/logs"

# Runs in its own `bash -eo pipefail` process (see build_cell): errexit is
# ignored for anything inside an `if` condition, so a failing step in an
# `if { ...; }` body was still reported as OK.
build_cell_steps() {
  local py="$1"
  local copy="$WORK/py$py"
  mkdir -p "$copy"
  echo "== py$py: copying tree =="
  tar -C "$ROOT" -cf - \
    --exclude=.git --exclude=.pixi --exclude=dist-local --exclude=dist \
    --exclude=build --exclude='*.egg-info' --exclude=__pycache__ \
    --exclude='.venv' --exclude='venv*' --exclude=docs \
    . | tar -C "$copy" -xf -
  if [[ -n "$PRERELEASE" ]]; then
    "$PIXI_PY" "$copy/scripts/release_lane.py" --apply --prerelease "$PRERELEASE"
  fi
  "$PIXI_PY" "$copy/scripts/release_lane.py" --check
  echo "== py$py: creating venv (uv pulls CPython $py) =="
  uv venv --python "$py" --no-project "$copy/.venv" --quiet
  uv pip install --python "$copy/.venv/bin/python" --quiet \
    pip "nanobind" "scikit-build-core" "cmake" "ninja"
  echo "== py$py: building wheel =="
  (cd "$copy" && unset PREFIX CONDA_PREFIX && \
    CMAKE_PREFIX_PATH="${PIXI_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
    MACOSX_DEPLOYMENT_TARGET=11.0 \
    ./.venv/bin/python -m pip wheel . \
    --no-deps --no-build-isolation -w "$OUT_DIR")
}

build_cell() {
  local py="$1"
  local log="$WORK/logs/py$py.log"
  if bash -eo pipefail -c 'build_cell_steps "$1"' _ "$py" >"$log" 2>&1; then
    echo "[ OK ] py$py"
  else
    echo "[FAIL] py$py (see $log)"
    tail -5 "$log"
    return 1
  fi
}

export -f build_cell build_cell_steps
export ROOT OUT_DIR WORK PIXI_PY PIXI_PREFIX PRERELEASE

echo "== building py ${PYTHONS} into $OUT_DIR (jobs=$JOBS) =="
echo "$PYTHONS" | tr ',' '\n' | xargs -P "$JOBS" -n1 bash -c 'build_cell "$1"' _

echo "== wheels =="
ls -lh "$OUT_DIR"/torchfits-*.whl
