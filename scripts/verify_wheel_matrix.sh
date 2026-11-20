#!/usr/bin/env bash
# Verify locally built wheels (scripts/build_wheels_local.sh output) in fresh
# venvs, mirroring the cibuildwheel test command: install the wheel plus a CPU
# torch from the PyTorch index, then run import + CLI + the release smoke test.
# Every wheel is checked against each requested torch minor -- by default the
# supported floor and the newest tested minor from scripts/torch_support.json.
# Exits non-zero if any cell fails.
#
# Usage:
#   bash scripts/verify_wheel_matrix.sh [--torch 2.12,2.15] [--python 3.13] [--jobs 2] [wheel-dir]
#   default wheel-dir: dist-local
#   --python overrides the CPython a cp312-abi3 wheel is tested on (default:
#   the version in the wheel tag).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCHES=""
PYTHON_OVERRIDE=""
JOBS=2
WHEEL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch) TORCHES="$2"; shift 2 ;;
    --python) PYTHON_OVERRIDE="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) WHEEL_DIR="$1"; shift ;;
  esac
done
WHEEL_DIR="${WHEEL_DIR:-dist-local}"

cd "$ROOT"
if [[ -z "$TORCHES" ]]; then
  TORCHES="$(python3 -c 'import json; s = json.load(open("scripts/torch_support.json")); print(",".join(dict.fromkeys([s["torch_min"], s["torch_tested"]])))')"
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/logs"

verify_cell_steps() {
  local wheel="$1" torch_minor="$2" py="$3"
  local venv="$WORK/venv-$(basename "$wheel")-torch$torch_minor"
  echo "== $(basename "$wheel"): fresh venv py$py, torch $torch_minor =="
  uv venv --python "$py" --no-project "$venv" --quiet
  uv pip install --python "$venv/bin/python" --quiet \
    "$wheel" \
    "torch==$torch_minor.*" \
    pytest numpy pyarrow \
    --extra-index-url https://download.pytorch.org/whl/cpu
  echo "== $(basename "$wheel") / torch $torch_minor: import + CLI + release smoke =="
  "$venv/bin/python" -c 'import torchfits, torch; print("import ok:", torchfits.__version__, "torch", torch.__version__)'
  "$venv/bin/torchfits" --help >/dev/null
  "$venv/bin/python" -m pytest -c /dev/null --noconftest \
    "$ROOT/tests/test_release_smoke.py" -q
}

verify_cell() {
  local wheel="$1" torch_minor="$2"
  local log="$WORK/logs/$(basename "$wheel")-torch$torch_minor.log"
  local py="$PYTHON_OVERRIDE"
  if [[ -z "$py" ]]; then
    # `-.*`, not an optional hyphen: BSD sed has no optional-escape
    # syntax, so the GNU spelling matched nothing on macOS.
    py="$(basename "$wheel" | sed -n 's/^torchfits-[^-]*-cp\([0-9][0-9][0-9]\)-.*/\1/p' | sed 's/^3/3./')"
  fi
  if [[ -z "$py" ]]; then
    echo "[FAIL] $(basename "$wheel"): cannot read the CPython version from the tag" >&2
    return 1
  fi
  # A separate `bash -eo pipefail` process: errexit is ignored inside an `if`
  # condition, so a failing step in an `if { ...; }` body reported OK.
  if bash -eo pipefail -c 'verify_cell_steps "$@"' _ "$wheel" "$torch_minor" "$py" \
    >"$log" 2>&1; then
    echo "[ OK ] $(basename "$wheel") torch $torch_minor"
  else
    echo "[FAIL] $(basename "$wheel") torch $torch_minor (see $log)"
    tail -20 "$log"
    return 1
  fi
}

export -f verify_cell verify_cell_steps
export ROOT WORK WHEEL_DIR PYTHON_OVERRIDE

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "no wheel directory $WHEEL_DIR" >&2
  exit 2
fi
# Absolute, so wheel paths work from any cwd (an absolute dir was previously
# glued onto $ROOT).
WHEEL_DIR="$(cd "$WHEEL_DIR" && pwd)"
shopt -s nullglob
wheels=("$WHEEL_DIR"/torchfits-*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "no wheels found in $WHEEL_DIR" >&2
  exit 2
fi

cells=()
for wheel in "${wheels[@]}"; do
  for torch_minor in $(echo "$TORCHES" | tr ',' ' '); do
    cells+=("$wheel $torch_minor")
  done
done
echo "== verifying ${#cells[@]} cells (torch ${TORCHES}, jobs=$JOBS) =="
printf '%s\n' "${cells[@]}" | xargs -P "$JOBS" -n2 bash -c 'verify_cell "$1" "$2"' _
