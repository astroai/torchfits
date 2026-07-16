#!/usr/bin/env bash
# Focused deficit-cluster benches from the suite registry (no full exhaustive).
# Usage:
#   bash scripts/bench_deficit_focus.sh              # all deficit-focus suites
#   bash scripts/bench_deficit_focus.sh hcompress
#   bash scripts/bench_deficit_focus.sh tiny_int8
#   bash scripts/bench_deficit_focus.sh predicate
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLUSTER="${1:-all}"
RUN_ID="${TORCHFITS_BENCH_RUN_ID:-deficit_focus_$(date -u +%Y%m%d_%H%M%S)}"
ENV_NAME="${TORCHFITS_BENCH_ENV:-bench-all}"

resolve_name() {
  case "$1" in
    hcompress) echo compressed_hcompress ;;
    tiny_int8) echo tiny_int8 ;;
    predicate) echo fitstable_predicate ;;
    all) echo all ;;
    *) echo "$1" ;;
  esac
}

run_suite() {
  local name="$1"
  local rid="${RUN_ID}_${name}"
  echo "==> deficit-focus suite=${name} run_id=${rid}"
  pixi run -e "$ENV_NAME" python benchmarks/bench_all.py \
    --suite "$name" \
    --run-id "$rid"
}

NAME="$(resolve_name "$CLUSTER")"
if [[ "$NAME" == "all" ]]; then
  SUITES="$(
    pixi run -e "$ENV_NAME" python -c \
      "from benchmarks.suites import DEFICIT_FOCUS_SUITES; print(' '.join(DEFICIT_FOCUS_SUITES))"
  )"
  for s in $SUITES; do
    run_suite "$s"
  done
else
  run_suite "$NAME"
fi
