#!/usr/bin/env bash
# Focused deficit-cluster benches (no full exhaustive).
# Usage:
#   bash scripts/bench_deficit_focus.sh              # all clusters
#   bash scripts/bench_deficit_focus.sh hcompress
#   bash scripts/bench_deficit_focus.sh tiny_int8
#   bash scripts/bench_deficit_focus.sh predicate
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLUSTER="${1:-all}"
RUN_ID="${TORCHFITS_BENCH_RUN_ID:-deficit_focus_$(date -u +%Y%m%d_%H%M%S)}"
PROFILE="${TORCHFITS_BENCH_PROFILE:-lab}"
ENV_NAME="${TORCHFITS_BENCH_ENV:-bench-all}"

run_cluster() {
  local name="$1"
  local scope="$2"
  local filter="$3"
  local rid="${RUN_ID}_${name}"
  echo "==> cluster=${name} scope=${scope} filter=${filter} run_id=${rid}"
  local -a cmd=(
    pixi run -e "$ENV_NAME" python benchmarks/bench_all.py
    --profile "$PROFILE"
    --scope "$scope"
    --filter "$filter"
    --mmap-matrix
    --run-id "$rid"
  )
  if [[ "$scope" == "fitstable" ]]; then
    cmd+=(--no-gpu)
  fi
  "${cmd[@]}"
}

case "$CLUSTER" in
  hcompress)
    run_cluster hcompress fits '^(compressed_hcompress_)'
    ;;
  tiny_int8)
    run_cluster tiny_int8 fits '^(tiny_int8_)'
    ;;
  predicate)
    run_cluster predicate fitstable '^(narrow_1000|narrow_10000)$'
    ;;
  all)
    run_cluster hcompress fits '^(compressed_hcompress_)'
    run_cluster tiny_int8 fits '^(tiny_int8_)'
    run_cluster predicate fitstable '^(narrow_1000|narrow_10000)$'
    ;;
  *)
    echo "Unknown cluster: $CLUSTER (expected: all|hcompress|tiny_int8|predicate)" >&2
    exit 2
    ;;
esac
