#!/usr/bin/env bash
# Patch docs from one or more exhaustive run directories (multi-host scorecard).
# Usage:
#   bash scripts/bench_release_scorecard.sh \
#     benchmarks_results/exhaustive_mps_... \
#     benchmarks_results/exhaustive_cpu_... \
#     benchmarks_results/exhaustive_cuda_...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [run_dir...]" >&2
  exit 2
fi

ARGS=()
for d in "$@"; do
  ARGS+=(--run-dir "$d")
done

pixi run -e default python scripts/patch_bench_docs.py "${ARGS[@]}"
echo "Patched docs/benchmarks.md from: $*"
