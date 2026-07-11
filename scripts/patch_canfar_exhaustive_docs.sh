#!/usr/bin/env bash
# Patch docs/benchmarks.md from a CANFAR exhaustive run (CSV must exist locally).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: $0 <exhaustive_cuda_run_id>}"
OUT_DIR="${ROOT_DIR}/benchmarks_results/${RUN_ID}"
CSV="${OUT_DIR}/results.csv"
DEFICITS="${OUT_DIR}/torchfits_deficits.csv"

if [[ ! -f "${CSV}" ]]; then
  cat >&2 <<EOF
missing ${CSV}

Copy scratch artifacts first, e.g. from canfar logs or:
  benchmarks_results/canfar_<run-id>/canfar_logs.txt

Expected layout:
  benchmarks_results/${RUN_ID}/results.csv
  benchmarks_results/${RUN_ID}/torchfits_deficits.csv
EOF
  exit 1
fi

python scripts/patch_bench_docs.py \
  --csv "${CSV}" \
  --deficits "${DEFICITS}" \
  --run-id "${RUN_ID}"

pixi run docs-build
pixi run pytest tests/test_docs_integrity.py -q

echo "Patched docs/benchmarks.md from ${RUN_ID}"
