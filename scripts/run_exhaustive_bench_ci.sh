#!/usr/bin/env bash
# Lab-profile bench-all (mmap matrix) + docs patch — pip/venv path for CI and CANFAR.
# Requires CUDA for GPU transport rows; CPU rows still emit when CUDA is absent.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-exhaustive_0.7.0_$(date -u +%Y%m%d_%H%M%S)}"
REQUIRE_CUDA="${TORCHFITS_BENCH_REQUIRE_CUDA:-0}"
LOG_DIR="${ROOT_DIR}/benchmarks_results"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
OUT_DIR="${LOG_DIR}/${RUN_ID}"

mkdir -p "$LOG_DIR"

echo "=== torchfits exhaustive CI bench: ${RUN_ID} ===" | tee "$LOG_FILE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())" | tee -a "$LOG_FILE"
if [[ "$REQUIRE_CUDA" == "1" ]]; then
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA required (set TORCHFITS_BENCH_REQUIRE_CUDA=0 to allow CPU-only)'" | tee -a "$LOG_FILE"
fi
python -c "import torchfits; print('torchfits', torchfits.__version__)" | tee -a "$LOG_FILE"

set +e
python benchmarks/bench_all.py \
  --profile lab \
  --scope all \
  --mmap-matrix \
  --run-id "$RUN_ID" \
  --keep-temp >>"$LOG_FILE" 2>&1
BENCH_RC=$?
set -e

echo "bench-all exit code: ${BENCH_RC}" | tee -a "$LOG_FILE"
echo "Finished bench-all: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

CSV="${OUT_DIR}/results.csv"
DEFICITS="${OUT_DIR}/torchfits_deficits.csv"

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: missing ${CSV}" | tee -a "$LOG_FILE"
  exit "${BENCH_RC:-1}"
fi

python scripts/patch_bench_docs.py \
  --csv "$CSV" \
  --deficits "$DEFICITS" \
  --run-id "$RUN_ID" >>"$LOG_FILE" 2>&1

echo "Patched docs/benchmarks.md from ${RUN_ID}" | tee -a "$LOG_FILE"
echo "Artifacts: ${OUT_DIR}/" | tee -a "$LOG_FILE"

exit "$BENCH_RC"
