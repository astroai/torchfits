#!/usr/bin/env bash
# Local Mac exhaustive: lab profile + mmap matrix + MPS GPU when available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-exhaustive_mps_$(date -u +%Y%m%d_%H%M%S)}"
ENV_NAME="${TORCHFITS_BENCH_ENV:-bench-all}"
LOG_DIR="${ROOT}/benchmarks_results"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "$LOG_DIR"

echo "=== torchfits local exhaustive (CPU + MPS): ${RUN_ID} ===" | tee "$LOG_FILE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

set +e
pixi run -e "$ENV_NAME" python benchmarks/bench_all.py \
  --suite release \
  --mmap-matrix \
  --profile lab \
  --run-id "$RUN_ID" \
  --keep-temp >>"$LOG_FILE" 2>&1
RC=$?
set -e

echo "bench-all exit code: ${RC}" | tee -a "$LOG_FILE"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "Artifacts: ${LOG_DIR}/${RUN_ID}/" | tee -a "$LOG_FILE"
exit "$RC"
