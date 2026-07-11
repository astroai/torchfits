#!/usr/bin/env bash
# Runs inside astroai/base on CANFAR (headless GPU session).
# Logs + benchmark CSVs land under ${TMP_SCRATCH_DIR}/torchfits-gpu-bench/<run-id>/.
set -euo pipefail

: "${TORCHFITS_BENCH_RUN_ID:=exhaustive_cuda_0.7.0_$(date -u +%Y%m%d_%H%M%S)}"
: "${TORCHFITS_BENCH_MODE:=exhaustive}"

SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="${SCRATCH}/torchfits-gpu-bench/${TORCHFITS_BENCH_RUN_ID}"
mkdir -p "$RUN_DIR"

# ponytail: notebook image ships pixi pointed at /usr/local/share (not writable); use scratch
export PIXI_HOME="${PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${SCRATCH}/torchfits-pixi-cache}"
mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}"

if [[ -z "${TORCHFITS_BENCH_LOG_REDIRECTED:-}" ]]; then
  export TORCHFITS_BENCH_LOG_REDIRECTED=1
  exec > >(tee -a "${RUN_DIR}/stdout.log") 2> >(tee -a "${RUN_DIR}/stderr.log" >&2)
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== torchfits CANFAR GPU bench ==="
echo "run_id=${TORCHFITS_BENCH_RUN_ID} mode=${TORCHFITS_BENCH_MODE}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "scratch=${SCRATCH} run_dir=${RUN_DIR}"
echo "TMP_SRC_DIR=${TMP_SRC_DIR:-unset}"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || true
fi

bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt

pixi install
pixi run -e bench-gpu gpu-bootstrap
pixi run -e bench-gpu bench-gpu-install
pixi run -e bench-gpu gpu-env-check

pixi run -e bench-gpu python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"

case "${TORCHFITS_BENCH_MODE}" in
  smoke)
    pixi run -e bench-gpu pytest tests/test_scale_on_device.py -q
    pixi run -e bench-gpu python benchmarks/bench_gpu_transports.py
    ;;
  release-gate)
    pixi run -e bench-gpu release-gate
    ;;
  exhaustive)
    pixi run -e bench-gpu python benchmarks/bench_all.py \
      --profile lab \
      --scope all \
      --mmap-matrix \
      --run-id "${TORCHFITS_BENCH_RUN_ID}" \
      --keep-temp
    ;;
  *)
    echo "unknown TORCHFITS_BENCH_MODE=${TORCHFITS_BENCH_MODE}" >&2
    exit 1
    ;;
esac

BENCH_OUT="benchmarks_results/${TORCHFITS_BENCH_RUN_ID}"
if [[ -d "${BENCH_OUT}" ]]; then
  cp -a "${BENCH_OUT}" "${RUN_DIR}/benchmarks_results"
fi

{
  echo "TORCHFITS_BENCH_RUN_DIR=${RUN_DIR}"
  echo "TORCHFITS_BENCH_RUN_ID=${TORCHFITS_BENCH_RUN_ID}"
  echo "TORCHFITS_BENCH_MODE=${TORCHFITS_BENCH_MODE}"
  echo "TORCHFITS_BENCH_GIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${RUN_DIR}/manifest.txt"

echo "=== done; artifacts on scratch: ${RUN_DIR} ==="
