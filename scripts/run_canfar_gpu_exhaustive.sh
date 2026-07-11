#!/usr/bin/env bash
# Full CUDA exhaustive bench inside astroai/base on a GPU host (CANFAR staging).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-exhaustive_cuda_0.7.0_$(date -u +%Y%m%d_%H%M%S)}"
IMAGE="${TORCHFITS_BENCH_IMAGE:-astroai/base:latest}"

if ! command -v docker >/dev/null; then
  echo "docker required" >&2
  exit 1
fi

docker run --gpus all --rm \
  -v "${ROOT_DIR}:/src" \
  -w /src \
  -e "TORCHFITS_BENCH_REQUIRE_CUDA=1" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt
    python -m pip install -q --upgrade pip wheel setuptools
  python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -q numpy astropy fitsio pytest nanobind scikit-build-core psutil pyarrow pandas
    python -m pip install -q -e . --no-build-isolation
    bash scripts/run_exhaustive_bench_ci.sh '"$RUN_ID"'
  '

echo "Done. Results: benchmarks_results/${RUN_ID}/"
