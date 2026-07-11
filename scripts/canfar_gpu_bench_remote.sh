#!/usr/bin/env bash
# Bootstrap on CANFAR headless: clone torchfits and run in-container bench driver.
set -euo pipefail

REF="${TORCHFITS_GIT_REF:-main}"
REPO="${TORCHFITS_GIT_URL:-https://github.com/astroai/torchfits.git}"
DIR=/tmp/torchfits

rm -rf "${DIR}"
git clone --depth 1 --branch "${REF}" "${REPO}" "${DIR}"
cd "${DIR}"
exec bash scripts/canfar_gpu_bench_incontainer.sh
