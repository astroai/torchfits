#!/usr/bin/env bash
# Download torchfits CANFAR bench artifacts from VOSpace to benchmarks_results/<run-id>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: $0 <run-id>}"
VOS_BASE="${TORCHFITS_VOS_BASE:-vos:sfabbro/torchfits-gpu-bench}"
VOS_URI="${VOS_BASE}/${RUN_ID}"
LOCAL_DIR="${ROOT_DIR}/benchmarks_results/${RUN_ID}"

if ! command -v vcp >/dev/null; then
  cat >&2 <<EOF
vcp not found. Install VOS tools locally:

  pip install vos
  cadc-get-cert -u <user>   # or use canfar x509 if already configured for vcp

Then re-run:
  bash scripts/fetch_canfar_bench_vos.sh ${RUN_ID}
EOF
  exit 1
fi

mkdir -p "${LOCAL_DIR}"
vcp "${VOS_URI}/" "${LOCAL_DIR}/"
echo "fetched ${VOS_URI} -> ${LOCAL_DIR}"
