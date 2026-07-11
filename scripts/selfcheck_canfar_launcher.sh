#!/usr/bin/env bash
# Pre-flight for launch_canfar_gpu_bench.sh (run locally before burning a GPU session).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TORCHFITS_BENCH_RUN_ID=selfcheck
export TORCHFITS_GIT_REF=main
export TORCHFITS_BENCH_MODE=smoke

REMOTE_PLAIN="git clone --depth 1 --branch main https://github.com/astroai/torchfits.git /scratch/torchfits; cd /scratch/torchfits; bash scripts/canfar_gpu_bench_incontainer.sh"
REMOTE_CMD="$(printf '%s' "${REMOTE_PLAIN}" | tr ' ' '\t')"

case "${REMOTE_CMD}" in
  *'$'*) echo "FAIL: remote cmd must not contain \$ (skaha regex)" >&2; exit 1 ;;
  *'&'*) echo "FAIL: remote cmd must not contain & (URL query split)" >&2; exit 1 ;;
  *' '*) echo "FAIL: remote cmd must be tab-encoded (no spaces)" >&2; exit 1 ;;
esac

command -v canfar >/dev/null || { echo "SKIP: canfar not in PATH" >&2; exit 0; }

canfar auth show >/dev/null 2>&1 || { echo "FAIL: canfar not authenticated" >&2; exit 1; }

echo "OK: canfar launcher preflight"
