#!/usr/bin/env bash
# Upload torchfits CANFAR bench artifacts to VOSpace via vcp.
set -euo pipefail

bench_out="${1:?usage: $0 <benchmarks_results/run-id-dir>}"
vos_dest="${2:?usage: $0 <bench-out> <vos:user/path/run-id>}"

# Avoid pip --user (breaks under PYTHONNOUSERSITE / some CANFAR images).
export PYTHONNOUSERSITE=1
if ! command -v vcp >/dev/null; then
  python3 -m pip install -q vos
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v vcp >/dev/null; then
  echo "vcp not available after vos install" >&2
  exit 1
fi

if command -v vmkdir >/dev/null; then
  vmkdir -p "${vos_dest}" 2>/dev/null || true
fi

# ponytail: copy directory *contents* (not the directory name) into vos_dest
vcp "${bench_out}/." "${vos_dest}/"
echo "TORCHFITS_VOS_URI=${vos_dest}"
