#!/usr/bin/env bash
# Upload torchfits CANFAR bench artifacts to VOSpace via vcp.
set -euo pipefail

bench_out="${1:?usage: $0 <benchmarks_results/run-id-dir>}"
vos_dest="${2:?usage: $0 <bench-out> <vos:user/path/run-id>}"

if ! command -v vcp >/dev/null; then
  python3 -m pip install --user -q vos
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v vcp >/dev/null; then
  echo "vcp not available after vos install" >&2
  exit 1
fi

if command -v vmkdir >/dev/null; then
  vmkdir -p "${vos_dest}" 2>/dev/null || true
fi

# ponytail: vcp is recursive; trailing slashes copy directory contents
vcp "${bench_out}/" "${vos_dest}/"
echo "TORCHFITS_VOS_URI=${vos_dest}"
