#!/usr/bin/env bash
# Build vendored-CFITSIO pure-C microbench and compare to torchfits/fitsio peers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f extern/cfitsio/CMakeLists.txt ]]; then
  bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt
fi

pixi run -e bench-all python benchmarks/run_cfitsio_direct_bench.py "$@"
