#!/usr/bin/env bash
# Build the published wheel matrix with cibuildwheel (same [tool.cibuildwheel]
# config CI uses).
#
# Linux x86_64 / aarch64: run on that arch (Docker + manylinux_2_28). GHA
# splits the two Linux jobs; locally this builds the host arch only.
# macOS arm64: run on a Mac (no Docker). GHA macos-15 also produces these.
# CUDA: there is no separate torchfits CUDA wheel — the CPU-linked wheel
# loads CUDA torch of the same minor. Verify on CANFAR with
#   bash scripts/verify_wheel_cuda_canfar.sh
#
# Usage:
#   bash scripts/cibuildwheel.sh              # host arch, cp311-cp315 (macOS: cp311-cp314)
#   CIBW_ARCHS=aarch64 bash scripts/cibuildwheel.sh
#   CIBW_BUILD=cp314-* bash scripts/cibuildwheel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v cibuildwheel >/dev/null 2>&1; then
  if command -v uvx >/dev/null 2>&1; then
    exec uvx --with 'cibuildwheel>=4.1.1,<5' cibuildwheel "$@"
  elif command -v pipx >/dev/null 2>&1; then
    exec pipx run 'cibuildwheel>=4.1.1,<5' "$@"
  else
    # No `pip install --user` fallback: user-site installs are banned
    # (AGENTS.md "Environment hygiene"), notably on CANFAR /arc/home.
    echo "cibuildwheel not found; install uv (uvx) or pipx, or put cibuildwheel on PATH" >&2
    exit 1
  fi
fi
cibuildwheel "$@"
