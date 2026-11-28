#!/usr/bin/env bash
# Mirror GitHub Comprehensive CI + Documentation build locally before pushing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FAST="${CI_LOCAL_FAST:-0}"

echo "=== ci_local: lint ==="
pixi run ruff check .
pixi run ruff format --check .
pixi run python scripts/check_duplicate_cpp.py
# Verifies the committed version and torch floor match scripts/torch_support.json.
pixi run check-lane

if [[ "${FAST}" == "1" ]]; then
  echo "=== ci_local: docs contract ==="
  # One task, not a hand-copied file list: pixi.toml's `docs-contract` is the
  # single copy of that contract (the CI job of the same name runs it too).
  # Fast mode does not rebuild, so it runs against the installed extension.
  pixi run docs-contract
  echo "=== ci_local: fast mode (skip build, mypy, changelog, tests, release-gate) ==="
  exit 0
fi

echo "=== ci_local: types + changelog ==="
pixi run mypy src/torchfits
pixi run changelog-check

# Build first so nothing below exercises a stale extension. A vendored
# extern/cfitsio older than extern/VERSIONS.txt or extern/patches is re-vendored
# first (CMake would do the same in a git checkout; this says so up front).
echo "=== ci_local: build ==="
if ! bash extern/vendor.sh --check-stamp; then
  bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt
fi
pixi run dev

echo "=== ci_local: full test suite ==="
pixi run test

echo "=== ci_local: release gate (incl. docs-contract + docs-links) ==="
pixi run release-gate

echo "=== ci_local: OK ==="
