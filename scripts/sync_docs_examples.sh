#!/usr/bin/env bash
# Copy runnable examples into docs/ so the published site can link them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/docs/published-examples"
rm -rf "${DEST}"
mkdir -p "${DEST}/cli"
shopt -s nullglob
for f in "${ROOT}/examples"/*.py; do
  cp "$f" "${DEST}/"
done
for f in "${ROOT}/examples/cli"/*; do
  [[ -f "$f" ]] || continue
  cp "$f" "${DEST}/cli/"
done
# Marker so empty trees are obvious in CI logs
printf '# Published examples\n\nCopied from `examples/` at docs-build time. Do not edit.\n' \
  > "${DEST}/README.md"
