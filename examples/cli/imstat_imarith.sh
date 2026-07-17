#!/usr/bin/env bash
# IRAF-flavoured torchfits CLI smoke on the HorseHead sample.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

HH="$(pixi run python -c "from examples._sample_data import ensure_sample; print(ensure_sample('horsehead'))")"
OUT="${TMPDIR:-/tmp}/torchfits_cli_demo_$$"
mkdir -p "$OUT"

echo "=== info / stats / header (imstat / fitsinfo) ==="
pixi run python -m torchfits.cli info "$HH"
pixi run python -m torchfits.cli stats "$HH" --hdu 0
pixi run python -m torchfits.cli header "$HH" --keyword BITPIX

echo "=== copy + setkey (imcopy / hedit) ==="
pixi run python -m torchfits.cli copy "$HH" "$OUT/copy.fits"
pixi run python -m torchfits.cli setkey "$OUT/copy.fits" --key OBJECT --value HORSEHEAD

echo "=== arith + cutout (imarith / imcopy section) ==="
pixi run python -m torchfits.cli arith "$HH" --op add --value 10 --out "$OUT/plus.fits"
pixi run python -m torchfits.cli cutout "$HH" "$OUT/cut.fits" --hdu 0 --box 50,50,128,128

echo "=== transform LogStretch (imfunction) ==="
pixi run python -m torchfits.cli transform "$HH" --name LogStretch --out "$OUT/log.fits"

echo "=== png preview ==="
pixi run python -m torchfits.cli convert "$HH" "$OUT/preview.png" --to png --bands 0,0,0

echo "CLI recipes OK → $OUT"
