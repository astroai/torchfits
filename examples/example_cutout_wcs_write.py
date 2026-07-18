"""Example: cutout + WCS-aware header write-back.

Reads a pixel subset with ``read_subset``, shifts ``CRPIX*`` by the cutout
origin, and writes the stamp with the translated WCS cards via
``write_tensor(header=...)``. Skips cleanly if no sample is cached (fetch via
``bash scripts/fetch_example_samples.sh``).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402

WCS_KEYS = (
    "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CUNIT1", "CUNIT2", "EQUINOX", "RADESYS",
)  # fmt: skip


def _wcs_subset_header(header: object, x1: int, y1: int) -> dict:
    subset = {k: header[k] for k in WCS_KEYS if k in header}  # type: ignore[index]
    if "CRPIX1" in subset:
        subset["CRPIX1"] = float(subset["CRPIX1"]) - x1
    if "CRPIX2" in subset:
        subset["CRPIX2"] = float(subset["CRPIX2"]) - y1
    return subset


def main() -> int:
    path = try_ensure_sample("spitzer_example") or try_ensure_sample("horsehead")
    if path is None:
        print(
            "SKIP: no 'spitzer_example' or 'horsehead' sample cached. "
            "Fetch via: bash scripts/fetch_example_samples.sh"
        )
        return 0

    header = torchfits.read_header(str(path), hdu=0)
    full_shape = torchfits.read_tensor(str(path), hdu=0).shape
    print(f"source: {path.name} shape={tuple(full_shape)}")

    h, w = int(full_shape[-2]), int(full_shape[-1])
    x1, y1 = w // 4, h // 4
    x2, y2 = min(x1 + 128, w), min(y1 + 128, h)
    stamp = torchfits.read_subset(str(path), hdu=0, x1=x1, y1=y1, x2=x2, y2=y2)
    print(f"cutout: x1={x1} y1={y1} x2={x2} y2={y2} shape={tuple(stamp.shape)}")

    wcs_header = _wcs_subset_header(header, x1, y1)
    print(f"WCS cards carried over: {sorted(wcs_header)}")

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        out_path = fh.name
    try:
        torchfits.write_tensor(out_path, stamp, header=wcs_header, overwrite=True)
        roundtrip, rt_header = torchfits.read(
            out_path, hdu=0, mode="image", return_header=True
        )
        print(f"wrote stamp: {out_path} shape={tuple(roundtrip.shape)}")
        if "CRPIX1" in wcs_header:
            print(f"  CRPIX1 written={rt_header.get('CRPIX1')}")
    finally:
        os.unlink(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
