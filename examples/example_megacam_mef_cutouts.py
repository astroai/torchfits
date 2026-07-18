"""Example: cutouts from a real multi-extension CFHT MegaCam exposure.

Scans ``benchmarks_data/cfht_megacam`` for ``*.fits.fz`` (tile-compressed MEF)
files. Skips cleanly if none are present — fetch via
``bash scripts/fetch_cfht_megacam_sample.sh``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import megacam_dir  # noqa: E402

import torchfits  # noqa: E402


def main() -> int:
    files = sorted(megacam_dir().glob("*.fits.fz"))
    if not files:
        print(
            "SKIP: no MegaCam samples in benchmarks_data/cfht_megacam. "
            "Fetch via: bash scripts/fetch_cfht_megacam_sample.sh"
        )
        return 0

    path = files[0]
    print(f"file: {path.name}")
    with torchfits.open(str(path)) as hdul:
        hdul.info()
        image_hdus = [
            i
            for i, hdu in enumerate(hdul)
            if hasattr(hdu.data, "shape") and len(hdu.data.shape) == 2
        ]

    for i in image_hdus[:3]:
        cutout = torchfits.read_subset(str(path), hdu=i, x1=0, y1=0, x2=64, y2=64)
        print(f"  hdu[{i}] cutout shape={tuple(cutout.shape)} dtype={cutout.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
