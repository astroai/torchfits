"""
Example: opt-in robust float → BITPIX=16 / TFORM=I packing.

Default write keeps native float. When size forces int16, use
quantize="robust" (percentile bulk range + clip) instead of global min→max.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchfits  # noqa: E402


def _skewed_image(n: int = 128) -> torch.Tensor:
    """Most pixels near background; a few bright outliers (HDR-like)."""
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 2.0, size=(n, n)).astype(np.float32)
    img[n // 2, n // 2] = 5.0e4
    img[n // 4, n // 4] = 2.0e4
    return torch.from_numpy(img)


def main() -> None:
    image = _skewed_image()
    with tempfile.TemporaryDirectory() as tmp:
        native = os.path.join(tmp, "native.fits")
        packed = os.path.join(tmp, "packed.fits")
        table_path = os.path.join(tmp, "table.fits")

        torchfits.write_tensor(native, image, overwrite=True)
        torchfits.write_tensor(packed, image, quantize="robust", overwrite=True)

        bitpix_n, _ = torchfits.read_shape(native, hdu=0)
        bitpix_p, _ = torchfits.read_shape(packed, hdu=0)
        hdr = torchfits.read_header(packed, hdu=0)
        recovered = torchfits.read_tensor(packed, hdu=0)
        bulk = image < np.percentile(image.numpy(), 99.0)
        mae_bulk = (recovered.cpu() - image.cpu()).abs()[bulk].mean().item()

        print(f"native BITPIX={bitpix_n} (float)")
        print(
            f"packed BITPIX={bitpix_p} BSCALE={hdr['BSCALE']:.6g} BZERO={hdr['BZERO']:.6g}"
        )
        print(f"bulk MAE (p99 interior): {mae_bulk:.4g}")

        flux = image.reshape(-1)[:1000]
        torchfits.table.write(
            table_path,
            {"ID": np.arange(flux.numel(), dtype=np.int32), "FLUX": flux.numpy()},
            quantize={"FLUX": "robust"},
            overwrite=True,
        )
        cols = torchfits.table.read_torch(table_path, hdu=1, columns=["FLUX"])
        mae = (cols["FLUX"].cpu().float() - flux.cpu()).abs().mean().item()
        print(f"table FLUX round-trip MAE: {mae:.4g}")


if __name__ == "__main__":
    main()
