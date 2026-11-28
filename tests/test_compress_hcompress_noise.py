"""Lossless HCOMPRESS_1 of full-range noise compresses (poorly) instead of failing.

CFITSIO sized the compressed-tile buffer for at most ~10% growth
(``nx * 2.2 + 26`` bytes for 8/16-bit, ``nx * 4.4 + 26`` for 32-bit pixels).
Full-range noise needs more -- up to ~2.3 / ~4.5 bytes per pixel on 16x16
tiles and several times that on 4x4 tiles -- so such images failed with
status 413 ("error compressing image"). cfitsio-4.7.0-hcompress-bufsize
sizes the buffer for them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits


def _noise(dtype: np.dtype, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    info = np.iinfo(dtype)
    rng = np.random.default_rng(seed)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint16, np.int32])
@pytest.mark.parametrize(
    ("shape", "tile"),
    [
        ((64, 16), None),  # default tiles: 16 rows x 16 columns
        ((48, 9), None),
        ((256, 256), None),
        ((32, 32), (16, 16)),
        ((16, 16), (4, 4)),
        ((15, 14), (5, 7)),
    ],
)
def test_full_range_noise_round_trips(tmp_path, dtype, shape, tile):
    path = str(tmp_path / "noise.fits")
    data = _noise(np.dtype(dtype), shape)
    torchfits.write(path, data, compress="HCOMPRESS_1", tile_shape=tile)
    assert torchfits.read_header(path, 1)["ZCMPTYPE"] == "HCOMPRESS_1"
    out = torchfits.read(path, hdu=1)
    assert torch.equal(out, torch.from_numpy(data))
