"""Lossless quantization for one compressed HDU must not leak to the next.

An integer HDU sets CFITSIO's quantize level to 0 (lossless). The request is
stored per file, so a float HDU written after it under RICE inherited it and
CFITSIO failed with status 413.
"""

import pytest
import torch

import torchfits


@pytest.mark.parametrize("compress", ["RICE", "GZIP"])
def test_int_then_float_compressed(tmp_path, compress):
    path = tmp_path / "mixed.fits"
    ints = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    floats = torch.linspace(0, 1, 64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), [ints, floats], compress=compress, overwrite=True)

    assert torch.equal(torchfits.read(str(path), hdu=1), ints)
    got = torchfits.read(str(path), hdu=2)
    assert got.shape == floats.shape
    assert torch.allclose(got, floats, atol=0.05)
