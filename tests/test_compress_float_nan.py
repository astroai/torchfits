"""NaN pixels survive lossy (quantized) tile compression of float images.

The writer handed CFITSIO float pixels without null flagging, so a quantized
tile turned NaN into an arbitrary integer (e.g. -2.3e9 after decompression,
for every reader). NaN is now written as CFITSIO's null value, which the tile
records through ZBLANK, and the image readers null-check HDUs that declare
ZBLANK so those pixels come back as NaN.
"""

from __future__ import annotations

import pytest
import torch

import torchfits


def _image_with_nans(dtype):
    g = torch.Generator().manual_seed(4)
    data = (torch.randn(24, 40, generator=g) * 10 + 24).to(dtype)
    data[0, 0] = float("nan")
    data[5, 17] = float("nan")
    data[23, :] = float("nan")  # a whole tile of NaN
    return data


def _readers(path):
    yield "read", torchfits.read(str(path), hdu=1)
    yield "read_subset", torchfits.read_subset(str(path), 1, 0, 0, 40, 24)
    with torchfits.open(str(path)) as hdul:
        yield "hdulist", hdul[1].to_tensor()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"compress": "RICE_1"},
        {"compress": "GZIP_2"},
        {"compress": "HCOMPRESS_1"},
        {"compress": "RICE_1", "quantize_level": -0.01},
        {"compress": "RICE_1", "quantize_method": "subtractive_dither_2"},
        {"compress": "GZIP_1", "quantize_level": 8},
    ],
)
def test_nan_survives_quantized_compression(tmp_path, dtype, kwargs):
    data = _image_with_nans(dtype)
    path = tmp_path / "nan.fits"
    torchfits.write(str(path), data, **kwargs)
    assert torchfits.read_header(str(path), 1).get("ZBLANK") is not None
    nan = torch.isnan(data)
    for name, got in _readers(path):
        assert torch.equal(torch.isnan(got), nan), name
        assert (got - data)[~nan].abs().max() < 10, name


def test_lossless_float_keeps_ieee_specials(tmp_path):
    data = _image_with_nans(torch.float32)
    data[1, 1] = float("inf")
    data[1, 2] = float("-inf")
    data[1, 3] = -0.0
    data[1, 4] = 1e-40
    path = tmp_path / "gzip.fits"
    torchfits.write(str(path), data, compress="GZIP_1")
    assert "ZBLANK" not in torchfits.read_header(str(path), 1)
    for name, got in _readers(path):
        assert torch.equal(got.view(torch.int32), data.view(torch.int32)), name


def test_nan_free_quantized_image_declares_no_zblank(tmp_path):
    data = _image_with_nans(torch.float32).nan_to_num(0.0)
    path = tmp_path / "clean.fits"
    torchfits.write(str(path), data, compress="RICE_1")
    assert "ZBLANK" not in torchfits.read_header(str(path), 1)
