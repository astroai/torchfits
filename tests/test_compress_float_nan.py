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


_SPECIALS = [-0.0, float("inf"), float("-inf"), float("nan"), 1e-45]
_SPECIALS += [1.0, 2.5, -1.25, 3.0]


def _mixed_tiles_image(dtype, layout):
    """Noise with NaN (quantized ZBLANK tiles) plus one tile of IEEE specials
    that CFITSIO cannot quantize and stores losslessly. Returns the image, the
    write kwargs, the lossless tile's (rows, cols) slices and an
    (x1, y1, x2, y2) window over a quantized tile holding a NaN."""
    g = torch.Generator().manual_seed(4)
    if layout == "rows":  # default one-row tiles: CFITSIO caches none
        data = (torch.randn(3, 45, generator=g) * 10 + 24).to(dtype)
        data[0, 5] = float("nan")
        data[2, 7] = float("nan")
        box = (slice(1, 2), slice(0, 45))
        data[box] = torch.tensor(_SPECIALS * 5, dtype=dtype)
        return data, {}, box, (0, 2, 45, 3)
    # 3x9 tiles: CFITSIO keeps the last decoded tile of each tile column
    data = (torch.randn(6, 18, generator=g) * 10 + 24).to(dtype)
    data[0, 5] = float("nan")
    data[4, 2] = float("nan")
    box = (slice(3, 6), slice(9, 18))
    data[box] = torch.tensor(_SPECIALS * 3, dtype=dtype).reshape(3, 9)
    return data, {"tile_shape": (3, 9)}, box, (0, 3, 9, 6)


@pytest.mark.parametrize("layout", ["rows", "blocks"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lossless_tile_next_to_zblank_tiles_keeps_ieee_specials(
    tmp_path, dtype, layout
):
    """A quantized float HDU with NaN declares ZBLANK, but a tile CFITSIO
    cannot quantize (here: one holding +-Inf) is stored losslessly as raw
    IEEE values. The NaN nulval the quantized tiles need made CFITSIO
    null-check that tile too, so Inf read back as NaN and -0.0 / subnormals
    as +0.0 (astropy reads them exactly). The tiled layout also pins that
    CFITSIO's decoded-tile cache (multi-row tiles only) never serves a tile
    decoded under one null handling to a read under the other.
    """
    data, kwargs, box, (x1, y1, x2, y2) = _mixed_tiles_image(dtype, layout)
    rows, cols = data.shape
    path = tmp_path / "mixed.fits"
    torchfits.write(str(path), data, compress="RICE_1", **kwargs)
    assert torchfits.read_header(str(path), 1).get("ZBLANK") is not None

    nan = torch.isnan(data)
    bits = torch.int32 if dtype == torch.float32 else torch.int64
    special = data[box]
    keep = ~torch.isnan(special)
    with torchfits.open(str(path)) as hdul:
        readers = {
            "read": torchfits.read(str(path), hdu=1),
            "read_subset": torchfits.read_subset(str(path), 1, 0, 0, cols, rows),
            "hdulist": hdul[1].to_tensor(),
            # Same handle again, then a window of the lossless tile only.
            "hdulist_again": hdul[1].to_tensor(),
        }
        window = hdul[1].data[box]
    for name, got in readers.items():
        assert torch.equal(torch.isnan(got), nan), name
        assert torch.equal(got[box][keep].view(bits), special[keep].view(bits)), name
        assert (got - data)[~nan & torch.isfinite(data)].abs().max() < 10, name
    assert torch.equal(window[keep].view(bits), special[keep].view(bits))

    # One reader, a full read then a window of a quantized tile with a NaN:
    # the window must not be served the tile the full read last decoded
    # without the nulval (its ZBLANK pixel is then a finite number).
    with torchfits.open_subset_reader(str(path), hdu=1) as reader:
        full = reader(0, 0, cols, rows)
        part = reader(x1, y1, x2, y2)
    assert torch.equal(torch.isnan(full), nan)
    want = data[y1:y2, x1:x2]
    assert torch.isnan(want).any()
    assert torch.equal(torch.isnan(part), torch.isnan(want))
    assert (part - want)[~torch.isnan(want)].abs().max() < 10
