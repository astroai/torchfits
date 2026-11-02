"""Tile compression of 64-bit integer images is refused up front.

CFITSIO has no tile codec for BITPIX=64 and fails mid-write with a bare
"error compressing image" (status 413). Every compress= entry point raises a
ValueError naming the problem instead, and leaves existing files untouched.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

import torchfits
from torchfits import HDUList, TensorHDU

MSG = "tile compression does not support 64-bit integer images"
DTYPES = [torch.int64, torch.uint64]


def _img(dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(64, dtype=torch.int64).reshape(8, 8).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("algo", [True, "GZIP_1", "RICE_1"])
def test_write_rejects_64bit_integer_compression(tmp_path, dtype, algo):
    path = tmp_path / "x.fits"
    with pytest.raises(ValueError, match=MSG):
        torchfits.write(str(path), _img(dtype), compress=algo)
    assert not path.exists()


@pytest.mark.parametrize("dtype", DTYPES)
def test_write_tensor_rejects_64bit_integer_compression(tmp_path, dtype):
    path = tmp_path / "x.fits"
    torchfits.write(str(path), torch.zeros(2, 2))
    before = path.read_bytes()
    with pytest.raises(ValueError, match=MSG):
        torchfits.write_tensor(str(path), _img(dtype), overwrite=True, compress=True)
    assert path.read_bytes() == before


def test_hdulist_write_rejects_64bit_integer_compression(tmp_path):
    path = tmp_path / "x.fits"
    hdul = HDUList([TensorHDU(torch.zeros(2, 2)), TensorHDU(_img(torch.int64))])
    with pytest.raises(ValueError, match=MSG):
        torchfits.write(str(path), hdul, compress="GZIP_1")


def test_insert_and_replace_hdu_reject_64bit_integer_compression(tmp_path):
    path = tmp_path / "x.fits"
    torchfits.write(str(path), torch.zeros(2, 2))
    torchfits.insert_hdu(str(path), torch.ones(3, 3), index=1)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=MSG):
        torchfits.insert_hdu(str(path), _img(torch.int64), index=1, compress=True)
    with pytest.raises(ValueError, match=MSG):
        torchfits.replace_hdu(str(path), 1, _img(torch.int64), compress="GZIP_1")
    assert path.read_bytes() == before


def test_int64_uncompressed_and_int32_compressed_still_write(tmp_path):
    a = tmp_path / "a.fits"
    torchfits.write(str(a), _img(torch.int64))
    assert torch.equal(torchfits.read(str(a)), _img(torch.int64))
    b = tmp_path / "b.fits"
    torchfits.write(str(b), _img(torch.int32), compress="GZIP_1")
    assert torch.equal(torchfits.read(str(b), hdu=1), _img(torch.int32))


def test_cli_compress_reports_64bit_integer_image(tmp_path):
    src = tmp_path / "in.fits"
    torchfits.write(str(src), _img(torch.int64))
    out = tmp_path / "out.fits"
    result = subprocess.run(
        [sys.executable, "-m", "torchfits.cli", "compress", str(src), str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3, result.stderr
    assert MSG in result.stderr
    assert not out.exists()
