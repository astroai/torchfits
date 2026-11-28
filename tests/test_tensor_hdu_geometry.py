"""A file-backed TensorHDU reports the image it holds, compressed or not.

A tile-compressed image is stored as a BINTABLE of tiles: its NAXISn/BITPIX
describe that table (``NAXIS1 = 8`` bytes per row, ``BITPIX = 8``), and the
image geometry lives in ZNAXISn/ZBITPIX. ``shape_str`` / ``dtype_str`` and
``HDUList.info()`` read the plain cards, so a CFHT MegaCam CCD of 2112x4644
int16 pixels was listed as ``(4644, 8) uint8``.
"""

from __future__ import annotations

import io

import torch

import torchfits


def test_compressed_image_hdu_reports_image_shape_and_dtype(tmp_path) -> None:
    path = str(tmp_path / "rice.fits")
    image = torch.arange(30 * 50, dtype=torch.int16).reshape(30, 50)
    torchfits.write(path, image, compress="RICE_1", overwrite=True)

    with torchfits.open(path) as hdul:
        hdu = hdul[1]
        assert hdu.header.get("XTENSION") == "BINTABLE"  # tile storage
        assert hdu.shape_str == "(30, 50)"
        assert hdu.dtype_str == "int16"
        assert "TensorHDU(name='PRIMARY', shape=(30, 50), dtype=int16)" in repr(hdu)
        buf = io.StringIO()
        hdul.info(buf)
    line = buf.getvalue().splitlines()[3]
    assert "(30, 50)" in line and "int16" in line, buf.getvalue()


def test_uncompressed_image_hdu_geometry_unchanged(tmp_path) -> None:
    path = str(tmp_path / "plain.fits")
    torchfits.write(path, torch.zeros(4, 6, 8, dtype=torch.float32), overwrite=True)
    with torchfits.open(path) as hdul:
        assert hdul[0].shape_str == "(4, 6, 8)"
        assert hdul[0].dtype_str == "float32"
