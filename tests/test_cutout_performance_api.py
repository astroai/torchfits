import os
import tempfile

import numpy as np
from astropy.io import fits

import torchfits


def test_read_subset_basic_roundtrip():
    data = (np.arange(256 * 256, dtype=np.float32).reshape(256, 256) * 0.5).astype(
        np.float32
    )
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        name = f.name
    try:
        fits.PrimaryHDU(data).writeto(name, overwrite=True)

        # 10x10 cutout at (x=5..14, y=7..16) using torchfits API coords.
        cut = torchfits.read_subset(name, 0, 5, 7, 15, 17)
        assert cut.shape == (10, 10)
        assert np.allclose(cut.numpy(), data[7:17, 5:15])

        with torchfits.open_subset_reader(name, hdu=0) as reader:
            cut2 = reader.read_subset(5, 7, 15, 17)
        assert cut2.shape == (10, 10)
        assert np.allclose(cut2.numpy(), data[7:17, 5:15])
    finally:
        try:
            os.unlink(name)
        except Exception:
            pass


def test_subset_reader_uint16_convention_roundtrip(tmp_path):
    """Unsigned SHORT (BZERO=32768) must match via mmap fast path."""
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    path = str(tmp_path / "u16.fits")
    fits.PrimaryHDU(data).writeto(path, overwrite=True)

    import torch

    with torchfits.open_subset_reader(path, hdu=0) as reader:
        cut = reader.read_subset(3, 5, 13, 15)
    assert cut.dtype == torch.uint16
    assert cut.shape == (10, 10)
    assert np.array_equal(cut.numpy(), data[5:15, 3:13])


def test_read_shape_warm_after_subset_reader(tmp_path):
    path = str(tmp_path / "shape.fits")
    fits.PrimaryHDU(np.zeros((32, 16), dtype=np.float32)).writeto(path, overwrite=True)
    with torchfits.open_subset_reader(path, hdu=0) as reader:
        reader.read_subset(0, 0, 4, 4)
    bitpix, shape = torchfits.read_shape(path, hdu=0)
    assert bitpix == -32
    assert shape == (32, 16)


def test_read_header_cache_isolated_from_mutation(tmp_path):
    path = str(tmp_path / "hdr.fits")
    hdu = fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    hdu.header["OBJECT"] = "orig"
    hdu.writeto(path, overwrite=True)

    hdr1 = torchfits.read_header(path, hdu=0)
    hdr1["OBJECT"] = "mutated"
    hdr2 = torchfits.read_header(path, hdu=0)
    assert hdr2["OBJECT"] == "orig"
