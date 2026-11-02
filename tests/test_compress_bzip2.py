"""BZIP2_1 tile compression (enabled by extern/patches/cfitsio-4.7.0-bzip2.patch).

- Float tiles >= 149 px wide failed to write: the output buffer size used the
  signed zbitpix (-32/-64), so it went negative (status 113) or too small.
- A tile whose bzip2 stream decoded to fewer bytes than the tile was accepted,
  returning uninitialized heap memory as pixels.
"""

import bz2

import pytest
import torch

import torchfits
from torchfits._io_engine.paths import has_bz2_support

pytestmark = pytest.mark.skipif(
    not has_bz2_support(), reason="this build has no BZIP2 support (HAVE_BZIP2)"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [8, 256])
def test_bzip2_float_images_write(tmp_path, dtype, width):
    path = str(tmp_path / "f.fits")
    data = torch.linspace(0, 1, width * width, dtype=dtype).reshape(width, width)
    torchfits.write(path, data, compress="BZIP2_1", overwrite=True)
    got = torchfits.read(path, hdu=1)
    assert got.shape == data.shape
    assert torch.allclose(got.to(torch.float64), data.to(torch.float64), atol=0.05)


def test_bzip2_short_tile_is_rejected(tmp_path):
    path = str(tmp_path / "s.fits")
    data = (torch.arange(64 * 64).reshape(64, 64) % 1000 + 7).to(torch.int16)
    torchfits.write(path, data, compress="BZIP2_1", overwrite=True)
    assert torch.equal(torchfits.read(path, hdu=1), data)

    raw = bytearray(open(path, "rb").read())
    row0 = data[0].numpy().astype(">i2").tobytes()
    full = bz2.compress(row0, 9)
    short = bz2.compress(row0[:8], 9)
    at = raw.find(full)
    assert at > 0 and len(short) <= len(full)
    raw[at : at + len(short)] = short
    open(path, "wb").write(bytes(raw))

    with pytest.raises(RuntimeError, match="uncompressing"):
        torchfits.read(path, hdu=1)
