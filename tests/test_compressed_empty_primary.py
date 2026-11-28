"""A compressed MEF whose primary is ``TensorHDU(torch.empty(0))`` writes.

The compressed writer emits its own empty primary, so the caller's empty
primary must be skipped. Only a ``NAXIS=0`` header card was recognised; the
examples' idiom (an empty tensor with no header) failed with CFITSIO status
414 for every ``compress=`` value.
"""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits import HDUList, TensorHDU


@pytest.mark.parametrize("compress", ["RICE", True])
def test_empty_tensor_primary_with_compression(tmp_path, compress):
    sci = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    err = torch.ones(20, 20, dtype=torch.float32)
    hdul = HDUList(
        [
            TensorHDU(torch.empty(0)),
            TensorHDU(sci, {"EXTNAME": "SCI"}),
            TensorHDU(err, {"EXTNAME": "ERR"}),
        ]
    )
    path = str(tmp_path / f"mef_{compress}.fits")
    torchfits.write(path, hdul, compress=compress, overwrite=True)
    assert torch.allclose(torchfits.read(path, hdu="SCI"), sci)
    assert torch.allclose(torchfits.read(path, hdu="ERR"), err)
    with torchfits.open(path) as opened:
        assert len(opened) == 3


@pytest.mark.parametrize(
    "header",
    [
        {"OBSERVER": "me", "TELESCOP": "CFHT"},
        {"SIMPLE": True, "NAXIS": 0, "OBSERVER": "me", "TELESCOP": "CFHT"},
    ],
)
def test_empty_primary_keeps_its_header_cards(tmp_path, header):
    """The skipped placeholder's user cards used to be dropped."""
    path = str(tmp_path / "mef.fits")
    hdul = HDUList(
        [
            TensorHDU(torch.empty(0), header=header),
            TensorHDU(torch.ones(8, 8), {"EXTNAME": "SCI"}),
        ]
    )
    torchfits.write(path, hdul, compress="RICE")
    primary = torchfits.read_header(path, 0)
    assert (primary["OBSERVER"], primary["TELESCOP"]) == ("me", "CFHT")
    assert primary["NAXIS"] == 0
    assert torchfits.read_header(path, "SCI").get("OBSERVER") is None


@pytest.mark.parametrize("compress", ["RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"])
def test_zero_size_image_hdu_is_stored_uncompressed(tmp_path, compress):
    # CFITSIO cannot tile an image with a zero-length axis: a compressed
    # write of one (an extension, or the primary of a re-opened file) killed
    # the process with SIGFPE. It is now stored uncompressed, like fpack does.
    path = str(tmp_path / "zero.fits")
    data = torch.ones(5, 6, dtype=torch.int16)
    torchfits.write(
        path,
        [torch.zeros(4, 4), torch.zeros(3, 0, 12), data, torch.zeros(0, 7)],
        compress=compress,
    )
    assert torchfits.read(path, hdu=2).shape == (3, 0, 12)
    assert torchfits.read(path, hdu=4).shape == (0, 7)
    assert "ZIMAGE" not in torchfits.read_header(path, 2)
    assert torchfits.read_header(path, 3).get("ZIMAGE") is True
    assert torch.equal(torchfits.read(path, hdu=3), data)

    cube = str(tmp_path / "cube.fits")
    torchfits.write(cube, torch.zeros(3, 0, 12, dtype=torch.float64))
    out = str(tmp_path / "cube_c.fits")
    with torchfits.open(cube) as hdul:
        torchfits.write(out, hdul, compress=compress)
    assert torchfits.read(out, hdu=1).shape == (3, 0, 12)


@pytest.mark.parametrize("compress", [True, "RICE_1", "GZIP_1"])
@pytest.mark.parametrize("shape", [(3, 0, 12), (0, 7)])
def test_compressed_write_of_zero_size_tensor_keeps_declared_shape(
    tmp_path, compress, shape
):
    # A bare zero-size tensor with declared axes was mistaken for the empty
    # placeholder primary and dropped: the file held only the writer's empty
    # primary, so the declared shape was lost. Like any compressed write the
    # image now lands in HDU 1 (stored uncompressed, as fpack does).
    path = str(tmp_path / "zero_c.fits")
    torchfits.write(path, torch.zeros(*shape), compress=compress)
    assert torchfits.read(path, hdu=1).shape == shape
    assert torchfits.read(path, hdu="auto").shape == shape
    with torchfits.open(path) as hdul:
        assert len(hdul) == 2


def test_compressed_write_of_shapeless_empty_tensor_is_a_placeholder(tmp_path):
    path = str(tmp_path / "empty_c.fits")
    torchfits.write(path, torch.empty(0), compress=True)
    with torchfits.open(path) as hdul:
        assert len(hdul) == 1
    assert torchfits.read(path).shape == (0,)
