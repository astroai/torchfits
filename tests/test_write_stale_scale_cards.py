"""BSCALE/BZERO copied from a header whose storage the tensor left are dropped.

A uint16 source header (BITPIX=16, BZERO=32768) written with an int32 tensor
(user arithmetic promoted the dtype) kept BZERO, so every value read back
shifted by 32768. Only floats were protected.
"""

import pytest
import torch

import torchfits


def _source_header(tmp_path):
    src = str(tmp_path / "src.fits")
    torchfits.write(src, torch.tensor([[0, 1000, 65535]], dtype=torch.uint16))
    return torchfits.read(src), torchfits.read_header(src)


@pytest.mark.parametrize(
    "wrap, kwargs",
    [
        (lambda y, h: (y, {"header": h}), {}),
        (lambda y, h: (y, {"header": h}), {"compress": True}),
        (lambda y, h: ({"data": y, "header": h}, {}), {"compress": True}),
        (lambda y, h: ([{"data": y, "header": h}], {}), {}),
    ],
)
def test_promoted_int_drops_stale_bzero(tmp_path, wrap, kwargs):
    x, h = _source_header(tmp_path)
    y = x.to(torch.int32) * 2
    data, extra = wrap(y, h)
    out = str(tmp_path / "out.fits")
    torchfits.write(out, data, **extra, **kwargs)
    hdu = 1 if kwargs.get("compress") else 0
    assert torchfits.read(out, hdu=hdu).tolist() == y.tolist()


def test_matching_storage_keeps_and_conventions_survive(tmp_path):
    x, h = _source_header(tmp_path)
    out = str(tmp_path / "same.fits")
    torchfits.write(out, x, header=h)  # uint16 again: convention kept
    got = torchfits.read(out)
    assert got.dtype == torch.uint16 and got.tolist() == x.tolist()
    # Stale BITPIX=32 header on a uint16 tensor: the convention cards win.
    torchfits.write(out, x, header={"BITPIX": 32, "BZERO": 7.0}, overwrite=True)
    for compress in (False, True):
        torchfits.write(
            out,
            x,
            header={"BITPIX": 32, "BZERO": 7.0},
            compress=compress,
            overwrite=True,
        )
        got = torchfits.read(out, hdu=1 if compress else 0)
        assert got.dtype == torch.uint16 and got.tolist() == x.tolist()


def test_quantize_keeps_its_scale_over_float_header(tmp_path):
    src = str(tmp_path / "f.fits")
    img = torch.linspace(0, 1, 64).reshape(8, 8)
    torchfits.write(src, img)
    h = torchfits.read_header(src)
    for compress in (False, True):
        out = str(tmp_path / f"q{compress}.fits")
        torchfits.write(out, img, header=h, quantize="robust", compress=compress)
        got = torchfits.read(out, hdu=1 if compress else 0)
        assert torch.allclose(got.float(), img, atol=1e-3)
