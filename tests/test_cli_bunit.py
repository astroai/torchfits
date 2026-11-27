"""Conservative BUNIT handling (no unit algebra) in ``arith`` and ``transform``."""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits.cli.main import main


def _image(path, bunit=None, value=2.0):
    header = {"OBJECT": "M31"}
    if bunit is not None:
        header["BUNIT"] = bunit
    torchfits.write(str(path), torch.full((4, 4), value), header=header, overwrite=True)


def _bunit(path):
    return torchfits.read_header(str(path), 0).get("BUNIT")


@pytest.mark.parametrize(
    ("op", "value", "kept"),
    [
        ("add", "5", True),
        ("sub", "5", True),
        ("mul", "1", True),
        ("div", "1", True),
        ("mul", "2", False),
        ("div", "2", False),
    ],
)
def test_arith_scalar_bunit(tmp_path, op, value, kept):
    src = tmp_path / "a.fits"
    _image(src, "adu")
    out = tmp_path / "out.fits"
    assert main(["arith", str(src), "--op", op, "--value", value, "-o", str(out)]) == 0
    assert _bunit(out) == ("adu" if kept else None)
    assert torchfits.read_header(str(out), 0)["OBJECT"] == "M31"


@pytest.mark.parametrize("op", ["mul", "div"])
def test_arith_image_mul_div_drops_bunit(tmp_path, op):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _image(a, "adu")
    _image(b, "adu")
    out = tmp_path / "out.fits"
    assert main(["arith", str(a), str(b), "--op", op, "-o", str(out)]) == 0
    assert _bunit(out) is None


@pytest.mark.parametrize("op", ["add", "sub"])
def test_arith_image_add_sub_same_bunit_kept(tmp_path, op):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _image(a, "adu")
    _image(b, "adu")
    out = tmp_path / "out.fits"
    assert main(["arith", str(a), str(b), "--op", op, "-o", str(out)]) == 0
    assert _bunit(out) == "adu"


@pytest.mark.parametrize("op", ["add", "sub"])
def test_arith_image_add_sub_bunit_mismatch_refused(tmp_path, op, capsys):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _image(a, "adu")
    _image(b, "electron")
    out = tmp_path / "out.fits"
    assert main(["arith", str(a), str(b), "--op", op, "-o", str(out)]) == 2
    assert "different BUNIT" in capsys.readouterr().err
    assert not out.exists()


def test_arith_image_add_one_side_without_bunit_keeps_a(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _image(a, "adu")
    _image(b, None)
    out = tmp_path / "out.fits"
    assert main(["arith", str(a), str(b), "--op", "add", "-o", str(out)]) == 0
    assert _bunit(out) == "adu"


@pytest.mark.parametrize(
    ("spec", "kept"),
    [
        ("MinMaxNormalize", False),
        ("LogStretch", False),
        ("ArcsinhStretch", False),
        ("ZScaleNormalize", False),
        ("AffineTransform:scale=2.0", False),
        ("AffineTransform:scale=1.0,offset=3.0", True),
        ("BackgroundSubtract", True),
        ("SigmaClip", True),
    ],
)
def test_transform_bunit(tmp_path, spec, kept):
    src = tmp_path / "a.fits"
    torchfits.write(
        str(src),
        torch.arange(1, 65, dtype=torch.float32).reshape(8, 8),
        header={"BUNIT": "adu", "OBJECT": "M31"},
        overwrite=True,
    )
    out = tmp_path / "out.fits"
    assert main(["transform", str(src), str(out), "--name", spec]) == 0
    assert _bunit(out) == ("adu" if kept else None)
    assert torchfits.read_header(str(out), 0)["OBJECT"] == "M31"
