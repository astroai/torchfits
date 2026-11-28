"""``arith`` and ``cutout`` drop stale DATAMIN/DATAMAX (as ``transform`` does)."""

from __future__ import annotations

import torch

import torchfits
from torchfits.cli.main import main


def _image(path, data=None):
    if data is None:
        data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(
        str(path),
        data,
        header={"DATAMIN": 0.0, "DATAMAX": 15.0, "OBJECT": "M31"},
        overwrite=True,
    )


def _assert_range_dropped(path):
    header = torchfits.read_header(str(path), 0)
    assert "DATAMIN" not in header
    assert "DATAMAX" not in header
    assert header["OBJECT"] == "M31"


def test_arith_scalar_drops_datamin_datamax(tmp_path):
    src = tmp_path / "a.fits"
    _image(src)
    out = tmp_path / "out.fits"
    assert (
        main(["arith", str(src), "--op", "mul", "--value", "10", "-o", str(out)]) == 0
    )
    _assert_range_dropped(out)
    assert float(torchfits.read_tensor(str(out)).max()) == 150.0


def test_arith_image_drops_datamin_datamax(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _image(a)
    _image(b)
    out = tmp_path / "out.fits"
    assert main(["arith", str(a), str(b), "--op", "add", "-o", str(out)]) == 0
    _assert_range_dropped(out)


def test_arith_split_hdu_drops_datamin_datamax(tmp_path):
    src = tmp_path / "a.fits"
    _image(src)
    out_dir = tmp_path / "split"
    code = main(
        [
            "arith",
            str(src),
            "--op",
            "add",
            "--value",
            "1",
            "--split",
            "hdu",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    _assert_range_dropped(out_dir / "a_hdu00.fits")


def test_cutout_box_drops_datamin_datamax(tmp_path):
    src = tmp_path / "a.fits"
    _image(src)
    out = tmp_path / "out.fits"
    assert main(["cutout", str(src), "--box", "0,0,2,2", "-o", str(out)]) == 0
    _assert_range_dropped(out)


def test_cutout_section_drops_datamin_datamax(tmp_path):
    src = tmp_path / "a.fits"
    _image(src)
    out = tmp_path / "out.fits"
    assert main(["cutout", f"{src}[1:2,1:2]", "-o", str(out)]) == 0
    _assert_range_dropped(out)
