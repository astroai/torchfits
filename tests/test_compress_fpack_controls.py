"""fpack-style tile-compression controls on write() and ``torchfits compress``.

Tile geometry (``tile_shape=`` / ``--tile``, fpack ``-t``).
"""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits.cli.main import main


def _ztile(path, hdu=1) -> tuple[int, ...]:
    header = torchfits.read_header(str(path), hdu)
    naxis = int(header["ZNAXIS"])
    return tuple(int(header[f"ZTILE{axis}"]) for axis in range(1, naxis + 1))


def _image(shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(*shape, generator=g) * 50 + 1000
    return data.to(dtype)


# --- tile geometry -----------------------------------------------------------


def test_default_tiles_unchanged(tmp_path):
    data = _image((37, 53))
    for algorithm, expected in [("GZIP_1", (53, 1)), ("HCOMPRESS_1", (53, 16))]:
        path = tmp_path / f"{algorithm}.fits"
        torchfits.write(str(path), data, compress=algorithm)
        assert _ztile(path) == expected


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int16, torch.int32, torch.uint8]
)
def test_tile_shape_sets_ztile_and_round_trips(tmp_path, dtype):
    data = _image((37, 53), dtype)
    path = tmp_path / "tiled.fits"
    torchfits.write(str(path), data, compress="GZIP_1", tile_shape=(10, 20))
    # tensor order (rows, cols) -> ZTILE1 = cols, ZTILE2 = rows
    assert _ztile(path) == (20, 10)
    assert torch.equal(torchfits.read(str(path), hdu=1), data)


def test_tile_shape_covers_trailing_axes_and_whole_axis(tmp_path):
    data = _image((3, 37, 53))
    path = tmp_path / "cube.fits"
    torchfits.write(str(path), data, compress="RICE_1", tile_shape=(-1, 100))
    # leading axis gets 1; -1 and sizes past the axis mean the whole axis
    assert _ztile(path) == (53, 37, 1)
    got = torchfits.read(str(path), hdu=1)
    assert got.shape == data.shape
    assert torch.allclose(got, data, atol=50)


def test_tile_shape_applies_to_every_image_hdu(tmp_path):
    path = tmp_path / "mef.fits"
    images = [_image((16, 16), torch.int16), _image((20, 24))]
    torchfits.write(str(path), images, compress="GZIP_1", tile_shape=(8, 8))
    assert _ztile(path, 1) == (8, 8)
    assert _ztile(path, 2) == (8, 8)


def test_hcompress_tile_shape(tmp_path):
    data = _image((40, 48), torch.int32)
    path = tmp_path / "h.fits"
    torchfits.write(str(path), data, compress="HCOMPRESS_1", tile_shape=(20, 24))
    assert _ztile(path) == (24, 20)
    assert torch.equal(torchfits.read(str(path), hdu=1), data)


@pytest.mark.parametrize(
    ("shape", "tile", "match"),
    [
        ((40, 48), (1, 48), "HCOMPRESS_1 needs 2-D tiles"),
        ((40, 48), (3, 48), "at least 4 pixels"),
        ((4, 40, 48), (2, 8, 8), "exactly two tile axes"),
    ],
)
def test_hcompress_tile_constraints(tmp_path, shape, tile, match):
    path = tmp_path / "h.fits"
    with pytest.raises(ValueError, match=match):
        torchfits.write(str(path), _image(shape), compress="H", tile_shape=tile)
    assert not path.exists()


@pytest.mark.parametrize(
    ("tile", "error", "match"),
    [
        ((1, 2, 3), ValueError, "has 3 axes but image HDU 0 has 2"),
        ((0, 4), ValueError, "must be positive"),
        ((-2, 4), ValueError, "must be positive"),
        ((), ValueError, "at least one axis"),
        ("10,10", TypeError, "sequence of ints"),
        ((1.5, 2), TypeError, "must be ints"),
    ],
)
def test_tile_shape_validation(tmp_path, tile, error, match):
    path = tmp_path / "bad.fits"
    with pytest.raises(error, match=match):
        torchfits.write(str(path), _image((8, 8)), compress="GZIP_1", tile_shape=tile)
    assert not path.exists()


def test_tile_shape_requires_compress(tmp_path):
    with pytest.raises(ValueError, match="pass compress="):
        torchfits.write(str(tmp_path / "x.fits"), _image((8, 8)), tile_shape=(4, 4))


def test_cli_tile_is_fits_axis_order(tmp_path):
    src = tmp_path / "src.fits"
    data = _image((37, 53), torch.int16)
    torchfits.write(str(src), data)
    out = tmp_path / "out.fits.fz"
    assert main(["compress", str(src), str(out), "--tile", "53,5"]) == 0
    assert _ztile(out) == (53, 5)
    assert torch.equal(torchfits.read(str(out), hdu=1), data)
    whole = tmp_path / "whole.fits.fz"
    assert main(["compress", str(src), str(whole), "-t=-1,-1"]) == 0
    assert _ztile(whole) == (53, 37)


def test_cli_tile_split_hdu(tmp_path):
    src = tmp_path / "mef.fits"
    torchfits.write(str(src), [_image((16, 20)), _image((12, 12), torch.int32)])
    out_dir = tmp_path / "split"
    rc = main(
        ["compress", str(src), "--split", "hdu", "--out-dir", str(out_dir), "-t", "4,4"]
    )
    assert rc == 0
    outputs = sorted(out_dir.glob("*.fits"))
    assert len(outputs) == 2
    assert all(_ztile(p) == (4, 4) for p in outputs)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--tile", "10,x"], "comma-separated integers"),
        (["--tile", "10,0"], "must be positive"),
        (["--tile", "1,2,3"], "has 3 axes"),
        (["--algorithm", "HCOMPRESS_1", "--tile", "48,2"], "HCOMPRESS_1 needs 2-D"),
    ],
)
def test_cli_tile_usage_errors(tmp_path, capsys, argv, message):
    src = tmp_path / "src.fits"
    torchfits.write(str(src), _image((40, 48)))
    out = tmp_path / "out.fits.fz"
    assert main(["compress", str(src), str(out), *argv]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()
