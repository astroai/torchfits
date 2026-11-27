"""fpack-style tile-compression controls on write() and ``torchfits compress``.

Tile geometry (``tile_shape=`` / ``--tile``, fpack ``-t``) and float
quantization (``quantize_level=`` / ``quantize_method=`` / ``dither_seed=``,
fpack ``-q``).
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


# --- float quantization / dithering -----------------------------------------


def _zcards(path, hdu=1) -> dict:
    header = torchfits.read_header(str(path), hdu)
    return {key: header.get(key) for key in ("ZQUANTIZ", "ZDITHER0", "ZBITPIX")}


@pytest.mark.parametrize("algorithm", ["RICE_1", "GZIP_2", "HCOMPRESS_1"])
def test_default_quantization_is_unchanged(tmp_path, algorithm):
    """Implicit defaults == CFITSIO's level 4 + SUBTRACTIVE_DITHER_1, byte for
    byte once the clock-picked dither seed is pinned to the same value."""
    data = _image((40, 48))
    default = tmp_path / "default.fits"
    torchfits.write(str(default), data, compress=algorithm)
    cards = _zcards(default)
    assert cards["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
    seed = int(cards["ZDITHER0"])
    assert 1 <= seed <= 10000
    explicit = tmp_path / "explicit.fits"
    torchfits.write(
        str(explicit),
        data,
        compress=algorithm,
        quantize_level=4.0,
        quantize_method="subtractive_dither_1",
        dither_seed=seed,
    )
    assert explicit.read_bytes() == default.read_bytes()


def test_default_gzip_float_stays_lossless(tmp_path):
    data = _image((40, 48))
    path = tmp_path / "g.fits"
    torchfits.write(str(path), data, compress="GZIP_1")
    assert _zcards(path)["ZQUANTIZ"] == "NONE"
    assert torch.equal(torchfits.read(str(path), hdu=1), data)


def test_dither_seed_is_reproducible(tmp_path):
    data = _image((40, 48))
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    for path in (a, b):
        torchfits.write(str(path), data, compress="RICE_1", dither_seed=1234)
    assert _zcards(a)["ZDITHER0"] == 1234
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("algorithm", ["GZIP_1", "GZIP_2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lossless_float_round_trips_exactly(tmp_path, algorithm, dtype):
    data = _image((33, 47), dtype)
    data[0, 0] = float("nan")
    data[1, 1] = float("inf")
    data[2, 2] = -0.0
    path = tmp_path / "lossless.fits"
    torchfits.write(str(path), data, compress=algorithm, quantize_level=0)
    assert _zcards(path)["ZQUANTIZ"] == "NONE"
    got = torchfits.read(str(path), hdu=1)
    assert got.dtype == dtype
    assert torch.equal(got.view(torch.uint8), data.view(torch.uint8))


def test_quantize_level_noise_relative_and_absolute(tmp_path):
    data = _image((64, 64))  # noise sigma ~50
    errors = {}
    for level in (2.0, 32.0, -0.01):
        path = tmp_path / f"q{level}.fits"
        torchfits.write(
            str(path), data, compress="RICE_1", quantize_level=level, dither_seed=3
        )
        assert _zcards(path)["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
        errors[level] = float((torchfits.read(str(path), hdu=1) - data).abs().max())
    assert errors[32.0] < errors[2.0]
    # negative level = absolute step: subtractive dither keeps |err| <= step / 2
    assert errors[-0.01] <= 0.005 + 1e-4


def test_quantize_method_cards(tmp_path):
    data = _image((40, 48))
    data[5, :] = 0.0
    nd = tmp_path / "nd.fits"
    torchfits.write(str(nd), data, compress="RICE_1", quantize_method="no_dither")
    assert _zcards(nd)["ZQUANTIZ"] == "NO_DITHER"
    assert _zcards(nd)["ZDITHER0"] is None
    # CFITSIO's default level without dithering is 16 (finer than 4)
    err = (torchfits.read(str(nd), hdu=1) - data).abs().max()
    assert err < 50 / 16
    d2 = tmp_path / "d2.fits"
    torchfits.write(
        str(d2),
        data,
        compress="RICE_1",
        quantize_method="SUBTRACTIVE_DITHER_2",
        dither_seed=77,
    )
    assert _zcards(d2) == {
        "ZQUANTIZ": "SUBTRACTIVE_DITHER_2",
        "ZDITHER0": 77,
        "ZBITPIX": -32,
    }
    got = torchfits.read(str(d2), hdu=1)
    assert torch.equal(got[5], torch.zeros(48))  # dither 2 keeps exact zeros


def test_quantize_options_skip_integer_hdus(tmp_path):
    ints = _image((16, 16), torch.int32)
    floats = _image((16, 16))
    path = tmp_path / "mixed.fits"
    torchfits.write(
        str(path), [ints, floats], compress="RICE_1", quantize_level=8, dither_seed=5
    )
    assert torch.equal(torchfits.read(str(path), hdu=1), ints)
    assert _zcards(path, 1)["ZQUANTIZ"] is None
    assert _zcards(path, 2)["ZDITHER0"] == 5


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"compress": "RICE_1", "quantize_level": 0}, ValueError, "GZIP_1' or 'GZIP_2"),
        ({"compress": "HCOMPRESS_1", "quantize_level": 0}, ValueError, "lossless"),
        ({"compress": "PLIO_1", "quantize_level": 4}, ValueError, "PLIO_1"),
        ({"compress": "GZIP_1", "dither_seed": 4}, ValueError, "unless quantize_level"),
        (
            {"compress": "GZIP_2", "quantize_level": 0, "quantize_method": "no_dither"},
            ValueError,
            "quantize_level=0 stores them losslessly",
        ),
        (
            {"compress": "RICE_1", "quantize_method": "no_dither", "dither_seed": 3},
            ValueError,
            "no effect",
        ),
        (
            {"compress": "H", "quantize_method": "subtractive_dither_2"},
            ValueError,
            "HCOMPRESS_1 does not support",
        ),
        ({"compress": "RICE_1", "quantize_method": "dither"}, ValueError, "one of"),
        ({"compress": "RICE_1", "quantize_method": 1}, TypeError, "one of"),
        ({"compress": "RICE_1", "dither_seed": 0}, ValueError, "1..10000"),
        ({"compress": "RICE_1", "dither_seed": 10001}, ValueError, "1..10000"),
        ({"compress": "RICE_1", "dither_seed": 2.0}, TypeError, "int"),
        ({"compress": "RICE_1", "quantize_level": "4"}, TypeError, "number"),
        ({"compress": "RICE_1", "quantize_level": float("nan")}, ValueError, "finite"),
        ({"quantize_level": 4}, ValueError, "pass compress="),
    ],
)
def test_quantize_option_validation(tmp_path, kwargs, error, match):
    path = tmp_path / "bad.fits"
    with pytest.raises(error, match=match):
        torchfits.write(str(path), _image((8, 8)), **kwargs)
    assert not path.exists()


def test_quantize_options_on_integer_image_are_refused(tmp_path):
    path = tmp_path / "int.fits"
    with pytest.raises(ValueError, match="every image HDU in this write is integer"):
        torchfits.write(
            str(path), _image((8, 8), torch.int16), compress="RICE_1", quantize_level=4
        )
    assert not path.exists()


def test_cli_quantize_flags(tmp_path):
    src = tmp_path / "src.fits"
    data = _image((40, 48))
    torchfits.write(str(src), data)
    out = tmp_path / "out.fits.fz"
    argv = ["compress", str(src), str(out), "--algorithm", "RICE_1", "-q", "-0.01"]
    argv += ["--quantize-method", "subtractive_dither_2", "--dither-seed", "42"]
    assert main(argv) == 0
    assert _zcards(out) == {
        "ZQUANTIZ": "SUBTRACTIVE_DITHER_2",
        "ZDITHER0": 42,
        "ZBITPIX": -32,
    }
    assert (torchfits.read(str(out), hdu=1) - data).abs().max() <= 0.005 + 1e-4
    lossless = tmp_path / "lossless.fits.fz"
    assert (
        main(["compress", str(src), str(lossless), "--algorithm", "GZIP_2", "-q", "0"])
        == 0
    )
    assert torch.equal(torchfits.read(str(lossless), hdu=1), data)


def test_cli_quantize_split_hdu_keeps_integer_hdus_lossless(tmp_path):
    src = tmp_path / "mef.fits"
    ints = _image((12, 12), torch.int32)
    torchfits.write(str(src), [_image((16, 20)), ints])
    out_dir = tmp_path / "split"
    argv = ["compress", str(src), "--split", "hdu", "--out-dir", str(out_dir)]
    assert main([*argv, "--algorithm", "RICE_1", "-q", "8", "--dither-seed", "9"]) == 0
    first, second = sorted(out_dir.glob("*.fits"))
    assert _zcards(first)["ZDITHER0"] == 9
    assert torch.equal(torchfits.read(str(second), hdu=1), ints)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--algorithm", "RICE_1", "-q", "0"], "--algorithm 'GZIP_1' or 'GZIP_2'"),
        (["--dither-seed", "5"], "unless --quantize-level is set"),
        (["--algorithm", "RICE_1", "--dither-seed", "0"], "1..10000"),
        (["--quantize-method", "bogus"], "invalid choice"),
    ],
)
def test_cli_quantize_usage_errors(tmp_path, capsys, argv, message):
    src = tmp_path / "src.fits"
    torchfits.write(str(src), _image((8, 8)))
    out = tmp_path / "out.fits.fz"
    assert _main_exit(["compress", str(src), str(out), *argv]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_cli_quantize_on_integer_file_is_usage_error(tmp_path, capsys):
    src = tmp_path / "int.fits"
    torchfits.write(str(src), _image((8, 8), torch.int16))
    out = tmp_path / "out.fits.fz"
    assert (
        main(["compress", str(src), str(out), "--algorithm", "RICE_1", "-q", "4"]) == 2
    )
    assert "every selected image HDU is integer" in capsys.readouterr().err
    assert not out.exists()


def _main_exit(argv) -> int:
    """main() exit code, including argparse's SystemExit for bad choices."""
    try:
        return main(argv)
    except SystemExit as exc:
        return int(exc.code)
