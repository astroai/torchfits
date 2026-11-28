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


# --- integer images as quantized floats (fpack -i2f) -------------------------


def _int_image(shape, dtype, base=1000, sigma=30.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * sigma + base).round().to(dtype)


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (torch.int16, 1000),
        (torch.int32, 100000),
        (torch.uint16, 40000),
        (torch.uint8, 120),
        (torch.int8, 0),
    ],
)
@pytest.mark.parametrize("algorithm", ["GZIP_1", "RICE_1", "HCOMPRESS_1"])
def test_int_to_float_quantizes_within_step(tmp_path, dtype, base, algorithm):
    sigma = 3.0 if dtype in (torch.uint8, torch.int8) else 30.0
    data = _int_image((40, 48), dtype, base=base, sigma=sigma)
    path = tmp_path / "i2f.fits"
    torchfits.write(
        str(path),
        data,
        compress=algorithm,
        int_to_float=True,
        quantize_level=-0.5,  # absolute step 0.5
        dither_seed=11,
    )
    assert _zcards(path) == {
        "ZQUANTIZ": "SUBTRACTIVE_DITHER_1",
        "ZDITHER0": 11,
        "ZBITPIX": -32,
    }
    header = torchfits.read_header(str(path), 1)
    assert "BZERO" not in header and "BSCALE" not in header
    got = torchfits.read(str(path), hdu=1)
    assert got.dtype == torch.float32
    tolerance = 0.25 + 1e-3 if algorithm != "HCOMPRESS_1" else 0.5
    assert (got.double() - data.double()).abs().max() <= tolerance


def test_int_to_float_default_level_is_fpack_4(tmp_path):
    data = _int_image((40, 48), torch.int32)
    default = tmp_path / "default.fits"
    torchfits.write(str(default), data, compress="GZIP_1", int_to_float=True)
    cards = _zcards(default)
    assert cards["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"  # quantized even under GZIP_1
    explicit = tmp_path / "explicit.fits"
    torchfits.write(
        str(explicit),
        data,
        compress="GZIP_1",
        int_to_float=True,
        quantize_level=4.0,
        dither_seed=int(cards["ZDITHER0"]),
    )
    assert explicit.read_bytes() == default.read_bytes()
    # noise sigma 30 / level 4 -> step ~7.5: error stays well below sigma
    got = torchfits.read(str(default), hdu=1)
    assert (got.double() - data.double()).abs().max() < 30


def test_int_to_float_applies_scale_and_blank(tmp_path):
    raw = torch.full((8, 8), 7, dtype=torch.int16)
    raw[0, 0] = -999
    path = tmp_path / "scaled.fits"
    torchfits.write(
        str(path),
        raw,
        header={"BSCALE": 2.0, "BZERO": 10.0, "BLANK": -999, "OBJECT": "M31"},
        compress="RICE_1",
        int_to_float=True,
        quantize_level=-0.01,
    )
    header = torchfits.read_header(str(path), 1)
    assert header.get("OBJECT") == "M31"
    assert not {"BSCALE", "BZERO", "BLANK"} & set(header.keys())
    got = torchfits.read(str(path), hdu=1)
    assert torch.isnan(got[0, 0])
    assert torch.allclose(got.flatten()[1:], torch.tensor(24.0), atol=0.01)


def test_int_to_float_leaves_float_images_alone(tmp_path):
    floats = _image((16, 16))
    ints = _int_image((16, 16), torch.int16)
    path = tmp_path / "mixed.fits"
    torchfits.write(str(path), [ints, floats], compress="GZIP_1", int_to_float=True)
    # the float HDU keeps GZIP_1's lossless default; the int HDU is quantized
    assert _zcards(path, 2)["ZQUANTIZ"] == "NONE"
    assert torch.equal(torchfits.read(str(path), hdu=2), floats)
    assert _zcards(path, 1)["ZBITPIX"] == -32


def test_quantize_options_with_int_to_float_on_integer_image(tmp_path):
    path = tmp_path / "i.fits"
    torchfits.write(
        str(path),
        _int_image((16, 16), torch.int16),
        compress="RICE_1",
        int_to_float=True,
        quantize_method="no_dither",
    )
    assert _zcards(path)["ZQUANTIZ"] == "NO_DITHER"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"compress": "PLIO_1", "int_to_float": True}, ValueError, "PLIO_1"),
        (
            {"compress": "GZIP_1", "int_to_float": True, "quantize_level": 0},
            ValueError,
            "drop one of the two",
        ),
        ({"compress": "RICE_1", "int_to_float": 1}, TypeError, "bool"),
        ({"int_to_float": True}, ValueError, "pass compress="),
    ],
)
def test_int_to_float_validation(tmp_path, kwargs, error, match):
    path = tmp_path / "bad.fits"
    with pytest.raises(error, match=match):
        torchfits.write(str(path), _int_image((8, 8), torch.int16), **kwargs)
    assert not path.exists()


def test_cli_int_to_float(tmp_path):
    src = tmp_path / "int.fits"
    data = _int_image((40, 48), torch.int16)
    torchfits.write(str(src), data, header={"OBJECT": "M31"})
    out = tmp_path / "out.fits.fz"
    argv = ["compress", str(src), str(out), "--algorithm", "RICE_1", "--int-to-float"]
    assert main([*argv, "-q", "-0.5", "--dither-seed", "3"]) == 0
    assert _zcards(out) == {
        "ZQUANTIZ": "SUBTRACTIVE_DITHER_1",
        "ZDITHER0": 3,
        "ZBITPIX": -32,
    }
    assert torchfits.read_header(str(out), 1).get("OBJECT") == "M31"
    got = torchfits.read(str(out), hdu=1)
    assert (got.double() - data.double()).abs().max() <= 0.25 + 1e-3
    restored = tmp_path / "restored.fits"
    assert main(["decompress", str(out), str(restored)]) == 0
    assert torchfits.read_header(str(restored), 0).get("BITPIX") == -32


def test_cli_int_to_float_split_hdu(tmp_path):
    src = tmp_path / "mef.fits"
    torchfits.write(str(src), [_image((16, 20)), _int_image((12, 12), torch.int32)])
    out_dir = tmp_path / "split"
    argv = ["compress", str(src), "--split", "hdu", "--out-dir", str(out_dir)]
    assert main([*argv, "--algorithm", "RICE_1", "--int-to-float", "-q", "8"]) == 0
    outputs = sorted(out_dir.glob("*.fits"))
    assert [_zcards(p)["ZBITPIX"] for p in outputs] == [-32, -32]


def test_cli_int_to_float_usage_errors(tmp_path, capsys):
    src = tmp_path / "int.fits"
    torchfits.write(str(src), _int_image((8, 8), torch.int16))
    out = tmp_path / "out.fits.fz"
    argv = ["compress", str(src), str(out), "--int-to-float"]
    assert main([*argv, "--algorithm", "PLIO_1"]) == 2
    assert "PLIO_1" in capsys.readouterr().err
    assert main([*argv, "-q", "0"]) == 2
    assert "--quantize-level 0" in capsys.readouterr().err
    assert not out.exists()


# --- the same controls on write_tensor / insert_hdu / replace_hdu ------------


def test_write_tensor_takes_compression_controls(tmp_path):
    data = _image((32, 40))
    path = tmp_path / "wt.fits"
    torchfits.write_tensor(
        str(path),
        data,
        compress="RICE_1",
        tile_shape=(8, 10),
        quantize_method="subtractive_dither_2",
        dither_seed=77,
    )
    assert _ztile(path) == (10, 8)
    cards = _zcards(path)
    assert cards["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_2"
    assert cards["ZDITHER0"] == 77


def test_write_tensor_int_to_float(tmp_path):
    data = torch.arange(32 * 40, dtype=torch.int32).reshape(32, 40)
    path = tmp_path / "i2f.fits"
    torchfits.write_tensor(
        str(path), data, compress="GZIP_1", int_to_float=True, dither_seed=3
    )
    assert _zcards(path)["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1"
    assert torchfits.read(str(path), hdu=1).dtype == torch.float32


@pytest.mark.parametrize("op", ["insert", "replace"])
def test_hdu_edit_takes_compression_controls(tmp_path, op):
    path = tmp_path / "mef.fits"
    base = _image((16, 16), torch.int16)
    torchfits.write(str(path), [base, base.clone()], overwrite=True)
    data = _image((24, 30))
    kwargs = dict(
        compress="RICE_1",
        tile_shape=(6, 15),
        quantize_level=-0.5,
        quantize_method="no_dither",
    )
    if op == "insert":
        torchfits.insert_hdu(str(path), data, index=2, **kwargs)
    else:
        torchfits.replace_hdu(str(path), 1, data, **kwargs)
    target = 2 if op == "insert" else 1
    assert _ztile(path, target) == (15, 6)
    assert _zcards(path, target)["ZQUANTIZ"] == "NO_DITHER"
    got = torchfits.read(str(path), hdu=target)
    assert torch.allclose(got, data, atol=0.5)
    # every other HDU untouched
    assert torch.equal(torchfits.read(str(path), hdu=0), base)


@pytest.mark.parametrize("op", ["insert", "replace"])
def test_hdu_edit_int_to_float_and_seed(tmp_path, op):
    path = tmp_path / "mef.fits"
    base = _image((16, 16), torch.int16)
    torchfits.write(str(path), [base, base.clone()], overwrite=True)
    data = torch.arange(20 * 20, dtype=torch.int32).reshape(20, 20)
    kwargs = dict(compress="GZIP_1", int_to_float=True, dither_seed=19)
    if op == "insert":
        torchfits.insert_hdu(str(path), data, index=1, **kwargs)
    else:
        torchfits.replace_hdu(str(path), 1, data, **kwargs)
    assert _zcards(path, 1)["ZDITHER0"] == 19
    assert torchfits.read(str(path), hdu=1).dtype == torch.float32


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"tile_shape": (4, 4)}, ValueError, "pass compress="),
        ({"compress": "RICE_1", "tile_shape": (0, 4)}, ValueError, "tile"),
        ({"compress": "RICE_1", "dither_seed": 0}, ValueError, "dither_seed"),
        ({"compress": "RICE_1", "int_to_float": 1}, TypeError, "int_to_float"),
    ],
)
@pytest.mark.parametrize("op", ["insert", "replace"])
def test_hdu_edit_compression_control_validation_leaves_file(
    tmp_path, op, kwargs, error, match
):
    path = tmp_path / "mef.fits"
    base = _image((16, 16), torch.int16)
    torchfits.write(str(path), [base, base.clone()], overwrite=True)
    before = path.read_bytes()
    with pytest.raises(error, match=match):
        if op == "insert":
            torchfits.insert_hdu(str(path), _image((8, 8)), index=1, **kwargs)
        else:
            torchfits.replace_hdu(str(path), 1, _image((8, 8)), **kwargs)
    assert path.read_bytes() == before
