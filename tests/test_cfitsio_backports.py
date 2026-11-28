"""Crafted inputs for the CFITSIO fixes carried in extern/patches/.

Each case below is a file that crashed or read out of bounds in the vendored
cfitsio-4.7.0 (SIGFPE, SIGSEGV, stack smashing, or an AddressSanitizer
report) and now fails with a Python exception, or reads correctly. The
crafted reads run in a child interpreter so that a regression shows up as a
failed test with the child's signal instead of killing the test session.
Under the sanitizer workflow the child inherits LD_PRELOAD, so an
out-of-bounds read that happens to return garbage still fails there.

Each test names the extern/patches/ file whose fix it covers.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import torchfits

# --------------------------------------------------------------------------
# FITS building blocks


def _card(key: str, value: str | int | bool | None = None) -> bytes:
    if value is None:
        text = key
    elif isinstance(value, bool):
        text = f"{key:<8}= {'T' if value else 'F':>20}"
    elif isinstance(value, str):
        text = f"{key:<8}= {value!r:<20}"
    else:
        text = f"{key:<8}= {value:>20}"
    return text.ljust(80).encode("ascii")


def _header(cards: list[bytes]) -> bytes:
    payload = b"".join([*cards, _card("END")])
    return payload.ljust((len(payload) + 2879) // 2880 * 2880, b" ")


def _pad(data: bytes, fill: bytes = b"\0") -> bytes:
    return data.ljust((len(data) + 2879) // 2880 * 2880, fill)


_PRIMARY_NULL = _header(
    [
        _card("SIMPLE", True),
        _card("BITPIX", 8),
        _card("NAXIS", 0),
        _card("EXTEND", True),
    ]
)


def _parse_hdus(buf: bytes) -> list[tuple[int, dict[str, str]]]:
    """(data start, cards) for every HDU."""
    hdus = []
    pos = 0
    while pos < len(buf):
        cards: dict[str, str] = {}
        done = False
        while not done:
            block = buf[pos : pos + 2880]
            pos += 2880
            for i in range(36):
                card = block[i * 80 : (i + 1) * 80].decode("ascii")
                key = card[:8].strip()
                if key == "END":
                    done = True
                    break
                if card[8:10] == "= ":
                    value = card[10:].split("/")[0].strip()
                    cards[key] = value.strip("'").strip()
        hdus.append((pos, cards))
        naxis = int(cards.get("NAXIS", 0))
        size = 0
        if naxis:
            count = 1
            for i in range(1, naxis + 1):
                count *= int(cards[f"NAXIS{i}"])
            size = (
                abs(int(cards["BITPIX"])) // 8 * (int(cards.get("PCOUNT", 0)) + count)
            )
        pos += (size + 2879) // 2880 * 2880
    return hdus


def _run_child(code: str, *args: str, cwd: Path | None = None) -> str:
    """Run `code` in a fresh interpreter; fail on any signal or traceback."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=cwd,
    )
    assert result.returncode == 0, (
        f"child exited with {result.returncode} "
        f"(negative = killed by that signal)\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout


# A read that must fail with an exception, not a crash or a silent result.
_EXPECT_ERROR = """
import sys
import torchfits
try:
    result = torchfits.read(sys.argv[1], hdu=int(sys.argv[2]))
except (RuntimeError, OSError, ValueError) as exc:
    print("RAISED", type(exc).__name__, exc)
else:
    print("READ", tuple(result.shape))
"""


def _assert_read_raises(target: str, hdu: int = 0, cwd: Path | None = None) -> str:
    out = _run_child(_EXPECT_ERROR, target, str(hdu), cwd=cwd)
    assert out.startswith("RAISED"), out
    return out


# --------------------------------------------------------------------------
# Tile-compressed images (RICE_1, HCOMPRESS_1)


def _compressed_tile(path: Path) -> tuple[int, int, int, str]:
    """(descriptor position, tile length, tile heap position, TFORM)."""
    buf = path.read_bytes()
    data_start, cards = next(h for h in _parse_hdus(buf) if h[1].get("ZIMAGE") == "T")
    assert cards["TTYPE1"] == "COMPRESSED_DATA", cards
    tform = cards["TFORM1"]
    code = ">qq" if "Q" in tform else ">ii"
    length, offset = struct.unpack_from(code, buf, data_start)
    heap = data_start + int(
        cards.get("THEAP", int(cards["NAXIS1"]) * int(cards["NAXIS2"]))
    )
    return data_start, length, heap + offset, tform


def _write_compressed(path: Path, dtype: torch.dtype, algorithm: str) -> Path:
    data = (torch.arange(64 * 64) % 97 * 3).reshape(64, 64).to(dtype)
    torchfits.write(str(path), data, compress=algorithm, tile_shape=(64, 64))
    return path


def _truncate_tile(path: Path) -> None:
    descriptor, length, _, tform = _compressed_tile(path)
    code = ">qq" if "Q" in tform else ">ii"
    buf = bytearray(path.read_bytes())
    _, offset = struct.unpack_from(code, buf, descriptor)
    # Half the compressed bytes: the decoder needs the rest and, unpatched,
    # read on past the end of the tile buffer to find them.
    struct.pack_into(code, buf, descriptor, length // 2, offset)
    path.write_bytes(bytes(buf))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32])
def test_truncated_rice_tile_raises(tmp_path: Path, dtype: torch.dtype) -> None:
    """ricecomp-input-bounds: uint8/int16/int32 exercise fits_rdecomp_byte,
    fits_rdecomp_short and fits_rdecomp."""
    path = _write_compressed(tmp_path / "rice.fits", dtype, "RICE_1")
    _truncate_tile(path)
    _assert_read_raises(str(path), hdu=1)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_truncated_hcompress_tile_raises(tmp_path: Path, dtype: torch.dtype) -> None:
    """hdecompress-input-bounds: int16 -> fits_hdecompress, int32 ->
    fits_hdecompress64."""
    path = _write_compressed(tmp_path / "hcomp.fits", dtype, "HCOMPRESS_1")
    _truncate_tile(path)
    _assert_read_raises(str(path), hdu=1)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_hcompress_tile_with_zero_ny_raises(tmp_path: Path, dtype: torch.dtype) -> None:
    """hdecompress-input-bounds: an H-compress stream is magic(2) nx(4) ny(4)
    ...; ny == 0 divided by zero in decode() (SIGFPE on x86)."""
    path = _write_compressed(tmp_path / "hcomp-ny0.fits", dtype, "HCOMPRESS_1")
    _, _, tile, _ = _compressed_tile(path)
    buf = bytearray(path.read_bytes())
    assert buf[tile : tile + 2] == b"\xdd\x99"
    buf[tile + 6 : tile + 10] = b"\0\0\0\0"
    path.write_bytes(bytes(buf))
    _assert_read_raises(str(path), hdu=1)


def test_intact_compressed_tiles_still_read(tmp_path: Path) -> None:
    """The decoder checks must not reject valid tiles."""
    for algorithm in ("RICE_1", "HCOMPRESS_1", "GZIP_1", "GZIP_2"):
        for dtype in (torch.int16, torch.int32):
            path = _write_compressed(
                tmp_path / f"{algorithm}-{dtype}.fits", dtype, algorithm
            )
            expected = (torch.arange(64 * 64) % 97 * 3).reshape(64, 64).to(dtype)
            assert torch.equal(torchfits.read(str(path), hdu=1), expected)


# --------------------------------------------------------------------------
# Images with many axes


def _write_many_axis_image(path: Path, naxis: int) -> Path:
    cards = [_card("SIMPLE", True), _card("BITPIX", 16), _card("NAXIS", naxis)]
    cards += [_card(f"NAXIS{i}", 1) for i in range(1, naxis + 1)]
    path.write_bytes(_header(cards) + _pad(b"\0\1"))
    return path


def test_image_section_of_many_axis_image_raises(tmp_path: Path) -> None:
    """copy-image-section-naxis: fits_copy_image_section (extended syntax
    "[a:b,c:d]") read all NAXISn into a 9-element stack array before
    rejecting NAXIS > 4."""
    path = _write_many_axis_image(tmp_path / "axes60.fits", 60)
    out = _assert_read_raises(f"{path}[1:1,1:1]")
    assert "dimensions" in out


def test_ten_axis_image_reads_raise(tmp_path: Path) -> None:
    """getcol-ndim: fits_read_subset/fits_read_pix walked NAXIS > 9 past a
    naxes[9] array; torchfits also refuses such images before calling them."""
    path = _write_many_axis_image(tmp_path / "axes10.fits", 10)
    with pytest.raises(RuntimeError, match="at most 9 axes"):
        torchfits.read(str(path))
    with pytest.raises(RuntimeError):
        torchfits.read_subset(str(path), 0, 0, 0, 1, 1)


# --------------------------------------------------------------------------
# Tables


def test_ascii_table_with_too_many_fields_raises(tmp_path: Path) -> None:
    """getkey-tfields: ffgttb compared instead of assigning BAD_TFIELDS, so
    TFIELDS > 999 opened as a valid ASCII table."""
    fields = 1000
    cards = [
        _card("XTENSION", "TABLE"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", fields),
        _card("NAXIS2", 1),
        _card("PCOUNT", 0),
        _card("GCOUNT", 1),
        _card("TFIELDS", fields),
    ]
    for i in range(1, fields + 1):
        cards += [_card(f"TBCOL{i}", i), _card(f"TFORM{i}", "A1")]
    path = tmp_path / "tfields.fits"
    path.write_bytes(_PRIMARY_NULL + _header(cards) + _pad(b"x" * fields, b" "))
    with pytest.raises((RuntimeError, OSError)):
        torchfits.table.read(str(path), hdu=1)


@pytest.mark.parametrize("mmap", [True, False])
def test_ascii_string_column_ignores_narrower_tdisp(tmp_path: Path, mmap: bool) -> None:
    """getcols-vla-strings: upstream 113781d cuts ffgcls2 strings to the
    TDISPn width; the vendored patch keeps whole fields (a display hint must
    not truncate data)."""
    cards = [
        _card("XTENSION", "TABLE"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", 14),
        _card("NAXIS2", 2),
        _card("PCOUNT", 0),
        _card("GCOUNT", 1),
        _card("TFIELDS", 2),
        _card("TTYPE1", "NAME"),
        _card("TBCOL1", 1),
        _card("TFORM1", "A10"),
        _card("TDISP1", "A5"),
        _card("TTYPE2", "N"),
        _card("TBCOL2", 11),
        _card("TFORM2", "I4"),
    ]
    path = tmp_path / "tdisp.fits"
    rows = b"abcdefghij   1" + b"klmnopqrst   2"
    path.write_bytes(_PRIMARY_NULL + _header(cards) + _pad(rows, b" "))
    table = torchfits.table.read(str(path), hdu=1, mmap=mmap)
    assert table.column("NAME").to_pylist() == ["abcdefghij", "klmnopqrst"]
    tensors = torchfits.table.read_torch(str(path), hdu=1, mmap=mmap)
    assert bytes(tensors["NAME"][1].tolist()) == b"klmnopqrst"


def test_vla_string_descriptor_past_heap_raises(tmp_path: Path) -> None:
    """getcols-vla-strings: a 1PA string whose descriptor points past the
    heap. torchfits reads it as bytes through its own descriptor checks, not
    through the ffgcls2 path the patch hardens."""
    heap = b"abcdefgh"
    cards = [
        _card("XTENSION", "BINTABLE"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", 8),
        _card("NAXIS2", 2),
        _card("PCOUNT", len(heap)),
        _card("GCOUNT", 1),
        _card("TFIELDS", 1),
        _card("TTYPE1", "S"),
        _card("TFORM1", "1PA(8)"),
    ]
    rows = struct.pack(">ii", 3, 0) + struct.pack(">ii", 5, 4096)
    path = tmp_path / "vla-string.fits"
    path.write_bytes(_PRIMARY_NULL + _header(cards) + _pad(rows + heap))
    script = """
    import sys
    import torchfits
    for kwargs in ({}, {"mmap": False}):
        try:
            torchfits.table.read(sys.argv[1], hdu=1, **kwargs)
        except (RuntimeError, OSError, ValueError) as exc:
            print("RAISED", type(exc).__name__, exc)
        else:
            print("READ")
    """
    out = _run_child(script, str(path))
    assert out.count("RAISED") == 2, out


# --------------------------------------------------------------------------
# IRAF .imh/.pix (CFITSIO converts them when the name ends in ".imh" and
# goes through the extended filename syntax, e.g. "x.imh[0]")


def _write_iraf(
    directory: Path,
    name: str,
    shape: tuple[int, ...] = (3, 4),
    physical: tuple[int, ...] | None = None,
    pixoff: int = 513,
    user: bytes = b"",
) -> str:
    """An IRAF v2 image (big-endian TY_SHORT); returns the .imh file name."""
    physical = physical or shape
    header = bytearray(2046)
    header[0:5] = b"imhv2"
    struct.pack_into(">i", header, 10, 3)  # pixel type TY_SHORT
    struct.pack_into(">i", header, 14, 0)  # pixels not byte-swapped
    struct.pack_into(">i", header, 18, len(shape))
    for i, n in enumerate(shape):
        struct.pack_into(">i", header, 22 + 4 * i, n)
    for i, n in enumerate(physical):
        struct.pack_into(">i", header, 50 + 4 * i, n)
    struct.pack_into(">i", header, 86, pixoff)  # in 2-byte units, 1-based
    pix = f"{name}.pix".encode()
    header[126 : 126 + len(pix)] = pix
    (directory / f"{name}.imh").write_bytes(bytes(header) + user + b"\0")
    pixels = np.arange(int(np.prod(physical)), dtype=">i2").tobytes()
    (directory / f"{name}.pix").write_bytes(b"impv2".ljust(1024, b"\0") + pixels)
    return f"{name}.imh"


_EXPECT_IRAF_IMAGE = """
import sys
import torchfits
image = torchfits.read(sys.argv[1] + "[0]")
print(tuple(image.shape), image.flatten().tolist())
"""


def _assert_iraf_reads(directory: Path, imh: str) -> None:
    out = _run_child(_EXPECT_IRAF_IMAGE, imh, cwd=directory)
    assert out.strip() == f"(4, 3) {list(range(12))}", out


def test_iraf_image_reads_through_extended_syntax(tmp_path: Path) -> None:
    """iraffits-bounds: the converter is reachable and still works."""
    _assert_iraf_reads(tmp_path, _write_iraf(tmp_path, "plain"))


@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param({"pixoff": 1}, id="pixel-offset-inside-magic"),
        pytest.param({"shape": (2, 4, 0), "physical": (3, 4, 1)}, id="zero-axis"),
        pytest.param(
            {"shape": (65536, 32768), "physical": (65537, 2)}, id="size-overflow"
        ),
    ],
)
def test_iraf_bad_geometry_raises(tmp_path: Path, geometry: dict) -> None:
    """iraffits-bounds: pixel offset, zero axis and size overflow checks."""
    imh = _write_iraf(tmp_path, "bad", **geometry)
    _assert_read_raises(imh + "[0]", cwd=tmp_path)
