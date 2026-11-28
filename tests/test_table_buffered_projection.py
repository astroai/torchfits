"""Buffered (mmap=False) table reads: single/few-column projections and windows.

Every fixed-width projection now goes through the row-buffered reader (a tiled
pread + per-column gather), single columns included; before, a single column
used CFITSIO's ``fits_read_col``. The values must not depend on that choice:
each read here is compared with the same read made column by column through
``fits_read_col`` (``TORCHFITS_TABLE_BUFFERED=0`` in a subprocess), across
TSCAL/TZERO, TNULL, unsigned, signed-byte, logical, string and vector columns,
and across tile boundaries.
"""

from __future__ import annotations

import gzip
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

import torchfits._C as cpp
import torchfits.table as table

N_ROWS = 30_000  # 56-byte rows: ~1.7 MB, several 512 KiB tiles

# name, TFORM, extra header cards, big-endian numpy dtype (per cell)
COLUMNS = [
    ("A", "E", {}, ">f4"),
    ("B", "J", {"TNULL": -999}, ">i4"),
    ("C", "I", {"TSCAL": 0.5, "TZERO": 10.0}, ">i2"),
    ("D", "I", {"TZERO": 32768}, ">i2"),
    ("E", "L", {}, "S1"),
    ("F", "6A", {}, "S6"),
    ("G", "3D", {}, (">f8", (3,))),
    ("H", "B", {"TZERO": -128}, "u1"),
    ("K", "K", {"TNULL": -1}, ">i8"),
    ("S", "J", {"TSCAL": 2.5, "TNULL": 7}, ">i4"),
]


def _card(key: str, value) -> bytes:
    if isinstance(value, str):
        text = f"{key:<8}= '{value:<8}'"
    elif isinstance(value, bool):
        text = f"{key:<8}= {'T' if value else 'F':>20}"
    else:
        text = f"{key:<8}= {value!r:>20}"
    return text.ljust(80).encode("ascii")


def _header(cards: list[tuple[str, object]]) -> bytes:
    raw = b"".join(_card(k, v) for k, v in cards) + b"END".ljust(80)
    return raw + b" " * (-len(raw) % 2880)


def _write_table(path: str) -> None:
    rng = np.random.default_rng(5)
    dtype = np.dtype([(name, dt) for name, _tf, _cards, dt in COLUMNS])
    rows = np.zeros(N_ROWS, dtype=dtype)
    rows["A"] = rng.standard_normal(N_ROWS)
    b = rng.integers(-1000, 1000, N_ROWS)
    b[::17] = -999
    rows["B"] = b
    rows["C"] = rng.integers(-30000, 30000, N_ROWS)
    rows["D"] = rng.integers(-32768, 32767, N_ROWS)
    rows["E"] = np.array([b"T", b"F", b"\x00"])[rng.integers(0, 3, N_ROWS)]
    rows["F"] = np.array([b"abc", b"", b"xyzxyz", b"q r"])[rng.integers(0, 4, N_ROWS)]
    rows["G"] = rng.standard_normal((N_ROWS, 3))
    rows["H"] = rng.integers(0, 256, N_ROWS)
    k = rng.integers(-(2**62), 2**62, N_ROWS)
    k[::23] = -1
    rows["K"] = k
    s = rng.integers(-100, 100, N_ROWS)
    s[::11] = 7
    rows["S"] = s
    data = rows.tobytes()
    width = dtype.itemsize
    assert width == 56

    cards: list[tuple[str, object]] = [
        ("XTENSION", "BINTABLE"),
        ("BITPIX", 8),
        ("NAXIS", 2),
        ("NAXIS1", width),
        ("NAXIS2", N_ROWS),
        ("PCOUNT", 0),
        ("GCOUNT", 1),
        ("TFIELDS", len(COLUMNS)),
    ]
    for i, (name, tform, extra, _dt) in enumerate(COLUMNS, start=1):
        cards.append((f"TTYPE{i}", name))
        cards.append((f"TFORM{i}", tform))
        for key, value in extra.items():
            cards.append((f"{key}{i}", value))
    primary = _header([("SIMPLE", True), ("BITPIX", 8), ("NAXIS", 0), ("EXTEND", True)])
    with open(path, "wb") as fh:
        fh.write(primary)
        fh.write(_header(cards))
        fh.write(data)
        fh.write(b"\x00" * (-len(data) % 2880))


# (columns, start_row, num_rows): full reads, tiny windows, windows that
# straddle a tile boundary (9362 rows per tile here), and the last rows.
READS = (
    [([name], 1, -1) for name, *_ in COLUMNS]
    + [([name], 9_300, 200) for name, *_ in COLUMNS]
    + [
        (["A"], 1, 1),
        (["G"], 5, 10),
        (["A", "K"], 1, -1),
        (["S", "C", "F"], 18_000, 9_500),
        ([], 1, -1),
        ([], N_ROWS - 3, 4),
        (["E", "H"], N_ROWS, 1),
    ]
)

_REFERENCE_SCRIPT = """
import pickle, sys
import torchfits._C as cpp
path, reads = sys.argv[1], pickle.loads(bytes.fromhex(sys.argv[2]))
out = []
for cols, start, n in reads:
    got = cpp.read_fits_table_rows(path, 1, cols, start, n, False)
    out.append({k: v.numpy() for k, v in got.items()})
sys.stdout.buffer.write(pickle.dumps(out))
"""


def _column_by_column(path: str) -> list[dict[str, np.ndarray]]:
    """The same reads through ``fits_read_col`` (buffered reader disabled)."""
    env = dict(os.environ, TORCHFITS_TABLE_BUFFERED="0", KMP_DUPLICATE_LIB_OK="TRUE")
    proc = subprocess.run(
        [sys.executable, "-c", _REFERENCE_SCRIPT, path, pickle.dumps(READS).hex()],
        env=env,
        check=True,
        capture_output=True,
    )
    return pickle.loads(proc.stdout)


def _assert_same(got: dict[str, torch.Tensor], want: dict[str, np.ndarray]) -> None:
    assert list(got) == list(want)
    for name, ref in want.items():
        value = got[name].numpy()
        assert value.dtype == ref.dtype, name
        assert value.shape == ref.shape, name
        np.testing.assert_array_equal(value, ref, err_msg=name)


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("buffered") / "t.fits")
    _write_table(path)
    return path


@pytest.fixture(scope="module")
def reference(table_path):
    return _column_by_column(table_path)


@pytest.mark.parametrize("case", range(len(READS)))
def test_buffered_projection_matches_fits_read_col(table_path, reference, case):
    cols, start, n = READS[case]
    got = cpp.read_fits_table_rows(table_path, 1, cols, start, n, False)
    _assert_same(got, reference[case])


def test_buffered_projection_matches_through_cfitsio_buffers(
    table_path, reference, tmp_path
):
    # A gzip file has no raw fd for pread: the buffered reader falls back to
    # fits_read_tblbytes chunks, and must still agree value for value.
    gz = str(tmp_path / "t.fits.gz")
    with open(table_path, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    for case, (cols, start, n) in enumerate(READS):
        got = cpp.read_fits_table_rows(gz, 1, cols, start, n, False)
        _assert_same(got, reference[case])


def test_buffered_single_column_matches_mmap_through_table_read(table_path):
    for name in ("A", "B", "C", "D", "E", "F", "G", "H", "K", "S"):
        buffered = table.read(table_path, columns=[name], mmap=False)
        full = table.read(table_path, mmap=False).select([name])
        assert buffered.equals(full), name
