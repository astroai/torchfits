"""Fixed-width column gathers against an independent big-endian reference.

The buffered (tiled pread) and mmap table readers copy every fixed-width
column out of whole big-endian rows with width-specialized loops: a strided
element loop, a strided cell loop for vector columns, and a dense block copy
when the rows hold nothing but the column. Their outputs share one
allocation per read. This builds tables byte by byte with NumPy, so the
expected values never pass through torchfits, and checks every shape of
column through both readers, for the full table, a window that starts and
ends inside a tile, and projections.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits._C as cpp

N_ROWS = 150_000  # several 512 KiB pread tiles and mmap row blocks

# (name, TFORM, big-endian NumPy storage dtype, cell shape)
COLUMNS = [
    ("i2", "I", ">i2", ()),
    ("i4", "J", ">i4", ()),
    ("i8", "K", ">i8", ()),
    ("f4", "E", ">f4", ()),
    ("f8", "D", ">f8", ()),
    ("u1", "B", "u1", ()),
    ("lg", "L", "S1", ()),
    ("s5", "5A", "S5", ()),
    ("v3i2", "3I", ">i2", (3,)),
    ("v2i4", "2J", ">i4", (2,)),
    ("v2f8", "2D", ">f8", (2,)),
    ("v3u1", "3B", "u1", (3,)),
    ("v2lg", "2L", "S1", (2,)),
]


def _card(key: str, value: str) -> str:
    return f"{key:<8}= {value:>20}".ljust(80)


def _block(cards: list[str]) -> bytes:
    text = "".join(cards) + "END".ljust(80)
    return (text + " " * ((-len(text)) % 2880)).encode()


def _values(rng: np.random.Generator, dtype: str, shape: tuple, n: int) -> np.ndarray:
    full = (n, *shape)
    if dtype == "S1":
        return rng.choice(np.array([b"T", b"F"]), size=full)
    if dtype == "S5":
        return rng.choice(np.array([b"abcde", b"x", b"", b"FITS!"]), size=full)
    if dtype.endswith(("f4", "f8")):
        return rng.standard_normal(full).astype(dtype)
    info = np.iinfo(np.dtype(dtype))
    return rng.integers(info.min, info.max, size=full, endpoint=True).astype(dtype)


def _write_table(path, columns, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    record = np.dtype([(name, dt, shape) for name, _, dt, shape in columns])
    rows = np.zeros(n, dtype=record)
    for name, _, dt, shape in columns:
        rows[name] = _values(rng, dt, shape, n)
    cards = [
        _card("XTENSION", "'BINTABLE'"),
        _card("BITPIX", "8"),
        _card("NAXIS", "2"),
        _card("NAXIS1", str(record.itemsize)),
        _card("NAXIS2", str(n)),
        _card("PCOUNT", "0"),
        _card("GCOUNT", "1"),
        _card("TFIELDS", str(len(columns))),
    ]
    for i, (name, tform, _, _) in enumerate(columns, start=1):
        cards += [_card(f"TTYPE{i}", f"'{name}'"), _card(f"TFORM{i}", f"'{tform}'")]
    primary = _block(
        [
            _card("SIMPLE", "T"),
            _card("BITPIX", "8"),
            _card("NAXIS", "0"),
            _card("EXTEND", "T"),
        ]
    )
    data = rows.tobytes()
    with open(path, "wb") as fh:
        fh.write(primary + _block(cards) + data + b"\0" * ((-len(data)) % 2880))
    return rows


def _expected(rows: np.ndarray, name: str, dtype: str) -> np.ndarray:
    values = rows[name]
    if dtype == "S1":
        return values == b"T"
    if dtype == "S5":
        # Raw cell bytes (NUL padded), one uint8 row per cell.
        return np.frombuffer(values.astype("S5").tobytes(), dtype=np.uint8).reshape(
            -1, 5
        )
    return values.astype(np.dtype(dtype).newbyteorder("="))


def _check(result: dict, rows: np.ndarray, columns, lo: int, hi: int) -> None:
    assert list(result) == [c[0] for c in columns]
    for name, _, dtype, _ in columns:
        got = np.asarray(result[name])
        np.testing.assert_array_equal(
            got, _expected(rows[lo:hi], name, dtype), err_msg=name
        )


@pytest.fixture(scope="module")
def wide_table(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("gather") / "wide.fits")
    return path, _write_table(path, COLUMNS, N_ROWS, seed=1)


@pytest.mark.parametrize("mmap", [False, True], ids=["buffered", "mmap"])
@pytest.mark.parametrize(
    "window", [(1, -1), (12_345, 101_011), (N_ROWS, 1)], ids=["full", "inner", "last"]
)
def test_every_column_shape_matches_the_reference(wide_table, mmap, window) -> None:
    path, rows = wide_table
    start, count = window
    hi = N_ROWS if count < 0 else start - 1 + count
    result = cpp.read_fits_table_rows(path, 1, [], start, count, mmap)
    _check(result, rows, COLUMNS, start - 1, hi)


@pytest.mark.parametrize("mmap", [False, True], ids=["buffered", "mmap"])
def test_projections_match_the_reference(wide_table, mmap) -> None:
    path, rows = wide_table
    for picked in (COLUMNS[5:7], COLUMNS[:1], COLUMNS[::-3]):
        names = [c[0] for c in picked]
        result = cpp.read_fits_table_rows(path, 1, names, 1, -1, mmap)
        _check(result, rows, picked, 0, N_ROWS)


@pytest.mark.parametrize("mmap", [False, True], ids=["buffered", "mmap"])
@pytest.mark.parametrize("column", COLUMNS, ids=[c[0] for c in COLUMNS])
def test_single_column_tables_take_the_dense_copy(tmp_path, mmap, column) -> None:
    # A row that is one cell: the gather degenerates to one dense block copy
    # (byte-swapped in bulk), split across tiles and threads.
    path = str(tmp_path / "one.fits")
    rows = _write_table(path, [column], N_ROWS, seed=2)
    result = cpp.read_fits_table_rows(path, 1, [], 1, -1, mmap)
    _check(result, rows, [column], 0, N_ROWS)
    windowed = cpp.read_fits_table_rows(path, 1, [], 777, 50_001, mmap)
    _check(windowed, rows, [column], 776, 776 + 50_001)


def test_columns_of_one_read_outlive_each_other(wide_table) -> None:
    # The outputs of one read share an allocation; each column must stay valid
    # (and independent) after the others are dropped or modified.
    path, rows = wide_table
    result = cpp.read_fits_table_rows(path, 1, ["i4", "f8", "u1"], 1, -1, False)
    keep = result["f8"]
    result["i4"].fill_(0)
    del result
    np.testing.assert_array_equal(np.asarray(keep), _expected(rows, "f8", ">f8"))
