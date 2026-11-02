"""where= results agree across every read backend and the reference evaluator.

Each case writes a small table, evaluates the predicate with
torchfits.where.evaluate_where on the decoded columns, and checks that every
table read path (auto, cpp, torch, mmap on/off, projection, scan, read_torch)
selects exactly the same rows.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import torch

import torchfits
from torchfits.where import evaluate_where, parse_where_expression


def _reference_ids(path: str, expr: str) -> list[int]:
    tab = torchfits.table.read(path, apply_fits_nulls=True)
    data = {}
    for name in tab.column_names:
        col = tab[name]
        arr = col.to_numpy(zero_copy_only=False)
        if col.null_count and arr.dtype.kind != "f":
            arr = np.array(col.to_pylist(), dtype=object)
        data[name] = arr
    mask = evaluate_where(parse_where_expression(expr), data)
    return sorted(np.asarray(tab["ID"])[mask].tolist())


def _backend_ids(path: str, expr: str) -> dict[str, list[int]]:
    def ids(t):
        if isinstance(t, dict):
            return sorted(t["ID"].tolist())
        return sorted(t["ID"].to_pylist())

    def scan(backend):
        batches = list(
            torchfits.table.scan(path, where=expr, batch_size=2, backend=backend)
        )
        return pa.Table.from_batches(batches) if batches else {"ID": torch.tensor([])}

    out = {
        "auto": ids(torchfits.table.read(path, where=expr)),
        "auto_nommap": ids(torchfits.table.read(path, where=expr, mmap=False)),
        "cpp": ids(torchfits.table.read(path, where=expr, backend="cpp")),
        "torch": ids(torchfits.table.read(path, where=expr, backend="torch")),
        "proj": ids(torchfits.table.read(path, where=expr, columns=["ID"])),
        "scan": ids(scan("auto")),
        "scan_torch": ids(scan("torch")),
    }
    try:
        out["read_torch"] = ids(torchfits.table.read_torch(path, where=expr))
    except ValueError as exc:
        # read_torch only takes AND-ed simple comparisons.
        if "Unsupported where expression for read_torch" not in str(exc):
            raise
    return out


def _assert_parity(path: str, expr: str, expected: list[int]) -> None:
    assert _reference_ids(path, expr) == expected
    got = _backend_ids(path, expr)
    assert got == {k: expected for k in got}, expr


@pytest.fixture
def null_table(tmp_path):
    path = str(tmp_path / "nulls.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(4, dtype=np.int32),
            "A": np.array([1, -999, -999, 5], dtype=np.int16),
            "B": np.array([0, 7, 0, 0], dtype=np.int16),
        },
        schema={"A": {"tnull": -999}},
        overwrite=True,
    )
    return path


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("A > 0 OR B == 7", [0, 1, 3]),
        ("B == 7 OR A > 0", [0, 1, 3]),
        ("A IS NULL OR A > 3", [1, 2, 3]),
        ("A > 3 OR A IS NULL", [1, 2, 3]),
        ("A > 3 AND A IS NULL", []),
    ],
)
def test_or_with_null_operand_uses_kleene_logic(null_table, expr, expected):
    _assert_parity(null_table, expr, expected)


@pytest.fixture
def wide_int_table(tmp_path):
    path = str(tmp_path / "wide.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(3, dtype=np.int32),
            "J": np.array([16777217, 16777216, 2147483647], dtype=np.int32),
            "K": np.array([2**53 + 1, 2**53, 5], dtype=np.int64),
        },
        overwrite=True,
    )
    return path


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("J > 16777216.5", [0, 2]),
        ("J == 16777216.0", [1]),
        ("J > 2147483646.5", [2]),
        ("J < 16777216.5", [1]),
        ("J != 16777216.5", [0, 1, 2]),
        ("K == 9007199254740992.0", [1]),
        ("K > 9007199254740992.0", [0]),
        ("K < 1e300", [0, 1, 2]),
        ("K > -1e300", [0, 1, 2]),
    ],
)
def test_integer_column_float_literal_is_exact(wide_int_table, expr, expected):
    # The reference evaluator compares in float64 and rounds 2**53 + 1, so
    # the exact answer is asserted directly here.
    got = _backend_ids(wide_int_table, expr)
    assert got == {k: expected for k in got}, expr


@pytest.fixture
def u64_table(tmp_path):
    path = str(tmp_path / "u64.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(3, dtype=np.int32),
            "U": np.array([1, 2**63, 2**64 - 1], dtype=np.uint64),
            "S": np.array(["aa", "bb", "cc"]),
        },
        overwrite=True,
    )
    return path


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("U >= 0", [0, 1, 2]),
        ("U == 1", [0]),
        ("U > 9223372036854775807", [1, 2]),
        ("U == 18446744073709551615", [2]),
        ("U < 18446744073709551616", [0, 1, 2]),
        ("U > -1", [0, 1, 2]),
        ("U >= 1.5", [1, 2]),
        ("U BETWEEN 0.5 AND 9.2233720368547758e18", [0, 1]),
    ],
)
def test_uint64_column_compares_exactly(u64_table, expr, expected):
    got = _backend_ids(u64_table, expr)
    assert got == {k: expected for k in got}, expr


def test_where_type_mismatch_raises_value_error(u64_table):
    for kwargs in ({}, {"backend": "cpp"}, {"backend": "torch"}):
        with pytest.raises(ValueError):
            torchfits.table.read(u64_table, where="S > 3", **kwargs)


def _card(path: str, hdu: int, key: str) -> str:
    raw = open(path, "rb").read()
    pos, current = 0, 0
    while True:
        cards = []
        while not cards or not cards[-1].startswith("END     "):
            cards.append(raw[pos : pos + 80].decode("ascii"))
            pos += 80
        pos = -(-pos // 2880) * 2880
        values = {c[:8].strip(): c[10:].split("/")[0].strip() for c in cards}
        if current == hdu:
            return values[key]
        size = abs(int(values["BITPIX"])) // 8 if int(values["NAXIS"]) else 0
        for a in range(1, int(values["NAXIS"]) + 1):
            size *= int(values[f"NAXIS{a}"])
        pos += -(-(size + int(values.get("PCOUNT", 0))) // 2880) * 2880
        current += 1


@pytest.fixture
def tnull_offset_table(tmp_path):
    """TNULL is the raw stored value, compared before TSCAL/TZERO."""
    path = str(tmp_path / "tnull_offset.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(3, dtype=np.int32),
            # Stored [7, 2, 0] with TZERO=5, TNULL=7 -> [NULL, 7.0, 5.0].
            "SC": np.array([7, 2, 0], dtype=np.int16),
            # uint16 values; TNULL given in the value domain like the data.
            "U16": np.array([65535, 0, 32769], dtype=np.uint16),
            "I8": np.array([-128, 0, 127], dtype=np.int8),
        },
        schema={
            "SC": {"format": "I", "bzero": 5.0, "tnull": 7},
            "U16": {"tnull": 65535},
            "I8": {"tnull": -128},
        },
        overwrite=True,
    )
    return path


def test_unsigned_tnull_is_written_as_the_stored_value(tnull_offset_table):
    assert int(_card(tnull_offset_table, 1, "TNULL2")) == 7
    assert int(_card(tnull_offset_table, 1, "TNULL3")) == 65535 - 32768
    assert int(_card(tnull_offset_table, 1, "TNULL4")) == 0


def test_tnull_is_compared_before_scaling(tnull_offset_table):
    for kwargs in (
        {},
        {"mmap": False},
        {"backend": "cpp"},
        {"backend": "torch"},
    ):
        tab = torchfits.table.read(tnull_offset_table, **kwargs)
        assert tab["SC"].to_pylist() == [None, 7.0, 5.0], kwargs
        assert tab["U16"].to_pylist() == [None, 0, 32769], kwargs
        assert tab["I8"].to_pylist() == [None, 0, 127], kwargs


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("SC == 7", [1]),
        ("SC > 6", [1]),
        ("SC IS NULL", [0]),
        ("SC IS NOT NULL", [1, 2]),
        ("U16 == 0", [1]),
        ("U16 > 1", [2]),
        ("U16 IS NULL", [0]),
        ("I8 >= 0", [1, 2]),
        ("I8 IS NULL", [0]),
    ],
)
def test_tnull_where_parity(tnull_offset_table, expr, expected):
    _assert_parity(tnull_offset_table, expr, expected)


def test_tnull_to_nan_from_header_uses_decoded_values(tnull_offset_table):
    from torchfits.transforms import TNullToNan

    header = torchfits.read_header(tnull_offset_table, hdu=1)
    data = torchfits.table.read_torch(tnull_offset_table, hdu=1)
    out = TNullToNan.from_header(header)(
        {"U16": data["U16"].to(torch.float64), "I8": data["I8"].to(torch.float64)}
    )
    assert torch.isnan(out["U16"]).tolist() == [True, False, False]
    assert torch.isnan(out["I8"]).tolist() == [True, False, False]


def test_append_rows_fills_unsigned_tnull(tnull_offset_table):
    torchfits.table.append_rows(
        tnull_offset_table, {"ID": np.array([3], dtype=np.int32)}
    )
    tab = torchfits.table.read(tnull_offset_table)
    assert tab["U16"].to_pylist()[-1] is None
    assert tab["I8"].to_pylist()[-1] is None


def test_nan_is_null_and_numeric_null_compare_is_rejected(tmp_path):
    path = str(tmp_path / "nan.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(3, dtype=np.int32),
            "F": np.array([1.0, np.nan, 3.0]),
        },
        overwrite=True,
    )
    _assert_parity(path, "F IS NULL", [1])
    _assert_parity(path, "F IS NOT NULL", [0, 2])
    for expr in ("F == NULL", "F != NULL"):
        with pytest.raises(ValueError, match="IS NULL"):
            torchfits.table.read(path, where=expr)


@pytest.fixture
def padded_string_table(tmp_path):
    path = str(tmp_path / "strs.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(3, dtype=np.int32),
            "S": np.array([b"ab  ", b"cafe", b"a   "], dtype="S4"),
        },
        overwrite=True,
    )
    raw = open(path, "rb").read()
    # A latin-1 cell: table.write only takes ASCII, so patch the stored byte.
    with open(path, "wb") as fh:
        fh.write(raw.replace(b"cafe", b"caf\xe9", 1))
    return path


@pytest.mark.parametrize(
    ("expr", "kwargs", "expected"),
    [
        ("S == 'ab'", {"strip": False}, []),
        ("S == 'ab  '", {"strip": False}, [0]),
        ("S == 'ab'", {}, [0]),
        ("S == 'café'", {"encoding": "latin-1"}, [1]),
        ("S == 'caf'", {"encoding": "latin-1"}, []),
    ],
)
def test_string_predicate_uses_callers_decoding(
    padded_string_table, expr, kwargs, expected
):
    for backend in ("auto", "cpp", "torch"):
        for mmap in (True, False):
            got = torchfits.table.read(
                padded_string_table,
                where=expr,
                backend=backend,
                mmap=mmap,
                columns=["ID", "S"],
                **kwargs,
            )
            assert got["ID"].to_pylist() == expected, (backend, mmap)


def test_vector_column_predicate_is_refused_on_every_backend(tmp_path):
    path = str(tmp_path / "vec.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(4, dtype=np.int32),
            "V": np.arange(12, dtype=np.float32).reshape(4, 3),
        },
        overwrite=True,
    )
    for columns in (None, ["V"], ["ID", "V"], ["ID"]):
        for backend in ("auto", "cpp", "torch"):
            for mmap in (True, False):
                with pytest.raises(ValueError, match="vector column 'V'"):
                    torchfits.table.read(
                        path,
                        where="V > 4",
                        columns=columns,
                        backend=backend,
                        mmap=mmap,
                    )
        with pytest.raises(ValueError, match="vector column 'V'"):
            torchfits.table.read_torch(path, where="V > 4", columns=columns)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("S == 'a\\nb'", ("cmp", "S", "==", "a\\nb")),
        ("S == 'a\\db'", ("cmp", "S", "==", "a\\db")),
        ("S == 'a\\'", ("cmp", "S", "==", "a\\")),
        ("S == 'O''NEIL'", ("cmp", "S", "==", "O'NEIL")),
        ('S == "say ""hi"""', ("cmp", "S", "==", 'say "hi"')),
        ("X BETWEEN .5 AND 2", ("between", "X", 0.5, 2, False)),
        ("X NOT BETWEEN -.5 AND +.5", ("between", "X", -0.5, 0.5, True)),
        ("S BETWEEN 'a' AND 'c'", ("between", "S", "a", "c", False)),
        ("S == 'x BETWEEN 1 AND 2'", ("cmp", "S", "==", "x BETWEEN 1 AND 2")),
    ],
)
def test_where_literals_are_taken_literally(expr, expected):
    import warnings

    from torchfits._where import _parse_where_expression

    _parse_where_expression.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no SyntaxWarning for '\d'
        assert parse_where_expression(expr) == expected


def test_string_between_and_backslash_literal_select_rows(tmp_path):
    path = str(tmp_path / "s.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(4, dtype=np.int32),
            "S": np.array([b"a", b"b\\n", b"c", b"d"], dtype="S3"),
        },
        overwrite=True,
    )
    for backend in ("auto", "cpp", "torch"):
        got = torchfits.table.read(path, where="S BETWEEN 'b' AND 'c'", backend=backend)
        assert got["ID"].to_pylist() == [1, 2]
        got = torchfits.table.read(path, where="S == 'b\\n'", backend=backend)
        assert got["ID"].to_pylist() == [1]


def test_read_torch_coerces_logical_and_numeric_words(tmp_path):
    path = str(tmp_path / "lw.fits")
    torchfits.table.write(
        path,
        {
            "ID": np.arange(3, dtype=np.int32),
            "L": np.array([True, False, True]),
            "E": np.array([1.0, np.inf, 2.0], dtype=np.float32),
            "D": np.array([1.0, np.inf, 2.0]),
        },
        overwrite=True,
    )
    for expr, expected in [
        ("L == T", [0, 2]),
        ("L == FALSE", [1]),
        ("E < inf", [0, 2]),
        ("D < inf", [0, 2]),
    ]:
        assert torchfits.table.read_torch(path, where=expr)["ID"].tolist() == expected
        assert torchfits.table.read(path, where=expr)["ID"].to_pylist() == expected
    with pytest.raises(ValueError):
        torchfits.table.read_torch(path, where="ID > abc")
    with pytest.raises(ValueError):
        torchfits.table.read_torch(path, where="E > abc")


def test_repeated_projection_column_is_returned_once(tmp_path):
    path = str(tmp_path / "dup.fits")
    torchfits.table.write(path, {"ID": np.arange(4, dtype=np.int32)}, overwrite=True)
    for where in (None, "ID > 1", "ID > 50"):
        for backend in ("auto", "cpp", "torch"):
            for mmap in (True, False):
                got = torchfits.table.read(
                    path, columns=["ID", "ID"], where=where, backend=backend, mmap=mmap
                )
                assert got.column_names == ["ID"], (where, backend, mmap)
