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
