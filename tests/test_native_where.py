"""Native, torch-free, windowed ``where=`` filtering.

``table.read`` / ``table.scan`` / ``table.scan_torch`` push a conjunction of
numeric comparisons down to the native filtered reader
(``read_fits_table_filtered_raw`` / ``read_fits_table_filtered``), which
evaluates it over the requested row window only and gathers just the selected
rows. Everything else stays on the Arrow evaluator. These tests pin the two
engines to the same rows and the same representation, and pin which engine
serves which request.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

import torchfits._C as C
from torchfits import table
from torchfits._table._read_where import _filter_table_with_where
from torchfits._where import parse_where_expression, where_columns_from_ast

N = 1000
# Every column but SC, which is scaled (TSCAL/TZERO) and so not gathered natively.
COLS = ["I16", "J", "K", "B", "SB", "U16", "U32", "U64", "E", "D", "S", "L", "V"]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> str:
    rng = np.random.default_rng(7)
    e = rng.uniform(-1, 1, N).astype(np.float32)
    e[::17] = np.nan
    d = rng.uniform(-2, 2, N)
    d[::23] = np.nan
    i16 = rng.integers(-50, 51, N).astype(np.int16)
    cols = {
        "I16": i16,
        "J": rng.integers(-(2**31), 2**31 - 1, N, dtype=np.int64).astype(np.int32),
        "K": rng.integers(-(2**62), 2**62, N, dtype=np.int64),
        "B": rng.integers(0, 256, N).astype(np.uint8),
        "SB": rng.integers(-128, 128, N).astype(np.int8),
        "U16": rng.integers(0, 2**16, N).astype(np.uint16),
        "U32": rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32),
        "U64": rng.integers(0, 2**63, N, dtype=np.uint64) * np.uint64(2),
        "E": e,
        "D": d,
        "S": np.array(["ab", "cde", "", "fghi"] * (N // 4)),
        "L": np.array([True, False, True, True] * (N // 4)),
        "V": np.arange(3 * N, dtype=np.int32).reshape(N, 3),
        "SC": rng.integers(0, 100, N).astype(np.int16),
    }
    path = tmp_path_factory.mktemp("native-where") / "cat.fits"
    table.write(
        str(path),
        cols,
        schema={
            "I16": {"tnull": -50},
            "J": {"tnull": int(cols["J"][5])},
            "SC": {"format": "I", "bscale": 0.5, "bzero": 10.0},
        },
        overwrite=True,
    )
    return str(path)


def _norm(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


def _assert_same(got: pa.Table, want: pa.Table) -> None:
    assert got.schema == want.schema
    assert got.num_rows == want.num_rows
    for name in want.column_names:
        assert _norm(got[name].to_pylist()) == _norm(want[name].to_pylist()), name


def _reference(
    path: str,
    where: str,
    *,
    columns: list[str] | None = None,
    row_slice: Any = None,
    **kwargs: Any,
) -> pa.Table:
    """The Arrow evaluator over an unfiltered read of the same window."""
    pred = where_columns_from_ast(parse_where_expression(where))
    read_cols = None if columns is None else list(dict.fromkeys(columns + pred))
    base = table.read(path, columns=read_cols, row_slice=row_slice, **kwargs)
    out = _filter_table_with_where(pa, base, where)
    return out if columns is None else out.select(columns)


@pytest.fixture()
def native_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []
    for name in ("read_fits_table_filtered_raw", "read_fits_table_filtered"):
        real = getattr(C, name)

        def spy(*args: Any, _real: Any = real, _name: str = name) -> Any:
            calls.append((_name, *args[4:]))
            return _real(*args)

        monkeypatch.setattr(C, name, spy)
    return calls


NATIVE_WHERE = [
    "I16 > 3",
    "I16 <= -5",
    "I16 != 0",
    "J > 0",
    "J != 7",
    "J > 2.5",
    "J < -1000.5",
    "K >= 1099511627776",
    "B == 200",
    "B > 100.5",
    "SB < 0",
    "SB >= -3",
    "U16 > 40000",
    "U16 >= 40000.5",
    "U32 < 3000000000",
    "U64 >= 9223372036854775808",
    "U64 < 4611686018427387904",
    "E < 0.5",
    "E < 0.2",
    "D >= -0.3 AND E <= 0.1",
    "E != 0.25",
    "E > -1e30",
    "D BETWEEN -1 AND 1",
    "D != 0",
    "I16 > 3 AND E < 0.5 AND U16 < 60000",
    "I16 > 1000",
]


@pytest.mark.parametrize("where", NATIVE_WHERE)
@pytest.mark.parametrize("row_slice", [None, (100, 637)])
def test_native_read_matches_evaluator(
    catalog: str, where: str, row_slice: Any, native_calls: list
) -> None:
    got = table.read(catalog, columns=COLS, where=where, row_slice=row_slice)
    assert [c[0] for c in native_calls] == ["read_fits_table_filtered_raw"]
    _assert_same(got, _reference(catalog, where, columns=COLS, row_slice=row_slice))


@pytest.mark.parametrize("where", ["I16 > 3", "J != 7", "J > 0 AND E < 0"])
@pytest.mark.parametrize("apply_fits_nulls", [True, False])
def test_native_read_tnull_and_projection(
    catalog: str, where: str, apply_fits_nulls: bool, native_calls: list
) -> None:
    """TNULL never matches with nulls applied, and is a plain value without.

    The projection leaves out the predicate columns, which the native scan
    still filters on (nothing is read just to be dropped)."""
    kwargs = {"apply_fits_nulls": apply_fits_nulls}
    for columns in (["S", "L", "V", "U64"], COLS):
        got = table.read(
            catalog, columns=columns, where=where, row_slice=(5, 900), **kwargs
        )
        want = _reference(catalog, where, columns=columns, row_slice=(5, 900), **kwargs)
        _assert_same(got, want)
    assert len(native_calls) == 2
    if where == "I16 > 3":
        # The I16 TNULL (-50) never satisfies > 3; check the != shape instead.
        nulls = table.read(catalog, columns=["I16"], where="I16 != 1", **kwargs)
        assert (nulls["I16"].null_count == 0) and (
            (-50 in nulls["I16"].to_pylist()) is (not apply_fits_nulls)
        )


def test_native_read_metadata_and_decode_options(
    catalog: str, native_calls: list
) -> None:
    for kwargs in (
        {"include_fits_metadata": True},
        {"decode_bytes": False},
        {"strip": False},
    ):
        got = table.read(catalog, columns=COLS, where="E < 0", **kwargs)
        _assert_same(got, _reference(catalog, "E < 0", columns=COLS, **kwargs))
    assert len(native_calls) == 3


def test_native_read_empty_result_keeps_schema(
    catalog: str, native_calls: list
) -> None:
    for row_slice in (None, (10, 20), (N, N + 5)):
        got = table.read(catalog, columns=COLS, where="I16 > 1000", row_slice=row_slice)
        want = _reference(catalog, "I16 > 1000", columns=COLS, row_slice=row_slice)
        assert got.num_rows == 0
        _assert_same(got, want)


@pytest.mark.parametrize(
    "where",
    [
        "S == 'ab'",  # string predicate: evaluator (decode semantics)
        "L == TRUE",  # logical predicate
        "I16 > 3 OR J < 0",  # not a conjunction
        "I16 IN (1, 2, 3)",
        "SC > 20",  # scaled predicate column
        "I16 IS NULL",
        "E < inf",  # non-finite literal
        "J == 2.5 AND I16 > 0",  # constant comparison
    ],
)
def test_evaluator_predicates_stay_off_the_native_scan(
    catalog: str, where: str, native_calls: list
) -> None:
    got = table.read(catalog, columns=COLS, where=where, row_slice=(3, 800))
    assert native_calls == []
    _assert_same(got, _reference(catalog, where, columns=COLS, row_slice=(3, 800)))


def test_scaled_output_column_falls_back(catalog: str, native_calls: list) -> None:
    got = table.read(catalog, where="I16 > 3")
    assert native_calls == []  # SC (scaled) is in the projection
    _assert_same(got, _reference(catalog, "I16 > 3"))
    got = table.read(catalog, columns=["I16", "E"], where="I16 > 3")
    assert len(native_calls) == 1
    _assert_same(got, _reference(catalog, "I16 > 3", columns=["I16", "E"]))


def test_mmap_false_keeps_buffered_evaluator(catalog: str, native_calls: list) -> None:
    got = table.read(catalog, columns=["I16"], where="I16 > 3", mmap=False)
    assert native_calls == []
    _assert_same(got, _reference(catalog, "I16 > 3", columns=["I16"]))


@pytest.mark.parametrize("batch_size", [37, 256, 5000])
@pytest.mark.parametrize("row_slice", [None, (90, 731)])
def test_native_scan_reads_each_window_once(
    catalog: str, batch_size: int, row_slice: Any, native_calls: list
) -> None:
    where = "E < 0.2 AND I16 >= -20"
    cols = ["I16", "E", "S", "U32"]
    batches = list(
        table.scan(
            catalog,
            columns=cols,
            where=where,
            row_slice=row_slice,
            batch_size=batch_size,
        )
    )
    start, stop = row_slice or (0, N)
    windows = [
        (row + 1, min(batch_size, stop - row)) for row in range(start, stop, batch_size)
    ]
    assert [c[1:] for c in native_calls] == windows
    got = pa.Table.from_batches(batches)
    _assert_same(got, _reference(catalog, where, columns=cols, row_slice=row_slice))


def test_native_scan_without_matches_yields_typed_empty_batch(
    catalog: str, native_calls: list
) -> None:
    batches = list(table.scan(catalog, columns=["I16"], where="I16 > 999"))
    assert len(batches) == 1 and batches[0].num_rows == 0
    assert batches[0].schema.field("I16").type == pa.int16()


def test_scan_falls_back_for_evaluator_predicates(
    catalog: str, native_calls: list
) -> None:
    got = pa.Table.from_batches(
        list(table.scan(catalog, columns=["I16"], where="S == 'ab'", batch_size=99))
    )
    assert native_calls == []
    _assert_same(got, _reference(catalog, "S == 'ab'", columns=["I16"]))


def _torch_reference(path: str, where: str, columns: list[str], row_slice: Any) -> dict:
    full = {}
    for chunk in table.scan_torch(path, columns=columns, row_slice=row_slice):
        for k, v in chunk.items():
            full.setdefault(k, []).append(v)
    full = {k: torch.cat(v) for k, v in full.items()}
    pred = where_columns_from_ast(parse_where_expression(where))
    mask_tbl = table.read(path, columns=pred, row_slice=row_slice)
    from torchfits._table._read_where import _where_mask_for_table

    mask = torch.from_numpy(
        np.asarray(
            _where_mask_for_table(mask_tbl, where).to_numpy(zero_copy_only=False)
        )
    )
    return {k: v[mask] for k, v in full.items()}


@pytest.mark.parametrize(
    "where,native", [("U16 > 30000 AND E < 0.5", True), ("S == 'cde'", False)]
)
@pytest.mark.parametrize("row_slice", [None, (33, 777)])
def test_scan_torch_where_matches_unfiltered_rows(
    catalog: str, where: str, native: bool, row_slice: Any, native_calls: list
) -> None:
    cols = ["I16", "U16", "U64", "SB", "S", "V", "L"]
    chunks = list(
        table.scan_torch(
            catalog, columns=cols, row_slice=row_slice, batch_size=101, where=where
        )
    )
    assert bool(native_calls) is native
    assert all(c[0] == "read_fits_table_filtered" for c in native_calls)
    assert all(next(iter(c.values())).shape[0] > 0 for c in chunks)
    want = _torch_reference(catalog, where, cols, row_slice)
    got = {k: torch.cat([c[k] for c in chunks]) for k in want}
    assert list(chunks[0].keys()) == cols
    for k in want:
        assert got[k].dtype == want[k].dtype, k
        assert torch.equal(got[k], want[k]), k


def test_scan_torch_where_validates_eagerly(catalog: str) -> None:
    with pytest.raises(ValueError):
        table.scan_torch(catalog, where="I16 >")


def test_native_binding_window_arguments(catalog: str) -> None:
    full = C.read_fits_table_filtered(catalog, 1, ["I16"], [("I16", ">", 0)])
    for start, num in ((1, 10), (500, 250), (990, 100), (N + 1, 5), (7, 0)):
        out = C.read_fits_table_filtered(
            catalog, 1, ["I16"], [("I16", ">", 0)], start, num
        )
        window = table.read(
            catalog,
            columns=["I16"],
            row_slice=(start - 1, start - 1 + num),
            apply_fits_nulls=False,
        )
        vals = [v for v in window["I16"].to_pylist() if v > 0]
        assert out["I16"].tolist() == vals
    assert full["I16"].shape[0] == sum(
        1
        for v in table.read(catalog, columns=["I16"], apply_fits_nulls=False)[
            "I16"
        ].to_pylist()
        if v > 0
    )
    with pytest.raises(RuntimeError):
        C.read_fits_table_filtered(catalog, 1, ["I16"], [("I16", ">", 0)], 0, 5)


def test_native_binding_refuses_vla_output(tmp_path: Path) -> None:
    path = str(tmp_path / "vla.fits")
    table.write(
        path,
        {
            "A": np.arange(3, dtype=np.int32),
            "V": [np.arange(k, dtype=np.int32) for k in range(3)],
        },
    )
    with pytest.raises(RuntimeError, match="VLA"):
        C.read_fits_table_filtered(path, 1, ["A", "V"], [("A", ">", 0)])
    # The public read still serves it, through the evaluator.
    assert table.read(path, where="A > 0")["A"].to_pylist() == [1, 2]


def test_native_where_read_is_torch_free(catalog: str) -> None:
    """An Arrow where= read takes the native path without importing torch."""
    script = f"""
import sys
import torchfits._C as C
calls = []
real = C.read_fits_table_filtered_raw
def spy(*a):
    calls.append(a[4:])
    return real(*a)
C.read_fits_table_filtered_raw = spy
import torchfits.table as T
t = T.read({catalog!r}, columns=['I16', 'S'], where='I16 > 3 AND E < 0.5', row_slice=(10, 500))
n = sum(b.num_rows for b in T.scan({catalog!r}, columns=['I16'], where='I16 > 3', batch_size=300))
assert calls[0] == (11, 490), calls
assert len(calls) == 1 + 4, calls
assert t.num_rows > 0 and n > 0
print("torch" in sys.modules)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
