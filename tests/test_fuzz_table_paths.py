"""Seeded differential check of the table read/write paths.

A small, deterministic slice of the release fuzz harness: random tables
(unsigned conventions, TSCAL/TZERO, TNULL, NaN, padded strings, vectors,
TDIM cells, VLA, logical and bit columns) are written once and read back
through every path, which must agree with each other: ``table.read`` with
and without mmap and on each backend, ``scan`` / ``reader`` in random batch
sizes and row windows, ``read_torch`` / ``scan_torch``, random ``where=``
predicates against the reference evaluator, and ``table.writer`` streaming
the same rows in random batch splits.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import torchfits
from torchfits.where import evaluate_where, parse_where_expression

INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


def _ints(rng, dtype, shape):
    info = np.iinfo(dtype)
    out = rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    if out.size:
        flat = out.reshape(-1)
        flat[rng.integers(0, flat.size, size=max(1, flat.size // 8))] = info.max
    return out


def _floats(rng, dtype, shape):
    out = (rng.standard_normal(shape) * 100).astype(dtype)
    if out.size:
        flat = out.reshape(-1)
        flat[rng.integers(0, flat.size, size=max(1, flat.size // 8))] = np.nan
    return out


def _table(rng, n):
    data: dict = {"ROWID": np.arange(n, dtype=np.int64)}
    schema: dict = {}
    for i in range(int(rng.integers(3, 8))):
        kind = rng.choice(
            [
                "int",
                "flt",
                "str",
                "vec",
                "tdim",
                "vla",
                "bool",
                "bit",
                "scaled",
                "tnull",
            ]
        )
        name = f"C{i}"
        if kind == "int":
            data[name] = _ints(rng, str(rng.choice(INTS)), n)
        elif kind == "flt":
            data[name] = _floats(rng, str(rng.choice(["float32", "float64"])), n)
        elif kind == "str":
            words = ["", "a", " b", "cc ", "x'y", "zz9"]
            data[name] = np.array([words[j] for j in rng.integers(0, len(words), n)])
            schema[name] = {"format": "5A"}
        elif kind == "vec":
            data[name] = _ints(rng, str(rng.choice(INTS)), (n, 3))
        elif kind == "tdim":
            data[name] = _floats(rng, "float32", (n, 2, 3))
        elif kind == "vla":
            data[name] = [
                _floats(rng, "float64", int(k)) for k in rng.integers(0, 4, n)
            ]
        elif kind == "bool":
            data[name] = rng.random(n) < 0.5
        elif kind == "bit":
            data[name] = rng.random((n, 5)) < 0.5
            schema[name] = {"format": "5X"}
        elif kind == "scaled":
            data[name] = rng.integers(-1000, 1000, n).astype(np.int16)
            schema[name] = {"format": "I", "bscale": 0.5, "bzero": 7.0, "tnull": -1000}
        else:
            data[name] = rng.integers(-3, 3, n).astype(np.int32)
            schema[name] = {"tnull": 0}
    return data, schema


def _rows(table: pa.Table) -> list:
    return [tuple(r.values()) for r in table.to_pylist()]


def _same(a, b) -> bool:
    return repr(a) == repr(b)  # NaN-safe structural equality


def _reference_rows(full: pa.Table, expr: str) -> list[int]:
    data = {}
    for name in full.column_names:
        col = full[name]
        if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
            continue
        arr = col.to_numpy(zero_copy_only=False)
        if col.null_count and arr.dtype.kind != "f":
            arr = np.array(col.to_pylist(), dtype=object)
        data[name] = arr
    mask = evaluate_where(parse_where_expression(expr), data)
    return np.asarray(full["ROWID"])[mask].tolist()


def _predicate(rng, data) -> str:
    names = [
        k
        for k, v in data.items()
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "iuf"
    ]
    parts = []
    for _ in range(int(rng.integers(1, 3))):
        name = str(rng.choice(names))
        col = data[name]
        lit = col[int(rng.integers(len(col)))].item() if len(col) else 0
        lit = 0 if isinstance(lit, float) and np.isnan(lit) else lit
        op = str(rng.choice(["<", "<=", ">", ">=", "==", "!="]))
        parts.append(f"{name} {op} {lit!r}")
    return " AND ".join(parts)


@pytest.mark.parametrize("seed", range(8))
def test_table_paths_agree(tmp_path, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([0, 1, 7, 300]))
    data, schema = _table(rng, n)
    path = str(tmp_path / "t.fits")
    torchfits.table.write(path, data, schema=schema)

    base = torchfits.table.read(path)
    assert base.num_rows == n
    for kwargs in ({"mmap": False}, {"backend": "cpp"}, {"backend": "torch"}):
        assert _same(_rows(torchfits.table.read(path, **kwargs)), _rows(base)), kwargs
    assert torchfits.table.schema(path) == base.schema

    batch = int(rng.integers(1, 64))
    scanned = list(torchfits.table.scan(path, batch_size=batch))
    if n:
        assert _same(_rows(pa.Table.from_batches(scanned)), _rows(base))
    streamed = torchfits.table.reader(path, batch_size=batch).read_all()
    assert _same(_rows(streamed.replace_schema_metadata(None)), _rows(base))

    if n:
        lo = int(rng.integers(0, n))
        hi = int(rng.integers(lo, n + 1))
        window = torchfits.table.read(path, row_slice=(lo, hi), mmap=False)
        assert _same(_rows(window), _rows(base)[lo:hi])
        ids = torchfits.table.read_torch(path, start_row=lo + 1, num_rows=hi - lo)
        assert ids["ROWID"].tolist() == list(range(lo, hi))

    chunks = list(torchfits.table.scan_torch(path, batch_size=batch))
    assert sum(len(c["ROWID"]) for c in chunks) == n

    if n:
        expr = _predicate(rng, data)
        expected = _reference_rows(base, expr)
        for kwargs in ({}, {"mmap": False}, {"backend": "torch"}):
            got = torchfits.table.read(path, where=expr, **kwargs)
            assert got["ROWID"].to_pylist() == expected, (expr, kwargs)
        got = [
            v
            for c in torchfits.table.scan_torch(path, batch_size=batch, where=expr)
            for v in c["ROWID"].tolist()
        ]
        assert got == expected, expr

    # The streaming writer, fed the same rows in random splits.
    cuts = sorted(int(c) for c in rng.integers(0, n + 1, size=3))
    bounds = [0, *cuts, n]
    out = str(tmp_path / "s.fits")
    with torchfits.table.writer(out, schema=schema) as writer:
        for a, b in zip(bounds[:-1], bounds[1:]):
            writer.write({k: v[a:b] for k, v in data.items()})
    assert _same(_rows(torchfits.table.read(out)), _rows(base))
