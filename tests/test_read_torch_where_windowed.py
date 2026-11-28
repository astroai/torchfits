"""read_torch(where=) is evaluated in row windows, with unchanged results.

It used to read every requested column whole and build a whole-table boolean
mask before keeping the matching rows: a 2.1 GB one-byte column peaked at
4.4 GB. Each window is now read and filtered on its own and only the matches
are kept, so memory is the result plus one window. The results must be the
ones the whole-table evaluation gives (same rows, order, dtypes, TNULL
handling), which these tests check against that path with tiny windows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import torchfits
from torchfits._io_engine import table_api

N = 23  # several 5-row windows plus a partial one


@pytest.fixture
def mixed_table(tmp_path) -> str:
    path = str(tmp_path / "mixed.fits")
    ids = np.arange(N, dtype=np.int32)
    a = (ids % 7 - 3).astype(np.int16)
    a[[2, 9, 17]] = -999
    f = (ids * 0.5).astype(np.float32)
    f[[4, 11]] = np.nan
    torchfits.table.write(
        path,
        {
            "ID": ids,
            "A": a,
            "U": (ids * 3000).astype(np.uint16),
            "Q": (np.uint64(2**63) + ids.astype(np.uint64)),
            "F": f,
            "L": ids % 3 == 0,
            "S": np.array([f"s{i}" for i in range(N)]),
            "V": np.arange(N * 3, dtype=np.float64).reshape(N, 3),
            "P": [np.arange(i % 4, dtype=np.int32) for i in range(N)],
            "C": (ids % 5).astype(np.int16),
        },
        schema={"A": {"tnull": -999}, "C": {"bscale": 0.5, "bzero": 10.0}},
        overwrite=True,
    )
    return path


def _whole_table(monkeypatch, **kwargs: Any) -> Any:
    with monkeypatch.context() as m:
        m.setattr(table_api, "_read_where_windowed", lambda *a, **k: None)
        return torchfits.table.read_torch(**kwargs)


def _windowed(monkeypatch, rows: int = 5, **kwargs: Any) -> Any:
    reads: list[int] = []
    original = table_api._thin_read_table_torch

    def spy(path: str, **kw: Any) -> Any:
        reads.append(kw["num_rows"])
        return original(path, **kw)

    with monkeypatch.context() as m:
        m.setattr(table_api, "_WHERE_WINDOW_MIN_ROWS", rows)
        m.setattr(table_api, "_WHERE_WINDOW_MAX_ROWS", rows)
        m.setattr(table_api, "_thin_read_table_torch", spy)
        out = torchfits.table.read_torch(**kwargs)
    # Every read is bounded by the window: nothing reads the table whole.
    assert reads and all(0 < n <= rows for n in reads), reads
    return out


def _assert_same(got: Any, want: Any) -> None:
    if isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        _assert_same(got[0], want[0])
        assert dict(got[1]) == dict(want[1])
        return
    assert list(got) == list(want)
    for name, value in want.items():
        other = got[name]
        if isinstance(value, torch.Tensor):
            assert isinstance(other, torch.Tensor), name
            assert other.dtype == value.dtype, name
            assert other.shape == value.shape, name
            assert torch.equal(other, value) or (
                value.dtype.is_floating_point
                and torch.equal(torch.isnan(other), torch.isnan(value))
                and torch.equal(other.nan_to_num(), value.nan_to_num())
            ), name
        elif isinstance(value, list):
            assert len(other) == len(value), name
            for x, y in zip(other, value):
                if isinstance(y, torch.Tensor):
                    assert torch.equal(x, y), name
                else:
                    assert x == y, name
        else:
            assert other == value, name


@pytest.mark.parametrize(
    "where",
    [
        "ID >= 0",
        "ID > 100",
        "A > -2",
        "A != 0",
        "U >= 30000",
        "Q > 9223372036854775815",
        "F < 7.5",
        "F != 3.0",
        "L == 1",
        "C >= 11",
        "ID BETWEEN 3 AND 19 AND A <= 1",
    ],
)
@pytest.mark.parametrize("columns", [None, ["ID", "S", "V", "P"], ["V", "ID"]])
def test_windowed_matches_whole_table(monkeypatch, mixed_table, where, columns):
    kwargs = {"path": mixed_table, "hdu": 1, "columns": columns, "where": where}
    want = _whole_table(monkeypatch, **kwargs)
    got = _windowed(monkeypatch, **kwargs)
    _assert_same(got, want)


@pytest.mark.parametrize(
    "start_row,num_rows", [(1, -1), (4, 12), (7, 1), (20, 50), (1, 5)]
)
@pytest.mark.parametrize("mmap", [True, False])
def test_windowed_row_window_matches(
    monkeypatch, mixed_table, start_row, num_rows, mmap
):
    kwargs = {
        "path": mixed_table,
        "hdu": 1,
        "where": "A < 3",
        "start_row": start_row,
        "num_rows": num_rows,
        "mmap": mmap,
        "return_header": True,
    }
    _assert_same(_windowed(monkeypatch, **kwargs), _whole_table(monkeypatch, **kwargs))


def test_vector_predicate_still_rejected(monkeypatch, mixed_table):
    with pytest.raises(ValueError, match="fixed-size vector column 'V'"):
        _windowed(monkeypatch, path=mixed_table, hdu=1, where="V > 1")


def test_default_window_is_one_read_for_a_small_table(monkeypatch, mixed_table):
    reads: list[int] = []
    original = table_api._thin_read_table_torch

    def spy(path: str, **kw: Any) -> Any:
        reads.append(kw["num_rows"])
        return original(path, **kw)

    monkeypatch.setattr(table_api, "_thin_read_table_torch", spy)
    out = torchfits.table.read_torch(mixed_table, 1, where="ID < 4")
    assert reads == [N]
    assert out["ID"].tolist() == [0, 1, 2, 3]
