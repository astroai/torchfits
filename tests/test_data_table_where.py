"""``where=`` on table datasets selects rows without changing their representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from torchfits import Header, HDUList, TableHDU, TensorHDU
from torchfits.data import FitsTableDataset, FitsTableIterableDataset
from torchfits.transforms import TNullToNan


@pytest.fixture()
def catalog(tmp_path: Path) -> str:
    n = 9
    cols = {
        "ID": np.arange(n, dtype=np.int64),
        "NAME": np.array(["a", "bb", "ccc", "d", "ee", "f", "gg", "h", "iii"]),
        "U16": np.array([0, 1, 40000, 65535, 2, 3, 4, 5, 6], dtype=np.uint16),
        "S": np.array([1, 2, -99, 4, 5, -99, 7, 8, 9], dtype=np.int16),
        "FLAG": np.array([True, False] * 4 + [True]),
        "VEC": np.arange(2 * n, dtype=np.float32).reshape(n, 2),
    }
    path = tmp_path / "cat.fits"
    HDUList(
        [TensorHDU(torch.empty(0)), TableHDU(cols, header=Header({"TNULL4": -99}))]
    ).write(str(path), overwrite=True)
    return str(path)


def _assert_rows_equal(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for key in want:
        assert isinstance(got[key], torch.Tensor), key
        assert got[key].dtype == want[key].dtype, key
        assert torch.equal(got[key], want[key]), key


WHERE = "ID >= 2 AND ID != 4 OR NAME == 'a'"
SELECTED = [0, 2, 3, 5, 6, 7, 8]


def test_map_dataset_where_matches_unfiltered_rows(catalog: str) -> None:
    full = FitsTableDataset(catalog)
    filtered = FitsTableDataset(catalog, where=WHERE)
    assert len(filtered) == len(SELECTED)
    for i, src in enumerate(SELECTED):
        _assert_rows_equal(filtered[i][0], full[src][0])
    # Nulls stay stored sentinels; TNullToNan is the documented conversion.
    assert filtered[1][0]["S"].item() == -99
    assert torch.isnan(TNullToNan({"S": -99})(filtered[1][0])["S"])


def test_map_dataset_where_without_matches(catalog: str) -> None:
    assert len(FitsTableDataset(catalog, where="ID > 100")) == 0


@pytest.mark.parametrize("batch_size", [2, 64])
def test_iterable_where_matches_unfiltered_rows(catalog: str, batch_size: int) -> None:
    full = list(FitsTableIterableDataset(catalog, batch_size=batch_size))
    filtered = list(
        FitsTableIterableDataset(catalog, where=WHERE, batch_size=batch_size)
    )
    assert len(filtered) == len(SELECTED)
    for row, src in zip(filtered, SELECTED):
        _assert_rows_equal(row, full[src])


@pytest.mark.parametrize("world_size", [2, 3])
def test_iterable_where_sharded_windows(catalog: str, world_size: int) -> None:
    full = list(FitsTableIterableDataset(catalog, batch_size=2))
    rows = []
    for rank in range(world_size):
        rows += list(
            FitsTableIterableDataset(
                catalog, where=WHERE, batch_size=2, rank=rank, world_size=world_size
            )
        )
    assert [int(r["ID"]) for r in rows] == SELECTED
    for row, src in zip(rows, SELECTED):
        _assert_rows_equal(row, full[src])


def test_iterable_where_as_batches_matches_unfiltered(catalog: str) -> None:
    (full,) = list(FitsTableIterableDataset(catalog, as_batches=True))
    batches = list(
        FitsTableIterableDataset(catalog, where=WHERE, as_batches=True, batch_size=4)
    )
    keep = torch.tensor(SELECTED)
    assert batches
    assert set(batches[0]) == set(full)
    for key in full:
        got = torch.cat([b[key] for b in batches])
        assert got.dtype == full[key].dtype, key
        assert torch.equal(got, full[key][keep]), key


def _count_native_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Spy on every native table read entry point (module functions)."""
    import torchfits._C as C

    calls: list[str] = []
    for name in dir(C):
        if not name.startswith("read_fits_table"):
            continue
        real = getattr(C, name)

        def spy(*args, _real=real, _name=name, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(C, name, spy)
    return calls


@pytest.mark.parametrize(
    "where,native",
    [("ID >= 2 AND S > 0", True), (WHERE, False)],
)
@pytest.mark.parametrize("as_batches", [False, True])
def test_iterable_where_reads_each_chunk_once(
    catalog: str,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
    native: bool,
    as_batches: bool,
) -> None:
    """One native read per chunk: the predicate columns are not read again."""
    want = list(FitsTableIterableDataset(catalog, where=where, as_batches=as_batches))
    calls = _count_native_reads(monkeypatch)
    got = list(
        FitsTableIterableDataset(
            catalog, where=where, batch_size=2, as_batches=as_batches
        )
    )
    n_chunks = 5  # 9 rows in windows of 2
    assert len(calls) == n_chunks, calls
    assert all(c == "read_fits_table_filtered" for c in calls) is native
    if as_batches:
        got = [{k: torch.cat([b[k] for b in got]) for k in got[0]}]
    assert len(got) == len(want)
    for row, src in zip(got, want):
        _assert_rows_equal(row, src)
