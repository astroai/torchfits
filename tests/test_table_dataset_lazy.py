"""``FitsTableDataset(lazy=True)``: on-demand blocks give the eager rows.

Lazy mode reads aligned ``block_rows`` blocks on demand and keeps the last
``cache_blocks`` of them per process; every row it returns must be the row
the eager (fully materialised) dataset returns at the same index.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import torchfits.table
from torchfits import Header, HDUList, TableHDU, TensorHDU
from torchfits.data import FitsTableDataset, fits_collate_fn

N = 23


@pytest.fixture()
def catalog(tmp_path: Path) -> str:
    rng = np.random.default_rng(3)
    s = rng.integers(-50, 50, N).astype(np.int16)
    s[::5] = -99
    cols = {
        "ID": np.arange(N, dtype=np.int64),
        "NAME": np.array([f"n{i}" for i in range(N)]),
        "U16": rng.integers(0, 65535, N).astype(np.uint16),
        "S": s,
        "FLAG": rng.integers(0, 2, N).astype(bool),
        "VEC": rng.standard_normal((N, 3)).astype(np.float32),
    }
    path = tmp_path / "cat.fits"
    HDUList(
        [TensorHDU(torch.empty(0)), TableHDU(cols, header=Header({"TNULL4": -99}))]
    ).write(str(path), overwrite=True)
    return str(path)


@pytest.fixture()
def vla_table(tmp_path: Path) -> str:
    path = tmp_path / "vla.fits"
    torchfits.table.write(
        str(path),
        {
            "V": [np.arange(i % 4 + 1, dtype=np.int32) for i in range(N)],
            "X": np.arange(N, dtype=np.int32),
        },
        schema={"V": {"format": "1PJ"}},
        overwrite=True,
    )
    return str(path)


def _assert_same_row(got: dict, want: dict) -> None:
    assert list(got) == list(want)
    for key in want:
        assert type(got[key]) is type(want[key]), key
        if isinstance(want[key], torch.Tensor):
            assert got[key].dtype == want[key].dtype, key
            assert torch.equal(got[key], want[key]), key
        else:
            assert got[key] == want[key], key


def _assert_same_dataset(lazy: FitsTableDataset, eager: FitsTableDataset) -> None:
    assert len(lazy) == len(eager)
    order = list(range(len(eager)))
    rng = np.random.default_rng(0)
    for idx in order + rng.permutation(len(eager)).tolist() + [-1, -len(eager)]:
        got, want = lazy[idx], eager[idx]
        if isinstance(want, tuple):
            _assert_same_row(got[0], want[0])
            assert torch.equal(got[1], want[1])
        else:
            _assert_same_row(got, want)


@pytest.mark.parametrize("block_rows,cache_blocks", [(1, 1), (4, 2), (5, 64)])
@pytest.mark.parametrize("mmap", ["auto", True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"columns": ["VEC", "NAME"]},
        {"where": "S > 0 AND ID != 7"},  # TNULL -99 never matches
        {"where": "S > 0 OR NAME == 'n3'", "columns": ["ID", "FLAG"]},
    ],
)
def test_lazy_rows_match_eager(catalog, block_rows, cache_blocks, mmap, kwargs):
    eager = FitsTableDataset(catalog, mmap=mmap, **kwargs)
    lazy = FitsTableDataset(
        catalog,
        mmap=mmap,
        lazy=True,
        block_rows=block_rows,
        cache_blocks=cache_blocks,
        **kwargs,
    )
    _assert_same_dataset(lazy, eager)
    assert len(lazy._lazy_blocks._blocks) <= cache_blocks


def test_lazy_labels_and_return_label(catalog):
    labels = list(range(N))
    eager = FitsTableDataset(catalog, labels=labels)
    lazy = FitsTableDataset(catalog, labels=labels, lazy=True, block_rows=4)
    _assert_same_dataset(lazy, eager)
    bare = FitsTableDataset(catalog, return_label=False, lazy=True, block_rows=4)
    _assert_same_dataset(bare, FitsTableDataset(catalog, return_label=False))
    with pytest.raises(IndexError):
        lazy[N]


def test_lazy_vla_rows_match_eager(vla_table):
    eager = FitsTableDataset(vla_table, return_label=False)
    lazy = FitsTableDataset(vla_table, return_label=False, lazy=True, block_rows=3)
    _assert_same_dataset(lazy, eager)
    assert lazy[5]["V"].tolist() == [0, 1]


def test_lazy_reads_one_block_per_miss(catalog, monkeypatch):
    import torchfits._C as C

    calls: list[tuple] = []
    real = C.read_fits_table_rows

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(C, "read_fits_table_rows", spy)
    ds = FitsTableDataset(catalog, lazy=True, block_rows=4, cache_blocks=2)
    assert calls == []  # nothing is read at __init__
    for idx in range(N):
        ds[idx]
    assert [(a[3], a[4]) for a in calls] == [
        (1, 4),
        (5, 4),
        (9, 4),
        (13, 4),
        (17, 4),
        (21, 3),
    ]
    ds[22]  # cached
    ds[0]  # evicted (cache_blocks=2): read again
    assert len(calls) == 7


def test_lazy_cache_is_per_process(catalog):
    ds = FitsTableDataset(catalog, lazy=True, block_rows=4)
    ds[0]
    cache = ds._lazy_blocks
    assert len(cache._blocks) == 1
    clone = pickle.loads(pickle.dumps(ds))  # spawn workers
    assert len(clone._lazy_blocks._blocks) == 0
    _assert_same_dataset(clone, ds)
    cache._pid = -1  # as seen from a forked child
    ds[1]
    assert list(cache._blocks) == [0] and cache._pid != -1


def test_lazy_dataloader_workers(catalog):
    eager = FitsTableDataset(catalog, return_label=False)
    lazy = FitsTableDataset(catalog, return_label=False, lazy=True, block_rows=4)
    loader = torch.utils.data.DataLoader(
        lazy, batch_size=5, num_workers=2, collate_fn=fits_collate_fn
    )
    got = {k: torch.cat([b[k] for b in loader]) for k in ("ID", "VEC", "S")}
    for key, value in got.items():
        assert torch.equal(value, eager._data[key]), key


@pytest.mark.parametrize(
    "bad", [{"block_rows": 0}, {"cache_blocks": -1}, {"block_rows": 2.5}]
)
def test_lazy_rejects_bad_block_settings(catalog, bad):
    with pytest.raises(ValueError, match="positive int"):
        FitsTableDataset(catalog, lazy=True, **bad)


def test_lazy_where_unknown_column(catalog):
    with pytest.raises(KeyError, match="NOPE"):
        FitsTableDataset(catalog, where="NOPE > 1", lazy=True)
