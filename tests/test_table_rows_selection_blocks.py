"""table.read(rows=...) reads dense scatters through block reads.

Rows close together share one ranged read (the gap rows are discarded) and
blocks are read in bounded batches; the result must equal ``full.take(rows)``
for any order and duplicates, for every column kind -- including VLA columns,
which used to come back in sorted row order while the other columns followed
the requested order.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits._table.engine as engine
from torchfits import table

N = 2000


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    p = tmp_path_factory.mktemp("rows") / "t.fits"
    rng = np.random.default_rng(0)
    table.write(
        str(p),
        {
            "i": np.arange(N, dtype=np.int64),
            "f": rng.random(N).astype(np.float32),
            "v3": rng.random((N, 3)),
            "u": rng.integers(0, 65535, N).astype(np.uint16),
            "s": np.array([f"r{i}" for i in range(N)]),
            "vla": [np.arange(i % 5, dtype=np.float64) + i for i in range(N)],
        },
        schema={"vla": {"format": "1PD"}},
    )
    return str(p)


SELECTIONS = {
    "unsorted_dups": [7, 2, 2, 1999, 0, 7, 1000],
    "dense_scatter": np.random.default_rng(1).choice(N, 600, replace=False).tolist(),
    "sparse": [5, 900, 1800],
    "contiguous": list(range(300, 420)),
    "reversed": list(range(N - 1, N - 50, -1)),
    "single": [1234],
}


@pytest.mark.parametrize("name", sorted(SELECTIONS))
@pytest.mark.parametrize("budget", ["default", "tiny"])
def test_rows_match_full_take(path, monkeypatch, name, budget):
    if budget == "tiny":
        # Many batches, wide read-through: exercises batch splitting.
        monkeypatch.setattr(engine, "_ROW_BLOCK_BATCH_BYTES", 4096)
        monkeypatch.setattr(engine, "_ROW_GAP_READ_THROUGH_BYTES", 1 << 20)
    rows = SELECTIONS[name]
    full = table.read(path)
    out = table.read(path, rows=rows)
    assert out.equals(full.take(rows))


def test_rows_with_column_projection(path):
    rows = [10, 3, 3, 1500, 11]
    full = table.read(path, columns=["vla", "i"])
    out = table.read(path, columns=["vla", "i"], rows=rows)
    assert out.equals(full.take(rows))


def test_row_blocks_merge_only_small_gaps():
    rows = np.array([0, 1, 5, 100, 101, 300], dtype=np.int64)
    starts, lengths = engine._row_blocks(rows, max_gap=3)
    assert starts.tolist() == [0, 100, 300]
    assert lengths.tolist() == [6, 2, 1]
    starts, lengths = engine._row_blocks(rows, max_gap=0)
    assert list(zip(starts.tolist(), lengths.tolist())) == [
        (0, 2),
        (5, 1),
        (100, 2),
        (300, 1),
    ]
