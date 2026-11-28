"""
Example: random-access training on a catalog that need not fit in memory.

``FitsTableDataset`` loads the whole (filtered) table at ``__init__`` by
default. ``lazy=True`` reads only the row count (and, with ``where=``, the
selected row numbers) up front; each ``ds[i]`` then reads the
``block_rows``-row block holding row ``i`` and keeps the last
``cache_blocks`` blocks per process (so per DataLoader worker). Rows are
identical to eager mode, and shuffled map-style sampling works as usual.

Synthetic data only; every claim is an assertion.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

import torchfits
from torchfits.data import FitsTableDataset

N_ROWS = 100_000


def _write_catalog(path: str) -> None:
    rng = np.random.default_rng(7)
    torchfits.table.write(
        path,
        {
            "ID": np.arange(N_ROWS, dtype=np.int64),
            "MAG": rng.normal(20.0, 1.5, N_ROWS).astype(np.float32),
            "COLOR": rng.normal(0.5, 0.3, N_ROWS).astype(np.float32),
            "SPEC": rng.normal(size=(N_ROWS, 8)).astype(np.float32),  # vector
        },
        overwrite=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchfits_lazy_") as tmp:
        path = os.path.join(tmp, "catalog.fits")
        _write_catalog(path)
        columns = ["ID", "MAG", "COLOR", "SPEC"]

        eager = FitsTableDataset(path, columns=columns, where="MAG < 21")
        lazy = FitsTableDataset(
            path,
            columns=columns,
            where="MAG < 21",
            lazy=True,
            block_rows=512,  # rows per native read
            cache_blocks=32,  # blocks kept per process (LRU)
        )
        assert len(lazy) == len(eager) > 0
        print(f"{N_ROWS} rows, where='MAG < 21' keeps {len(lazy)}")

        # Same rows, same dtypes, in any order.
        rng = np.random.default_rng(0)
        for i in rng.integers(0, len(lazy), 200).tolist() + [0, -1]:
            (row_l, label_l), (row_e, _) = lazy[i], eager[i]
            for name in columns:
                assert row_l[name].dtype == row_e[name].dtype
                assert torch.equal(row_l[name], row_e[name])
            assert int(label_l) == 0  # no labels= given
        print("lazy rows == eager rows (random indices)")

        # Shuffled mini-batches with two workers: each worker reads its own
        # blocks and keeps its own cache; every selected row is seen once.
        loader = DataLoader(
            lazy,
            batch_size=1024,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(0),
        )
        ids = []
        for rows, labels in loader:
            assert rows["SPEC"].shape[1:] == (8,)
            assert bool((rows["MAG"] < 21).all())
            ids.append(rows["ID"])
        ids = torch.cat(ids)
        assert ids.numel() == len(lazy) and ids.unique().numel() == len(lazy)
        print(f"shuffled epoch, 2 workers: {ids.numel()} rows, each once")

        # Access with locality (sequential or windowed shuffles) mostly hits
        # the block cache; uniformly random access costs about one block read
        # per row. Larger blocks favour locality, smaller ones random access.

    print("All lazy dataset checks passed.")


if __name__ == "__main__":
    main()
