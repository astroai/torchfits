"""
Example: native ``where=`` filters, from Arrow reads to a training loader.

A conjunction of numeric comparisons (``AND`` / ``BETWEEN``) on scalar
columns is evaluated natively, one row window at a time, and only the
matching rows are gathered:

- ``table.read`` / ``table.scan`` with ``where=`` return Arrow data and never
  import PyTorch (checked in a fresh interpreter below);
- ``table.scan_torch(where=...)`` streams the matching rows as tensor batches;
- ``FitsTableIterableDataset(where=...)`` feeds them to a ``DataLoader``.

Every path returns the same rows as a NumPy mask over the whole table.
Synthetic data only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

import torchfits

N_ROWS = 300_000
WHERE = "MAG BETWEEN 18 AND 20 AND DEC > 0"

# Run in a fresh interpreter: proves the Arrow filter path stays torch-free.
_TORCH_FREE_CHECK = """
import sys
import torchfits

path, where = sys.argv[1], sys.argv[2]
table = torchfits.table.read(path, columns=["ID", "MAG"], where=where)
streamed = sum(b.num_rows for b in torchfits.table.scan(path, where=where,
                                                        batch_size=50_000))
assert table.num_rows == streamed, (table.num_rows, streamed)
assert "torch" not in sys.modules, "an Arrow where= read imported torch"
print(table.num_rows)
"""


def _write_catalog(path: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    cols = {
        "ID": np.arange(N_ROWS, dtype=np.int64),
        "DEC": rng.uniform(-90.0, 90.0, N_ROWS),
        "MAG": rng.normal(20.0, 1.5, N_ROWS).astype(np.float32),
        "FLUX": rng.lognormal(0.0, 1.0, N_ROWS).astype(np.float32),
    }
    torchfits.table.write(path, cols, overwrite=True)
    return cols


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from torchfits.data import FitsTableIterableDataset, make_loader

    with tempfile.TemporaryDirectory(prefix="torchfits_where_") as tmp:
        path = os.path.join(tmp, "catalog.fits")
        cols = _write_catalog(path)
        mask = (cols["MAG"] >= 18) & (cols["MAG"] <= 20) & (cols["DEC"] > 0)
        expected_ids = cols["ID"][mask]
        print(f"catalog: {N_ROWS} rows; where={WHERE!r} selects {mask.sum()}")

        # 1. Arrow: table.read / table.scan, in a process that never loads torch.
        out = subprocess.run(
            [sys.executable, "-c", _TORCH_FREE_CHECK, path, WHERE],
            check=True,
            capture_output=True,
            text=True,
        )
        assert int(out.stdout.strip()) == expected_ids.size
        print(f"table.read / table.scan: {out.stdout.strip()} rows, torch not imported")

        # 2. Tensors: scan_torch(where=) reads each 64k-row window once and
        #    yields only its matching rows (windows with no match are skipped).
        ids = []
        for batch in torchfits.table.scan_torch(
            path, columns=["ID", "FLUX"], where=WHERE, batch_size=65_536
        ):
            assert isinstance(batch["FLUX"], torch.Tensor)
            ids.append(batch["ID"])
        got = torch.cat(ids).numpy()
        assert np.array_equal(got, expected_ids)
        print(f"table.scan_torch(where=): {got.size} rows in {len(ids)} batches")

        # 3. Training input: the iterable dataset streams filtered chunks
        #    (as_batches=True keeps them as tensors end to end). With workers,
        #    each one filters its own contiguous row range, so every matching
        #    row arrives exactly once; batch_size=None passes chunks through.
        ds = FitsTableIterableDataset(
            path,
            columns=["ID", "MAG", "FLUX"],
            where=WHERE,
            batch_size=32_768,
            as_batches=True,
        )
        chunks = []
        for chunk in DataLoader(ds, batch_size=None, num_workers=2):
            assert bool(((chunk["MAG"] >= 18) & (chunk["MAG"] <= 20)).all())
            chunks.append(chunk["ID"])
        seen = torch.cat(chunks).sort().values.numpy()
        assert np.array_equal(seen, expected_ids)
        print(f"FitsTableIterableDataset(where=, 2 workers): {seen.size} rows")

        # Per-row mode, collated by make_loader as usual.
        rows = FitsTableIterableDataset(path, columns=["MAG"], where=WHERE)
        first = next(iter(make_loader(rows, batch_size=256, num_workers=0)))
        assert first["MAG"].shape == (256,)

    print("All where= checks passed.")


if __name__ == "__main__":
    main()
