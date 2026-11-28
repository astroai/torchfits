"""
Example: stream batches into a FITS table with ``table.writer()``.

A table larger than memory is written one batch at a time: ``table.writer()``
(or ``table.write`` given a stream of batches) keeps one batch in memory, and
the file appears at its path only when the stream completes. Covers:

- a ``with table.writer(...)`` loop over generated batches;
- Parquet -> FITS out of core (``pyarrow.parquet`` batches into ``table.write``);
- FITS -> FITS filtering (``table.reader(where=...)`` into ``table.write``);
- what a failed stream leaves behind (nothing: the old file is untouched).

Synthetic data only; every claim is an assertion.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import torchfits

N_BATCHES = 8
BATCH_ROWS = 25_000
N_ROWS = N_BATCHES * BATCH_ROWS


def _batches(seed: int = 0):
    """A deterministic catalog produced batch by batch (never all at once)."""
    rng = np.random.default_rng(seed)
    for i in range(N_BATCHES):
        start = i * BATCH_ROWS
        yield {
            "ID": np.arange(start, start + BATCH_ROWS, dtype=np.int64),
            "RA": rng.uniform(0.0, 360.0, BATCH_ROWS),
            "MAG": rng.normal(21.0, 1.5, BATCH_ROWS).astype(np.float32),
            "BAND": np.array(["g", "r", "i", "z"])[rng.integers(0, 4, BATCH_ROWS)],
        }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchfits_stream_write_") as tmp:
        catalog = os.path.join(tmp, "catalog.fits")

        # 1. Incremental writer: the first batch fixes the layout (TFORM,
        #    string widths); later batches must match it. schema= sizes the
        #    BAND column up front so a longer value cannot arrive later.
        with torchfits.table.writer(
            catalog,
            extname="CATALOG",
            schema={"BAND": {"format": "2A"}},
            checksum=True,
            overwrite=True,
        ) as writer:
            for batch in _batches():
                writer.write(batch)
            assert writer.num_rows == N_ROWS
        assert torchfits.read_nrows(catalog, hdu=1) == N_ROWS
        header = torchfits.read_header(catalog, hdu=1)
        assert header["EXTNAME"] == "CATALOG" and header["TFORM4"] == "2A"
        assert torchfits.verify_checksums(catalog, hdu=1)["ok"]
        print(f"table.writer: {N_ROWS} rows in {N_BATCHES} batches -> CATALOG")

        # 2. Parquet -> FITS without loading the Parquet file: iter_batches()
        #    is a stream, which table.write() hands to the same writer.
        parquet = os.path.join(tmp, "catalog.parquet")
        with pq.ParquetWriter(
            parquet, pa.schema([("ID", pa.int64()), ("MAG", pa.float32())])
        ) as pw:
            for batch in _batches(seed=1):
                pw.write_table(pa.table({"ID": batch["ID"], "MAG": batch["MAG"]}))
        from_parquet = os.path.join(tmp, "from_parquet.fits")
        stream = pq.ParquetFile(parquet).iter_batches(batch_size=10_000)
        torchfits.table.write(from_parquet, stream, overwrite=True)
        back = torchfits.table.read(from_parquet)
        expected = pq.read_table(parquet)
        assert back.num_rows == expected.num_rows == N_ROWS
        assert back.column("ID").equals(expected.column("ID"))
        assert np.array_equal(
            back.column("MAG").to_numpy(), expected.column("MAG").to_numpy()
        )
        print(f"Parquet -> FITS: {back.num_rows} rows streamed in 10k-row batches")

        # 3. FITS -> FITS filter: table.reader() is a RecordBatchReader, so the
        #    filtered rows stream straight into the new file.
        bright = os.path.join(tmp, "bright.fits")
        torchfits.table.write(
            bright,
            torchfits.table.reader(catalog, where="MAG < 19.5", batch_size=20_000),
            overwrite=True,
        )
        n_bright = torchfits.read_nrows(bright, hdu=1)
        n_expected = torchfits.table.read(catalog, where="MAG < 19.5").num_rows
        assert n_bright == n_expected > 0
        print(f"FITS -> FITS where='MAG < 19.5': {n_bright} rows")

        # 4. Atomic: a stream that fails part way publishes nothing, and an
        #    existing file at the path stays byte-identical.
        before = open(bright, "rb").read()

        def _failing():
            yield from _batches()
            raise RuntimeError("upstream producer failed")

        try:
            torchfits.table.write(bright, _failing(), overwrite=True)
        except RuntimeError as exc:
            print(f"failed stream: {exc!s} -> existing file untouched")
        else:
            raise AssertionError("the failing stream should have raised")
        assert open(bright, "rb").read() == before
        assert sorted(os.listdir(tmp)) == sorted(
            ["catalog.fits", "catalog.parquet", "from_parquet.fits", "bright.fits"]
        ), os.listdir(tmp)

    print("All streaming-write checks passed.")


if __name__ == "__main__":
    main()
