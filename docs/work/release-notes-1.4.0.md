<!--
Draft user-facing notes for 1.4.0 (temporary; lives in docs/work/, deleted in
the commit that tags 1.4.0). Extend it as each milestone of
docs/work/1.4.0-best-in-class.md lands. At release time:
`pixi run changelog-release -- 1.4.0`, paste "Highlights" and "Behaviour
changes" at the top of the stamped section of docs/changelog.md (the shape
1.3.2rc1 / 1.3.3rc1 use), and use the whole draft as the GitHub Release body
(docs/release.md section 10).
-->

### Highlights

Write tables larger than memory, filter them without PyTorch, train on them
without loading them, and compress images with fpack's controls.

- **Streaming table writes.** `torchfits.table.writer()` writes a FITS table
  one batch at a time and publishes it atomically on close; `table.write`
  takes a stream of batches (a `RecordBatchReader`, a generator, Parquet
  `iter_batches()`) the same way, and `torchfits convert --to fits` streams
  through it.

  ```python
  import pyarrow.parquet as pq
  import torchfits

  batches = pq.ParquetFile("huge.parquet").iter_batches(batch_size=100_000)
  torchfits.table.write("huge.fits", batches)

  with torchfits.table.writer("bright.fits", extname="BRIGHT") as w:
      for batch in torchfits.table.scan("huge.fits", where="MAG < 18"):
          w.write(batch)
  ```

- **Native `where=` without PyTorch.** `table.read`, `table.scan`,
  `table.scan_torch(where=...)` (new) and the table datasets evaluate a
  conjunction of numeric comparisons natively, one row window at a time, and
  gather only the matching rows. Arrow reads never import PyTorch.
- **Lazy map-style tables.** `FitsTableDataset(path, lazy=True)` reads cached
  row blocks on index (`block_rows=`, `cache_blocks=`) instead of the whole
  table up front: shuffled training on a catalog that does not fit in memory.
- **fpack-style compression controls.** `tile_shape=`, `quantize_level=`,
  `quantize_method=`, `dither_seed=` and `int_to_float=` on `write`,
  `write_tensor`, `insert_hdu` and `replace_hdu`; `-t/--tile`, `-q`,
  `--quantize-method`, `--dither-seed` and `--int-to-float` on
  `torchfits compress`.

  ```python
  torchfits.write(
      "sky.fits.fz",
      image,
      compress="RICE_1",
      tile_shape=(128, 128),
      quantize_level=16,
      dither_seed=42,
  )
  ```

- **N-D table cells.** A `(N, d1, d2, ...)` column is written with `TDIM`;
  `shape_cells=True` reads such columns back with their cell shape (tensors,
  nested Arrow lists, Polars `Array`).
- **Typed column errors.** `torchfits.UnknownColumnError` and
  `torchfits.ColumnExistsError` (both `KeyError` and `ValueError`).
- **Faster buffered reads.** `mmap=False` column projections read the table
  in parallel tiles.
- **Correctness.** A sweep of silent-wrong-result and partial-output bugs:
  NaN pixels survive quantized compression, repeated `TTYPE` names no longer
  drop columns, every CLI output is written atomically, 64-bit `arith` is
  exact, `verify` catches truncated files.

Runnable examples for each feature:
`example_table_streaming_write.py`, `example_table_where_streaming.py`,
`example_lazy_table_dataset.py`, `example_fpack_compression.py`,
`example_tdim_cells.py`, `example_decode_errors.py`.

### Behaviour changes

- **Breaking: string columns decode strictly.** A byte the encoding cannot
  decode raises `UnicodeDecodeError` naming the column, on every read path
  and in `where=` string comparisons. 1.3 dropped such bytes silently. Pass
  `encoding="latin-1"` (or the file's real encoding), or
  `decode_errors="ignore"` for the old result (`"replace"` keeps U+FFFD).
- A repeated `TTYPE` column is named `COL<n>` on every read path.
- `num_rows` raises on a malformed `NAXIS2` instead of returning 0.
- `append_rows` / `insert_rows` fill an omitted float column with NaN (was 0).
- Quantized float compression keeps NaN pixels (1.3 wrote garbage values).
- CLI: atomic outputs; `convert --to fits` keeps the stored table layout and
  cards; exact, saturating 64-bit `arith` (a NaN into an integer image is exit
  2); `BUNIT` / `DATAMIN` / `DATAMAX` dropped or checked when stale; `verify`
  exits 3 on a truncated file; strict-JSON `table` previews.

The full list, with what to change, is in
[Compatibility → Upgrading from 1.3](https://astroai.github.io/torchfits/compatibility/#upgrading-from-13).
