# Roadmap

torchfits reads and writes FITS as tensors and tables. World coordinates, units, and source finding stay in other packages.

## Where things stand

**1.3.3** (2026-11-20) dropped the per-PyTorch-minor release lanes. The native
extension no longer links libtorch: decoded buffers reach PyTorch through
DLPack. It targets the CPython stable ABI, so one wheel per platform works with
PyTorch 2.12 and every later release.

Earlier in the **1.3 release series**, torchfits moved
to Python 3.11–3.15 (3.10 is dropped), made `torchfits.open()` and header/HDU
inspection run without importing PyTorch, and wrapped native table buffers in
Arrow instead of staging them through a copy. The [changelog](changelog.md) has
the notes. The changes that matter for what you call:

- Header and metadata reads go through `libtorchfits_core` and do not load PyTorch.
- Arrow reads (`table.read`) do not import PyTorch either: fixed columns, numeric
  vectors, and fixed-width byte strings are zero-copy views over the decoded
  column storage. Tensor reads stay on `read_torch`.
- Transforms can carry inverse-variance and masks, and they refuse to calibrate data that is already calibrated.
- Scaled BITPIX=32 and 64 images are read as float64. Scaled BITPIX=8 and 16, including `quantize="robust"`, are float32.
- The legacy `torchfits.cpp` module is removed; the native surface is internal (`torchfits._cpp`).

**1.1** added checksummed writes, streaming `where=` filters, and the remote downloader. **1.0** (2026-08-09) was memory-mapped image reads, table filters, the datasets, and the CLI.

## Next: 1.4

- **Correctness and data integrity** (in progress): a sweep of silent-wrong-result
  and partial-output bugs across reads, writes, the CLI and the datasets.
- **Streaming table writes**: an incremental, atomic FITS table writer that
  takes an Arrow `RecordBatchReader` or any iterable of batches, so tables larger
  than memory can be written; `torchfits convert` streams through it.
- **Native filtered reads without PyTorch**: `scan(where=)`, `scan_torch(where=)`
  and the datasets evaluate the filter natively and push row windows down to the
  reader instead of materializing every row first.
- **Measure before optimizing the table reader further.** Buffered reads
  (`mmap=False`) still trail fitsio on some narrow tables. Single-column
  projections already use `fits_read_col`, but on a narrow table that call still
  reads every byte of every row: one column of a 20-column, 48 MB table read all
  48 MB and was slower than a buffered two-column read. Closing that gap needs a
  strided per-column read. Whether it is worth building will be decided by
  re-benchmarking in [torchfits-harness](https://github.com/astroai/torchfits-harness),
  not by the October 2026 figures, which predate 1.3's zero-copy transport.

## Later

- More `fpack` controls on `torchfits compress`: tile geometry,
  dither/quantization settings, and `-i2f`.
- Lazy, map-style table access (rows fetched on index instead of read up front).
- Shaped cells for `TDIM` columns.

A later major version may read tiles with its own codecs and move bytes from storage into GPU memory without a host copy. That is not scheduled. The 1.x read and write calls stay as they are.
