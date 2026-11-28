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

**1.4** (unreleased, on `main`) is implemented:

- **Correctness and data integrity**: a sweep of silent-wrong-result and
  partial-output bugs across reads, writes, the CLI and the datasets. String
  columns now decode strictly by default; `decode_errors="replace"` or
  `"ignore"` opts into a lossy decode. CLI outputs (FITS, Parquet, CSV,
  Arrow) are written to a temporary file and renamed into place, and
  `torchfits verify` reports a truncated uncompressed file as an I/O error.
  A new column name the table already has raises `torchfits.ColumnExistsError`.
- **Streaming table writes**: `torchfits.table.writer()` (and `table.write`
  given a stream of batches) writes a table larger than memory incrementally
  and atomically; `torchfits convert` streams through it.
- **Native filtered reads without PyTorch**: `table.read`, `scan`,
  `scan_torch` and the table datasets evaluate eligible `where=` predicates
  natively, one row window at a time, and gather only the matching rows.
- **Faster buffered table reads**: `mmap=False` column projections read the
  table in parallel tiles instead of whole rows.
- **Faster full table reads**: a full `read_torch` or `table.read` of a narrow
  table (many rows of a few small columns) is about three times faster with
  `mmap=True` or `mmap=False` and no longer trails `fitsio.read`.
- **fpack-style compression controls**: `tile_shape=`, `quantize_level=`,
  `quantize_method=`, `dither_seed=` and `int_to_float=` on the writers, and
  `--tile`, `-q`, `--quantize-method`, `--dither-seed` and `--int-to-float`
  on `torchfits compress`.
- **Lazy map-style tables** (opt-in): `FitsTableDataset(..., lazy=True)`
  reads cached row blocks on index instead of the whole table up front.
- **Shaped `TDIM` cells** (opt-in): `shape_cells=True` reads a column with a
  multi-axis `TDIMn` as N-D cells; N-D cells are written with `TDIM`.

## Next

The **1.4.x** line aims to be the FITS release to beat: faster than every
earlier torchfits release and every other FITS library on each benchmark case,
and complete on the FITS 4.0 format. Planned work, in order:

- **ML pre-processing**: batched transforms that run on the GPU without host
  round trips, fast robust statistics (percentiles, zscale, sigma clipping,
  background) on the device, `torch.compile`-friendly pipelines, and loaders
  that prefetch into device memory so data loading never stalls training.
- **Remote and cached fetching**: S3 and other object stores alongside HTTP,
  VOSpace and shared filesystems such as `$PROJECT`, staged into a managed
  local cache on node NVMe (`$SLURM_TMPDIR`, CANFAR `/scratch`), with an
  optional decoded, device-ready cache layout and a staging API.
- **Native throughput**: parallel tile decompression and compression,
  64-bit heap descriptors on write, faster variable-length columns, and
  native `where=` for `read_torch`.
- **Out of core and remote**: a streaming image writer, in-place image
  section updates, and partial remote reads of compressed images, cubes and
  tables.
- **Format completeness**: tiled table compression, random groups,
  `A3DTABLE`, the `INHERIT` convention, shaped string arrays, and lossless
  nested columns with nulls.
- **Usability, CLI and docs**: an error-message audit, CLI equivalents for
  common fpack, ftools and STILTS tasks, migration guides per library, and a
  performance guide backed by the published benchmarks.

Every step is measured in
[torchfits-harness](https://github.com/astroai/torchfits-harness) against
earlier releases and the other FITS libraries.

Out of scope: world coordinates, units, catalog cross-matching and source
finding stay in other packages, and so do spectroscopy and continuum
transforms.

A later major version may read tiles with its own codecs and move bytes from storage into GPU memory without a host copy. That is not scheduled. The 1.x read and write calls stay as they are.
