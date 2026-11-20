# Roadmap

torchfits reads and writes FITS as tensors and tables. World coordinates, units, and source finding stay in other packages.

## Where things stand

**1.4** drops the per-PyTorch-minor release lanes: the native extension no
longer links libtorch (decoded buffers reach PyTorch through DLPack) and
targets the CPython stable ABI, so one wheel per platform works with PyTorch
2.12 and every later release.

The **1.3 release series** used the PyTorch 2.14 lane. It moved
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

## Next

- Buffered table reads (`mmap=False`) still trail fitsio on narrow full reads and
  ASCII predicate filters. The published October 2026 figures predate 1.3.0's
  zero-copy transport and will be refreshed in
  [torchfits-harness](https://github.com/astroai/torchfits-harness).
- Single-column projection on the buffered path still `pread`s whole rows; a
  strided per-column read would close the remaining `mmap=False` gap.

## Later

A later major version may read tiles with its own codecs and move bytes from storage into GPU memory without a host copy. That is not scheduled. The 1.x read and write calls stay as they are.
