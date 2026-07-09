# Changelog

All notable changes to torchfits are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0b2] - 2026-07-09

### Added

- **Predicate filter improvements (all sizes now use C++ pushdown):** The
  `predicate_filter` path delegates to C++ for all table sizes, eliminating the
  Python fallback for narrow tables.  The `read_policy.py` size threshold is
  removed — safe non-VLA tables always use C++ pushdown.  Narrow-table
  predicate_filter lag vs fitsio reduced from ~2.86× (smallest) to ≤1.07×.
- **Lightweight is_compressed check:** Compressed images use a fast O(1) header
  probe instead of opening and parsing the full HDU, reducing overhead on
  batched compressed-image reads.
- **Thread-safe caches:** CacheManager internal data structures use
  `std::shared_mutex` for concurrent reader access, safe under multi-worker
  DataLoader patterns without global GIL serialisation.
- **Parallel scan with sequential fallback:** The C++ mmap pushdown scan is now
  parallelised via `at::parallel_for` when `torch::get_num_threads() > 1`,
  with a zero-overhead sequential path when single-threaded.  Added
  `posix_madvise(POSIX_MADV_SEQUENTIAL)` to the filtered scan path for kernel
  prefetch hints.
- **Lab benchmark refresh (mmap-on+off, 0.6.0b2):** 2754 rows, **3 deficits**
  in `20260709_163739` — *down* from 0.6.0b1's 14 deficits and 0.5.0b4's 22
  deficits.  Remaining 3-deficit breakdown:
  - 3 fitstable (narrow): `predicate_filter` on `narrow_{10000,100000,1000000}`
    (1.07–1.25× behind fitsio; `narrow_1000` dropped below the deficit
    threshold).  The gap is now dominated by Python dispatch + Arrow
    conversion overhead, not the C++ scan itself (which reaches near-parity
    with fitsio at ~11.4 ms vs ~11.0 ms for 1 M rows).
  All compressed-image deficits eliminated.  The uint16/uint32 mmap-on
  regression that motivated 0.5.0b4's bswap+BZERO merge is no longer in
  the deficit table.
- **Multi-worker DataLoader coverage:** `tests/test_data.py` now exercises
  ``make_loader(..., num_workers=2)`` for both ``FitsImageDataset`` and
  ``FitsImageIterableDataset``.  Tests fork a subprocess to keep CFITSIO's
  threadpool away from pytest's own threadpool, and verify that every file is
  seen exactly once regardless of ``num_workers`` and shuffle seed.
- **End-to-end FITS round-trip coverage** in `tests/test_transforms_e2e.py`:
  - `TestEndToEndImageRoundTrip` — write / read / scale / inverse for INT16
    with custom BSCALE/BZERO, BZERO=32768 unsigned convention, and INT32 with
    rescaling.
  - `TestEndToEndTableRoundTrip` — FITS binary tables with TSCAL/TZERO use
    real on-disk encoding (via astropy), then ``FITSScaleColumns.from_header`` +
    ``TNullToNan.from_header`` round-trip is verified to within the storage
    precision.
  - `TestEndToEndFITSHeaderNormalize` — full int16 BZERO=32768 round-trip
    through the header-driven normaliser.
- **Release-gate now includes** `tests/test_data.py`, `tests/test_transforms.py`,
  and `tests/test_transforms_e2e.py`.  This closes the *torchfits.data
  documented with multi-worker test coverage* and *torchfits.transforms
  round-trip tests for scaled images and tables* gate items.
- **`AsymmetricLeastSquares(lam, p, max_iter, dim)`** — Eilers 2003 penalised
  baseline correction with asymmetric weights. Iteratively solves the Whittaker
  smoother `(W + λD^T D)z = Wy` with differential weighting (p above baseline,
  1-p below). Standard in Raman/NIR spectroscopy. D^T D penalty matrix built in
  float64 for numerical stability at large λ. Additive decomposition (invertible).
- **`AlphaShapeContinuum(half_window, iterations, dim)`** — Morphological closing
  (dilation→erosion) via `unfold` + max/min. Produces a guaranteed upper envelope
  (always ≥ signal). Practical approximation to the full alpha-shape algorithm
  (RASSINE). Additive decomposition (invertible).
- **`AsymmetricSigmaClip(n_low, n_high, dim)`** — Simple one-pass asymmetric
  sigma-clipping outlier rejection using `estimate_background` (median + MAD).
  Supports different lower/upper sigma thresholds; replaces outliers with per-group
  median. Lossy (no inverse).
- `_build_d2_matrix` internal helper for the n×n pentadiagonal second-difference
  penalty matrix D^T D used by the Whittaker smoother / AsLS.
- 27 new tests for the three transforms (201 transforms tests total, all passing).
- Sections 7–9 in `examples/example_hyperspectral.py` demonstrating
  `AsymmetricLeastSquares`, `AlphaShapeContinuum`, and `AsymmetricSigmaClip`.
- All three transforms exported to the root package for direct
  `from torchfits import AsymmetricLeastSquares` access.
- Documentation for all three transforms in `docs/api.md` and `README.md`
  transform tables.

## [0.6.0b1] - 2026-07-08

### Removed

- Removed deprecated `read_large_table` function (use `stream_table` or `read_table` instead).
- **Custom WHERE AST evaluator runtime** (~120 lines) from `_where.py`: `_evaluate_cmp`,
  `_evaluate_in`, `_evaluate_between`, `_evaluate_isnull`, `_evaluate_where`, and the
  `evaluate_where` public alias. Replaced with `pyarrow.compute` native predicates via
  `_where_mask_for_table`. The parser, tokenizer, normalizers, and `where_columns_from_ast`
  stay for C++ pushdown path compatibility.
- **Compressed parallel decompression path** (~350 lines): `try_read_compressed_rows_parallel`,
  `compressed_parallel_enabled/min_pixels/min_rows_per_thread/max_threads/hcompress_enabled`
  helpers, `load_bswap` templates, `FitsHandleGuard` local class, `is_parallel_compressed_codec_cached`,
  `compressed_parallel_cache`, and `hardware_concurrency` dependency. CFITSIO's built-in decompression
  already covers this serially — the 2-thread cap meant the heuristic rarely activated.
- **Unused `read_rice_parallel`** (~320 lines) from `compression.cpp` — vendored Rice
  decompression, nanobind binding, and the entire `compression.cpp`/`compression.h` files.
  Dead after compressed parallel path removal.
- `bind_compression` from `bindings.cpp` — only bound `read_rice_parallel`.

### Changed

- **C++ hardening (Rule-of-5 fixes, RAII guards, unified BITPIX mapping):**
  All C++ resource-owning classes now follow the Rule-of-5 with explicit
  destructors, copy/move constructors, and copy/move assignment operators.
  RAII guards (`FitsHandleGuard`, `MmapGuard`) replace manual
  open/close pairs. BITPIX-to-dtype mapping is unified in a single
  `bitpix_to_dtype` lookup table instead of scattered `switch` statements,
  reducing the risk of type mismatch bugs.

### Fixed

- **BackgroundSubtract inverse:** Fixed incorrect inverse computation when
  `BackgroundSubtract` was used in a transform pipeline — the reconstruction
  was applying the offset in the wrong direction.
- **Writable mmap O_RDONLY:** Fixed a bug where opening a FITS file with
  `mode="r"` (read-only) would still request a writable mmap, causing
  `mmap failed: Permission denied` on read-only filesystems.

### Deprecated

- **`table_module` parameter removed:** The `table_module` parameter from
  `table.read_table` and related functions has been removed. Import
  `torchfits.table` directly instead.

## [0.6.0b1] - 2026-07-08

### Removed

- Removed deprecated `read_large_table` function (use `stream_table` or `read_table` instead).
- **Custom WHERE AST evaluator runtime** (~120 lines) from `_where.py`: `_evaluate_cmp`,
  `_evaluate_in`, `_evaluate_between`, `_evaluate_isnull`, `_evaluate_where`, and the
  `evaluate_where` public alias. Replaced with `pyarrow.compute` native predicates via
  `_where_mask_for_table`. The parser, tokenizer, normalizers, and `where_columns_from_ast`
  stay for C++ pushdown path compatibility.
- **Compressed parallel decompression path** (~350 lines): `try_read_compressed_rows_parallel`,
  `compressed_parallel_enabled/min_pixels/min_rows_per_thread/max_threads/hcompress_enabled`
  helpers, `load_bswap` templates, `FitsHandleGuard` local class, `is_parallel_compressed_codec_cached`,
  `compressed_parallel_cache`, and `hardware_concurrency` dependency. CFITSIO's built-in decompression
  already covers this serially — the 2-thread cap meant the heuristic rarely activated.
- **Unused `read_rice_parallel`** (~320 lines) from `compression.cpp` — vendored Rice
  decompression, nanobind binding, and the entire `compression.cpp`/`compression.h` files.
  Dead after compressed parallel path removal.
- `bind_compression` from `bindings.cpp` — only bound `read_rice_parallel`.

### Changed

- **3→1 C++ read path merge:** Extracted a single `read_tensor_canonical()` in `fits_detail.h`
  and converted three read paths (`read_full_cached`, `read_full_nocache`, `FITSFile::read_tensor`)
  into thin wrappers, eliminating ~455 lines of duplication.
- **API naming consistency:** Renamed `read_image_canonical` → `read_tensor_canonical` and
  `FITSFile::read_image` → `FITSFile::read_tensor`, aligning C++ with the Python `read_tensor`/`write_tensor` API.
- **bswap+BZERO merge:** Merged the two-pass byte-swap and BZERO offset into a single `parallel_for`
  in the multi-byte mmap fast path. For unsigned images (uint16 with BZERO=32768, uint32 with
  BZERO=2147483648), `bswap + add` executes in one traversal instead of two.
- **Unsigned mmap fast path unlocked:** `_read_unsigned_image_if_needed` now defers to the C++
  path when `mmap=True`, letting `read_tensor_canonical` handle unsigned conventions natively
  (single-pass bswap+BZERO returning uint16/uint32 directly). Previously Python preempted C++
  by calling `read_full_raw` and doing a second offset pass — making the bswap+BZERO merge dead code.
  **uint32_2d: 8.3× faster (now beats fitsio); uint16_2d: 3.5× faster; 5 deficits eliminated.**
- **Vectorized string decode:** Replaced per-row Python `for` loops in `interop.py`
  (`to_pandas`, `to_arrow`), `table_hdu.py` (`get_string_column`, `to_fits`), and
  `table_hdu_ref.py` (`get_string_column`) with `np.char.decode()` + `np.char.rstrip()`
  for significant speedup on large string columns.
- **Deduplicated `fits_schema.py`:** `column_tnull_map()` delegates to `_iter_tfields_indexed()`
  instead of reimplementing the TTYPE/TNULL iteration loop.
- **Deduplicated unsigned dtype and TFORM parsing:** `_table/read.py` now delegates to
  `fits_schema.unsigned_column_dtypes_from_header()` and `fits_schema.iter_table_columns()`
  instead of reimplementing TZERO/unsigned detection and TTYPE/TFORM header walks.
- **Table schema fast path:** `table.schema()` skips data reads when `where=None`,
  inferring the Arrow schema directly from FITS TFORM header cards (≤1 header pass).
- **C++ source extraction:** Split `fits.cpp` (4552 lines) into `fits_detail.h`,
  `fits_file.h`/`.cpp`, `fits_rw.h`; split `table.cpp` (3432 lines) into
  `table_types.h`, `table_reader.h` (header-only), `table_mutation.h`/`.cpp`.
  Removed `extern "C"` linkage from table mutation functions to fix UB from
  C++ exceptions crossing C ABI boundaries. Removed dead declarations,
  unused types, stale comments, and double includes.
- **Merged cache stats:** `CacheManager.get_stats()` now pulls I/O engine metrics
  (`io_hits`, `io_misses`, `io_total_requests`) from the cache subsystem.
- **WHERE evaluator → Arrow compute:** `TableHDU.filter()` now builds a minimal Arrow
  table and delegates to `_where_mask_for_table` (pyarrow.compute native predicates)
  instead of running the old NumPy-based custom evaluator. The parser stays for
  C++ pushdown path compatibility.
- **Table read unification:** Extracted `_read_ranges_as_chunk` from `_read_cpp_numpy_table`
  into shared `_table/engine.py`, removing ~50 lines of duplicated code.
- **CI:** Added non-blocking `mypy src/` step to the GitHub Actions lint job.
- **Benchmark fairness fix:** fitsio is no longer unconditionally skipped — runs when
  `mmap=off` for fair buffered-read comparisons (449 fitsio OK rows in fits domain,
  180 in fitstable).
- `examples/example_image_dataset.py`: `optimize_for_dataset` + correct `pin_memory` when
  reading directly to CUDA.
- `scripts/run_exhaustive_bench_and_patch_docs.sh` skips rebuild when extension imports.

### Fixed

- **Vectorized NULL evaluation:** `_where.py` replaced Python-loop `np.array([v is None for v in val])`
  with vectorized `(val == None)` for element-wise null checks.
- Fixed unused imports in `tests/test_cache_config.py`.
- **Security:** Block CFITSIO pipe injection bypass via leading `!` prefix (`!|command`) in
  `check_fits_filename_security`; also enforced on unified cache open path.
- GPU `scale_on_device` preserves narrow integer H2D for FITS signed-byte (int8) and
  unsigned uint16/uint32 conventions instead of promoting through float32 or int64 on CPU.

### Added

- **Header:** O(N) construction for large dict inputs via keyed fast-path in `_set_card`
  (2000 keys ~0.002s locally vs ~2.5s pre-fix).
- **Jupyter:** Scrollable, sticky-header HTML repr for `Header` and `HDUList`.
- `tests/test_scale_on_device.py` — signed-byte, unsigned, and fitsio parity checks.
- Release gate includes `test_scale_on_device.py`.
- `.cursor/skills/release-api-freeze-review/` — pre-tag API/feature freeze audit workflow.
- **`_table/engine.py`** — shared C++ table read dispatch module with extracted
  `_read_ranges_as_chunk` helper (de-duplicated from `_read_cpp_numpy_table`).

### Performance notes

- Local `bench_ml_loader.py` diagnostic (30×512² float32, CPU, 2 epochs): Rice-compressed
  **1.12×** vs fitsio; uncompressed within ~4% (tune handle cache for your file count).
- Lab exhaustive refresh (`exhaustive_mmap_0.5.0b4_20260630_162835`, H100 MIG): **3626 rows**,
  **13 deficits** (down from 22). Integer CUDA gaps closed; remaining are marginal int8 (≤1.2×)
  and cold `large_uint32_2d` CPU vs astropy (~1.5×).
- User-profile refresh (`unsigned_mmap_fix_20260708`, CPU, mmap=on): **1,377 rows**,
  **25 deficits**. `torchfits_specialized` uint32_2d now beats fitsio (was 5–10× behind);
  uint16_2d at ~1.6× vs fitsio (was ~5×). Remaining deficits dominated by medium-size
  unsigned reads and compressed HCOMPRESS.
  Torchfits dominates table I/O (886×–2,318× vs astropy), image reads (7.92× vs astropy,
  1.76× vs fitsio on large float32), and repeated cutouts (17× vs astropy, 1.09× vs fitsio).

## [0.5.0b4] - 2026-06-30

### Changed

- Centralized FITS binary-table header parsing in `fits_schema` (TFORM/VLA/string/bit/unsigned).
- `table.read` no longer recurses for `where=`; strategy lives in `_table_engine.read_policy`.
- Table C++ handle caches moved to `_table.cache`; I/O cache invalidation no longer depends on
  importing `torchfits.table`.
- README highlights 0.5.0 features and published benchmark speedups; API docs document table
  backends and `where=` tuning environment variables.

### Added

- Unit tests for `fits_schema`, table where-read policy, and runnable example scripts.
- Public `torchfits.table.TABLE_BACKENDS` constant.
- `pixi run release-gate` task matching the release checklist parity/docs/examples gates.

## [0.5.0b3] - 2026-06-30

### Changed

- Refocused torchfits as a FITS I/O package: images, HDUs, headers, checksums,
  compression, FITS tables, caching, and table interop.
- Removed stale public claims that torchfits owns WCS, sphere geometry, HEALPix,
  sky-domain simulation, or training pipelines. Those domains belong outside
  torchfits.
- Added a roadmap and compatibility matrix that distinguish supported, partial,
  unsupported, and out-of-scope behavior.
- Replaced broad parity claims with test-backed parity tiers for common fitsio,
  Astropy, and selected CFITSIO-backed workflows.

### Added

- Extended benchmark matrix: native **uint16/uint32** 2D image fixtures, **typed**
  binary tables (BIT/complex/string columns), and **ASCII** table fixtures.
- `bench_all.py --mmap-matrix` runs mmap-on and mmap-off passes in one CSV so the
  I/O transport table can populate both `disk→CPU` and `disk→RAM→CPU` (plus GPU
  `disk→CPU→GPU` / `disk→RAM→GPU` when CUDA/MPS is available).
- `scripts/run_exhaustive_bench_and_patch_docs.sh` for lab-profile `bench-all` on
  CUDA/MPS hardware with automatic `docs/benchmarks.md` refresh.
- Lab CUDA benchmark snapshot `exhaustive_mmap_0.5.0b3_20260630_063118` (3474 rows,
  mmap on+off matrix, 720 GPU transport rows on H100).
- `docs/parity.md` for the public compatibility matrix.
- Astropy upstream smoke coverage for common image, HDU, compressed-image,
  table, ASCII table, VLA, complex column, and scaled-image workflows.
- Documentation integrity checks for stale WCS/sphere/HEALPix ownership claims.
- Supported-status promotion for in-place mmap table updates on COMPLEX
  (`1C`/`1M`), BIT (`8X`), and fixed-width STRING (`12A`-style) columns.
  `torchfits.table.update_rows(..., mmap=True)` now writes these column
  types correctly on disk. Verified via raw byte inspection and an astropy
  upstream-reader roundtrip. VLA columns remain explicitly unsupported in the mmap
  fast path by design.
- Astropy and fitsio upstream smoke coverage that exercises the
  COMPLEX / BIT / fixed-width STRING mmap-update parity shift,
  including right-padding to the declared column width and verification
  vs the upstream readers. The 8A-string assertion falls back to
  astropy because the local fitsio upstream misdecodes updated `8A`
  rows (the on-disk bytes are bit-exact to the expected layout; this
  is an upstream-reader limitation, not a torchfits writer bug).
- `tests/test_astropy_upstream_smoke.py::test_astropy_compimage_compression_variants_match_torchfits`
  exercising additional `astropy.io.fits.CompImageHDU` compression
  variants (RICE / HCOMPRESS / PLIO) round-tripped against torchfits.

### Fixed

- API docs and install guide now reference `torchfits.cache` for cache tuning
  (`configure_for_environment`, `get_cache_stats`, `clear_cache`) and the root
  I/O helpers `get_cache_performance` / `clear_file_cache` where appropriate.
- Roadmap mmap limitations updated to match the parity matrix (BIT and
  fixed-width STRING mmap updates are supported; VLA and scaled columns remain
  partial).

### Removed

- Dataset/training helper namespace from the torchfits package contract.

## [0.5.0b2] - 2026-06-30

### Fixed

- Patched `fitstable` specialised column projection and row slicing benchmark errors due to invalid `policy` argument.
- Cleaned up C++ build flags in `bench-gpu` to remove strict CUDA and Torch pins.
- Audited C++ codebase for potential memory leaks, redundant hardware heuristics, and API bounds.

### Added

- Restored core FITS benchmarks from v0.3.2: ML DataLoader performance (`bench_ml_loader.py`) and GPU Memory usage/leak validator (`bench_gpu_memory.py`).
- Added exhaustive progress print logging during benchmark execution.
- Added persistent cutout / multi-cutout repeated read benchmarks (`SubsetReader` / `open_subset_reader`) for both CPU and GPU.
- Added `read_tensor` for reading N-dimensional arrays (1D spectra, 2D images, 3D cubes, xD arrays) directly to a single PyTorch `Tensor`.
- Added `write_tensor` as the specialized PyTorch-native writer for writing single PyTorch `Tensor`s directly to FITS files.

### Deprecated

- Deprecated `read_image` in favor of the more general and PyTorch-native `read_tensor`.

## [0.5.0b1] - 2026-06-29

### Changed

- Repository home: `github.com/astroai/torchfits`.
- Default development Python is **3.13** (pixi); supported install range remains **3.10+**.
- Development Status classifier promoted to **Beta**.
- Removed obsolete diagnostic benchmarks, scratch scripts, and legacy HEALPix/WCS artifacts.
- CI rewritten: ruff-only lint, multi-OS/Python test matrix, CFITSIO vendoring via `extern/VERSIONS.txt`.
- Wheel builds: portable flags (no `-march=native`), `cp310`–`cp313` on macOS and Linux.

### Added

- GPU I/O transport benchmark rows (`bench_gpu_transports.py`) with **MPS** on Apple Silicon and **CUDA** on Linux.
- `pixi run bench-mps` for Apple Silicon accelerator benchmarks.
- Automated benchmark report workflow (`.github/workflows/bench-report.yml`).
- `scripts/render_bench_deficits.py` for documenting performance deficits without fixing them.

### Fixed

- Table mutations now invalidate FITS path caches via internal `io` helper (fixes `torchfits._invalidate_path_caches` AttributeError).

## Earlier releases

Earlier 0.1.x through 0.3.x releases included broader experimental astronomy
domains. The current package contract is FITS I/O only; consult the current
README, API reference, roadmap, and parity matrix for supported behavior.

[0.1.0]: https://github.com/astroai/torchfits/releases/tag/v0.1.0
[0.1.1]: https://github.com/astroai/torchfits/releases/tag/v0.1.1
[0.2.0]: https://github.com/astroai/torchfits/releases/tag/v0.2.0
[0.2.1]: https://github.com/astroai/torchfits/releases/tag/v0.2.1
[0.3.0]: https://github.com/astroai/torchfits/releases/tag/v0.3.0
[0.3.1]: https://github.com/astroai/torchfits/releases/tag/v0.3.1
[Unreleased]: https://github.com/astroai/torchfits/compare/v0.6.0b1...HEAD
[0.6.0b1]: https://github.com/astroai/torchfits/releases/tag/v0.6.0b1
[0.5.0b4]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b4
[0.5.0b3]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b3
[0.5.0b2]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b2
[0.5.0b1]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b1
[0.3.2]: https://github.com/astroai/torchfits/releases/tag/v0.3.2
