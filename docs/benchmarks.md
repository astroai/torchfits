# Benchmarks

`torchfits` benchmarks cover FITS image I/O and FITS table I/O. WCS, HEALPix,
sphere, and sky-domain benchmarks are out of scope for this repository.

## Comparison Targets

| Domain | torchfits module | Compared against |
|---|---|---|
| FITS image I/O | `torchfits.read` / `torchfits.write` | `astropy.io.fits`, `fitsio` |
| FITS table I/O | `torchfits.table` | `astropy.io.fits`, `fitsio` |

## Methodology

Each case measures median wall-clock time over multiple repetitions. Cases are
grouped into two families:

- **smart** — the idiomatic high-level API, such as `torchfits.read()` vs
  `astropy.io.fits.getdata()` plus `torch.from_numpy()`.
- **specialized** — lower-level paths with explicit mmap, compression, or table
  streaming controls.

Fairness controls:

- Rows with mismatched mmap behavior are marked `SKIPPED` and excluded from
  rankings.
- FITS comparators must be official released distributions.
- Warm-cache and cold-cache profiles are kept separate.

## Correctness Gates

| Gate | Command | Validates |
|---|---|---|
| fitsio parity | `pixi run pytest tests/test_fitsio_upstream_smoke.py -q` | Common fitsio image, header, table, compression, and checksum workflows |
| Astropy parity | `pixi run pytest tests/test_astropy_upstream_smoke.py -q` | Common Astropy HDU, header, image, compressed-image, table, and scaled-data workflows |
| Package isolation | `pixi run pytest tests/test_package_isolation.py tests/test_docs_integrity.py -q` | Clean FITS-only package boundary and docs contract |

## Reproducing

```bash
pixi run bench-fits
pixi run bench-fitstable
pixi run bench-all
# Full transport matrix (mmap on + off, doubles CPU rows; GPU rows for both when CUDA/MPS):
pixi run -e bench-gpu python benchmarks/bench_all.py --profile lab --scope all --mmap-matrix
```

For focused FITS partitions:

```bash
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(tiny_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(small_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(medium_|large_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(scaled_|compressed_|mef_)'
```

## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | fits / fitstable | FITS benchmark orchestrator |
| `bench_fits_io.py` | fits | Image I/O across dtypes, sizes, compression, scaling, MEF, and cutouts |
| `bench_fitstable_io.py` | fitstable | Table I/O across row counts, schemas, projection, row slicing, predicates, and streaming |
| `bench_fast.py` | fits | Low-level image/header fast-path checks |
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |
| `bench_gpu_transports.py` | fits (GPU) | CUDA/MPS image reads, cutouts, repeated cutouts (`disk→CPU→GPU` / `disk→RAM→GPU` rows) |
| `bench_ml_loader.py` | fits (diagnostic) | PyTorch `DataLoader` throughput (not merged into `bench-all` CSV) |
| `bench_gpu_memory.py` | fits (diagnostic) | GPU memory/leak checks (non-gating) |

## Coverage matrix

What the exhaustive `bench-all` suite measures today, and what is intentionally out of
scope or not yet wired into the published tables.

| Dimension | Covered? | Where | Gap / caveat |
|---|---|---|---|
| Backends (torchfits / astropy / fitsio) | Yes | `bench_fits_io.py`, `bench_fitstable_io.py` | `fitsio` often excluded from mmap-fairness summaries; **uint** image comparators may be torchfits-only when astropy requires buffered fallback |
| CPU vs GPU device | Partial | CPU: full matrix; GPU: image reads only | GPU requires CUDA/MPS hardware (`pixi run -e bench-gpu bench-gpu` or local CUDA); **CI weekly bench is CPU-only** |
| I/O transport `disk→RAM→CPU` | Yes | `bench-all` mmap-on pass | Median mixes many ops/sizes — coarse aggregate |
| I/O transport `disk→CPU` (non-mmap) | Yes | `bench-all --mmap-matrix` mmap-off pass | Buffered host decode; use `--mmap-matrix` (or `--no-mmap`) to populate |
| I/O transport `disk→RAM→GPU` | Partial | `bench_gpu_transports.py` (mmap on) | Image `read_full`, cutouts, repeated cutouts only; **no tables** |
| I/O transport `disk→CPU→GPU` | Partial | `bench_gpu_transports.py` (mmap off) | Same GPU ops with buffered host decode + H2D copy |
| I/O transport `disk→GPU` | No | — | No Python FITS backend supports true disk→GPU (GPUDirect / cuFile); row stays empty by design |
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D fixtures (`small/medium/large_uint*_2d`); unsigned via BZERO also in `scaled_*` |
| Image dimensions / sizes | Yes | tiny → large categories | Large 3D cubes skipped (size cap) |
| Compression | Yes | gzip, rice, hcompress, plio | Write-side compression not benchmarked |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` repeated cutouts | MEF random ext reads only on selected fixtures |
| Multi-extension (MEF) | Yes | `mef_*`, `multi_mef_10ext` | — |
| Table full read / projection / slice | Yes | `bench_fitstable_io.py` | — |
| Table predicate / scan | Yes | `predicate_filter`, `scan_count` | Arrow `table.scan` streaming not identical to `scan_count` row |
| Table schemas | Partial | mixed / narrow / wide / varlen | **typed** (BIT/complex/string) and **ascii** table fixtures at selected row counts |
| Table GPU | No | — | All comparators are CPU-resident; not a meaningful apples-to-apples GPU row today |
| Writes | No | — | Read-heavy suite; write parity validated in tests, not bench CSV |
| FITS physical units (BUNIT/TUNIT) | No | — | Metadata semantics, not I/O transport — covered by parity tests only |
| ML DataLoader pattern | Diagnostic | `bench_ml_loader.py` | Not merged into `docs/benchmarks.md` tables; README cites local CPU diagnostic (Rice **1.12×** vs fitsio, 30×512² files) |

### Why the I/O transport table looks sparse on GPU

1. **`disk→GPU` is always empty** — every backend decodes on the host first (CFITSIO /
   astropy / fitsio into host RAM), then copies with `.to(device)`. `device="cuda"` does
   **not** mean a native disk→GPU bypass (that would require GPUDirect Storage / cuFile,
   which none of these Python FITS stacks use).
2. **`disk→CPU→GPU` vs `disk→RAM→GPU`** — the former is the mmap-off GPU path (buffered
   host decode + H2D); the latter is mmap-on decode + H2D. Both still touch host memory.
3. **`disk→RAM→GPU` is populated only when GPU rows exist in the CSV** — produced by
   `bench_gpu_transports.py` inside `bench-all` when `torch.cuda.is_available()` or MPS
   is available. GitHub Actions `bench-report` installs **CPU PyTorch**, so weekly CI
   runs will **not** refresh GPU cells; the published CUDA numbers come from a manual
   lab run (`exhaustive_mmap_0.5.0b4_20260630_162835`, via `pixi run -e bench-gpu bench-exhaustive`).
4. **FITS tables have no GPU transport rows** — astropy/fitsio/torchfits table paths are
   CPU-buffered; GPU table benchmarks would mostly measure PyTorch copy overhead, not FITS
   decode, and are deliberately omitted.

### GPU integer dtype comparisons (0.5.0+)

The **deficit table** below compares default
`torchfits.read(..., scale_on_device=True)` against `torch.from_numpy(fitsio.read(...)).to(cuda)`.
That pairing is **not dtype-equivalent** for generic scaled integer FITS (see table).
After 0.5.0 narrow-integer H2D fixes, the lab snapshot dropped from **22 → 13** deficits;
remaining gaps are mostly **≤20% on tiny CUDA int8** or **cold CPU uint32** vs astropy.

| FITS convention | fitsio @ CUDA | default `read` @ CUDA (before 0.5.0 fixes) | 0.5.0 behavior |
|---|---|---|---|
| Signed byte (BITPIX=8, BZERO=-128) | native `int8` H2D | promoted to `float32` on GPU | narrow `int8` H2D + offset on device |
| Unsigned uint16/uint32 (BZERO offset) | native uint H2D | int64 widen on CPU, then cast | narrow storage H2D, offset on device |
| Generic BSCALE/BZERO scaling | often native storage dtype | `float32` on device (intentional for ML) | unchanged `float32` on device |

For apples-to-apples integer GPU timing, the exhaustive suite also records
**`torchfits_dtype_fair_device`** (`read_tensor(..., raw_scale=True)`).

**Training loops:** cold single-shot reads can lose to astropy on native uint32 CPU;
call `torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=…)` before
`DataLoader` epochs so handle caches stay warm (see `examples/example_image_dataset.py`).

### Refreshing GPU numbers

```bash
# Linux + NVIDIA
pixi run -e bench-gpu bench-gpu

# Apple Silicon (MPS transport rows; separate from CUDA lab numbers)
pixi run bench-mps

# Re-render docs from the merged CSV
pixi run -e bench-gpu bench-exhaustive
# or, from an existing run directory:
pixi run bench-table-render -- --csv benchmarks_results/<run-id>/results.csv
python scripts/patch_bench_docs.py --csv ... --deficits ... --run-id <run-id>
```

## I/O Transport × Backend

> **GPU summary:** Image **`disk→CPU→GPU`** and **`disk→RAM→GPU`** rows appear only when the benchmark CSV was
> produced on CUDA or MPS hardware. **`disk→GPU`** is intentionally empty (unsupported by
> all backends). **Table GPU transports are not benchmarked.** CI weekly `bench-report`
> uses CPU PyTorch and will not update GPU cells.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/20260708_230314/results.csv` (CPU mmap-on run.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | — | — | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.06 ms` (n=269) | `0.46 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | — | — | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.05 ms` (n=180) | `2.13 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |
<!-- BENCH_IOPATH_END -->

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`,
  `disk→CPU→GPU`, `disk→RAM→GPU`).
- Columns are **backends** (`torchfits` / `astropy` / `fitsio` / `cfitsio-direct`).
- `cfitsio` is the C engine used by `torchfits`; no standalone `cfitsio`-only
  benchmark row is generated by `bench-all`, so the cell is documented as
  "engine exposed under `torchfits`".
- Cell `n=` counts comparable OK rows in the bucket; `—` indicates the
  bucket is empty (no rows match, or rows were excluded under
  `strict_mmap_fairness` in the original `bench-all` summary).
- Median is computed over heterogeneous operations (`read_full`,
  `cutout_100x100`, `header_read`, `predicate_filter`, `projection`,
  `row_slice`, etc.) and payload sizes; treat the per-cell ms as a
  coarse representative number, not a precise benchmark.

## Performance Highlights

<!-- BENCH_HIGHLIGHTS_BEGIN -->
The following table showcases median wall-clock execution times of key representative FITS benchmarks.
In almost all core I/O paths, `torchfits` is significantly faster than standard astronomical tools, with extra performance wins from persistent handle caches and direct-to-device transfers.

| Benchmark Case | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **1.76 ms** | 1.76 ms | 9.17 ms | 2.86 ms | **5.20x** | **1.62x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **7.18 ms** | 7.20 ms | 19.03 ms | 7.25 ms | **2.65x** | **1.01x** |
| Repeated Cutouts (50x 100x100) | CPU | **3.18 ms** | 2.97 ms | 26.81 ms | 3.36 ms | **9.02x** | **1.13x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **47.0 μs** | 48.5 μs | 4.07 ms | 43.48 ms | **86.52x** | **924.74x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **48.5 μs** | 50.2 μs | 2.03 ms | 123.42 ms | **41.75x** | **2542.95x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 88.9 μs | 1.47 ms | 154.7 μs | **16.55x** | **1.74x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **23.58 ms** | 23.62 ms | 45.80 ms | 26.45 ms | **1.94x** | **1.12x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 86.8 μs | 1.48 ms | 154.2 μs | **17.02x** | **1.78x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **20.17 ms** | 20.26 ms | 72.17 ms | 23.04 ms | **3.58x** | **1.14x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 91.1 μs | 1.53 ms | 172.3 μs | **16.84x** | **1.89x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **45.51 ms** | 26.29 ms | 29.77 ms | 25.52 ms | **1.13x** | **0.97x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **714.9 μs** | 702.3 μs | 5.84 ms | 813.2 μs | **8.32x** | **1.16x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 91.9 μs | 1.53 ms | 172.4 μs | **16.64x** | **1.88x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **7.18 ms** | 7.20 ms | 19.03 ms | 7.25 ms | **2.65x** | **1.01x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 47.1 μs | 401.2 μs | 55.7 μs | **8.52x** | **1.18x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **448.8 μs** | 455.2 μs | 1.06 ms | 766.4 μs | **2.35x** | **1.71x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 47.3 μs | 422.7 μs | 54.5 μs | **8.93x** | **1.15x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **1.76 ms** | 1.76 ms | 9.17 ms | 2.86 ms | **5.20x** | **1.62x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 44.2 μs | 401.0 μs | 54.0 μs | **9.07x** | **1.22x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **806.5 μs** | 810.3 μs | 1.67 ms | 1.23 ms | **2.07x** | **1.53x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 45.5 μs | 426.2 μs | 55.4 μs | **9.36x** | **1.22x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **9.05 ms** | 8.82 ms | 18.31 ms | 10.61 ms | **2.08x** | **1.20x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 43.3 μs | 396.5 μs | 54.1 μs | **9.16x** | **1.25x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **631.1 μs** | 638.9 μs | 751.8 μs | 380.2 μs | **1.19x** | **0.60x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 44.8 μs | 423.9 μs | 55.0 μs | **9.45x** | **1.23x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **2.50 ms** | 2.49 ms | 1.76 ms | 1.29 ms | **0.71x** | **0.52x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 42.7 μs | 395.8 μs | 52.2 μs | **9.27x** | **1.22x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **455.1 μs** | 455.4 μs | 1.05 ms | 753.5 μs | **2.32x** | **1.66x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 43.8 μs | 416.8 μs | 56.4 μs | **9.52x** | **1.29x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **1.77 ms** | 1.77 ms | 9.14 ms | 2.87 ms | **5.16x** | **1.62x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 43.0 μs | 385.3 μs | 53.6 μs | **8.96x** | **1.25x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **802.4 μs** | 801.2 μs | 1.68 ms | 1.28 ms | **2.10x** | **1.59x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 45.0 μs | 417.4 μs | 55.8 μs | **9.28x** | **1.24x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **8.98 ms** | 8.86 ms | 18.36 ms | 10.52 ms | **2.07x** | **1.19x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 46.4 μs | 440.1 μs | 57.8 μs | **9.49x** | **1.24x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **164.0 μs** | 150.8 μs | — | 207.7 μs | **—** | **1.38x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 47.8 μs | 463.0 μs | 61.3 μs | **9.69x** | **1.28x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **523.5 μs** | 527.5 μs | — | 700.0 μs | **—** | **1.34x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 48.2 μs | 464.6 μs | 61.8 μs | **9.64x** | **1.28x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **2.46 ms** | 2.46 ms | — | 1.53 ms | **—** | **0.62x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 47.5 μs | 474.4 μs | 62.5 μs | **9.98x** | **1.32x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **1.75 ms** | 1.76 ms | — | 3.31 ms | **—** | **1.89x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 43.5 μs | 400.3 μs | 54.1 μs | **9.19x** | **1.24x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **76.4 μs** | 80.7 μs | 472.3 μs | 144.6 μs | **6.18x** | **1.89x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 46.0 μs | 421.5 μs | 55.2 μs | **9.16x** | **1.20x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **487.8 μs** | 496.0 μs | 1.13 ms | 828.2 μs | **2.31x** | **1.70x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 47.5 μs | 437.5 μs | 56.6 μs | **9.20x** | **1.19x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **710.1 μs** | 713.3 μs | 1.48 ms | 1.21 ms | **2.08x** | **1.71x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 45.9 μs | 408.1 μs | 53.8 μs | **8.89x** | **1.17x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **122.8 μs** | 127.9 μs | 546.3 μs | 175.8 μs | **4.45x** | **1.43x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 47.6 μs | 419.9 μs | 56.2 μs | **8.82x** | **1.18x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **840.0 μs** | 846.1 μs | 1.75 ms | 1.29 ms | **2.08x** | **1.54x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 48.1 μs | 442.8 μs | 58.1 μs | **9.21x** | **1.21x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **1.28 ms** | 1.28 ms | 2.52 ms | 1.97 ms | **1.98x** | **1.54x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 43.4 μs | 394.5 μs | 54.2 μs | **9.08x** | **1.25x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **104.5 μs** | 109.8 μs | 441.6 μs | 106.0 μs | **4.23x** | **1.01x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 46.0 μs | 421.6 μs | 55.1 μs | **9.17x** | **1.20x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **663.1 μs** | 674.1 μs | 779.6 μs | 382.6 μs | **1.18x** | **0.58x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 46.1 μs | 432.6 μs | 57.9 μs | **9.38x** | **1.26x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **1.01 ms** | 1.02 ms | 973.5 μs | 572.7 μs | **0.96x** | **0.57x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 43.6 μs | 396.3 μs | 53.7 μs | **9.08x** | **1.23x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **78.6 μs** | 81.1 μs | 475.2 μs | 143.7 μs | **6.04x** | **1.83x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 45.6 μs | 411.8 μs | 55.9 μs | **9.04x** | **1.23x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **495.0 μs** | 496.2 μs | 1.11 ms | 834.7 μs | **2.25x** | **1.69x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 46.4 μs | 434.2 μs | 57.5 μs | **9.36x** | **1.24x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **771.6 μs** | 761.1 μs | 1.49 ms | 1.20 ms | **1.96x** | **1.58x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 42.6 μs | 403.8 μs | 53.5 μs | **9.48x** | **1.26x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **126.0 μs** | 129.6 μs | 540.3 μs | 185.2 μs | **4.29x** | **1.47x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 45.7 μs | 423.4 μs | 55.7 μs | **9.26x** | **1.22x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **853.7 μs** | 858.9 μs | 1.78 ms | 1.35 ms | **2.08x** | **1.58x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 46.7 μs | 431.6 μs | 57.3 μs | **9.24x** | **1.23x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **1.30 ms** | 1.30 ms | 2.53 ms | 2.05 ms | **1.95x** | **1.58x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 45.1 μs | 434.8 μs | 58.7 μs | **9.64x** | **1.30x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **48.7 μs** | 52.6 μs | — | 94.0 μs | **—** | **1.93x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 47.7 μs | 458.2 μs | 60.9 μs | **9.60x** | **1.28x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **159.4 μs** | 163.7 μs | — | 224.1 μs | **—** | **1.41x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 52.3 μs | 479.7 μs | 63.5 μs | **9.18x** | **1.22x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **226.9 μs** | 236.9 μs | — | 319.7 μs | **—** | **1.41x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 48.0 μs | 459.9 μs | 60.6 μs | **9.58x** | **1.26x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **654.6 μs** | 663.8 μs | — | 432.1 μs | **—** | **0.66x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 48.9 μs | 475.0 μs | 62.5 μs | **9.71x** | **1.28x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **504.7 μs** | 517.7 μs | — | 897.8 μs | **—** | **1.78x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 55.6 μs | 666.9 μs | 68.4 μs | **12.00x** | **1.23x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **167.5 μs** | 157.6 μs | — | 251.8 μs | **—** | **1.60x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 54.2 μs | 679.3 μs | 68.6 μs | **12.54x** | **1.27x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **47.0 μs** | 54.8 μs | — | 130.8 μs | **—** | **2.78x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **35.7 μs** | 34.5 μs | — | 203.2 μs | **—** | **5.90x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 55.2 μs | 672.4 μs | 70.5 μs | **12.18x** | **1.28x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **4.97 ms** | 5.00 ms | — | 6.83 ms | **—** | **1.37x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **45.1 μs** | 54.2 μs | — | 185.1 μs | **—** | **4.11x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **3.18 ms** | 2.97 ms | 26.81 ms | 3.36 ms | **9.02x** | **1.13x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 49.8 μs | 477.8 μs | 60.5 μs | **9.59x** | **1.22x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **2.40 ms** | 2.40 ms | — | 2.92 ms | **—** | **1.22x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 48.4 μs | 469.0 μs | 60.3 μs | **9.69x** | **1.25x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **633.2 μs** | 638.9 μs | — | 802.6 μs | **—** | **1.27x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 56.1 μs | 490.8 μs | 67.9 μs | **8.75x** | **1.21x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **75.4 μs** | 82.6 μs | — | 127.2 μs | **—** | **1.69x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 53.4 μs | 414.5 μs | 63.6 μs | **7.77x** | **1.19x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **39.7 μs** | 46.3 μs | 411.7 μs | 86.5 μs | **10.38x** | **2.18x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 49.8 μs | 438.2 μs | 69.8 μs | **8.80x** | **1.40x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **65.6 μs** | 70.8 μs | 466.9 μs | 121.1 μs | **7.11x** | **1.84x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 47.8 μs | 440.8 μs | 58.9 μs | **9.23x** | **1.23x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **105.7 μs** | 111.5 μs | 537.6 μs | 184.8 μs | **5.09x** | **1.75x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 43.1 μs | 404.1 μs | 53.8 μs | **9.37x** | **1.25x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **60.0 μs** | 60.0 μs | 424.7 μs | 90.5 μs | **7.08x** | **1.51x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 46.0 μs | 419.8 μs | 55.4 μs | **9.13x** | **1.21x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **105.1 μs** | 105.1 μs | 503.1 μs | 146.4 μs | **4.79x** | **1.39x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 46.8 μs | 441.1 μs | 57.9 μs | **9.43x** | **1.24x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **181.1 μs** | 185.4 μs | 660.2 μs | 259.4 μs | **3.65x** | **1.43x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 42.0 μs | 404.9 μs | 52.7 μs | **9.64x** | **1.26x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **52.1 μs** | 54.6 μs | 405.9 μs | 81.8 μs | **7.80x** | **1.57x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 45.0 μs | 415.7 μs | 54.0 μs | **9.23x** | **1.20x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **94.2 μs** | 96.1 μs | 439.9 μs | 95.7 μs | **4.67x** | **1.02x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 46.1 μs | 442.2 μs | 56.3 μs | **9.58x** | **1.22x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **157.1 μs** | 150.6 μs | 480.7 μs | 121.0 μs | **3.19x** | **0.80x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 43.4 μs | 394.9 μs | 55.1 μs | **9.09x** | **1.27x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **41.9 μs** | 46.4 μs | 410.8 μs | 86.6 μs | **9.80x** | **2.07x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 44.0 μs | 420.8 μs | 54.9 μs | **9.56x** | **1.25x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **64.7 μs** | 69.7 μs | 460.0 μs | 122.8 μs | **7.12x** | **1.90x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 46.2 μs | 440.4 μs | 57.0 μs | **9.54x** | **1.23x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **107.1 μs** | 111.0 μs | 542.6 μs | 184.9 μs | **5.07x** | **1.73x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 43.4 μs | 393.0 μs | 52.9 μs | **9.05x** | **1.22x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **60.3 μs** | 60.9 μs | 417.4 μs | 90.5 μs | **6.92x** | **1.50x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 43.4 μs | 425.6 μs | 55.2 μs | **9.81x** | **1.27x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **99.4 μs** | 107.1 μs | 513.4 μs | 144.3 μs | **5.16x** | **1.45x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 45.6 μs | 445.9 μs | 55.7 μs | **9.79x** | **1.22x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **199.4 μs** | 184.9 μs | 655.2 μs | 268.7 μs | **3.54x** | **1.45x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 47.8 μs | 441.9 μs | 57.9 μs | **9.24x** | **1.21x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **36.5 μs** | 44.1 μs | — | 82.9 μs | **—** | **2.27x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 48.3 μs | 466.7 μs | 60.3 μs | **9.67x** | **1.25x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **44.0 μs** | 50.9 μs | — | 89.7 μs | **—** | **2.04x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 49.1 μs | 484.9 μs | 63.0 μs | **9.88x** | **1.28x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **58.7 μs** | 62.5 μs | — | 101.8 μs | **—** | **1.73x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 47.9 μs | 461.9 μs | 60.8 μs | **9.65x** | **1.27x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **88.9 μs** | 95.1 μs | — | 101.7 μs | **—** | **1.14x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 47.6 μs | 468.8 μs | 60.8 μs | **9.84x** | **1.28x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **105.8 μs** | 85.5 μs | — | 124.6 μs | **—** | **1.46x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 46.3 μs | 414.7 μs | 55.0 μs | **8.95x** | **1.19x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **62.4 μs** | 69.1 μs | 480.9 μs | 120.0 μs | **7.70x** | **1.92x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 46.1 μs | 421.8 μs | 55.4 μs | **9.16x** | **1.20x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **60.5 μs** | 67.9 μs | 462.4 μs | 117.1 μs | **7.65x** | **1.94x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 45.2 μs | 416.8 μs | 55.4 μs | **9.22x** | **1.23x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **62.8 μs** | 68.0 μs | 460.3 μs | 119.0 μs | **7.33x** | **1.89x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 45.4 μs | 425.1 μs | 54.9 μs | **9.37x** | **1.21x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **63.5 μs** | 68.0 μs | 464.5 μs | 117.5 μs | **7.31x** | **1.85x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 44.6 μs | 419.0 μs | 55.7 μs | **9.39x** | **1.25x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **61.8 μs** | 69.0 μs | 465.2 μs | 118.7 μs | **7.53x** | **1.92x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 44.8 μs | 402.5 μs | 52.3 μs | **8.98x** | **1.17x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **33.9 μs** | 38.8 μs | 407.1 μs | 79.8 μs | **12.02x** | **2.36x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 44.7 μs | 423.4 μs | 54.8 μs | **9.48x** | **1.23x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **39.3 μs** | 46.8 μs | 428.4 μs | 82.8 μs | **10.91x** | **2.11x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 46.8 μs | 452.4 μs | 57.0 μs | **9.66x** | **1.22x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **40.0 μs** | 48.0 μs | 436.5 μs | 83.8 μs | **10.90x** | **2.09x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 44.5 μs | 393.4 μs | 52.7 μs | **8.83x** | **1.18x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **46.1 μs** | 51.4 μs | 410.5 μs | 80.2 μs | **8.89x** | **1.74x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 45.5 μs | 419.7 μs | 54.3 μs | **9.22x** | **1.19x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **55.6 μs** | 56.5 μs | 432.1 μs | 84.4 μs | **7.77x** | **1.52x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 47.9 μs | 447.6 μs | 56.8 μs | **9.34x** | **1.19x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **54.1 μs** | 57.7 μs | 445.9 μs | 85.8 μs | **8.24x** | **1.59x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 43.2 μs | 399.3 μs | 52.7 μs | **9.25x** | **1.22x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **45.2 μs** | 49.9 μs | 405.4 μs | 78.4 μs | **8.97x** | **1.73x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 45.9 μs | 416.3 μs | 55.1 μs | **9.07x** | **1.20x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **47.7 μs** | 54.9 μs | 420.8 μs | 84.1 μs | **8.82x** | **1.76x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 48.0 μs | 437.4 μs | 56.9 μs | **9.11x** | **1.18x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **49.1 μs** | 54.4 μs | 430.6 μs | 81.9 μs | **8.77x** | **1.67x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 44.5 μs | 402.0 μs | 54.3 μs | **9.04x** | **1.22x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **33.9 μs** | 38.0 μs | 401.0 μs | 78.8 μs | **11.83x** | **2.33x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 45.6 μs | 425.3 μs | 55.1 μs | **9.32x** | **1.21x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **38.7 μs** | 45.7 μs | 416.5 μs | 81.6 μs | **10.77x** | **2.11x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 45.6 μs | 440.4 μs | 57.1 μs | **9.65x** | **1.25x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **40.2 μs** | 46.2 μs | 435.2 μs | 82.9 μs | **10.83x** | **2.06x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 45.0 μs | 403.8 μs | 53.9 μs | **8.98x** | **1.20x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **45.1 μs** | 53.0 μs | 403.3 μs | 79.7 μs | **8.94x** | **1.77x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 46.3 μs | 414.5 μs | 54.5 μs | **8.94x** | **1.18x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **56.0 μs** | 55.8 μs | 427.8 μs | 83.4 μs | **7.66x** | **1.49x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 45.7 μs | 444.7 μs | 57.5 μs | **9.73x** | **1.26x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **54.6 μs** | 58.7 μs | 445.6 μs | 90.7 μs | **8.17x** | **1.66x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 45.3 μs | 440.6 μs | 57.5 μs | **9.72x** | **1.27x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **35.5 μs** | 41.5 μs | — | 79.6 μs | **—** | **2.24x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 49.7 μs | 461.8 μs | 59.4 μs | **9.30x** | **1.20x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **36.1 μs** | 44.6 μs | — | 81.5 μs | **—** | **2.26x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 49.5 μs | 464.8 μs | 69.7 μs | **9.39x** | **1.41x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **36.7 μs** | 46.1 μs | — | 81.6 μs | **—** | **2.22x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **680.1 μs** | 129.0 μs | 3.35 ms | 5.18 ms | **25.95x** | **40.16x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **47.5 μs** | 49.3 μs | 8.11 ms | 5.14 ms | **170.55x** | **108.23x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **45.2 μs** | 48.1 μs | 1.42 ms | 5.12 ms | **31.46x** | **113.34x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **46.9 μs** | 50.4 μs | 1.75 ms | 2.38 ms | **37.23x** | **50.69x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **70.2 μs** | 49.9 μs | 2.72 ms | 2.00 ms | **54.48x** | **40.13x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **472.4 μs** | 90.9 μs | 2.06 ms | 757.6 μs | **22.63x** | **8.33x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **48.8 μs** | 50.4 μs | 2.29 ms | 757.9 μs | **46.85x** | **15.52x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **46.9 μs** | 52.5 μs | 1.38 ms | 746.8 μs | **29.31x** | **15.92x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **49.2 μs** | 51.0 μs | 1.76 ms | 452.2 μs | **35.71x** | **9.20x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **71.5 μs** | 50.7 μs | 1.63 ms | 370.3 μs | **32.04x** | **7.30x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **12.27 ms** | 6.31 ms | 73.51 ms | 325.71 ms | **11.64x** | **51.58x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **51.4 μs** | 50.4 μs | 5.36 ms | 42.13 ms | **106.40x** | **836.69x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **49.3 μs** | 49.0 μs | 33.55 ms | 523.79 ms | **684.29x** | **10684.04x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **49.0 μs** | 51.1 μs | 6.67 ms | 123.53 ms | **136.22x** | **2521.63x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **83.0 μs** | 53.8 μs | 5.74 ms | 122.86 ms | **106.66x** | **2282.58x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **1.92 ms** | 709.0 μs | 8.33 ms | 26.83 ms | **11.75x** | **37.84x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **47.5 μs** | 49.2 μs | 2.28 ms | 3.47 ms | **47.99x** | **73.02x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **47.0 μs** | 48.5 μs | 4.07 ms | 43.48 ms | **86.52x** | **924.74x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **49.8 μs** | 50.8 μs | 3.19 ms | 11.94 ms | **64.16x** | **239.71x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **77.7 μs** | 50.5 μs | 2.34 ms | 8.55 ms | **46.33x** | **169.32x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **643.6 μs** | 143.0 μs | 3.17 ms | 2.85 ms | **22.19x** | **19.93x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **47.1 μs** | 48.5 μs | 1.97 ms | 534.3 μs | **41.86x** | **11.34x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **45.6 μs** | 47.2 μs | 2.03 ms | 4.38 ms | **44.52x** | **95.99x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **46.9 μs** | 48.6 μs | 2.72 ms | 1.43 ms | **58.09x** | **30.45x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **76.0 μs** | 51.1 μs | 1.96 ms | 1.05 ms | **38.35x** | **20.64x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **497.8 μs** | 85.7 μs | 2.69 ms | 565.9 μs | **31.34x** | **6.60x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **45.7 μs** | 49.1 μs | 1.90 ms | 250.1 μs | **41.67x** | **5.47x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **45.6 μs** | 48.3 μs | 1.94 ms | 716.1 μs | **42.45x** | **15.70x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **48.3 μs** | 48.0 μs | 2.62 ms | 409.4 μs | **54.62x** | **8.53x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **76.7 μs** | 51.1 μs | 1.88 ms | 319.5 μs | **36.85x** | **6.26x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **10.13 ms** | 6.18 ms | 28.38 ms | 9.30 ms | **4.59x** | **1.50x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **47.5 μs** | 48.9 μs | 2.91 ms | 24.75 ms | **61.19x** | **520.50x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **48.4 μs** | 50.4 μs | 5.79 ms | 5.53 ms | **119.42x** | **114.13x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **47.8 μs** | 49.0 μs | 3.53 ms | 2.75 ms | **73.77x** | **57.56x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **70.6 μs** | 51.1 μs | 3.04 ms | 2.70 ms | **59.54x** | **52.83x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.47 ms** | 702.1 μs | 4.58 ms | 1.05 ms | **6.52x** | **1.50x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **47.9 μs** | 51.0 μs | 1.59 ms | 2.63 ms | **33.28x** | **54.94x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **47.7 μs** | 50.5 μs | 1.82 ms | 647.0 μs | **38.23x** | **13.56x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **47.8 μs** | 49.2 μs | 2.09 ms | 470.8 μs | **43.70x** | **9.84x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **67.7 μs** | 52.4 μs | 1.60 ms | 422.0 μs | **30.57x** | **8.06x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **536.6 μs** | 146.3 μs | 2.13 ms | 302.6 μs | **14.59x** | **2.07x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **47.4 μs** | 48.8 μs | 1.46 ms | 447.2 μs | **30.78x** | **9.44x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **45.8 μs** | 47.3 μs | 1.46 ms | 242.1 μs | **31.76x** | **5.28x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **46.6 μs** | 48.3 μs | 1.86 ms | 227.0 μs | **39.91x** | **4.87x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **68.4 μs** | 50.3 μs | 1.44 ms | 203.6 μs | **28.67x** | **4.05x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **457.1 μs** | 93.7 μs | 1.88 ms | 216.3 μs | **20.10x** | **2.31x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **48.0 μs** | 48.8 μs | 1.43 ms | 233.7 μs | **29.80x** | **4.87x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **46.3 μs** | 48.8 μs | 1.44 ms | 206.6 μs | **31.07x** | **4.46x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **47.3 μs** | 49.6 μs | 1.81 ms | 202.4 μs | **38.25x** | **4.28x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **68.5 μs** | 51.8 μs | 1.42 ms | 184.1 μs | **27.47x** | **3.55x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.29 ms** | 477.3 μs | 3.82 ms | 46.78 ms | **8.00x** | **98.02x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **51.1 μs** | 51.5 μs | 28.31 ms | 44.70 ms | **554.14x** | **874.97x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **47.9 μs** | 49.0 μs | 2.20 ms | 46.40 ms | **45.90x** | **968.98x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **49.1 μs** | 51.5 μs | 2.01 ms | 16.41 ms | **41.04x** | **334.49x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **71.0 μs** | 52.5 μs | 1.49 ms | 13.23 ms | **28.40x** | **252.29x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **554.1 μs** | 127.3 μs | 2.13 ms | 4.85 ms | **16.72x** | **38.11x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **49.1 μs** | 50.7 μs | 4.11 ms | 4.65 ms | **83.68x** | **94.56x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **48.0 μs** | 49.4 μs | 1.56 ms | 4.83 ms | **32.50x** | **100.72x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **48.3 μs** | 51.0 μs | 1.95 ms | 1.93 ms | **40.44x** | **39.89x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **72.5 μs** | 51.8 μs | 1.49 ms | 1.55 ms | **28.78x** | **29.90x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **826.9 μs** | 467.8 μs | 3.88 ms | 128.00 ms | **8.29x** | **273.61x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **51.5 μs** | 51.4 μs | 500.80 ms | 128.30 ms | **9739.85x** | **2495.36x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **48.5 μs** | 50.2 μs | 2.03 ms | 123.42 ms | **41.75x** | **2542.95x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **50.4 μs** | 50.0 μs | 1.81 ms | 125.45 ms | **36.32x** | **2510.78x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **69.1 μs** | 52.1 μs | 1.36 ms | 127.19 ms | **26.02x** | **2441.69x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **305.8 μs** | 129.8 μs | 1.96 ms | 11.79 ms | **15.14x** | **90.87x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **48.1 μs** | 49.3 μs | 50.61 ms | 11.70 ms | **1052.29x** | **243.28x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **46.9 μs** | 47.1 μs | 1.43 ms | 11.53 ms | **30.53x** | **245.63x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **46.8 μs** | 49.4 μs | 1.74 ms | 11.70 ms | **37.28x** | **249.98x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **67.7 μs** | 51.3 μs | 1.36 ms | 11.87 ms | **26.52x** | **231.36x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **442.0 μs** | 90.4 μs | 1.80 ms | 1.32 ms | **19.85x** | **14.64x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **46.8 μs** | 49.6 μs | 6.34 ms | 1.33 ms | **135.34x** | **28.48x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **45.9 μs** | 47.3 μs | 1.39 ms | 1.31 ms | **30.17x** | **28.55x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **47.2 μs** | 48.5 μs | 1.72 ms | 1.31 ms | **36.54x** | **27.74x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **65.0 μs** | 50.5 μs | 1.36 ms | 1.32 ms | **27.00x** | **26.04x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.24 ms** | 725.3 μs | 35.96 ms | 114.17 ms | **49.58x** | **157.41x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **47.9 μs** | 48.6 μs | 6.55 ms | 5.47 ms | **136.90x** | **114.31x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **47.0 μs** | 49.2 μs | 19.12 ms | 182.53 ms | **406.59x** | **3881.09x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **46.3 μs** | 47.9 μs | 10.52 ms | 53.07 ms | **227.35x** | **1147.48x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **148.2 μs** | 49.8 μs | 6.68 ms | 39.74 ms | **134.23x** | **798.41x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.25 ms** | 153.2 μs | 10.63 ms | 10.93 ms | **69.43x** | **71.38x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **48.0 μs** | 49.8 μs | 5.79 ms | 971.1 μs | **120.59x** | **20.22x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **47.3 μs** | 48.9 μs | 6.34 ms | 17.20 ms | **133.99x** | **363.37x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **47.1 μs** | 48.5 μs | 9.12 ms | 5.32 ms | **193.67x** | **112.90x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **143.9 μs** | 50.8 μs | 5.79 ms | 3.86 ms | **113.89x** | **76.03x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **884.3 μs** | 95.0 μs | 9.19 ms | 1.60 ms | **96.69x** | **16.80x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **46.4 μs** | 48.2 μs | 5.70 ms | 486.2 μs | **122.86x** | **10.49x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **46.4 μs** | 48.2 μs | 5.79 ms | 2.27 ms | **124.85x** | **48.93x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **46.1 μs** | 48.2 μs | 8.97 ms | 1.10 ms | **194.60x** | **23.97x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **143.1 μs** | 49.9 μs | 5.73 ms | 858.9 μs | **114.92x** | **17.23x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | large_int16_2d [read_full] | 0.0025003328919410706 | fitsio/fitsio_torch | 1.9262535264624594 |
| fits | compressed_hcompress_1 [read_full] | 0.04551075119525194 | fitsio/fitsio_torch | 1.7856094488645875 |
| fits | medium_int16_3d [read_full] | 0.0010118158534169197 | fitsio/fitsio_torch | 1.7404640686426858 |
| fits | medium_int16_2d [read_full] | 0.0006631268188357353 | fitsio/fitsio_torch | 1.6877236978709889 |
| fits | large_int16_1d [read_full] | 0.0006311405450105667 | fitsio/fitsio_torch | 1.6171131722938228 |
| fits | large_uint16_2d [read_full] | 0.00245534535497427 | fitsio/fitsio_torch | 1.5868605836771201 |
| fits | medium_uint16_2d [read_full] | 0.0006545921787619591 | fitsio/fitsio_torch | 1.4948997605141927 |
| fits | small_int16_3d [read_full] | 0.00015708059072494507 | fitsio/fitsio_torch | 1.2338890799089932 |
| fits | large_int16_2d [read_full] | 0.002487124875187874 | fitsio/fitsio | 1.9235320852783921 |
| fits | medium_int16_3d [read_full] | 0.0010156063362956047 | fitsio/fitsio | 1.773313277502236 |
| fits | medium_int16_2d [read_full] | 0.0006740503013134003 | fitsio/fitsio | 1.7618851710748933 |
| fits | large_int16_1d [read_full] | 0.0006388500332832336 | fitsio/fitsio | 1.680405476579514 |
| fits | large_uint16_2d [read_full] | 0.0024611353874206543 | fitsio/fitsio | 1.605010434355876 |
| fits | medium_uint16_2d [read_full] | 0.0006637647747993469 | fitsio/fitsio | 1.5360668987143982 |
| fits | small_int16_3d [read_full] | 0.00015055760741233826 | fitsio/fitsio | 1.244093518646781 |
| fits | medium_int16_1d [read_full] | 0.00010975822806358337 | fitsio/fitsio | 1.0359158272243025 |
| fits | compressed_hcompress_1 [read_full] | 0.026286441832780838 | fitsio/fitsio | 1.0299538421347805 |
| fits | small_int16_2d [read_full] | 9.606312960386276e-05 | fitsio/fitsio | 1.003766056831452 |
| fitstable | narrow_1000 [predicate_filter] | 0.0004570716992020607 | fitsio/fitsio_torch | 2.112667992526969 |
| fitstable | narrow_10000 [predicate_filter] | 0.0005366336554288864 | fitsio/fitsio_torch | 1.773187056669385 |
| fitstable | narrow_100000 [predicate_filter] | 0.0014708200469613075 | fitsio/fitsio_torch | 1.3951492075831728 |
| fitstable | narrow_1000000 [predicate_filter] | 0.010133324190974236 | fitsio/fitsio_torch | 1.0897940958711803 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `bench_all_force_v060b2_20260708` | fits + fitstable (lab) | 1377 | 22 | lab bench-all (mmap-on only) |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
