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
Source: `benchmarks_results/exhaustive_v060b1_20260708/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.06 ms` (n=269) | `0.48 ms` (n=269) | `0.10 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.06 ms` (n=269) | `0.46 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.06 ms` (n=180) | `2.39 ms` (n=162) | `2.19 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.05 ms` (n=180) | `2.37 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **1.75 ms** | 1.71 ms | 12.96 ms | 2.81 ms | **7.57x** | **1.64x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **12.44 ms** | 7.19 ms | 19.03 ms | 7.24 ms | **2.65x** | **1.01x** |
| Repeated Cutouts (50x 100x100) | CPU | **3.22 ms** | 3.01 ms | 51.59 ms | 3.27 ms | **17.16x** | **1.09x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **54.1 μs** | 54.4 μs | 4.42 ms | 44.32 ms | **81.71x** | **819.45x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **53.2 μs** | 56.4 μs | 2.29 ms | 124.36 ms | **43.05x** | **2338.20x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 83.9 μs | 1.50 ms | 158.9 μs | **17.85x** | **1.89x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **23.64 ms** | 23.61 ms | 45.48 ms | 26.42 ms | **1.93x** | **1.12x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 82.4 μs | 1.49 ms | 154.2 μs | **18.06x** | **1.87x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **20.13 ms** | 20.09 ms | 71.58 ms | 23.03 ms | **3.56x** | **1.15x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 90.7 μs | 1.58 ms | 172.5 μs | **17.43x** | **1.90x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **45.68 ms** | 45.57 ms | 51.63 ms | 44.07 ms | **1.13x** | **0.97x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **710.3 μs** | 705.4 μs | 6.64 ms | 819.6 μs | **9.42x** | **1.16x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 90.5 μs | 1.56 ms | 173.2 μs | **17.27x** | **1.91x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **12.44 ms** | 7.19 ms | 19.03 ms | 7.24 ms | **2.65x** | **1.01x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 42.6 μs | 401.7 μs | 52.5 μs | **9.43x** | **1.23x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **478.3 μs** | 453.3 μs | 1.21 ms | 766.1 μs | **2.68x** | **1.69x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 44.9 μs | 426.6 μs | 55.0 μs | **9.50x** | **1.23x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **1.75 ms** | 1.71 ms | 12.96 ms | 2.81 ms | **7.57x** | **1.64x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 43.1 μs | 402.5 μs | 53.8 μs | **9.34x** | **1.25x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **884.4 μs** | 850.1 μs | 1.94 ms | 1.21 ms | **2.29x** | **1.42x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 44.2 μs | 420.3 μs | 54.3 μs | **9.51x** | **1.23x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **10.07 ms** | 9.86 ms | 28.00 ms | 10.92 ms | **2.84x** | **1.11x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 42.8 μs | 398.3 μs | 54.4 μs | **9.30x** | **1.27x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **278.5 μs** | 261.8 μs | 813.5 μs | 351.0 μs | **3.11x** | **1.34x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 44.9 μs | 431.5 μs | 54.9 μs | **9.62x** | **1.22x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **945.1 μs** | 939.1 μs | 2.03 ms | 1.23 ms | **2.16x** | **1.30x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 42.1 μs | 404.6 μs | 54.1 μs | **9.61x** | **1.28x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **450.6 μs** | 451.5 μs | 1.20 ms | 759.3 μs | **2.66x** | **1.69x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 42.0 μs | 426.6 μs | 56.0 μs | **10.16x** | **1.33x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **1.73 ms** | 1.71 ms | 12.79 ms | 2.80 ms | **7.48x** | **1.64x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 42.5 μs | 404.9 μs | 55.8 μs | **9.52x** | **1.31x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **860.1 μs** | 848.2 μs | 1.95 ms | 1.21 ms | **2.30x** | **1.42x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 43.5 μs | 430.5 μs | 56.0 μs | **9.90x** | **1.29x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.21 ms** | 9.97 ms | 28.02 ms | 11.04 ms | **2.81x** | **1.11x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 46.1 μs | 457.8 μs | 61.7 μs | **9.93x** | **1.34x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **156.2 μs** | 155.2 μs | 732.4 μs | 211.2 μs | **4.72x** | **1.36x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 46.6 μs | 472.1 μs | 60.4 μs | **10.13x** | **1.30x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **499.7 μs** | 498.2 μs | 1.52 ms | 659.9 μs | **3.04x** | **1.32x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 46.2 μs | 471.0 μs | 64.2 μs | **10.20x** | **1.39x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **14.17 ms** | 8.62 ms | 3.87 ms | 1.47 ms | **0.45x** | **0.17x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 48.5 μs | 472.4 μs | 61.1 μs | **9.73x** | **1.26x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **31.64 ms** | 24.58 ms | 11.60 ms | 3.30 ms | **0.47x** | **0.13x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 43.0 μs | 409.9 μs | 54.3 μs | **9.54x** | **1.26x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **105.3 μs** | 80.2 μs | 475.2 μs | 142.0 μs | **5.93x** | **1.77x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 45.5 μs | 431.9 μs | 56.7 μs | **9.50x** | **1.25x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **499.1 μs** | 475.0 μs | 1.24 ms | 780.1 μs | **2.61x** | **1.64x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 47.3 μs | 446.8 μs | 58.5 μs | **9.46x** | **1.24x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **736.2 μs** | 709.0 μs | 1.71 ms | 1.16 ms | **2.41x** | **1.64x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 43.0 μs | 408.3 μs | 55.1 μs | **9.51x** | **1.28x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **122.6 μs** | 119.9 μs | 553.3 μs | 183.2 μs | **4.61x** | **1.53x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 46.9 μs | 429.0 μs | 56.5 μs | **9.14x** | **1.20x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **901.2 μs** | 895.1 μs | 2.03 ms | 1.27 ms | **2.27x** | **1.42x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 49.2 μs | 456.2 μs | 57.7 μs | **9.27x** | **1.17x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **1.36 ms** | 1.34 ms | 10.24 ms | 1.90 ms | **7.62x** | **1.41x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 41.4 μs | 409.0 μs | 53.7 μs | **9.87x** | **1.30x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **68.5 μs** | 64.9 μs | 438.5 μs | 107.7 μs | **6.76x** | **1.66x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 44.3 μs | 415.2 μs | 55.7 μs | **9.37x** | **1.26x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **278.2 μs** | 277.1 μs | 841.7 μs | 371.4 μs | **3.04x** | **1.34x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 43.8 μs | 445.6 μs | 56.9 μs | **10.17x** | **1.30x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **408.7 μs** | 407.5 μs | 1.08 ms | 540.0 μs | **2.66x** | **1.33x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 43.2 μs | 409.0 μs | 53.8 μs | **9.47x** | **1.25x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **85.6 μs** | 80.3 μs | 465.8 μs | 142.0 μs | **5.80x** | **1.77x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 43.8 μs | 427.7 μs | 55.9 μs | **9.77x** | **1.28x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **477.4 μs** | 474.3 μs | 1.24 ms | 779.5 μs | **2.60x** | **1.64x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 45.9 μs | 447.1 μs | 58.7 μs | **9.74x** | **1.28x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **713.3 μs** | 708.0 μs | 1.75 ms | 1.21 ms | **2.47x** | **1.71x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 42.3 μs | 405.6 μs | 56.2 μs | **9.58x** | **1.33x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **124.3 μs** | 120.4 μs | 557.1 μs | 182.3 μs | **4.63x** | **1.51x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 45.7 μs | 424.0 μs | 55.9 μs | **9.28x** | **1.22x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **908.5 μs** | 904.0 μs | 2.08 ms | 1.31 ms | **2.30x** | **1.45x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 44.2 μs | 446.8 μs | 57.6 μs | **10.11x** | **1.30x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **1.37 ms** | 1.36 ms | 10.31 ms | 1.91 ms | **7.59x** | **1.41x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 46.5 μs | 446.1 μs | 59.8 μs | **9.60x** | **1.29x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **56.3 μs** | 56.3 μs | 522.5 μs | 94.5 μs | **9.29x** | **1.68x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 52.6 μs | 471.5 μs | 62.9 μs | **8.97x** | **1.20x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **162.7 μs** | 162.0 μs | 758.8 μs | 220.2 μs | **4.68x** | **1.36x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 49.4 μs | 480.1 μs | 63.8 μs | **9.71x** | **1.29x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **227.8 μs** | 228.7 μs | 941.9 μs | 311.3 μs | **4.13x** | **1.37x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 48.4 μs | 471.9 μs | 60.6 μs | **9.75x** | **1.25x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **1.58 ms** | 944.0 μs | 1.34 ms | 420.5 μs | **1.42x** | **0.45x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 48.4 μs | 478.5 μs | 61.9 μs | **9.88x** | **1.28x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **2.51 ms** | 1.75 ms | 1.77 ms | 925.6 μs | **1.01x** | **0.53x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 53.0 μs | 687.7 μs | 70.4 μs | **12.98x** | **1.33x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **161.4 μs** | 168.3 μs | 950.4 μs | 264.3 μs | **5.89x** | **1.64x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 52.4 μs | 688.8 μs | 68.9 μs | **13.14x** | **1.31x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **56.5 μs** | 60.0 μs | 706.1 μs | 134.0 μs | **12.49x** | **2.37x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **34.5 μs** | 35.5 μs | 2.29 ms | 208.8 μs | **66.47x** | **6.06x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 52.3 μs | 686.1 μs | 70.7 μs | **13.12x** | **1.35x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **5.20 ms** | 5.21 ms | 6.78 ms | 7.07 ms | **1.30x** | **1.36x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **56.5 μs** | 56.1 μs | 722.8 μs | 189.6 μs | **12.90x** | **3.38x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **3.22 ms** | 3.01 ms | 51.59 ms | 3.27 ms | **17.16x** | **1.09x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 47.8 μs | 487.1 μs | 62.5 μs | **10.19x** | **1.31x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **2.46 ms** | 2.42 ms | 10.60 ms | 2.95 ms | **4.37x** | **1.22x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 47.6 μs | 473.9 μs | 62.5 μs | **9.96x** | **1.31x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **675.3 μs** | 649.0 μs | 1.47 ms | 807.9 μs | **2.26x** | **1.24x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 50.9 μs | 483.8 μs | 60.7 μs | **9.51x** | **1.19x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **108.7 μs** | 84.9 μs | 589.0 μs | 128.6 μs | **6.94x** | **1.51x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 46.6 μs | 409.4 μs | 54.7 μs | **8.79x** | **1.18x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **71.7 μs** | 47.0 μs | 407.0 μs | 89.1 μs | **8.66x** | **1.90x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 44.3 μs | 427.7 μs | 55.1 μs | **9.65x** | **1.24x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **91.9 μs** | 69.3 μs | 464.4 μs | 124.7 μs | **6.70x** | **1.80x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 47.0 μs | 454.5 μs | 58.1 μs | **9.68x** | **1.24x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **138.4 μs** | 108.8 μs | 546.8 μs | 184.8 μs | **5.02x** | **1.70x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 41.4 μs | 410.1 μs | 53.3 μs | **9.91x** | **1.29x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **54.7 μs** | 50.8 μs | 416.3 μs | 93.6 μs | **8.20x** | **1.84x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 42.9 μs | 433.0 μs | 56.0 μs | **10.10x** | **1.31x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **106.0 μs** | 93.6 μs | 507.4 μs | 143.0 μs | **5.42x** | **1.53x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 46.2 μs | 448.4 μs | 57.6 μs | **9.70x** | **1.25x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **177.7 μs** | 176.5 μs | 702.1 μs | 258.4 μs | **3.98x** | **1.46x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 40.4 μs | 408.0 μs | 53.5 μs | **10.09x** | **1.32x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **49.0 μs** | 45.7 μs | 396.8 μs | 82.7 μs | **8.69x** | **1.81x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 44.6 μs | 429.5 μs | 54.6 μs | **9.63x** | **1.22x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **61.3 μs** | 60.3 μs | 438.1 μs | 96.0 μs | **7.27x** | **1.59x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 44.2 μs | 459.0 μs | 57.8 μs | **10.37x** | **1.31x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **83.1 μs** | 81.9 μs | 474.0 μs | 121.5 μs | **5.79x** | **1.48x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 41.6 μs | 409.8 μs | 55.3 μs | **9.86x** | **1.33x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **51.6 μs** | 47.8 μs | 404.3 μs | 88.0 μs | **8.45x** | **1.84x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 43.8 μs | 426.1 μs | 56.7 μs | **9.72x** | **1.29x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **73.6 μs** | 70.2 μs | 457.8 μs | 121.4 μs | **6.52x** | **1.73x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 46.9 μs | 451.7 μs | 59.0 μs | **9.63x** | **1.26x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **109.8 μs** | 105.8 μs | 543.5 μs | 180.8 μs | **5.14x** | **1.71x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 42.1 μs | 410.0 μs | 55.7 μs | **9.73x** | **1.32x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **54.9 μs** | 51.5 μs | 412.8 μs | 90.2 μs | **8.02x** | **1.75x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 45.5 μs | 434.2 μs | 56.3 μs | **9.54x** | **1.24x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **100.1 μs** | 92.0 μs | 506.7 μs | 144.5 μs | **5.51x** | **1.57x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 45.8 μs | 460.8 μs | 57.2 μs | **10.06x** | **1.25x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **180.2 μs** | 179.7 μs | 695.0 μs | 262.7 μs | **3.87x** | **1.46x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 46.3 μs | 457.3 μs | 58.9 μs | **9.88x** | **1.27x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **41.9 μs** | 40.9 μs | 515.6 μs | 82.9 μs | **12.62x** | **2.03x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 46.2 μs | 481.0 μs | 60.9 μs | **10.41x** | **1.32x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **52.4 μs** | 55.6 μs | 547.9 μs | 92.5 μs | **10.46x** | **1.77x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 50.0 μs | 503.8 μs | 63.2 μs | **10.08x** | **1.26x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **65.2 μs** | 69.4 μs | 573.9 μs | 104.0 μs | **8.81x** | **1.60x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 46.7 μs | 480.7 μs | 61.8 μs | **10.29x** | **1.32x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **180.6 μs** | 116.8 μs | 514.7 μs | 101.7 μs | **4.41x** | **0.87x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 47.5 μs | 469.7 μs | 61.2 μs | **9.89x** | **1.29x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **218.0 μs** | 151.1 μs | 540.4 μs | 130.1 μs | **3.58x** | **0.86x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 44.8 μs | 433.9 μs | 55.6 μs | **9.70x** | **1.24x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **104.9 μs** | 71.2 μs | 457.7 μs | 120.4 μs | **6.43x** | **1.69x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 43.9 μs | 436.6 μs | 55.2 μs | **9.94x** | **1.26x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **97.5 μs** | 72.8 μs | 457.2 μs | 121.9 μs | **6.28x** | **1.68x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 44.5 μs | 440.6 μs | 56.6 μs | **9.89x** | **1.27x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **95.1 μs** | 69.9 μs | 461.7 μs | 121.4 μs | **6.61x** | **1.74x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 44.3 μs | 429.2 μs | 56.6 μs | **9.69x** | **1.28x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **93.1 μs** | 71.4 μs | 460.4 μs | 120.6 μs | **6.45x** | **1.69x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 44.5 μs | 431.8 μs | 55.5 μs | **9.70x** | **1.25x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **90.7 μs** | 72.0 μs | 463.5 μs | 120.6 μs | **6.44x** | **1.67x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 43.6 μs | 411.5 μs | 53.5 μs | **9.45x** | **1.23x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **63.6 μs** | 43.2 μs | 398.6 μs | 84.0 μs | **9.23x** | **1.95x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 45.1 μs | 433.5 μs | 56.0 μs | **9.62x** | **1.24x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **69.3 μs** | 47.6 μs | 416.6 μs | 85.6 μs | **8.76x** | **1.80x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 46.6 μs | 461.2 μs | 56.6 μs | **9.90x** | **1.22x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **70.0 μs** | 47.3 μs | 428.5 μs | 84.4 μs | **9.06x** | **1.79x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 43.7 μs | 415.5 μs | 54.7 μs | **9.51x** | **1.25x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **44.1 μs** | 40.8 μs | 399.8 μs | 81.6 μs | **9.80x** | **2.00x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 44.0 μs | 441.2 μs | 56.1 μs | **10.02x** | **1.27x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **49.2 μs** | 49.1 μs | 415.9 μs | 87.5 μs | **8.47x** | **1.78x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 46.4 μs | 452.8 μs | 57.0 μs | **9.76x** | **1.23x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **51.4 μs** | 50.3 μs | 437.0 μs | 86.9 μs | **8.70x** | **1.73x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 42.1 μs | 409.7 μs | 55.0 μs | **9.73x** | **1.31x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **42.5 μs** | 39.9 μs | 395.2 μs | 81.1 μs | **9.92x** | **2.03x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 44.1 μs | 426.7 μs | 56.4 μs | **9.68x** | **1.28x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **44.9 μs** | 42.6 μs | 412.1 μs | 81.1 μs | **9.66x** | **1.90x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 44.8 μs | 458.4 μs | 57.9 μs | **10.23x** | **1.29x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **44.5 μs** | 43.5 μs | 432.7 μs | 83.1 μs | **9.95x** | **1.91x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 45.7 μs | 408.8 μs | 54.0 μs | **8.95x** | **1.18x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **43.9 μs** | 40.8 μs | 392.5 μs | 81.1 μs | **9.62x** | **1.99x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 43.0 μs | 432.0 μs | 56.0 μs | **10.05x** | **1.30x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **49.8 μs** | 47.4 μs | 413.0 μs | 83.6 μs | **8.71x** | **1.76x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 44.2 μs | 447.4 μs | 56.8 μs | **10.13x** | **1.29x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **49.5 μs** | 48.3 μs | 429.8 μs | 85.0 μs | **8.91x** | **1.76x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 41.9 μs | 409.8 μs | 54.5 μs | **9.79x** | **1.30x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **44.2 μs** | 39.8 μs | 399.9 μs | 82.2 μs | **10.05x** | **2.07x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 44.7 μs | 434.1 μs | 55.3 μs | **9.71x** | **1.24x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **50.4 μs** | 47.9 μs | 417.3 μs | 84.9 μs | **8.71x** | **1.77x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 45.7 μs | 451.2 μs | 58.0 μs | **9.87x** | **1.27x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **52.0 μs** | 51.5 μs | 427.1 μs | 86.3 μs | **8.30x** | **1.68x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 47.2 μs | 459.7 μs | 58.0 μs | **9.74x** | **1.23x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **42.0 μs** | 40.6 μs | 516.4 μs | 80.7 μs | **12.71x** | **1.99x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 46.7 μs | 479.1 μs | 60.3 μs | **10.26x** | **1.29x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **44.4 μs** | 44.9 μs | 511.3 μs | 85.1 μs | **11.52x** | **1.92x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 49.9 μs | 499.5 μs | 64.1 μs | **10.01x** | **1.28x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **44.8 μs** | 45.4 μs | 529.9 μs | 84.4 μs | **11.82x** | **1.88x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **708.7 μs** | 146.5 μs | 3.43 ms | 5.31 ms | **23.39x** | **36.25x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **54.2 μs** | 56.2 μs | 8.26 ms | 5.27 ms | **152.56x** | **97.25x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **52.6 μs** | 54.4 μs | 1.51 ms | 5.24 ms | **28.63x** | **99.68x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **61.0 μs** | 56.3 μs | 1.84 ms | 2.44 ms | **32.69x** | **43.33x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **70.9 μs** | 57.4 μs | 2.76 ms | 2.05 ms | **48.10x** | **35.78x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **496.4 μs** | 119.9 μs | 2.12 ms | 782.5 μs | **17.65x** | **6.53x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **55.6 μs** | 57.8 μs | 2.33 ms | 785.2 μs | **41.82x** | **14.12x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **53.8 μs** | 56.3 μs | 1.40 ms | 770.5 μs | **26.01x** | **14.33x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **55.0 μs** | 56.7 μs | 1.82 ms | 475.4 μs | **33.09x** | **8.64x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **73.4 μs** | 59.0 μs | 1.64 ms | 387.1 μs | **27.87x** | **6.57x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **12.44 ms** | 6.70 ms | 93.08 ms | 329.83 ms | **13.89x** | **49.22x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **61.6 μs** | 57.8 μs | 24.11 ms | 43.48 ms | **416.85x** | **751.91x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **52.5 μs** | 55.1 μs | 49.50 ms | 519.07 ms | **943.39x** | **9891.75x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **60.4 μs** | 56.5 μs | 22.08 ms | 135.13 ms | **391.17x** | **2393.80x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **79.8 μs** | 61.5 μs | 20.28 ms | 119.86 ms | **329.53x** | **1948.03x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **1.93 ms** | 767.0 μs | 8.84 ms | 27.07 ms | **11.52x** | **35.30x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **56.6 μs** | 60.9 μs | 2.72 ms | 3.76 ms | **48.05x** | **66.37x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **54.1 μs** | 54.4 μs | 4.42 ms | 44.32 ms | **81.71x** | **819.45x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **57.4 μs** | 54.6 μs | 3.59 ms | 12.25 ms | **65.82x** | **224.55x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **89.1 μs** | 57.8 μs | 2.72 ms | 8.70 ms | **47.00x** | **150.43x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **661.5 μs** | 158.7 μs | 3.26 ms | 2.90 ms | **20.52x** | **18.28x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **53.2 μs** | 55.8 μs | 2.01 ms | 543.8 μs | **37.82x** | **10.22x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **58.8 μs** | 55.5 μs | 2.17 ms | 4.45 ms | **39.19x** | **80.12x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **55.7 μs** | 56.4 μs | 2.82 ms | 1.47 ms | **50.62x** | **26.29x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **83.7 μs** | 57.6 μs | 2.04 ms | 1.11 ms | **35.38x** | **19.23x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **507.1 μs** | 100.3 μs | 2.74 ms | 584.7 μs | **27.34x** | **5.83x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **54.9 μs** | 55.3 μs | 1.94 ms | 262.0 μs | **35.35x** | **4.77x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **53.3 μs** | 55.0 μs | 1.95 ms | 726.7 μs | **36.54x** | **13.64x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **52.4 μs** | 54.9 μs | 2.65 ms | 429.7 μs | **50.55x** | **8.21x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **77.3 μs** | 57.2 μs | 1.94 ms | 330.6 μs | **33.96x** | **5.78x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **10.38 ms** | 6.21 ms | 29.36 ms | 9.58 ms | **4.73x** | **1.54x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **57.8 μs** | 55.8 μs | 3.70 ms | 24.97 ms | **66.24x** | **447.18x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **54.7 μs** | 55.2 μs | 6.67 ms | 5.84 ms | **122.02x** | **106.79x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **53.9 μs** | 56.1 μs | 4.16 ms | 2.82 ms | **77.13x** | **52.28x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **76.4 μs** | 60.8 μs | 3.66 ms | 2.76 ms | **60.23x** | **45.34x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.35 ms** | 731.4 μs | 4.73 ms | 1.09 ms | **6.47x** | **1.49x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **54.1 μs** | 55.5 μs | 1.76 ms | 2.71 ms | **32.56x** | **50.07x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **54.5 μs** | 54.2 μs | 2.02 ms | 660.8 μs | **37.23x** | **12.19x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **60.3 μs** | 54.1 μs | 2.25 ms | 498.5 μs | **41.61x** | **9.22x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **71.0 μs** | 60.0 μs | 1.72 ms | 432.4 μs | **28.67x** | **7.21x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **520.3 μs** | 160.6 μs | 2.17 ms | 335.2 μs | **13.48x** | **2.09x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **53.1 μs** | 57.2 μs | 1.49 ms | 467.7 μs | **28.11x** | **8.81x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **53.8 μs** | 54.5 μs | 1.49 ms | 257.1 μs | **27.69x** | **4.78x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **53.7 μs** | 55.7 μs | 1.88 ms | 238.6 μs | **35.01x** | **4.45x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **71.9 μs** | 58.8 μs | 1.51 ms | 219.7 μs | **25.71x** | **3.73x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **671.5 μs** | 105.1 μs | 1.90 ms | 235.0 μs | **18.11x** | **2.24x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **69.3 μs** | 72.6 μs | 2.36 ms | 347.8 μs | **34.07x** | **5.02x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **66.6 μs** | 70.5 μs | 2.41 ms | 291.8 μs | **36.26x** | **4.38x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **79.1 μs** | 72.7 μs | 3.07 ms | 298.2 μs | **42.21x** | **4.10x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **76.1 μs** | 58.6 μs | 1.44 ms | 192.3 μs | **24.58x** | **3.28x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.34 ms** | 577.1 μs | 4.13 ms | 46.76 ms | **7.16x** | **81.02x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **55.8 μs** | 56.8 μs | 28.85 ms | 45.09 ms | **517.16x** | **808.34x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **56.5 μs** | 55.4 μs | 2.37 ms | 45.97 ms | **42.70x** | **829.70x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **55.0 μs** | 57.6 μs | 2.33 ms | 16.88 ms | **42.39x** | **307.13x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **73.7 μs** | 60.5 μs | 1.79 ms | 13.11 ms | **29.55x** | **216.69x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **613.9 μs** | 142.5 μs | 2.19 ms | 4.91 ms | **15.34x** | **34.47x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **55.0 μs** | 57.4 μs | 4.23 ms | 4.73 ms | **76.86x** | **86.02x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **54.1 μs** | 55.8 μs | 1.58 ms | 4.80 ms | **29.29x** | **88.72x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **56.4 μs** | 57.3 μs | 2.02 ms | 1.98 ms | **35.91x** | **35.19x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **72.4 μs** | 58.8 μs | 1.56 ms | 1.55 ms | **26.49x** | **26.38x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **830.4 μs** | 550.6 μs | 4.18 ms | 129.84 ms | **7.59x** | **235.81x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **54.7 μs** | 57.7 μs | 510.33 ms | 127.89 ms | **9325.70x** | **2337.09x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **53.2 μs** | 56.4 μs | 2.29 ms | 124.36 ms | **43.05x** | **2338.20x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **59.6 μs** | 58.1 μs | 2.12 ms | 127.05 ms | **36.42x** | **2185.27x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **69.5 μs** | 58.8 μs | 1.70 ms | 126.18 ms | **28.84x** | **2145.40x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **312.8 μs** | 143.8 μs | 2.03 ms | 12.00 ms | **14.10x** | **83.46x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **55.6 μs** | 57.6 μs | 52.18 ms | 12.19 ms | **938.58x** | **219.35x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **55.4 μs** | 54.4 μs | 1.47 ms | 11.73 ms | **27.00x** | **215.44x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **54.2 μs** | 56.1 μs | 1.79 ms | 11.81 ms | **32.99x** | **217.71x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **71.0 μs** | 58.7 μs | 1.41 ms | 11.98 ms | **23.94x** | **204.04x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **440.5 μs** | 111.8 μs | 1.80 ms | 1.34 ms | **16.11x** | **11.98x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **55.6 μs** | 56.6 μs | 6.66 ms | 1.38 ms | **119.76x** | **24.78x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **52.6 μs** | 55.6 μs | 1.43 ms | 1.34 ms | **27.17x** | **25.57x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **53.2 μs** | 55.4 μs | 1.76 ms | 1.37 ms | **33.14x** | **25.75x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **67.5 μs** | 57.8 μs | 1.37 ms | 1.34 ms | **23.64x** | **23.19x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.22 ms** | 735.4 μs | 48.49 ms | 116.75 ms | **65.93x** | **158.75x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **55.3 μs** | 55.0 μs | 7.80 ms | 5.44 ms | **141.70x** | **98.85x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **54.0 μs** | 55.2 μs | 28.97 ms | 182.27 ms | **536.63x** | **3376.72x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **55.1 μs** | 54.3 μs | 11.66 ms | 53.24 ms | **214.62x** | **979.55x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **149.9 μs** | 57.5 μs | 7.89 ms | 39.91 ms | **137.15x** | **693.68x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.25 ms** | 164.5 μs | 10.95 ms | 11.12 ms | **66.58x** | **67.60x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **53.1 μs** | 55.7 μs | 6.06 ms | 981.2 μs | **114.14x** | **18.48x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **53.4 μs** | 54.9 μs | 6.98 ms | 17.84 ms | **130.73x** | **334.01x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **53.4 μs** | 55.4 μs | 9.42 ms | 5.42 ms | **176.23x** | **101.35x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **147.2 μs** | 56.8 μs | 6.08 ms | 3.99 ms | **107.06x** | **70.32x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **904.3 μs** | 104.3 μs | 9.24 ms | 1.62 ms | **88.61x** | **15.54x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **53.3 μs** | 55.5 μs | 5.74 ms | 508.4 μs | **107.76x** | **9.54x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **51.9 μs** | 55.4 μs | 5.86 ms | 2.30 ms | **112.88x** | **44.33x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **53.1 μs** | 54.4 μs | 9.03 ms | 1.12 ms | **170.03x** | **21.17x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **145.8 μs** | 57.1 μs | 5.84 ms | 876.4 μs | **102.22x** | **15.34x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | large_uint16_2d [read_full] | 0.014053785242140293 | fitsio/fitsio_torch | 9.772265538991569 |
| fits | large_uint32_2d [read_full] | 0.031021260656416416 | fitsio/fitsio_torch | 9.500000285209708 |
| fits | medium_uint16_2d [read_full] | 0.0015531685203313828 | fitsio/fitsio_torch | 3.7934562559994176 |
| fits | medium_uint32_2d [read_full] | 0.0024644164368510246 | fitsio/fitsio_torch | 2.758673283222652 |
| fits | compressed_hcompress_1 [read_full] | 0.04568122047930956 | fitsio/fitsio_torch | 1.7902758099683618 |
| fits | small_uint16_2d [read_full] | 0.0001806328073143959 | fitsio/fitsio_torch | 1.7094243837089396 |
| fits | small_uint32_2d [read_full] | 0.00021802261471748352 | fitsio/fitsio_torch | 1.6697813092911453 |
| fits | large_uint16_2d [read_full] | 0.0023646680638194084 | fitsio/fitsio | 1.656403531948345 |
| fits | medium_uint16_2d [read_full] | 0.0006341412663459778 | fitsio/fitsio | 1.5551931589261487 |
| fits | compressed_hcompress_1 [read_full] | 0.02638348564505577 | fitsio/fitsio | 1.0357009244589013 |
| fitstable | narrow_1000 [predicate_filter] | 0.0006715217605233192 | fitsio/fitsio_torch | 2.8577922927227544 |
| fitstable | narrow_10000 [predicate_filter] | 0.0005139168351888657 | fitsio/fitsio_torch | 1.688335847313203 |
| fitstable | narrow_100000 [predicate_filter] | 0.001295565627515316 | fitsio/fitsio_torch | 1.2219282550206465 |
| fitstable | narrow_1000000 [predicate_filter] | 0.0103476382791996 | fitsio/fitsio_torch | 1.0988455324128859 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_v060b1_20260708` | fits + fitstable (lab) | 2754 | 14 | lab bench-all + `--mmap-matrix` |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
