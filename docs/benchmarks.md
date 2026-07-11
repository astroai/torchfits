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
Source: `benchmarks_results/20260710_020649/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.08 ms` (n=269) | `0.50 ms` (n=269) | `0.16 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.08 ms` (n=269) | `0.48 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.24 ms` (n=223) | `0.46 ms` (n=79) | `0.27 ms` (n=79) | — |
| `disk→RAM→GPU` | `0.25 ms` (n=223) | `0.67 ms` (n=79) | `0.29 ms` (n=79) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.07 ms` (n=180) | `2.01 ms` (n=162) | `2.73 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.07 ms` (n=180) | `2.52 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **2.80 ms** | 2.78 ms | 6.13 ms | 3.07 ms | **2.21x** | **1.11x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.09 ms** | 3.30 ms | 5.70 ms | 3.26 ms | **1.84x** | **1.05x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **6.72 ms** | 6.74 ms | 18.78 ms | 6.82 ms | **2.80x** | **1.02x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **6.89 ms** | 6.95 ms | 18.64 ms | 6.87 ms | **2.71x** | **1.00x** |
| Repeated Cutouts (50x 100x100) | CPU | **5.16 ms** | 4.92 ms | 65.38 ms | 5.10 ms | **13.28x** | **1.04x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **76.4 μs** | 67.9 μs | 3.51 ms | 53.01 ms | **51.72x** | **781.01x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **71.3 μs** | 74.7 μs | 2.77 ms | 120.38 ms | **38.91x** | **1688.52x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 101.6 μs | 1.25 ms | 214.1 μs | **12.32x** | **2.11x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **10.97 ms** | 10.89 ms | 23.75 ms | 13.48 ms | **2.18x** | **1.24x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 99.3 μs | 1.29 ms | 209.1 μs | **13.03x** | **2.11x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **11.52 ms** | 11.78 ms | 42.05 ms | 14.76 ms | **3.65x** | **1.28x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 95.0 μs | 1.35 ms | 238.7 μs | **14.18x** | **2.51x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **22.73 ms** | 22.21 ms | 27.51 ms | 22.76 ms | **1.24x** | **1.02x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **620.2 μs** | 620.5 μs | 5.63 ms | 813.1 μs | **9.08x** | **1.31x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 101.7 μs | 1.24 ms | 238.8 μs | **12.24x** | **2.35x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **6.72 ms** | 6.74 ms | 18.78 ms | 6.82 ms | **2.80x** | **1.02x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 54.4 μs | 531.4 μs | 133.0 μs | **9.77x** | **2.45x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **644.7 μs** | 676.2 μs | 1.80 ms | 832.7 μs | **2.79x** | **1.29x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 52.8 μs | 427.7 μs | 117.3 μs | **8.10x** | **2.22x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **2.80 ms** | 2.78 ms | 6.13 ms | 3.07 ms | **2.21x** | **1.11x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 56.6 μs | 412.4 μs | 102.5 μs | **7.29x** | **1.81x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **1.65 ms** | 1.64 ms | 2.98 ms | 1.64 ms | **1.82x** | **1.00x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 55.1 μs | 423.4 μs | 111.2 μs | **7.69x** | **2.02x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **5.93 ms** | 5.77 ms | 9.92 ms | 6.27 ms | **1.72x** | **1.09x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 52.6 μs | 377.3 μs | 108.9 μs | **7.17x** | **2.07x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **348.6 μs** | 345.9 μs | 1.10 ms | 436.2 μs | **3.18x** | **1.26x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 52.1 μs | 448.4 μs | 112.2 μs | **8.61x** | **2.15x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **1.25 ms** | 1.29 ms | 3.00 ms | 1.45 ms | **2.40x** | **1.16x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 51.8 μs | 402.8 μs | 100.8 μs | **7.77x** | **1.95x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **636.1 μs** | 650.3 μs | 1.66 ms | 767.4 μs | **2.61x** | **1.21x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 51.2 μs | 419.9 μs | 109.0 μs | **8.21x** | **2.13x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **2.50 ms** | 2.49 ms | 5.22 ms | 2.75 ms | **2.09x** | **1.10x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 53.2 μs | 396.6 μs | 106.9 μs | **7.45x** | **2.01x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **1.46 ms** | 1.42 ms | 2.75 ms | 1.62 ms | **1.94x** | **1.15x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 56.7 μs | 427.3 μs | 106.5 μs | **7.54x** | **1.88x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **5.72 ms** | 6.41 ms | 10.95 ms | 6.28 ms | **1.91x** | **1.10x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 53.9 μs | 429.7 μs | 115.8 μs | **7.97x** | **2.15x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **568.2 μs** | 566.0 μs | 693.0 μs | 677.1 μs | **1.22x** | **1.20x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 60.0 μs | 477.3 μs | 112.9 μs | **7.96x** | **1.88x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **2.40 ms** | 2.43 ms | 1.87 ms | 2.54 ms | **0.78x** | **1.06x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 56.3 μs | 472.4 μs | 127.8 μs | **8.39x** | **2.27x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **2.87 ms** | 2.89 ms | 4.59 ms | 3.05 ms | **1.60x** | **1.06x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 61.6 μs | 448.6 μs | 110.5 μs | **7.28x** | **1.80x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **4.16 ms** | 4.12 ms | 7.55 ms | 4.39 ms | **1.83x** | **1.06x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 52.4 μs | 427.5 μs | 109.1 μs | **8.16x** | **2.08x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **105.1 μs** | 111.8 μs | 526.7 μs | 170.7 μs | **5.01x** | **1.62x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 52.9 μs | 440.5 μs | 123.0 μs | **8.33x** | **2.33x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **667.8 μs** | 678.0 μs | 1.72 ms | 776.9 μs | **2.57x** | **1.16x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 54.0 μs | 449.6 μs | 108.0 μs | **8.33x** | **2.00x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **999.9 μs** | 1.02 ms | 2.40 ms | 1.15 ms | **2.40x** | **1.15x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 54.2 μs | 406.8 μs | 114.5 μs | **7.51x** | **2.11x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **168.5 μs** | 169.4 μs | 629.8 μs | 237.2 μs | **3.74x** | **1.41x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 50.3 μs | 407.0 μs | 107.5 μs | **8.09x** | **2.14x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **1.48 ms** | 1.50 ms | 2.84 ms | 1.69 ms | **1.92x** | **1.14x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 50.3 μs | 435.2 μs | 110.1 μs | **8.65x** | **2.19x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **2.27 ms** | 2.31 ms | 3.61 ms | 2.52 ms | **1.59x** | **1.11x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 53.4 μs | 363.2 μs | 107.0 μs | **6.80x** | **2.00x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **84.1 μs** | 88.1 μs | 460.3 μs | 143.3 μs | **5.47x** | **1.70x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 50.5 μs | 422.4 μs | 107.3 μs | **8.36x** | **2.12x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **350.0 μs** | 366.6 μs | 1.14 ms | 452.0 μs | **3.25x** | **1.29x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 51.0 μs | 434.9 μs | 116.2 μs | **8.53x** | **2.28x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **536.5 μs** | 542.7 μs | 1.43 ms | 627.7 μs | **2.66x** | **1.17x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 47.2 μs | 394.8 μs | 112.7 μs | **8.36x** | **2.38x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **97.4 μs** | 110.8 μs | 484.2 μs | 182.0 μs | **4.97x** | **1.87x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 55.1 μs | 400.4 μs | 115.7 μs | **7.27x** | **2.10x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **667.1 μs** | 675.4 μs | 1.73 ms | 793.1 μs | **2.59x** | **1.19x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 52.5 μs | 434.5 μs | 118.7 μs | **8.28x** | **2.26x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **994.9 μs** | 1.02 ms | 2.37 ms | 1.17 ms | **2.38x** | **1.18x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 54.8 μs | 395.2 μs | 100.3 μs | **7.21x** | **1.83x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **175.9 μs** | 180.1 μs | 593.8 μs | 246.3 μs | **3.38x** | **1.40x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 54.1 μs | 397.0 μs | 111.3 μs | **7.33x** | **2.06x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **1.46 ms** | 1.51 ms | 2.87 ms | 1.65 ms | **1.97x** | **1.13x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 55.4 μs | 425.8 μs | 113.0 μs | **7.69x** | **2.04x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **2.32 ms** | 2.31 ms | 3.65 ms | 2.47 ms | **1.58x** | **1.07x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 54.7 μs | 430.8 μs | 127.0 μs | **7.88x** | **2.32x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **116.0 μs** | 126.7 μs | 584.6 μs | 190.8 μs | **5.04x** | **1.64x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 54.8 μs | 445.0 μs | 116.1 μs | **8.12x** | **2.12x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **619.2 μs** | 601.4 μs | 728.5 μs | 696.5 μs | **1.21x** | **1.16x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 55.7 μs | 459.4 μs | 131.3 μs | **8.25x** | **2.36x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **942.2 μs** | 1.01 ms | 1.09 ms | 1.08 ms | **1.16x** | **1.14x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 59.1 μs | 419.6 μs | 110.9 μs | **7.10x** | **1.88x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **741.5 μs** | 741.8 μs | 1.58 ms | 862.6 μs | **2.13x** | **1.16x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 54.6 μs | 461.1 μs | 120.7 μs | **8.45x** | **2.21x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **1.05 ms** | 1.06 ms | 2.34 ms | 1.18 ms | **2.22x** | **1.12x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 57.5 μs | 599.2 μs | 120.6 μs | **10.41x** | **2.10x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **578.2 μs** | 588.8 μs | 873.9 μs | 722.7 μs | **1.51x** | **1.25x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 56.6 μs | 598.0 μs | 126.0 μs | **10.56x** | **2.23x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **85.4 μs** | 100.3 μs | 694.6 μs | 203.8 μs | **8.14x** | **2.39x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **63.5 μs** | 52.2 μs | 2.38 ms | 270.3 μs | **45.48x** | **5.17x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 62.5 μs | 624.0 μs | 126.4 μs | **9.99x** | **2.02x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **6.73 ms** | 6.47 ms | 5.84 ms | 6.43 ms | **0.90x** | **0.99x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **92.1 μs** | 108.1 μs | 698.5 μs | 266.2 μs | **7.58x** | **2.89x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **5.16 ms** | 4.92 ms | 65.38 ms | 5.10 ms | **13.28x** | **1.04x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 60.6 μs | 452.1 μs | 122.1 μs | **7.46x** | **2.02x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **2.42 ms** | 2.46 ms | 6.41 ms | 2.60 ms | **2.65x** | **1.08x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 62.1 μs | 447.3 μs | 118.6 μs | **7.20x** | **1.91x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **654.0 μs** | 669.4 μs | 2.00 ms | 766.0 μs | **3.07x** | **1.17x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 53.6 μs | 460.9 μs | 111.7 μs | **8.60x** | **2.08x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **87.3 μs** | 102.8 μs | 605.6 μs | 178.0 μs | **6.93x** | **2.04x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 49.0 μs | 398.7 μs | 106.7 μs | **8.14x** | **2.18x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **61.4 μs** | 67.4 μs | 409.5 μs | 136.4 μs | **6.67x** | **2.22x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 56.0 μs | 407.5 μs | 124.4 μs | **7.27x** | **2.22x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **90.8 μs** | 90.1 μs | 455.9 μs | 163.7 μs | **5.06x** | **1.82x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 53.8 μs | 423.6 μs | 112.1 μs | **7.87x** | **2.08x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **136.2 μs** | 126.9 μs | 569.7 μs | 212.7 μs | **4.49x** | **1.68x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 57.2 μs | 391.0 μs | 110.7 μs | **6.84x** | **1.94x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **71.2 μs** | 81.6 μs | 469.0 μs | 165.0 μs | **6.59x** | **2.32x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 52.4 μs | 438.4 μs | 111.0 μs | **8.36x** | **2.12x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **129.1 μs** | 127.1 μs | 534.3 μs | 205.9 μs | **4.20x** | **1.62x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 49.0 μs | 432.6 μs | 114.5 μs | **8.82x** | **2.33x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **239.3 μs** | 232.9 μs | 710.7 μs | 310.5 μs | **3.05x** | **1.33x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 52.8 μs | 395.8 μs | 121.7 μs | **7.50x** | **2.31x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **54.7 μs** | 67.4 μs | 429.0 μs | 137.3 μs | **7.85x** | **2.51x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 61.9 μs | 449.7 μs | 116.6 μs | **7.27x** | **1.88x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **81.4 μs** | 71.5 μs | 458.3 μs | 155.1 μs | **6.41x** | **2.17x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 42.4 μs | 435.5 μs | 115.8 μs | **10.27x** | **2.73x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **95.1 μs** | 103.5 μs | 503.4 μs | 178.5 μs | **5.29x** | **1.88x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 52.5 μs | 384.6 μs | 110.1 μs | **7.33x** | **2.10x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **59.4 μs** | 67.9 μs | 436.3 μs | 150.6 μs | **7.34x** | **2.54x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 52.1 μs | 432.0 μs | 105.8 μs | **8.29x** | **2.03x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **86.4 μs** | 96.4 μs | 496.6 μs | 163.8 μs | **5.75x** | **1.90x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 60.6 μs | 423.4 μs | 110.4 μs | **6.99x** | **1.82x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **144.1 μs** | 133.0 μs | 536.5 μs | 211.0 μs | **4.03x** | **1.59x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 50.9 μs | 415.2 μs | 111.9 μs | **8.15x** | **2.20x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **69.5 μs** | 72.2 μs | 440.0 μs | 143.7 μs | **6.34x** | **2.07x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 50.5 μs | 394.0 μs | 118.1 μs | **7.80x** | **2.34x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **119.8 μs** | 133.0 μs | 519.1 μs | 209.2 μs | **4.33x** | **1.75x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 54.3 μs | 402.7 μs | 117.2 μs | **7.42x** | **2.16x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **231.2 μs** | 248.7 μs | 754.5 μs | 321.2 μs | **3.26x** | **1.39x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 59.7 μs | 432.6 μs | 125.2 μs | **7.24x** | **2.09x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **62.1 μs** | 59.8 μs | 496.5 μs | 146.6 μs | **8.30x** | **2.45x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 55.3 μs | 473.3 μs | 115.0 μs | **8.55x** | **2.08x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **97.5 μs** | 104.6 μs | 559.8 μs | 170.4 μs | **5.74x** | **1.75x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 55.5 μs | 479.7 μs | 119.0 μs | **8.64x** | **2.14x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **140.9 μs** | 148.3 μs | 567.8 μs | 217.5 μs | **4.03x** | **1.54x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 55.8 μs | 468.6 μs | 122.9 μs | **8.40x** | **2.20x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **94.9 μs** | 98.2 μs | 521.8 μs | 173.8 μs | **5.50x** | **1.83x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 50.0 μs | 446.0 μs | 123.7 μs | **8.92x** | **2.47x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **112.3 μs** | 111.5 μs | 577.9 μs | 193.5 μs | **5.18x** | **1.73x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 51.0 μs | 413.9 μs | 117.7 μs | **8.12x** | **2.31x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **91.9 μs** | 92.5 μs | 531.1 μs | 166.1 μs | **5.78x** | **1.81x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 55.9 μs | 450.5 μs | 110.1 μs | **8.06x** | **1.97x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **85.8 μs** | 98.8 μs | 492.5 μs | 171.6 μs | **5.74x** | **2.00x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 55.7 μs | 408.4 μs | 117.4 μs | **7.33x** | **2.11x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **90.0 μs** | 103.4 μs | 490.8 μs | 158.9 μs | **5.46x** | **1.77x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 56.0 μs | 413.1 μs | 125.1 μs | **7.38x** | **2.24x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **91.3 μs** | 88.1 μs | 479.4 μs | 162.2 μs | **5.44x** | **1.84x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 48.0 μs | 432.7 μs | 118.9 μs | **9.02x** | **2.48x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **86.0 μs** | 90.6 μs | 466.5 μs | 154.2 μs | **5.43x** | **1.79x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 52.6 μs | 418.8 μs | 104.0 μs | **7.97x** | **1.98x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **42.7 μs** | 54.4 μs | 403.4 μs | 137.3 μs | **9.44x** | **3.22x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 58.8 μs | 406.0 μs | 107.7 μs | **6.91x** | **1.83x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **60.9 μs** | 70.6 μs | 424.3 μs | 142.9 μs | **6.97x** | **2.35x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 54.3 μs | 423.7 μs | 117.1 μs | **7.80x** | **2.16x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **52.2 μs** | 68.8 μs | 455.8 μs | 134.0 μs | **8.72x** | **2.56x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 56.0 μs | 375.8 μs | 107.0 μs | **6.71x** | **1.91x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **51.3 μs** | 53.8 μs | 418.0 μs | 118.1 μs | **8.15x** | **2.30x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 51.2 μs | 433.5 μs | 104.1 μs | **8.46x** | **2.03x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **62.3 μs** | 66.4 μs | 413.9 μs | 136.8 μs | **6.64x** | **2.20x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 59.0 μs | 409.7 μs | 111.6 μs | **6.95x** | **1.89x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **69.0 μs** | 76.5 μs | 431.8 μs | 144.4 μs | **6.26x** | **2.09x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 52.9 μs | 389.5 μs | 114.9 μs | **7.37x** | **2.17x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **42.6 μs** | 44.1 μs | 405.6 μs | 134.3 μs | **9.53x** | **3.15x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 57.7 μs | 423.2 μs | 119.6 μs | **7.33x** | **2.07x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **40.4 μs** | 52.2 μs | 436.2 μs | 144.2 μs | **10.79x** | **3.57x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 52.4 μs | 413.2 μs | 121.0 μs | **7.89x** | **2.31x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **46.8 μs** | 58.8 μs | 433.0 μs | 136.5 μs | **9.26x** | **2.92x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 52.8 μs | 413.2 μs | 117.3 μs | **7.83x** | **2.22x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **50.5 μs** | 59.6 μs | 434.5 μs | 139.6 μs | **8.60x** | **2.76x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 51.5 μs | 411.0 μs | 113.9 μs | **7.99x** | **2.21x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **61.5 μs** | 70.1 μs | 441.5 μs | 127.2 μs | **7.18x** | **2.07x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 47.7 μs | 431.0 μs | 113.9 μs | **9.04x** | **2.39x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **64.3 μs** | 75.4 μs | 458.8 μs | 143.8 μs | **7.14x** | **2.24x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 55.0 μs | 425.9 μs | 112.3 μs | **7.74x** | **2.04x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **47.0 μs** | 51.6 μs | 425.9 μs | 139.6 μs | **9.07x** | **2.97x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 56.5 μs | 412.2 μs | 113.0 μs | **7.29x** | **2.00x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **60.0 μs** | 80.7 μs | 524.1 μs | 168.5 μs | **8.74x** | **2.81x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 52.7 μs | 416.8 μs | 107.8 μs | **7.91x** | **2.05x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **72.8 μs** | 82.5 μs | 481.3 μs | 150.3 μs | **6.61x** | **2.06x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 56.4 μs | 429.0 μs | 119.7 μs | **7.61x** | **2.12x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **57.4 μs** | 56.8 μs | 515.0 μs | 140.9 μs | **9.07x** | **2.48x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 55.2 μs | 466.3 μs | 113.4 μs | **8.45x** | **2.05x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **55.6 μs** | 56.1 μs | 516.7 μs | 143.2 μs | **9.29x** | **2.58x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 59.1 μs | 468.1 μs | 115.1 μs | **7.92x** | **1.95x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **50.2 μs** | 56.1 μs | 516.8 μs | 145.9 μs | **10.30x** | **2.91x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | **11.28 ms** | 11.27 ms | 23.18 ms | 13.88 ms | **2.06x** | **1.23x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | **11.86 ms** | 11.95 ms | 40.97 ms | 14.56 ms | **3.45x** | **1.23x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | **21.51 ms** | 21.64 ms | 26.40 ms | 21.77 ms | **1.23x** | **1.01x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | **710.6 μs** | 722.9 μs | 5.50 ms | 820.0 μs | **7.74x** | **1.15x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | **6.89 ms** | 6.95 ms | 18.64 ms | 6.87 ms | **2.71x** | **1.00x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | **717.4 μs** | 699.3 μs | 1.67 ms | 766.9 μs | **2.38x** | **1.10x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | **3.09 ms** | 3.30 ms | 5.70 ms | 3.26 ms | **1.84x** | **1.05x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | **438.3 μs** | 492.4 μs | 864.5 μs | 495.6 μs | **1.97x** | **1.13x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | **1.36 ms** | 1.38 ms | 3.24 ms | 1.47 ms | **2.38x** | **1.08x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | **696.6 μs** | 714.0 μs | 1.61 ms | 792.0 μs | **2.31x** | **1.14x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | **3.09 ms** | 3.13 ms | 5.68 ms | 3.23 ms | **1.84x** | **1.04x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | **494.5 μs** | 1.76 ms | 3.13 ms | 1.67 ms | **6.32x** | **3.39x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | **1.54 ms** | 7.47 ms | 11.53 ms | 7.93 ms | **7.48x** | **5.14x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | **250.5 μs** | 715.1 μs | 559.0 μs | 733.8 μs | **2.23x** | **2.93x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | **657.9 μs** | 2.50 ms | 1.75 ms | 2.68 ms | **2.66x** | **4.07x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | **1.39 ms** | 3.10 ms | 4.93 ms | 3.26 ms | **3.53x** | **2.34x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | **3.08 ms** | 4.91 ms | 7.94 ms | 4.96 ms | **2.58x** | **1.61x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | **205.2 μs** | 215.1 μs | 389.6 μs | 242.0 μs | **1.90x** | **1.18x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | **753.1 μs** | 708.4 μs | 1.66 ms | 779.8 μs | **2.35x** | **1.10x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | **1.09 ms** | 1.04 ms | 2.51 ms | 1.15 ms | **2.41x** | **1.11x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | **207.6 μs** | 221.0 μs | 386.5 μs | 245.5 μs | **1.86x** | **1.18x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | **361.7 μs** | 367.7 μs | 813.0 μs | 419.5 μs | **2.25x** | **1.16x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | **559.3 μs** | 646.1 μs | 1.53 ms | 613.2 μs | **2.73x** | **1.10x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | **204.0 μs** | 197.6 μs | 401.9 μs | 238.2 μs | **2.03x** | **1.21x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | **725.3 μs** | 719.9 μs | 1.69 ms | 765.9 μs | **2.34x** | **1.06x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | **1.03 ms** | 1.05 ms | 2.57 ms | 1.15 ms | **2.48x** | **1.11x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | **222.6 μs** | 277.6 μs | 472.6 μs | 313.0 μs | **2.12x** | **1.41x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | **507.0 μs** | 1.66 ms | 3.20 ms | 1.77 ms | **6.31x** | **3.49x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | **676.6 μs** | 2.76 ms | 3.73 ms | 2.30 ms | **5.52x** | **3.40x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | **173.7 μs** | 223.5 μs | 426.0 μs | 254.3 μs | **2.45x** | **1.46x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | **259.0 μs** | 712.1 μs | 620.8 μs | 759.3 μs | **2.40x** | **2.93x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | **300.5 μs** | 1.02 ms | 756.6 μs | 1.08 ms | **2.52x** | **3.60x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | **360.7 μs** | 763.9 μs | 1.25 ms | 790.1 μs | **3.46x** | **2.19x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | **710.5 μs** | 1.11 ms | 2.03 ms | 1.18 ms | **2.85x** | **1.66x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | **255.6 μs** | 707.4 μs | 780.8 μs | 803.2 μs | **3.05x** | **3.14x** |
| fits | mef_small | read_full | 0.45 MB | MPS | **172.3 μs** | 207.0 μs | 541.9 μs | 263.8 μs | **3.15x** | **1.53x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | **140.3 μs** | 162.3 μs | 2.46 ms | 269.0 μs | **17.51x** | **1.92x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | **170.1 μs** | 198.0 μs | 561.2 μs | 367.0 μs | **3.30x** | **2.16x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | **11.35 ms** | 11.40 ms | 78.27 ms | 11.49 ms | **6.90x** | **1.01x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | **1.43 ms** | 3.13 ms | 6.19 ms | 3.17 ms | **4.33x** | **2.22x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | **383.1 μs** | 814.2 μs | 1.66 ms | 831.5 μs | **4.32x** | **2.17x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | **182.8 μs** | 224.8 μs | 517.5 μs | 239.8 μs | **2.83x** | **1.31x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | **157.5 μs** | 175.5 μs | 353.1 μs | 207.8 μs | **2.24x** | **1.32x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | **216.2 μs** | 217.1 μs | 400.0 μs | 258.7 μs | **1.85x** | **1.20x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | **243.4 μs** | 242.5 μs | 439.5 μs | 274.1 μs | **1.81x** | **1.13x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | **158.2 μs** | 164.2 μs | 337.4 μs | 192.1 μs | **2.13x** | **1.21x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | **177.2 μs** | 181.4 μs | 366.7 μs | 227.8 μs | **2.07x** | **1.29x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | **218.2 μs** | 226.6 μs | 428.8 μs | 255.9 μs | **1.97x** | **1.17x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | **162.2 μs** | 174.5 μs | 344.0 μs | 199.8 μs | **2.12x** | **1.23x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | **190.5 μs** | 188.0 μs | 398.2 μs | 213.6 μs | **2.12x** | **1.14x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | **237.8 μs** | 236.3 μs | 455.9 μs | 275.7 μs | **1.93x** | **1.17x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | **162.2 μs** | 166.0 μs | 339.7 μs | 201.0 μs | **2.09x** | **1.24x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | **205.7 μs** | 238.7 μs | 408.9 μs | 253.8 μs | **1.99x** | **1.23x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | **209.7 μs** | 320.7 μs | 573.3 μs | 362.0 μs | **2.73x** | **1.73x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | **139.2 μs** | 145.0 μs | 391.7 μs | 191.4 μs | **2.81x** | **1.38x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | **150.9 μs** | 189.3 μs | 410.5 μs | 217.8 μs | **2.72x** | **1.44x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | **161.2 μs** | 239.0 μs | 434.7 μs | 273.7 μs | **2.70x** | **1.70x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | **160.4 μs** | 188.5 μs | 403.3 μs | 231.6 μs | **2.51x** | **1.44x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | **167.2 μs** | 200.0 μs | 421.0 μs | 229.6 μs | **2.52x** | **1.37x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | **166.7 μs** | 168.1 μs | 377.2 μs | 202.5 μs | **2.26x** | **1.21x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | **165.6 μs** | 169.8 μs | 398.0 μs | 242.5 μs | **2.40x** | **1.46x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | **169.8 μs** | 183.1 μs | 390.4 μs | 224.1 μs | **2.30x** | **1.32x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | **157.5 μs** | 162.0 μs | 396.9 μs | 194.9 μs | **2.52x** | **1.24x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | **156.1 μs** | 166.4 μs | 368.2 μs | 201.5 μs | **2.36x** | **1.29x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | **126.5 μs** | 128.4 μs | 319.2 μs | 169.1 μs | **2.52x** | **1.34x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | **136.0 μs** | 140.5 μs | 330.9 μs | 164.8 μs | **2.43x** | **1.21x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | **132.2 μs** | 149.7 μs | 348.5 μs | 165.5 μs | **2.64x** | **1.25x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | **131.6 μs** | 163.4 μs | 326.7 μs | 172.6 μs | **2.48x** | **1.31x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | **134.7 μs** | 139.8 μs | 342.2 μs | 174.5 μs | **2.54x** | **1.30x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | **132.5 μs** | 137.6 μs | 345.0 μs | 179.4 μs | **2.60x** | **1.35x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | **131.1 μs** | 137.3 μs | 316.3 μs | 170.5 μs | **2.41x** | **1.30x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | **134.9 μs** | 143.5 μs | 333.4 μs | 173.8 μs | **2.47x** | **1.29x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | **139.0 μs** | 144.6 μs | 344.0 μs | 175.5 μs | **2.48x** | **1.26x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | **154.3 μs** | 139.2 μs | 326.5 μs | 180.1 μs | **2.35x** | **1.29x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | **154.0 μs** | 152.5 μs | 336.5 μs | 190.2 μs | **2.21x** | **1.25x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | **156.2 μs** | 158.8 μs | 340.4 μs | 188.2 μs | **2.18x** | **1.21x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | **133.7 μs** | 139.8 μs | 375.4 μs | 172.6 μs | **2.81x** | **1.29x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | **131.3 μs** | 139.7 μs | 390.4 μs | 174.0 μs | **2.97x** | **1.32x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | **127.8 μs** | 133.8 μs | 409.9 μs | 168.0 μs | **3.21x** | **1.31x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **362.5 μs** | 186.5 μs | 3.43 ms | 6.64 ms | **18.39x** | **35.59x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **75.1 μs** | 76.3 μs | 9.16 ms | 7.31 ms | **122.00x** | **97.35x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **71.5 μs** | 69.4 μs | 1.27 ms | 7.56 ms | **18.36x** | **108.97x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **69.0 μs** | 70.8 μs | 1.44 ms | 2.82 ms | **20.91x** | **40.89x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **68.5 μs** | 71.9 μs | 2.87 ms | 2.36 ms | **41.95x** | **34.47x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **290.1 μs** | 124.3 μs | 1.82 ms | 874.0 μs | **14.62x** | **7.03x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **75.7 μs** | 73.2 μs | 2.05 ms | 898.0 μs | **27.96x** | **12.27x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **69.2 μs** | 67.6 μs | 1.15 ms | 896.6 μs | **17.03x** | **13.27x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **71.5 μs** | 75.4 μs | 1.44 ms | 504.4 μs | **20.09x** | **7.05x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **81.0 μs** | 82.0 μs | 1.36 ms | 426.8 μs | **16.77x** | **5.27x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **14.36 ms** | 2.28 ms | 56.39 ms | 320.00 ms | **24.78x** | **140.62x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **63.4 μs** | 68.5 μs | 10.25 ms | 65.62 ms | **161.60x** | **1034.72x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **77.9 μs** | 75.5 μs | 25.29 ms | 529.58 ms | **334.73x** | **7010.40x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **72.1 μs** | 69.1 μs | 11.70 ms | 100.67 ms | **169.19x** | **1456.31x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **94.9 μs** | 88.9 μs | 10.65 ms | 95.92 ms | **119.85x** | **1079.31x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **1.80 ms** | 358.5 μs | 7.56 ms | 31.65 ms | **21.09x** | **88.28x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **66.1 μs** | 67.3 μs | 2.44 ms | 6.55 ms | **36.97x** | **99.12x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **76.4 μs** | 67.9 μs | 3.51 ms | 53.01 ms | **51.72x** | **781.01x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **73.4 μs** | 66.0 μs | 3.23 ms | 14.71 ms | **48.97x** | **222.96x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **88.7 μs** | 79.0 μs | 2.63 ms | 9.57 ms | **33.35x** | **121.15x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **447.7 μs** | 197.5 μs | 2.66 ms | 3.75 ms | **13.49x** | **18.97x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **69.6 μs** | 68.1 μs | 1.66 ms | 947.7 μs | **24.39x** | **13.91x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **75.1 μs** | 65.3 μs | 1.72 ms | 5.79 ms | **26.36x** | **88.62x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **69.1 μs** | 72.8 μs | 2.21 ms | 1.62 ms | **32.05x** | **23.44x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **75.1 μs** | 86.5 μs | 1.65 ms | 1.13 ms | **21.95x** | **14.99x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **356.3 μs** | 132.6 μs | 2.21 ms | 692.2 μs | **16.69x** | **5.22x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **66.6 μs** | 63.6 μs | 1.57 ms | 331.8 μs | **24.76x** | **5.22x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **68.0 μs** | 64.1 μs | 1.58 ms | 816.4 μs | **24.68x** | **12.74x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **63.4 μs** | 65.5 μs | 2.10 ms | 465.8 μs | **33.07x** | **7.35x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **85.8 μs** | 80.5 μs | 1.55 ms | 392.3 μs | **19.21x** | **4.87x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **10.17 ms** | 2.01 ms | 26.99 ms | 10.16 ms | **13.41x** | **5.05x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **71.0 μs** | 65.2 μs | 4.27 ms | 32.06 ms | **65.58x** | **491.92x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **75.9 μs** | 77.1 μs | 7.56 ms | 6.39 ms | **99.59x** | **84.22x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **77.5 μs** | 74.6 μs | 4.89 ms | 3.20 ms | **65.55x** | **42.94x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **76.9 μs** | 75.9 μs | 4.67 ms | 3.17 ms | **61.49x** | **41.79x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.50 ms** | 375.6 μs | 4.19 ms | 1.14 ms | **11.14x** | **3.03x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **64.3 μs** | 67.2 μs | 1.41 ms | 3.32 ms | **21.92x** | **51.60x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **66.0 μs** | 76.1 μs | 1.86 ms | 778.5 μs | **28.11x** | **11.79x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **73.8 μs** | 75.8 μs | 1.89 ms | 1.01 ms | **25.65x** | **13.63x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **77.0 μs** | 67.5 μs | 1.47 ms | 449.5 μs | **21.82x** | **6.66x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **414.9 μs** | 203.2 μs | 1.78 ms | 376.2 μs | **8.76x** | **1.85x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **76.4 μs** | 70.6 μs | 1.30 ms | 590.1 μs | **18.42x** | **8.36x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **72.1 μs** | 64.8 μs | 1.27 ms | 307.1 μs | **19.58x** | **4.74x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **61.3 μs** | 66.8 μs | 1.49 ms | 301.0 μs | **24.37x** | **4.91x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **77.4 μs** | 77.1 μs | 1.26 ms | 268.2 μs | **16.33x** | **3.48x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **298.2 μs** | 128.8 μs | 1.49 ms | 267.0 μs | **11.57x** | **2.07x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **66.0 μs** | 65.0 μs | 1.25 ms | 297.2 μs | **19.19x** | **4.57x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **69.0 μs** | 70.1 μs | 1.24 ms | 279.0 μs | **17.93x** | **4.05x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **68.3 μs** | 72.3 μs | 1.47 ms | 255.4 μs | **21.60x** | **3.74x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **81.3 μs** | 67.0 μs | 1.18 ms | 247.4 μs | **17.58x** | **3.69x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.00 ms** | 350.3 μs | 4.10 ms | 59.64 ms | **11.71x** | **170.26x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **70.6 μs** | 75.8 μs | 30.09 ms | 54.62 ms | **426.04x** | **773.36x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **73.5 μs** | 75.1 μs | 2.93 ms | 56.29 ms | **39.87x** | **765.43x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **78.0 μs** | 69.8 μs | 1.95 ms | 19.64 ms | **27.92x** | **281.23x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **85.5 μs** | 77.8 μs | 1.65 ms | 15.72 ms | **21.22x** | **202.01x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **1.01 ms** | 190.6 μs | 1.86 ms | 5.98 ms | **9.77x** | **31.38x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **71.3 μs** | 74.5 μs | 4.18 ms | 5.91 ms | **58.60x** | **82.91x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **77.6 μs** | 75.9 μs | 1.35 ms | 5.92 ms | **17.74x** | **78.07x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **70.5 μs** | 71.6 μs | 1.59 ms | 2.18 ms | **22.50x** | **30.98x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **83.5 μs** | 79.6 μs | 1.29 ms | 1.68 ms | **16.24x** | **21.08x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **1.41 ms** | 348.6 μs | 3.79 ms | 123.97 ms | **10.86x** | **355.64x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **77.8 μs** | 77.9 μs | 454.52 ms | 136.78 ms | **5839.71x** | **1757.36x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **71.3 μs** | 74.7 μs | 2.77 ms | 120.38 ms | **38.91x** | **1688.52x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **75.0 μs** | 67.3 μs | 1.96 ms | 121.54 ms | **29.20x** | **1806.12x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **83.7 μs** | 76.4 μs | 1.56 ms | 126.26 ms | **20.36x** | **1652.31x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **380.0 μs** | 187.6 μs | 1.66 ms | 12.54 ms | **8.83x** | **66.84x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **76.9 μs** | 79.6 μs | 46.43 ms | 12.64 ms | **604.00x** | **164.38x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **67.9 μs** | 64.5 μs | 1.22 ms | 12.29 ms | **18.92x** | **190.46x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **65.4 μs** | 65.0 μs | 1.47 ms | 12.48 ms | **22.58x** | **191.81x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **71.5 μs** | 83.5 μs | 1.16 ms | 11.99 ms | **16.26x** | **167.58x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **294.8 μs** | 126.5 μs | 1.43 ms | 1.42 ms | **11.29x** | **11.23x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **72.2 μs** | 68.7 μs | 5.66 ms | 1.45 ms | **82.36x** | **21.16x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **73.0 μs** | 68.7 μs | 1.17 ms | 1.40 ms | **17.10x** | **20.35x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **68.0 μs** | 71.2 μs | 1.42 ms | 1.44 ms | **20.82x** | **21.16x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **66.1 μs** | 68.4 μs | 1.15 ms | 1.38 ms | **17.44x** | **20.83x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.65 ms** | 680.0 μs | 35.68 ms | 134.61 ms | **52.48x** | **197.97x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **82.4 μs** | 73.6 μs | 7.96 ms | 7.48 ms | **108.11x** | **101.69x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **68.5 μs** | 73.2 μs | 21.19 ms | 221.81 ms | **309.30x** | **3238.06x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **65.0 μs** | 62.0 μs | 11.68 ms | 62.71 ms | **188.59x** | **1012.15x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **154.4 μs** | 69.1 μs | 8.12 ms | 46.05 ms | **117.48x** | **666.19x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.14 ms** | 257.3 μs | 9.48 ms | 13.94 ms | **36.84x** | **54.18x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **69.0 μs** | 66.7 μs | 4.87 ms | 1.17 ms | **73.04x** | **17.61x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **71.5 μs** | 69.9 μs | 6.25 ms | 21.95 ms | **89.33x** | **313.92x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **75.3 μs** | 65.1 μs | 7.46 ms | 7.28 ms | **114.55x** | **111.89x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **156.7 μs** | 71.9 μs | 4.85 ms | 4.94 ms | **67.54x** | **68.67x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **782.5 μs** | 131.3 μs | 7.23 ms | 2.00 ms | **55.03x** | **15.25x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **67.1 μs** | 67.0 μs | 4.59 ms | 529.4 μs | **68.55x** | **7.91x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **65.7 μs** | 69.4 μs | 4.62 ms | 2.64 ms | **70.40x** | **40.23x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **72.4 μs** | 69.6 μs | 7.08 ms | 1.27 ms | **101.76x** | **18.22x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **161.2 μs** | 69.8 μs | 4.51 ms | 933.7 μs | **64.64x** | **13.37x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | tiny_int8_3d [read_full @ mps] | 0.0003131669946014881 | fitsio/fitsio_torch_device | 1.864545356248735 |
| fits | tiny_int8_1d [read_full @ mps] | 0.0003046670462936163 | fitsio/fitsio_torch_device | 1.7649062136254938 |
| fits | tiny_int8_2d [read_full @ mps] | 0.00030229100957512856 | fitsio/fitsio_torch_device | 1.7377239193515575 |
| fits | scaled_small [read_full @ mps] | 0.00039204093627631664 | fitsio/fitsio_torch_device | 1.635206303855712 |
| fits | medium_int8_1d [read_full @ mps] | 0.00040020886808633804 | fitsio/fitsio_torch_device | 1.6332261862117523 |
| fits | small_int8_1d [read_full @ mps] | 0.0003105420619249344 | fitsio/fitsio_torch_device | 1.6223322345614442 |
| fits | small_uint32_2d [read_full @ mps] | 0.0003572918940335512 | fitsio/fitsio_torch_device | 1.5562637670224289 |
| fits | small_int8_2d [read_full @ mps] | 0.0003369168844074011 | fitsio/fitsio_torch_device | 1.5466744906400909 |
| fits | small_uint16_2d [read_full @ mps] | 0.0003452498931437731 | fitsio/fitsio_torch_device | 1.5447425580123446 |
| fits | mef_small [read_full @ mps] | 0.0003667909186333418 | fitsio/fitsio_torch_device | 1.5206364190952757 |
| fits | medium_uint32_2d [read_full @ mps] | 0.0016021248884499073 | fitsio/fitsio_torch_device | 1.4970220872018498 |
| fits | small_int8_3d [read_full @ mps] | 0.00035024993121623993 | fitsio/fitsio_torch_device | 1.3136432504180693 |
| fits | multi_mef_10ext [read_full @ mps] | 0.00040679192170500755 | fitsio/fitsio_torch_device | 1.240847535148249 |
| fits | scaled_large [read_full @ mps] | 0.0036521251313388348 | fitsio/fitsio_torch_device | 1.2279489116522297 |
| fits | medium_uint16_2d [read_full @ mps] | 0.0009322080295532942 | fitsio/fitsio_torch_device | 1.1798860782211016 |
| fits | large_int8_2d [read_full @ mps] | 0.0020512090995907784 | astropy/astropy_torch_device | 1.1703639484570307 |
| fits | scaled_medium [read_full @ mps] | 0.0008894579950720072 | fitsio/fitsio_torch_device | 1.1606043344255224 |
| fits | large_int8_1d [read_full @ mps] | 0.0006448328495025635 | astropy/astropy_torch_device | 1.1534629259885834 |
| fits | medium_int8_2d [read_full @ mps] | 0.0007052919827401638 | astropy/astropy_torch_device | 1.136116560857614 |
| fits | large_uint32_2d [read_full @ mps] | 0.005370875122025609 | fitsio/fitsio_torch_device | 1.083556880982901 |
| fits | medium_int16_3d [read_full @ mps] | 0.0006449581123888493 | fitsio/fitsio_torch_device | 1.0517751934246442 |
| fits | medium_int32_3d [read_full @ mps] | 0.0011759581975638866 | fitsio/fitsio_torch_device | 1.0205382560758545 |
| fits | compressed_hcompress_1 [read_full] | 0.0223173750564456 | fitsio/fitsio_torch | 1.010813632222697 |
| fits | compressed_hcompress_1 [read_full @ mps] | 0.021751708816736937 | fitsio/fitsio_torch_device | 1.010104146941212 |
| fits | medium_float32_2d [read_full @ mps] | 0.0007799582090228796 | fitsio/fitsio_torch_device | 1.0066147177093419 |
| fits | large_float32_1d [read_full @ mps] | 0.0007708331104367971 | fitsio/fitsio_torch_device | 1.0051613639724044 |
| fits | medium_int32_2d [read_full @ mps] | 0.0007695420645177364 | fitsio/fitsio_torch_device | 1.0047880956086932 |
| fits | multi_mef_10ext [random_ext_full_reads_200] | 0.006324833957478404 | astropy/astropy | 1.0821856098196725 |
| fits | compressed_rice_1 [read_full] | 0.006735249888151884 | fitsio/fitsio | 1.0169612376241035 |
| fits | compressed_hcompress_1 [read_full] | 0.022212665993720293 | fitsio/fitsio | 1.004505216499974 |
| fitstable | narrow_100000 [predicate_filter] | 0.0014969578478485346 | fitsio/fitsio_torch | 1.31673036250822 |
| fitstable | narrow_10000 [predicate_filter] | 0.00041487510316073895 | fitsio/fitsio_torch | 1.209989291318906 |
| fitstable | narrow_1000 [predicate_filter] | 0.0002982090227305889 | fitsio/fitsio_torch | 1.1168870126801058 |
| fitstable | narrow_1000000 [predicate_filter] | 0.010168708162382245 | fitsio/fitsio_torch | 1.0007463045836047 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `20260710_020649` | fits + fitstable (lab) | 3516 | 34 | lab bench-all + `--mmap-matrix` + CUDA/MPS |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

<!-- BENCH_QUICK_BEGIN -->
| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| — | FITS image I/O | _(no run yet)_ | — | — |
| — | FITS table I/O | _(no run yet)_ | — | — |
<!-- BENCH_QUICK_END -->

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
