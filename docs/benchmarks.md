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
Source: `benchmarks_results/exhaustive_mmap_v060b2_20260708_232039/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.05 ms` (n=269) | `0.49 ms` (n=269) | `0.10 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.06 ms` (n=269) | `0.47 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.05 ms` (n=180) | `2.60 ms` (n=162) | `2.18 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.05 ms` (n=180) | `2.24 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **1.77 ms** | 1.76 ms | 13.32 ms | 2.92 ms | **7.56x** | **1.65x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **7.17 ms** | 7.20 ms | 19.02 ms | 7.26 ms | **2.65x** | **1.01x** |
| Repeated Cutouts (50x 100x100) | CPU | **3.14 ms** | 2.93 ms | 51.76 ms | 3.30 ms | **17.64x** | **1.12x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **46.0 μs** | 48.5 μs | 4.34 ms | 43.11 ms | **94.51x** | **937.91x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **48.5 μs** | 49.6 μs | 2.24 ms | 119.54 ms | **46.18x** | **2466.25x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 88.9 μs | 1.50 ms | 157.7 μs | **16.89x** | **1.77x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **23.60 ms** | 23.67 ms | 45.33 ms | 26.45 ms | **1.92x** | **1.12x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 88.5 μs | 1.50 ms | 156.4 μs | **16.99x** | **1.77x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **20.11 ms** | 20.12 ms | 73.63 ms | 17.55 ms | **3.66x** | **0.87x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 93.6 μs | 1.58 ms | 173.5 μs | **16.90x** | **1.85x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **26.27 ms** | 26.30 ms | 29.91 ms | 25.59 ms | **1.14x** | **0.97x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **713.5 μs** | 708.6 μs | 6.69 ms | 833.7 μs | **9.45x** | **1.18x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 94.5 μs | 1.55 ms | 178.4 μs | **16.44x** | **1.89x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **7.17 ms** | 7.20 ms | 19.02 ms | 7.26 ms | **2.65x** | **1.01x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 47.6 μs | 419.7 μs | 55.5 μs | **8.81x** | **1.16x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **461.0 μs** | 462.7 μs | 1.21 ms | 775.1 μs | **2.63x** | **1.68x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 48.0 μs | 448.2 μs | 56.6 μs | **9.33x** | **1.18x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **1.77 ms** | 1.76 ms | 13.32 ms | 2.92 ms | **7.56x** | **1.65x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 45.7 μs | 425.7 μs | 53.8 μs | **9.32x** | **1.18x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **872.8 μs** | 872.2 μs | 2.00 ms | 1.27 ms | **2.30x** | **1.45x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 46.5 μs | 445.1 μs | 56.1 μs | **9.58x** | **1.21x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **9.93 ms** | 9.80 ms | 27.43 ms | 10.74 ms | **2.80x** | **1.10x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 45.6 μs | 419.6 μs | 55.0 μs | **9.20x** | **1.21x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **262.5 μs** | 265.6 μs | 825.6 μs | 372.7 μs | **3.15x** | **1.42x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 46.2 μs | 436.5 μs | 57.0 μs | **9.44x** | **1.23x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **1.00 ms** | 976.8 μs | 2.10 ms | 1.30 ms | **2.15x** | **1.33x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 45.5 μs | 408.6 μs | 55.2 μs | **8.98x** | **1.21x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **457.1 μs** | 458.3 μs | 1.20 ms | 758.3 μs | **2.62x** | **1.66x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 47.3 μs | 438.1 μs | 55.1 μs | **9.27x** | **1.17x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **1.78 ms** | 1.78 ms | 13.36 ms | 2.94 ms | **7.50x** | **1.65x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 44.5 μs | 414.9 μs | 55.1 μs | **9.33x** | **1.24x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **903.2 μs** | 875.9 μs | 2.00 ms | 1.26 ms | **2.28x** | **1.44x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 46.4 μs | 440.0 μs | 56.4 μs | **9.48x** | **1.22x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.34 ms** | 9.98 ms | 27.75 ms | 10.99 ms | **2.78x** | **1.10x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 49.2 μs | 464.9 μs | 60.0 μs | **9.44x** | **1.22x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **173.1 μs** | 156.0 μs | 736.0 μs | 214.2 μs | **4.72x** | **1.37x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 49.0 μs | 485.8 μs | 62.2 μs | **9.92x** | **1.27x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **507.7 μs** | 506.7 μs | 1.53 ms | 684.5 μs | **3.02x** | **1.35x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 50.7 μs | 482.4 μs | 63.0 μs | **9.52x** | **1.24x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **1.18 ms** | 1.19 ms | 3.88 ms | 1.49 ms | **3.29x** | **1.26x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 48.9 μs | 484.2 μs | 62.7 μs | **9.89x** | **1.28x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **2.21 ms** | 2.20 ms | 11.80 ms | 3.33 ms | **5.37x** | **1.51x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 47.0 μs | 411.2 μs | 55.0 μs | **8.74x** | **1.17x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **97.0 μs** | 80.2 μs | 475.4 μs | 142.6 μs | **5.93x** | **1.78x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 47.9 μs | 440.9 μs | 56.7 μs | **9.21x** | **1.18x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **482.4 μs** | 484.2 μs | 1.28 ms | 808.5 μs | **2.66x** | **1.68x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 49.4 μs | 462.5 μs | 60.4 μs | **9.37x** | **1.22x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **729.2 μs** | 731.7 μs | 1.75 ms | 1.21 ms | **2.39x** | **1.66x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 46.5 μs | 417.7 μs | 55.4 μs | **8.99x** | **1.19x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **119.4 μs** | 121.2 μs | 560.1 μs | 182.2 μs | **4.69x** | **1.53x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 48.3 μs | 440.7 μs | 56.9 μs | **9.12x** | **1.18x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **917.4 μs** | 916.3 μs | 2.10 ms | 1.32 ms | **2.29x** | **1.44x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 49.3 μs | 458.2 μs | 58.5 μs | **9.30x** | **1.19x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **1.40 ms** | 1.40 ms | 10.48 ms | 2.01 ms | **7.47x** | **1.43x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 47.4 μs | 414.3 μs | 56.3 μs | **8.74x** | **1.19x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **60.4 μs** | 64.4 μs | 442.3 μs | 107.8 μs | **7.32x** | **1.79x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 47.0 μs | 434.4 μs | 57.4 μs | **9.25x** | **1.22x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **276.5 μs** | 285.1 μs | 852.0 μs | 387.4 μs | **3.08x** | **1.40x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 48.2 μs | 455.1 μs | 59.3 μs | **9.44x** | **1.23x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **408.8 μs** | 415.2 μs | 1.09 ms | 560.9 μs | **2.67x** | **1.37x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 44.3 μs | 416.9 μs | 53.5 μs | **9.42x** | **1.21x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **77.2 μs** | 82.5 μs | 478.8 μs | 143.7 μs | **6.21x** | **1.86x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 46.9 μs | 434.9 μs | 55.7 μs | **9.27x** | **1.19x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **527.0 μs** | 487.7 μs | 1.27 ms | 811.6 μs | **2.61x** | **1.66x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 46.8 μs | 461.7 μs | 58.3 μs | **9.86x** | **1.24x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **736.3 μs** | 730.4 μs | 1.74 ms | 1.21 ms | **2.39x** | **1.65x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 44.7 μs | 416.0 μs | 54.5 μs | **9.31x** | **1.22x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **118.1 μs** | 120.1 μs | 557.0 μs | 182.0 μs | **4.72x** | **1.54x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 46.8 μs | 440.1 μs | 56.8 μs | **9.41x** | **1.21x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **911.2 μs** | 910.7 μs | 2.06 ms | 1.29 ms | **2.26x** | **1.41x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 48.6 μs | 460.2 μs | 59.2 μs | **9.46x** | **1.22x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **1.44 ms** | 1.40 ms | 10.53 ms | 2.01 ms | **7.50x** | **1.43x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 48.1 μs | 456.4 μs | 59.0 μs | **9.48x** | **1.23x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **49.1 μs** | 54.6 μs | 528.9 μs | 95.8 μs | **10.78x** | **1.95x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 49.0 μs | 480.7 μs | 61.0 μs | **9.81x** | **1.25x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **171.6 μs** | 164.0 μs | 759.6 μs | 224.5 μs | **4.63x** | **1.37x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 49.6 μs | 503.1 μs | 64.9 μs | **10.15x** | **1.31x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **224.4 μs** | 229.4 μs | 938.2 μs | 311.7 μs | **4.18x** | **1.39x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 51.8 μs | 481.7 μs | 62.7 μs | **9.30x** | **1.21x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **315.7 μs** | 318.0 μs | 1.35 ms | 419.5 μs | **4.27x** | **1.33x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 50.4 μs | 478.4 μs | 62.1 μs | **9.50x** | **1.23x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **606.2 μs** | 607.8 μs | 1.78 ms | 931.2 μs | **2.94x** | **1.54x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 56.2 μs | 680.7 μs | 72.5 μs | **12.11x** | **1.29x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **152.9 μs** | 161.3 μs | 943.0 μs | 257.0 μs | **6.17x** | **1.68x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 57.4 μs | 692.0 μs | 70.9 μs | **12.05x** | **1.23x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **46.6 μs** | 57.7 μs | 732.8 μs | 131.8 μs | **15.72x** | **2.83x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **40.6 μs** | 36.8 μs | 2.32 ms | 209.2 μs | **63.03x** | **5.69x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 56.2 μs | 687.8 μs | 70.5 μs | **12.24x** | **1.25x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **5.04 ms** | 5.05 ms | 6.72 ms | 6.86 ms | **1.33x** | **1.36x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **45.3 μs** | 54.9 μs | 729.5 μs | 190.8 μs | **16.09x** | **4.21x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **3.14 ms** | 2.93 ms | 51.76 ms | 3.30 ms | **17.64x** | **1.12x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 51.1 μs | 480.9 μs | 62.5 μs | **9.42x** | **1.22x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **2.42 ms** | 2.42 ms | 10.55 ms | 2.94 ms | **4.36x** | **1.21x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 52.3 μs | 489.1 μs | 62.8 μs | **9.35x** | **1.20x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **637.5 μs** | 645.5 μs | 1.46 ms | 806.9 μs | **2.29x** | **1.27x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 50.8 μs | 486.5 μs | 61.7 μs | **9.57x** | **1.21x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **85.0 μs** | 83.4 μs | 580.5 μs | 129.6 μs | **6.96x** | **1.55x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 47.2 μs | 416.8 μs | 54.2 μs | **8.84x** | **1.15x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **43.1 μs** | 46.2 μs | 412.7 μs | 89.4 μs | **9.57x** | **2.07x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 46.1 μs | 437.0 μs | 56.8 μs | **9.49x** | **1.23x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **64.2 μs** | 69.4 μs | 466.9 μs | 124.3 μs | **7.27x** | **1.94x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 49.0 μs | 459.4 μs | 57.8 μs | **9.37x** | **1.18x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **103.3 μs** | 108.8 μs | 551.8 μs | 184.8 μs | **5.34x** | **1.79x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 45.3 μs | 413.1 μs | 54.1 μs | **9.11x** | **1.19x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **49.8 μs** | 49.7 μs | 418.0 μs | 92.1 μs | **8.41x** | **1.85x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 48.4 μs | 433.7 μs | 57.0 μs | **8.96x** | **1.18x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **94.2 μs** | 96.0 μs | 511.2 μs | 150.2 μs | **5.43x** | **1.59x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 49.1 μs | 455.6 μs | 58.1 μs | **9.28x** | **1.18x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **175.1 μs** | 178.5 μs | 705.2 μs | 263.4 μs | **4.03x** | **1.50x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 45.3 μs | 410.6 μs | 54.3 μs | **9.06x** | **1.20x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **40.8 μs** | 44.6 μs | 403.9 μs | 83.1 μs | **9.89x** | **2.03x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 45.6 μs | 438.1 μs | 56.1 μs | **9.60x** | **1.23x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **53.2 μs** | 58.5 μs | 439.5 μs | 99.1 μs | **8.27x** | **1.86x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 49.8 μs | 608.8 μs | 61.1 μs | **12.21x** | **1.23x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **77.1 μs** | 81.8 μs | 489.2 μs | 125.5 μs | **6.35x** | **1.63x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 45.3 μs | 412.5 μs | 54.4 μs | **9.10x** | **1.20x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **42.8 μs** | 46.0 μs | 406.2 μs | 87.3 μs | **9.48x** | **2.04x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 45.6 μs | 427.5 μs | 56.4 μs | **9.37x** | **1.24x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **66.9 μs** | 70.7 μs | 463.0 μs | 124.2 μs | **6.92x** | **1.85x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 47.3 μs | 455.0 μs | 58.5 μs | **9.63x** | **1.24x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **105.4 μs** | 110.3 μs | 558.3 μs | 182.3 μs | **5.30x** | **1.73x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 46.2 μs | 412.1 μs | 54.1 μs | **8.91x** | **1.17x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **46.6 μs** | 51.1 μs | 422.7 μs | 92.0 μs | **9.07x** | **1.97x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 45.6 μs | 432.0 μs | 56.8 μs | **9.48x** | **1.25x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **90.9 μs** | 96.4 μs | 508.9 μs | 145.4 μs | **5.60x** | **1.60x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 46.7 μs | 456.0 μs | 58.3 μs | **9.76x** | **1.25x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **179.3 μs** | 179.0 μs | 705.7 μs | 265.5 μs | **3.94x** | **1.48x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 47.7 μs | 451.9 μs | 59.2 μs | **9.47x** | **1.24x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **41.9 μs** | 47.4 μs | 518.2 μs | 85.7 μs | **12.36x** | **2.05x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 48.5 μs | 474.3 μs | 63.7 μs | **9.78x** | **1.31x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **46.0 μs** | 53.2 μs | 541.7 μs | 94.3 μs | **11.76x** | **2.05x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 52.2 μs | 498.9 μs | 64.2 μs | **9.56x** | **1.23x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **64.5 μs** | 64.5 μs | 576.9 μs | 104.3 μs | **8.95x** | **1.62x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 49.9 μs | 478.0 μs | 62.8 μs | **9.59x** | **1.26x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **56.2 μs** | 62.0 μs | 525.8 μs | 101.1 μs | **9.36x** | **1.80x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 50.2 μs | 480.5 μs | 62.2 μs | **9.58x** | **1.24x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **68.9 μs** | 75.4 μs | 539.1 μs | 127.9 μs | **7.83x** | **1.86x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 46.0 μs | 441.3 μs | 55.4 μs | **9.59x** | **1.20x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **62.8 μs** | 67.7 μs | 459.6 μs | 122.3 μs | **7.32x** | **1.95x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 47.9 μs | 436.6 μs | 56.2 μs | **9.12x** | **1.17x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **64.3 μs** | 68.1 μs | 457.8 μs | 121.1 μs | **7.12x** | **1.88x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 46.4 μs | 438.3 μs | 56.6 μs | **9.45x** | **1.22x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **64.7 μs** | 69.3 μs | 458.8 μs | 121.5 μs | **7.10x** | **1.88x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 47.8 μs | 437.0 μs | 56.7 μs | **9.15x** | **1.19x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **64.5 μs** | 68.3 μs | 456.0 μs | 120.5 μs | **7.07x** | **1.87x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 47.0 μs | 433.6 μs | 56.5 μs | **9.22x** | **1.20x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **64.9 μs** | 68.8 μs | 462.1 μs | 123.0 μs | **7.12x** | **1.90x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 45.9 μs | 415.9 μs | 53.9 μs | **9.06x** | **1.18x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **36.1 μs** | 39.7 μs | 400.6 μs | 83.1 μs | **11.09x** | **2.30x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 47.1 μs | 436.2 μs | 56.5 μs | **9.26x** | **1.20x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **40.9 μs** | 46.3 μs | 423.8 μs | 86.1 μs | **10.36x** | **2.10x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 48.0 μs | 455.9 μs | 58.0 μs | **9.50x** | **1.21x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **43.3 μs** | 47.5 μs | 434.5 μs | 86.4 μs | **10.03x** | **1.99x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 45.2 μs | 409.9 μs | 53.1 μs | **9.07x** | **1.18x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **35.2 μs** | 40.1 μs | 403.5 μs | 84.4 μs | **11.46x** | **2.40x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 45.3 μs | 438.2 μs | 55.2 μs | **9.67x** | **1.22x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **42.9 μs** | 48.9 μs | 415.7 μs | 86.8 μs | **9.69x** | **2.02x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 48.3 μs | 462.5 μs | 59.5 μs | **9.57x** | **1.23x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **42.7 μs** | 49.4 μs | 436.1 μs | 88.2 μs | **10.21x** | **2.06x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 46.5 μs | 412.6 μs | 54.1 μs | **8.87x** | **1.16x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **35.3 μs** | 40.8 μs | 407.5 μs | 81.0 μs | **11.55x** | **2.29x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 47.3 μs | 437.3 μs | 55.3 μs | **9.25x** | **1.17x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **37.3 μs** | 43.1 μs | 418.9 μs | 83.7 μs | **11.24x** | **2.24x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 48.8 μs | 456.7 μs | 58.3 μs | **9.36x** | **1.19x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **38.0 μs** | 43.2 μs | 428.2 μs | 84.2 μs | **11.26x** | **2.21x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 45.8 μs | 417.6 μs | 55.0 μs | **9.11x** | **1.20x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **34.4 μs** | 38.9 μs | 405.6 μs | 83.3 μs | **11.78x** | **2.42x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 47.0 μs | 435.0 μs | 56.2 μs | **9.25x** | **1.20x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **42.5 μs** | 45.8 μs | 423.0 μs | 88.1 μs | **9.95x** | **2.07x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 48.6 μs | 453.4 μs | 58.5 μs | **9.32x** | **1.20x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **42.2 μs** | 48.8 μs | 443.1 μs | 86.9 μs | **10.50x** | **2.06x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 44.9 μs | 409.0 μs | 55.4 μs | **9.12x** | **1.23x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **35.3 μs** | 39.5 μs | 404.4 μs | 82.1 μs | **11.44x** | **2.32x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 46.5 μs | 426.7 μs | 56.9 μs | **9.17x** | **1.22x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **43.5 μs** | 46.5 μs | 423.5 μs | 87.1 μs | **9.74x** | **2.00x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 48.3 μs | 457.1 μs | 58.0 μs | **9.46x** | **1.20x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **44.3 μs** | 49.7 μs | 435.5 μs | 87.9 μs | **9.82x** | **1.98x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 50.1 μs | 459.7 μs | 59.3 μs | **9.17x** | **1.18x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **33.8 μs** | 41.6 μs | 519.2 μs | 82.7 μs | **15.34x** | **2.44x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 50.9 μs | 475.8 μs | 63.4 μs | **9.34x** | **1.24x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **34.7 μs** | 42.4 μs | 538.6 μs | 83.9 μs | **15.50x** | **2.41x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 51.1 μs | 494.8 μs | 63.9 μs | **9.69x** | **1.25x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **36.0 μs** | 42.8 μs | 555.2 μs | 84.7 μs | **15.44x** | **2.36x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **667.0 μs** | 131.4 μs | 3.32 ms | 5.17 ms | **25.27x** | **39.38x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **46.9 μs** | 48.0 μs | 8.08 ms | 5.15 ms | **172.49x** | **109.96x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **45.1 μs** | 46.3 μs | 1.44 ms | 5.15 ms | **31.84x** | **114.12x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **45.4 μs** | 48.0 μs | 1.77 ms | 2.39 ms | **39.07x** | **52.69x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **68.4 μs** | 50.1 μs | 2.71 ms | 2.00 ms | **54.00x** | **39.86x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **464.5 μs** | 100.3 μs | 2.03 ms | 762.9 μs | **20.20x** | **7.61x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **47.9 μs** | 49.7 μs | 2.26 ms | 764.5 μs | **47.28x** | **15.96x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **47.4 μs** | 50.4 μs | 1.37 ms | 753.7 μs | **28.85x** | **15.88x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **46.9 μs** | 49.3 μs | 1.73 ms | 461.0 μs | **36.95x** | **9.83x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **69.8 μs** | 52.4 μs | 1.58 ms | 365.9 μs | **30.11x** | **6.98x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **12.70 ms** | 6.34 ms | 94.27 ms | 328.23 ms | **14.88x** | **51.81x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **52.5 μs** | 51.6 μs | 26.66 ms | 42.61 ms | **516.82x** | **826.02x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **48.3 μs** | 47.6 μs | 52.84 ms | 518.52 ms | **1111.24x** | **10904.31x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **52.2 μs** | 49.8 μs | 22.25 ms | 126.01 ms | **446.66x** | **2529.67x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **85.2 μs** | 52.4 μs | 21.83 ms | 123.53 ms | **416.31x** | **2355.51x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **1.85 ms** | 705.1 μs | 8.57 ms | 26.82 ms | **12.15x** | **38.04x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **46.5 μs** | 47.8 μs | 2.63 ms | 3.46 ms | **56.50x** | **74.34x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **46.0 μs** | 48.5 μs | 4.34 ms | 43.11 ms | **94.51x** | **937.91x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **47.3 μs** | 48.9 μs | 3.46 ms | 11.81 ms | **73.14x** | **249.80x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **76.6 μs** | 49.1 μs | 2.61 ms | 8.45 ms | **53.05x** | **171.95x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **630.1 μs** | 132.8 μs | 3.22 ms | 2.86 ms | **24.23x** | **21.57x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **46.9 μs** | 47.3 μs | 1.98 ms | 524.1 μs | **42.18x** | **11.18x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **44.8 μs** | 40.2 μs | 2.10 ms | 4.36 ms | **52.33x** | **108.51x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **47.9 μs** | 48.3 μs | 2.76 ms | 1.46 ms | **57.58x** | **30.50x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **77.5 μs** | 49.9 μs | 2.01 ms | 1.03 ms | **40.35x** | **20.59x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **450.7 μs** | 92.1 μs | 2.70 ms | 503.1 μs | **29.31x** | **5.46x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **50.6 μs** | 50.2 μs | 1.91 ms | 261.1 μs | **37.99x** | **5.20x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **69.0 μs** | 79.0 μs | 1.91 ms | 722.0 μs | **27.69x** | **10.47x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **49.3 μs** | 52.7 μs | 2.63 ms | 410.3 μs | **53.40x** | **8.33x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **73.7 μs** | 49.2 μs | 1.87 ms | 311.1 μs | **38.11x** | **6.32x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **10.13 ms** | 6.21 ms | 28.91 ms | 9.22 ms | **4.66x** | **1.48x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **48.9 μs** | 48.9 μs | 3.60 ms | 24.63 ms | **73.59x** | **503.91x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **49.4 μs** | 47.5 μs | 6.28 ms | 5.42 ms | **132.11x** | **114.05x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **51.0 μs** | 49.8 μs | 4.04 ms | 2.69 ms | **81.00x** | **53.91x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **69.7 μs** | 50.1 μs | 3.54 ms | 2.64 ms | **70.62x** | **52.69x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.33 ms** | 703.6 μs | 4.63 ms | 1.05 ms | **6.59x** | **1.49x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **47.4 μs** | 49.7 μs | 1.72 ms | 2.61 ms | **36.29x** | **55.17x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **46.1 μs** | 46.6 μs | 1.91 ms | 629.3 μs | **41.46x** | **13.65x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **48.5 μs** | 49.4 μs | 2.15 ms | 461.2 μs | **44.24x** | **9.50x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **66.7 μs** | 50.2 μs | 1.70 ms | 415.3 μs | **33.75x** | **8.27x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **460.1 μs** | 139.3 μs | 2.13 ms | 299.0 μs | **15.32x** | **2.15x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **45.7 μs** | 41.5 μs | 1.45 ms | 447.9 μs | **34.92x** | **10.78x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **45.6 μs** | 45.2 μs | 1.44 ms | 239.3 μs | **31.79x** | **5.29x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **46.5 μs** | 48.9 μs | 1.87 ms | 228.8 μs | **40.21x** | **4.92x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **68.5 μs** | 49.5 μs | 1.44 ms | 205.2 μs | **29.07x** | **4.14x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **900.8 μs** | 219.9 μs | 3.38 ms | 390.7 μs | **15.35x** | **1.78x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **73.5 μs** | 75.2 μs | 2.40 ms | 405.6 μs | **32.60x** | **5.52x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **76.0 μs** | 105.4 μs | 2.63 ms | 347.6 μs | **34.55x** | **4.57x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **67.0 μs** | 72.3 μs | 3.17 ms | 404.9 μs | **47.32x** | **6.05x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **132.8 μs** | 83.3 μs | 2.38 ms | 310.3 μs | **28.58x** | **3.73x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.31 ms** | 543.6 μs | 3.97 ms | 46.83 ms | **7.30x** | **86.13x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **47.5 μs** | 50.8 μs | 28.69 ms | 44.48 ms | **603.61x** | **935.78x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **47.9 μs** | 49.8 μs | 2.34 ms | 45.77 ms | **48.90x** | **954.88x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **51.3 μs** | 50.5 μs | 2.26 ms | 16.26 ms | **44.73x** | **321.88x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **68.9 μs** | 53.1 μs | 1.74 ms | 13.07 ms | **32.84x** | **246.11x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **572.0 μs** | 138.2 μs | 2.12 ms | 4.85 ms | **15.35x** | **35.06x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **47.3 μs** | 52.2 μs | 4.15 ms | 4.61 ms | **87.58x** | **97.29x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **48.3 μs** | 50.2 μs | 1.55 ms | 4.79 ms | **32.09x** | **99.12x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **49.3 μs** | 50.4 μs | 1.96 ms | 1.92 ms | **39.79x** | **38.94x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **68.2 μs** | 51.5 μs | 1.53 ms | 1.53 ms | **29.60x** | **29.72x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **802.0 μs** | 502.1 μs | 4.05 ms | 124.65 ms | **8.06x** | **248.25x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **55.6 μs** | 51.6 μs | 501.04 ms | 124.50 ms | **9707.06x** | **2412.15x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **48.5 μs** | 49.6 μs | 2.24 ms | 119.54 ms | **46.18x** | **2466.25x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **50.6 μs** | 49.4 μs | 2.10 ms | 122.73 ms | **42.55x** | **2484.67x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **66.5 μs** | 51.8 μs | 1.66 ms | 122.75 ms | **32.06x** | **2369.62x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **292.0 μs** | 132.9 μs | 1.97 ms | 11.71 ms | **14.79x** | **88.09x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **47.0 μs** | 47.9 μs | 51.14 ms | 11.60 ms | **1087.79x** | **246.82x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **46.5 μs** | 48.5 μs | 1.44 ms | 11.43 ms | **30.99x** | **245.85x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **49.3 μs** | 48.3 μs | 1.76 ms | 11.57 ms | **36.49x** | **239.73x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **66.2 μs** | 51.6 μs | 1.38 ms | 11.73 ms | **26.79x** | **227.47x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **441.0 μs** | 93.4 μs | 1.77 ms | 1.32 ms | **18.96x** | **14.17x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **46.9 μs** | 48.3 μs | 6.43 ms | 1.33 ms | **137.14x** | **28.28x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **46.1 μs** | 46.9 μs | 1.39 ms | 1.31 ms | **30.05x** | **28.50x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **45.7 μs** | 48.0 μs | 1.72 ms | 1.30 ms | **37.61x** | **28.50x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **66.3 μs** | 49.5 μs | 1.34 ms | 1.31 ms | **27.11x** | **26.43x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.26 ms** | 719.2 μs | 48.03 ms | 115.59 ms | **66.77x** | **160.71x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **48.4 μs** | 48.3 μs | 7.62 ms | 5.36 ms | **157.78x** | **110.98x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **46.6 μs** | 48.4 μs | 28.74 ms | 182.52 ms | **616.63x** | **3916.06x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **48.2 μs** | 50.3 μs | 11.37 ms | 52.65 ms | **235.88x** | **1092.28x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **144.4 μs** | 50.9 μs | 7.54 ms | 40.02 ms | **148.27x** | **786.47x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.24 ms** | 151.7 μs | 10.69 ms | 10.92 ms | **70.46x** | **71.98x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **46.0 μs** | 47.1 μs | 5.94 ms | 961.3 μs | **129.34x** | **20.92x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **45.0 μs** | 46.6 μs | 6.81 ms | 17.50 ms | **151.38x** | **388.89x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **46.6 μs** | 47.8 μs | 9.19 ms | 5.27 ms | **197.45x** | **113.08x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **147.0 μs** | 48.8 μs | 5.88 ms | 3.85 ms | **120.55x** | **78.95x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **847.2 μs** | 87.4 μs | 9.05 ms | 1.60 ms | **103.52x** | **18.27x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **44.8 μs** | 48.4 μs | 5.69 ms | 422.0 μs | **126.85x** | **9.41x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **36.9 μs** | 46.3 μs | 5.77 ms | 2.30 ms | **156.36x** | **62.30x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **38.4 μs** | 41.1 μs | 8.88 ms | 1.01 ms | **231.36x** | **26.44x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **132.5 μs** | 44.1 μs | 5.65 ms | 772.5 μs | **128.13x** | **17.51x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | compressed_hcompress_1 [read_full] | 0.02626755740493536 | fitsio/fitsio_torch | 1.0273285158712901 |
| fits | compressed_gzip_2 [read_full] | 0.020118449814617634 | fitsio/fitsio | 1.1460293083368114 |
| fits | compressed_hcompress_1 [read_full] | 0.02630484290421009 | fitsio/fitsio | 1.028083538828516 |
| fitstable | narrow_1000 [predicate_filter] | 0.00044406019151210785 | fitsio/fitsio_torch | 2.062113466711645 |
| fitstable | narrow_10000 [predicate_filter] | 0.00046013854444026947 | fitsio/fitsio_torch | 1.5436262579240232 |
| fitstable | narrow_100000 [predicate_filter] | 0.001331373117864132 | fitsio/fitsio_torch | 1.267858473196326 |
| fitstable | narrow_1000000 [predicate_filter] | 0.010126314125955105 | fitsio/fitsio_torch | 1.098466992032941 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mmap_v060b2_20260708_232039` | fits + fitstable (lab) | 2754 | 7 | lab bench-all + `--mmap-matrix` |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
