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
Source: `benchmarks_results/uint_fix_20260708/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.06 ms` (n=269) | `0.49 ms` (n=269) | `0.10 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | — | — | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.05 ms` (n=180) | `2.23 ms` (n=342) | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | — | — | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
## Performance Highlights

The following table showcases median wall-clock execution times of key representative FITS benchmarks.
In almost all core I/O paths, `torchfits` is significantly faster than standard astronomical tools, with extra performance wins from persistent handle caches and direct-to-device transfers.

| Benchmark Case | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **1.89 ms** | 1.77 ms | 12.91 ms | 2.86 ms | **7.29x** | **1.62x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **12.88 ms** | 12.81 ms | 33.91 ms | 11.35 ms | **2.65x** | **0.89x** |
| Repeated Cutouts (50x 100x100) | CPU | **3.32 ms** | 3.10 ms | 51.60 ms | 3.35 ms | **16.62x** | **1.08x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **51.7 μs** | 54.4 μs | 46.55 ms | — | **900.17x** | **—** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **51.1 μs** | 55.4 μs | 122.28 ms | — | **2392.60x** | **—** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 88.9 μs | 1.50 ms | 155.5 μs | **16.92x** | **1.75x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **24.06 ms** | 24.05 ms | 46.56 ms | 27.04 ms | **1.94x** | **1.12x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 87.1 μs | 1.51 ms | 154.3 μs | **17.28x** | **1.77x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **20.61 ms** | 20.56 ms | 73.05 ms | 23.61 ms | **3.55x** | **1.15x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 94.3 μs | 1.58 ms | 172.6 μs | **16.72x** | **1.83x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **46.14 ms** | 46.02 ms | 52.60 ms | 44.72 ms | **1.14x** | **0.97x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **715.5 μs** | 708.9 μs | 6.66 ms | 824.0 μs | **9.40x** | **1.16x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 95.6 μs | 1.58 ms | 173.4 μs | **16.50x** | **1.81x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **12.88 ms** | 12.81 ms | 33.91 ms | 11.35 ms | **2.65x** | **0.89x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 45.8 μs | 410.8 μs | 53.6 μs | **8.96x** | **1.17x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **656.4 μs** | 460.6 μs | 1.22 ms | 770.1 μs | **2.65x** | **1.67x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 47.4 μs | 434.9 μs | 55.8 μs | **9.18x** | **1.18x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **1.89 ms** | 1.77 ms | 12.91 ms | 2.86 ms | **7.29x** | **1.62x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 44.0 μs | 406.8 μs | 53.7 μs | **9.24x** | **1.22x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **936.3 μs** | 863.5 μs | 2.01 ms | 1.24 ms | **2.33x** | **1.44x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 47.0 μs | 436.0 μs | 55.8 μs | **9.28x** | **1.19x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **10.46 ms** | 9.67 ms | 27.81 ms | 10.43 ms | **2.88x** | **1.08x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 43.0 μs | 407.6 μs | 54.4 μs | **9.47x** | **1.26x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **327.2 μs** | 279.9 μs | 834.1 μs | 380.2 μs | **2.98x** | **1.36x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 44.9 μs | 428.0 μs | 55.6 μs | **9.53x** | **1.24x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **1.07 ms** | 984.1 μs | 2.12 ms | 1.35 ms | **2.16x** | **1.37x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 42.6 μs | 405.1 μs | 53.8 μs | **9.52x** | **1.26x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **521.4 μs** | 454.6 μs | 1.23 ms | 759.7 μs | **2.70x** | **1.67x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 46.3 μs | 439.8 μs | 56.0 μs | **9.49x** | **1.21x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **1.84 ms** | 1.77 ms | 12.90 ms | 2.87 ms | **7.27x** | **1.62x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 45.4 μs | 412.3 μs | 55.2 μs | **9.08x** | **1.22x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **947.1 μs** | 870.7 μs | 1.97 ms | 1.24 ms | **2.26x** | **1.43x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 46.5 μs | 433.8 μs | 55.4 μs | **9.33x** | **1.19x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.26 ms** | 9.84 ms | 28.12 ms | 10.83 ms | **2.86x** | **1.10x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 46.3 μs | 458.4 μs | 57.9 μs | **9.90x** | **1.25x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **233.5 μs** | 160.0 μs | 742.2 μs | 207.6 μs | **4.64x** | **1.30x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 48.7 μs | 481.5 μs | 61.1 μs | **9.88x** | **1.25x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **563.4 μs** | 521.1 μs | 1.55 ms | 695.9 μs | **2.97x** | **1.34x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 49.6 μs | 482.7 μs | 60.9 μs | **9.73x** | **1.23x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **14.12 ms** | 8.57 ms | 3.91 ms | 1.50 ms | **0.46x** | **0.17x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 55.2 μs | 485.4 μs | 60.4 μs | **8.79x** | **1.09x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **31.33 ms** | 24.22 ms | 11.52 ms | 3.31 ms | **0.48x** | **0.14x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 45.9 μs | 413.7 μs | 52.9 μs | **9.02x** | **1.15x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **138.5 μs** | 87.9 μs | 488.9 μs | 144.1 μs | **5.56x** | **1.64x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 46.0 μs | 442.9 μs | 55.9 μs | **9.63x** | **1.21x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **594.6 μs** | 508.2 μs | 1.28 ms | 819.9 μs | **2.51x** | **1.61x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 47.8 μs | 457.5 μs | 58.1 μs | **9.57x** | **1.22x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **849.5 μs** | 746.8 μs | 1.76 ms | 1.23 ms | **2.36x** | **1.64x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 45.0 μs | 417.5 μs | 53.4 μs | **9.28x** | **1.19x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **126.8 μs** | 123.8 μs | 567.7 μs | 185.4 μs | **4.59x** | **1.50x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 45.8 μs | 441.7 μs | 56.0 μs | **9.64x** | **1.22x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **988.2 μs** | 913.2 μs | 2.13 ms | 1.30 ms | **2.33x** | **1.43x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 47.7 μs | 468.1 μs | 59.1 μs | **9.81x** | **1.24x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **1.47 ms** | 1.39 ms | 10.09 ms | 1.97 ms | **7.24x** | **1.41x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 44.8 μs | 413.4 μs | 53.7 μs | **9.22x** | **1.20x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **79.1 μs** | 67.7 μs | 444.9 μs | 108.1 μs | **6.57x** | **1.60x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 45.1 μs | 443.3 μs | 56.6 μs | **9.83x** | **1.25x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **337.3 μs** | 287.7 μs | 873.9 μs | 396.8 μs | **3.04x** | **1.38x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 45.8 μs | 457.3 μs | 58.8 μs | **9.97x** | **1.28x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **499.1 μs** | 424.5 μs | 1.13 ms | 570.4 μs | **2.66x** | **1.34x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 44.2 μs | 414.7 μs | 55.0 μs | **9.39x** | **1.24x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **90.9 μs** | 82.1 μs | 476.8 μs | 145.7 μs | **5.80x** | **1.77x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 44.9 μs | 439.7 μs | 55.7 μs | **9.78x** | **1.24x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **540.7 μs** | 505.0 μs | 1.29 ms | 825.8 μs | **2.55x** | **1.64x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 46.8 μs | 461.8 μs | 57.6 μs | **9.88x** | **1.23x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **825.0 μs** | 744.5 μs | 1.80 ms | 1.25 ms | **2.41x** | **1.67x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 44.7 μs | 415.5 μs | 53.5 μs | **9.30x** | **1.20x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **156.7 μs** | 126.1 μs | 563.2 μs | 185.0 μs | **4.46x** | **1.47x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 46.5 μs | 443.2 μs | 56.0 μs | **9.54x** | **1.21x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **1.03 ms** | 947.0 μs | 2.12 ms | 1.35 ms | **2.24x** | **1.43x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 44.6 μs | 459.7 μs | 57.2 μs | **10.31x** | **1.28x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **1.84 ms** | 1.45 ms | 10.39 ms | 2.05 ms | **7.14x** | **1.41x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 46.5 μs | 460.2 μs | 59.8 μs | **9.89x** | **1.28x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **62.2 μs** | 57.6 μs | 554.3 μs | 96.8 μs | **9.62x** | **1.68x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 48.3 μs | 479.2 μs | 61.3 μs | **9.91x** | **1.27x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **201.0 μs** | 173.8 μs | 787.9 μs | 229.8 μs | **4.53x** | **1.32x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 50.2 μs | 493.7 μs | 63.7 μs | **9.83x** | **1.27x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **241.3 μs** | 239.5 μs | 959.2 μs | 316.1 μs | **4.01x** | **1.32x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 49.0 μs | 474.9 μs | 61.5 μs | **9.69x** | **1.25x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **1.60 ms** | 959.6 μs | 1.39 ms | 418.6 μs | **1.45x** | **0.44x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 49.1 μs | 482.1 μs | 61.5 μs | **9.82x** | **1.25x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **2.54 ms** | 1.80 ms | 1.81 ms | 923.6 μs | **1.01x** | **0.51x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 55.1 μs | 691.5 μs | 68.8 μs | **12.56x** | **1.25x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **161.6 μs** | 166.8 μs | 954.0 μs | 254.0 μs | **5.90x** | **1.57x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 55.3 μs | 686.3 μs | 70.0 μs | **12.42x** | **1.27x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **56.1 μs** | 58.4 μs | 740.1 μs | 138.6 μs | **13.20x** | **2.47x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **45.3 μs** | 35.6 μs | 2.33 ms | 208.4 μs | **65.45x** | **5.86x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 54.3 μs | 687.1 μs | 71.2 μs | **12.66x** | **1.31x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **4.72 ms** | 4.71 ms | 6.90 ms | 6.74 ms | **1.46x** | **1.43x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **60.4 μs** | 59.3 μs | 719.1 μs | 190.8 μs | **12.12x** | **3.22x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **3.32 ms** | 3.10 ms | 51.60 ms | 3.35 ms | **16.62x** | **1.08x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 50.1 μs | 488.4 μs | 65.7 μs | **9.74x** | **1.31x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **2.46 ms** | 2.40 ms | 11.86 ms | 2.94 ms | **4.94x** | **1.22x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 50.7 μs | 502.7 μs | 63.6 μs | **9.91x** | **1.25x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **675.4 μs** | 641.8 μs | 1.48 ms | 805.6 μs | **2.30x** | **1.26x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 48.9 μs | 493.7 μs | 62.7 μs | **10.10x** | **1.28x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **114.5 μs** | 86.8 μs | 596.4 μs | 132.8 μs | **6.87x** | **1.53x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 45.0 μs | 418.7 μs | 54.4 μs | **9.30x** | **1.21x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **70.7 μs** | 49.5 μs | 423.5 μs | 88.9 μs | **8.55x** | **1.80x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 45.9 μs | 441.4 μs | 56.6 μs | **9.62x** | **1.23x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **96.1 μs** | 75.8 μs | 469.5 μs | 127.3 μs | **6.19x** | **1.68x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 45.9 μs | 458.9 μs | 58.6 μs | **10.01x** | **1.28x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **141.9 μs** | 113.1 μs | 563.0 μs | 184.7 μs | **4.98x** | **1.63x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 45.5 μs | 415.8 μs | 54.3 μs | **9.13x** | **1.19x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **57.5 μs** | 52.6 μs | 419.1 μs | 92.9 μs | **7.97x** | **1.77x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 46.1 μs | 444.9 μs | 55.2 μs | **9.66x** | **1.20x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **102.9 μs** | 101.9 μs | 514.6 μs | 148.6 μs | **5.05x** | **1.46x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 47.7 μs | 458.8 μs | 57.7 μs | **9.61x** | **1.21x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **237.6 μs** | 184.5 μs | 708.2 μs | 270.4 μs | **3.84x** | **1.47x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 44.0 μs | 413.0 μs | 54.7 μs | **9.39x** | **1.24x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **52.0 μs** | 46.3 μs | 402.8 μs | 83.3 μs | **8.69x** | **1.80x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 46.1 μs | 437.5 μs | 56.1 μs | **9.49x** | **1.22x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **65.4 μs** | 59.6 μs | 444.1 μs | 99.5 μs | **7.45x** | **1.67x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 47.2 μs | 455.3 μs | 58.7 μs | **9.64x** | **1.24x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **105.6 μs** | 83.2 μs | 494.8 μs | 127.1 μs | **5.95x** | **1.53x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 42.7 μs | 419.3 μs | 54.4 μs | **9.83x** | **1.28x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **55.6 μs** | 47.9 μs | 414.6 μs | 87.7 μs | **8.66x** | **1.83x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 44.8 μs | 433.5 μs | 56.2 μs | **9.68x** | **1.25x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **86.4 μs** | 75.7 μs | 454.3 μs | 126.9 μs | **6.00x** | **1.68x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 45.6 μs | 455.2 μs | 59.5 μs | **9.99x** | **1.30x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **147.4 μs** | 115.2 μs | 570.5 μs | 186.9 μs | **4.95x** | **1.62x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 44.7 μs | 410.4 μs | 54.6 μs | **9.18x** | **1.22x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **56.5 μs** | 52.8 μs | 424.1 μs | 91.5 μs | **8.03x** | **1.73x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 43.9 μs | 433.9 μs | 55.3 μs | **9.88x** | **1.26x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **105.2 μs** | 101.9 μs | 529.2 μs | 147.5 μs | **5.19x** | **1.45x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 45.4 μs | 454.0 μs | 57.9 μs | **10.00x** | **1.27x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **195.2 μs** | 183.0 μs | 716.7 μs | 267.2 μs | **3.92x** | **1.46x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 51.3 μs | 463.5 μs | 58.9 μs | **9.03x** | **1.15x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **50.8 μs** | 44.7 μs | 529.1 μs | 86.0 μs | **11.84x** | **1.92x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 47.4 μs | 482.3 μs | 60.5 μs | **10.18x** | **1.28x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **58.1 μs** | 58.3 μs | 536.3 μs | 92.3 μs | **9.22x** | **1.59x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 46.9 μs | 510.0 μs | 63.3 μs | **10.87x** | **1.35x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **70.1 μs** | 69.2 μs | 577.3 μs | 105.0 μs | **8.34x** | **1.52x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 48.6 μs | 478.8 μs | 60.9 μs | **9.85x** | **1.25x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **182.8 μs** | 116.6 μs | 521.2 μs | 104.6 μs | **4.47x** | **0.90x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 46.6 μs | 484.5 μs | 59.0 μs | **10.39x** | **1.26x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **226.9 μs** | 151.3 μs | 543.3 μs | 128.8 μs | **3.59x** | **0.85x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 45.5 μs | 436.6 μs | 56.4 μs | **9.61x** | **1.24x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **93.8 μs** | 70.9 μs | 460.0 μs | 120.0 μs | **6.49x** | **1.69x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 46.3 μs | 440.5 μs | 55.8 μs | **9.51x** | **1.20x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **92.5 μs** | 69.3 μs | 466.9 μs | 121.3 μs | **6.74x** | **1.75x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 45.0 μs | 436.8 μs | 55.8 μs | **9.70x** | **1.24x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **96.5 μs** | 70.4 μs | 460.5 μs | 121.1 μs | **6.54x** | **1.72x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 45.0 μs | 438.0 μs | 55.9 μs | **9.73x** | **1.24x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **103.2 μs** | 67.4 μs | 465.8 μs | 121.9 μs | **6.91x** | **1.81x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 46.5 μs | 449.0 μs | 56.8 μs | **9.66x** | **1.22x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **97.8 μs** | 71.3 μs | 460.8 μs | 125.8 μs | **6.47x** | **1.77x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 45.2 μs | 416.3 μs | 53.7 μs | **9.21x** | **1.19x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **65.4 μs** | 41.3 μs | 413.0 μs | 82.6 μs | **10.00x** | **2.00x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 45.9 μs | 437.1 μs | 55.4 μs | **9.52x** | **1.21x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **73.7 μs** | 47.5 μs | 424.3 μs | 84.3 μs | **8.93x** | **1.78x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 46.3 μs | 464.4 μs | 57.6 μs | **10.03x** | **1.24x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **70.7 μs** | 51.1 μs | 431.4 μs | 86.0 μs | **8.45x** | **1.68x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 44.2 μs | 420.9 μs | 53.7 μs | **9.51x** | **1.21x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **45.5 μs** | 42.2 μs | 407.3 μs | 84.2 μs | **9.65x** | **2.00x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 46.3 μs | 440.5 μs | 54.9 μs | **9.51x** | **1.19x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **52.5 μs** | 51.4 μs | 427.5 μs | 87.7 μs | **8.32x** | **1.71x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 47.8 μs | 461.6 μs | 58.6 μs | **9.65x** | **1.23x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **53.8 μs** | 51.2 μs | 437.3 μs | 88.6 μs | **8.54x** | **1.73x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 43.4 μs | 413.6 μs | 53.0 μs | **9.53x** | **1.22x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **43.3 μs** | 41.2 μs | 399.3 μs | 83.1 μs | **9.69x** | **2.02x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 44.2 μs | 440.8 μs | 55.9 μs | **9.98x** | **1.27x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **47.8 μs** | 44.3 μs | 422.6 μs | 83.2 μs | **9.55x** | **1.88x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 46.3 μs | 464.0 μs | 58.5 μs | **10.03x** | **1.26x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **47.3 μs** | 43.6 μs | 440.7 μs | 84.4 μs | **10.10x** | **1.93x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 44.9 μs | 414.0 μs | 54.0 μs | **9.22x** | **1.20x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **46.5 μs** | 41.2 μs | 414.2 μs | 81.4 μs | **10.05x** | **1.97x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 45.4 μs | 437.9 μs | 56.3 μs | **9.65x** | **1.24x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **50.8 μs** | 48.9 μs | 415.8 μs | 84.7 μs | **8.50x** | **1.73x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 45.9 μs | 458.9 μs | 57.2 μs | **10.01x** | **1.25x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **50.8 μs** | 49.2 μs | 437.5 μs | 86.2 μs | **8.89x** | **1.75x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 42.6 μs | 417.2 μs | 54.4 μs | **9.78x** | **1.27x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **46.8 μs** | 40.6 μs | 410.4 μs | 83.1 μs | **10.11x** | **2.05x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 44.5 μs | 438.4 μs | 55.6 μs | **9.86x** | **1.25x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **54.4 μs** | 52.6 μs | 415.6 μs | 89.5 μs | **7.90x** | **1.70x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 46.1 μs | 461.7 μs | 57.3 μs | **10.02x** | **1.24x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **56.6 μs** | 53.5 μs | 443.2 μs | 89.7 μs | **8.28x** | **1.68x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 45.9 μs | 458.6 μs | 57.1 μs | **10.00x** | **1.24x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **44.9 μs** | 46.8 μs | 516.3 μs | 85.2 μs | **11.51x** | **1.90x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 48.8 μs | 480.0 μs | 61.4 μs | **9.84x** | **1.26x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **47.9 μs** | 43.1 μs | 524.6 μs | 84.6 μs | **12.16x** | **1.96x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 50.1 μs | 500.4 μs | 63.0 μs | **9.98x** | **1.26x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **46.8 μs** | 44.7 μs | 551.2 μs | 84.0 μs | **12.32x** | **1.88x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **648.4 μs** | 133.9 μs | 5.50 ms | — | **41.11x** | **—** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **51.4 μs** | 53.4 μs | 5.46 ms | — | **106.35x** | **—** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **51.7 μs** | 52.1 μs | 5.48 ms | — | **105.92x** | **—** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **51.6 μs** | 53.3 μs | 2.41 ms | — | **46.66x** | **—** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **67.2 μs** | 55.3 μs | 1.99 ms | — | **36.05x** | **—** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **464.6 μs** | 99.8 μs | 793.7 μs | — | **7.95x** | **—** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **52.4 μs** | 54.5 μs | 796.5 μs | — | **15.21x** | **—** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **51.7 μs** | 53.7 μs | 791.7 μs | — | **15.32x** | **—** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **52.6 μs** | 54.7 μs | 458.7 μs | — | **8.71x** | **—** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **71.2 μs** | 56.6 μs | 372.0 μs | — | **6.57x** | **—** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **12.37 ms** | 6.30 ms | 340.25 ms | — | **54.01x** | **—** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **61.4 μs** | 54.5 μs | 41.87 ms | — | **767.96x** | **—** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **50.9 μs** | 53.0 μs | 545.67 ms | — | **10720.43x** | **—** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **57.9 μs** | 54.1 μs | 121.40 ms | — | **2242.40x** | **—** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **88.1 μs** | 66.6 μs | 118.91 ms | — | **1786.41x** | **—** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **2.50 ms** | 719.5 μs | 28.15 ms | — | **39.13x** | **—** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **52.1 μs** | 53.8 μs | 3.45 ms | — | **66.21x** | **—** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **51.7 μs** | 54.4 μs | 46.55 ms | — | **900.17x** | **—** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **50.8 μs** | 54.2 μs | 12.26 ms | — | **241.10x** | **—** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **77.5 μs** | 56.9 μs | 8.62 ms | — | **151.56x** | **—** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **639.4 μs** | 147.7 μs | 3.00 ms | — | **20.32x** | **—** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **50.1 μs** | 52.3 μs | 531.1 μs | — | **10.61x** | **—** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **51.5 μs** | 51.5 μs | 4.67 ms | — | **90.73x** | **—** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **50.5 μs** | 52.9 μs | 1.49 ms | — | **29.57x** | **—** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **79.3 μs** | 53.4 μs | 1.06 ms | — | **19.87x** | **—** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **489.9 μs** | 89.9 μs | 575.4 μs | — | **6.40x** | **—** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **50.5 μs** | 51.9 μs | 252.3 μs | — | **4.99x** | **—** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **48.6 μs** | 50.8 μs | 743.9 μs | — | **15.30x** | **—** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **48.8 μs** | 52.7 μs | 408.2 μs | — | **8.37x** | **—** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **76.5 μs** | 53.6 μs | 319.5 μs | — | **5.96x** | **—** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **10.32 ms** | 6.18 ms | 9.29 ms | — | **1.50x** | **—** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **53.8 μs** | 54.3 μs | 24.62 ms | — | **457.93x** | **—** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **50.7 μs** | 52.7 μs | 5.38 ms | — | **106.06x** | **—** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **52.1 μs** | 53.5 μs | 2.73 ms | — | **52.49x** | **—** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **72.7 μs** | 56.8 μs | 2.69 ms | — | **47.28x** | **—** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.30 ms** | 706.0 μs | 1.05 ms | — | **1.49x** | **—** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **51.9 μs** | 53.9 μs | 2.62 ms | — | **50.58x** | **—** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **50.2 μs** | 52.0 μs | 637.8 μs | — | **12.71x** | **—** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **51.7 μs** | 52.9 μs | 460.2 μs | — | **8.90x** | **—** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **69.2 μs** | 55.5 μs | 413.8 μs | — | **7.46x** | **—** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **520.3 μs** | 152.2 μs | 301.4 μs | — | **1.98x** | **—** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **50.7 μs** | 52.4 μs | 448.8 μs | — | **8.85x** | **—** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **48.6 μs** | 51.5 μs | 245.6 μs | — | **5.05x** | **—** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **49.5 μs** | 51.4 μs | 228.4 μs | — | **4.62x** | **—** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **68.3 μs** | 55.0 μs | 205.9 μs | — | **3.74x** | **—** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **443.7 μs** | 97.8 μs | 221.2 μs | — | **2.26x** | **—** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **52.4 μs** | 53.9 μs | 231.4 μs | — | **4.42x** | **—** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **51.2 μs** | 52.3 μs | 210.1 μs | — | **4.11x** | **—** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **51.1 μs** | 53.8 μs | 203.6 μs | — | **3.98x** | **—** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **71.6 μs** | 54.1 μs | 184.1 μs | — | **3.41x** | **—** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.24 ms** | 494.7 μs | 49.88 ms | — | **100.83x** | **—** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **54.0 μs** | 54.0 μs | 48.16 ms | — | **892.05x** | **—** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **53.0 μs** | 53.7 μs | 49.31 ms | — | **930.59x** | **—** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **53.8 μs** | 55.3 μs | 16.70 ms | — | **310.16x** | **—** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **73.9 μs** | 58.2 μs | 13.17 ms | — | **226.36x** | **—** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **554.6 μs** | 135.8 μs | 5.11 ms | — | **37.67x** | **—** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **56.9 μs** | 62.3 μs | 4.94 ms | — | **86.85x** | **—** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **50.9 μs** | 53.6 μs | 5.10 ms | — | **100.19x** | **—** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **52.7 μs** | 55.1 μs | 1.94 ms | — | **36.86x** | **—** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **70.2 μs** | 57.4 μs | 1.56 ms | — | **27.10x** | **—** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **818.3 μs** | 507.3 μs | 127.27 ms | — | **250.87x** | **—** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **53.4 μs** | 55.9 μs | 125.91 ms | — | **2355.98x** | **—** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **51.1 μs** | 55.4 μs | 122.28 ms | — | **2392.60x** | **—** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **52.3 μs** | 55.0 μs | 125.50 ms | — | **2401.10x** | **—** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **69.9 μs** | 56.7 μs | 126.05 ms | — | **2221.87x** | **—** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **300.8 μs** | 133.4 μs | 11.88 ms | — | **89.03x** | **—** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **51.5 μs** | 53.0 μs | 11.85 ms | — | **229.96x** | **—** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **50.6 μs** | 51.3 μs | 11.67 ms | — | **230.69x** | **—** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **51.3 μs** | 53.7 μs | 11.79 ms | — | **229.83x** | **—** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **70.3 μs** | 56.4 μs | 11.95 ms | — | **211.96x** | **—** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **434.4 μs** | 96.7 μs | 1.35 ms | — | **13.98x** | **—** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **50.9 μs** | 52.5 μs | 1.35 ms | — | **26.54x** | **—** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **49.8 μs** | 52.4 μs | 1.34 ms | — | **26.82x** | **—** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **50.4 μs** | 52.0 μs | 1.34 ms | — | **26.52x** | **—** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **66.0 μs** | 54.9 μs | 1.34 ms | — | **24.35x** | **—** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.11 ms** | 725.6 μs | 120.40 ms | — | **165.94x** | **—** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **50.9 μs** | 53.3 μs | 5.36 ms | — | **105.31x** | **—** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **51.1 μs** | 51.9 μs | 193.24 ms | — | **3780.73x** | **—** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **52.1 μs** | 53.8 μs | 54.40 ms | — | **1044.24x** | **—** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **146.5 μs** | 57.2 μs | 39.80 ms | — | **695.29x** | **—** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.23 ms** | 155.7 μs | 11.55 ms | — | **74.15x** | **—** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **51.1 μs** | 53.2 μs | 951.3 μs | — | **18.63x** | **—** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **52.7 μs** | 53.3 μs | 18.75 ms | — | **356.02x** | **—** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **50.6 μs** | 53.1 μs | 5.47 ms | — | **108.19x** | **—** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **145.2 μs** | 54.6 μs | 3.85 ms | — | **70.56x** | **—** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **875.1 μs** | 97.7 μs | 1.66 ms | — | **17.04x** | **—** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **49.8 μs** | 53.8 μs | 492.0 μs | — | **9.89x** | **—** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **51.0 μs** | 51.1 μs | 2.40 ms | — | **47.16x** | **—** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **49.2 μs** | 52.2 μs | 1.12 ms | — | **22.76x** | **—** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **143.5 μs** | 54.4 μs | 867.8 μs | — | **15.96x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
## Performance deficits

Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | large_uint32_2d [read_full] | 0.03132591862231493 | fitsio/fitsio_torch | 9.46314093141013 |
| fits | large_uint16_2d [read_full] | 0.014118923805654049 | fitsio/fitsio_torch | 9.387295760815922 |
| fits | medium_uint16_2d [read_full] | 0.001598881557583809 | fitsio/fitsio_torch | 3.7724513003065363 |
| fits | medium_uint32_2d [read_full] | 0.00254021305590868 | fitsio/fitsio_torch | 2.7331715328723156 |
| fits | small_uint32_2d [read_full] | 0.00022694654762744904 | fitsio/fitsio_torch | 1.7039269431935782 |
| fits | small_uint16_2d [read_full] | 0.0001827627420425415 | fitsio/fitsio_torch | 1.6501988748644034 |
| fits | large_int8_1d [read_full] | 0.000233539380133152 | fitsio/fitsio_torch | 1.0806102001680635 |
| fits | compressed_hcompress_1 [read_full] | 0.04614310059696436 | fitsio/fitsio_torch | 1.0304858461260527 |
| fits | large_uint32_2d [read_full] | 0.024223418906331062 | fitsio/fitsio | 7.32870069165592 |
| fits | large_uint16_2d [read_full] | 0.008565553463995457 | fitsio/fitsio | 5.724816812196088 |
| fits | medium_uint16_2d [read_full] | 0.000959642231464386 | fitsio/fitsio | 2.2927302502759086 |
| fits | medium_uint32_2d [read_full] | 0.001800677739083767 | fitsio/fitsio | 1.9496942042503844 |
| fits | small_uint32_2d [read_full] | 0.00015132594853639603 | fitsio/fitsio | 1.1747715310312916 |
| fits | compressed_rice_1 [read_full] | 0.012809901498258114 | fitsio/fitsio | 1.1284395412905757 |
| fits | small_uint16_2d [read_full] | 0.00011655781418085098 | fitsio/fitsio | 1.1146905839182015 |
| fits | compressed_hcompress_1 [read_full] | 0.04601987078785896 | fitsio/fitsio | 1.0291023126189744 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `uint_fix_20260708` | fits + fitstable (user) | 1377 | 16 | user bench-all --no-mmap (uint mmap fast-path fix applied) |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
