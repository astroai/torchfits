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
Source: `benchmarks_results/20260709_163739/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.05 ms` (n=269) | `0.49 ms` (n=269) | `0.16 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.06 ms` (n=269) | `0.47 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.05 ms` (n=180) | `2.41 ms` (n=162) | `2.88 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.05 ms` (n=180) | `2.16 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **1.99 ms** | 1.86 ms | 13.76 ms | 4.80 ms | **7.38x** | **2.57x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **7.27 ms** | 7.31 ms | 18.55 ms | 16.17 ms | **2.55x** | **2.22x** |
| Repeated Cutouts (50x 100x100) | CPU | **3.38 ms** | 3.16 ms | 52.14 ms | 4.69 ms | **16.51x** | **1.49x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **48.3 μs** | 48.2 μs | 4.35 ms | 45.42 ms | **90.20x** | **942.50x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **46.9 μs** | 49.2 μs | 2.26 ms | 129.59 ms | **48.23x** | **2762.98x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 86.2 μs | 1.53 ms | 225.2 μs | **17.70x** | **2.61x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **13.73 ms** | 13.72 ms | 26.39 ms | 28.53 ms | **1.92x** | **2.08x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 86.9 μs | 1.51 ms | 220.4 μs | **17.34x** | **2.54x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **11.74 ms** | 11.77 ms | 42.57 ms | 26.64 ms | **3.63x** | **2.27x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 90.3 μs | 1.59 ms | 298.8 μs | **17.63x** | **3.31x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **26.45 ms** | 26.41 ms | 30.80 ms | 54.67 ms | **1.17x** | **2.07x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **752.3 μs** | 739.3 μs | 6.89 ms | 1.83 ms | **9.31x** | **2.47x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 91.9 μs | 1.58 ms | 258.8 μs | **17.21x** | **2.82x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **7.27 ms** | 7.31 ms | 18.55 ms | 16.17 ms | **2.55x** | **2.22x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 42.6 μs | 421.6 μs | 62.6 μs | **9.89x** | **1.47x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **491.0 μs** | 480.6 μs | 1.24 ms | 1.18 ms | **2.58x** | **2.45x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 47.9 μs | 444.9 μs | 65.3 μs | **9.28x** | **1.36x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **1.99 ms** | 1.86 ms | 13.76 ms | 4.80 ms | **7.38x** | **2.57x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 44.0 μs | 420.2 μs | 61.9 μs | **9.55x** | **1.41x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **957.0 μs** | 900.6 μs | 2.02 ms | 1.63 ms | **2.24x** | **1.81x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 49.2 μs | 432.9 μs | 66.1 μs | **8.79x** | **1.34x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **11.01 ms** | 10.24 ms | 29.31 ms | 13.01 ms | **2.86x** | **1.27x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 42.3 μs | 402.7 μs | 61.8 μs | **9.52x** | **1.46x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **333.9 μs** | 273.1 μs | 845.8 μs | 859.8 μs | **3.10x** | **3.15x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 44.4 μs | 433.1 μs | 64.9 μs | **9.75x** | **1.46x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **1.06 ms** | 1.00 ms | 2.11 ms | 3.25 ms | **2.11x** | **3.25x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 42.7 μs | 417.0 μs | 62.5 μs | **9.77x** | **1.46x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **528.1 μs** | 475.6 μs | 1.26 ms | 1.20 ms | **2.64x** | **2.52x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 46.7 μs | 442.6 μs | 66.3 μs | **9.48x** | **1.42x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **1.98 ms** | 1.86 ms | 13.55 ms | 4.63 ms | **7.28x** | **2.49x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 43.8 μs | 420.4 μs | 63.4 μs | **9.59x** | **1.45x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **980.5 μs** | 903.5 μs | 2.01 ms | 1.63 ms | **2.22x** | **1.81x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 45.5 μs | 447.8 μs | 65.8 μs | **9.83x** | **1.45x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.92 ms** | 10.20 ms | 29.74 ms | 13.01 ms | **2.91x** | **1.28x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 48.3 μs | 469.2 μs | 71.5 μs | **9.72x** | **1.48x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **254.8 μs** | 165.7 μs | 751.1 μs | 1.22 ms | **4.53x** | **7.34x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 48.4 μs | 489.1 μs | 73.6 μs | **10.11x** | **1.52x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **591.9 μs** | 521.9 μs | 1.55 ms | 5.06 ms | **2.97x** | **9.70x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 49.6 μs | 486.1 μs | 76.9 μs | **9.81x** | **1.55x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **1.31 ms** | 1.26 ms | 3.98 ms | 8.73 ms | **3.16x** | **6.94x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 49.9 μs | 493.9 μs | 76.0 μs | **9.90x** | **1.52x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **2.48 ms** | 2.37 ms | 12.44 ms | 8.21 ms | **5.24x** | **3.46x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 45.2 μs | 419.3 μs | 62.7 μs | **9.27x** | **1.39x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **114.7 μs** | 82.2 μs | 493.2 μs | 202.1 μs | **6.00x** | **2.46x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 45.5 μs | 446.6 μs | 64.9 μs | **9.81x** | **1.43x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **551.0 μs** | 498.1 μs | 1.28 ms | 1.22 ms | **2.57x** | **2.44x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 47.9 μs | 460.2 μs | 68.8 μs | **9.60x** | **1.44x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **796.3 μs** | 751.6 μs | 1.76 ms | 1.85 ms | **2.34x** | **2.46x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 45.6 μs | 423.1 μs | 62.5 μs | **9.28x** | **1.37x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **122.8 μs** | 126.9 μs | 560.9 μs | 244.6 μs | **4.57x** | **1.99x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 46.5 μs | 446.8 μs | 65.8 μs | **9.60x** | **1.41x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **976.1 μs** | 947.0 μs | 2.11 ms | 1.70 ms | **2.22x** | **1.79x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 47.3 μs | 459.8 μs | 71.2 μs | **9.71x** | **1.50x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **1.51 ms** | 1.47 ms | 10.73 ms | 2.68 ms | **7.30x** | **1.82x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 43.9 μs | 414.0 μs | 63.2 μs | **9.43x** | **1.44x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **68.7 μs** | 66.0 μs | 451.9 μs | 169.3 μs | **6.85x** | **2.57x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 46.0 μs | 444.7 μs | 65.9 μs | **9.66x** | **1.43x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **344.5 μs** | 291.3 μs | 874.0 μs | 884.6 μs | **3.00x** | **3.04x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 48.4 μs | 451.5 μs | 70.2 μs | **9.32x** | **1.45x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **483.9 μs** | 424.6 μs | 1.11 ms | 1.35 ms | **2.61x** | **3.17x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 41.5 μs | 414.6 μs | 62.4 μs | **9.98x** | **1.50x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **95.8 μs** | 86.3 μs | 478.9 μs | 200.0 μs | **5.55x** | **2.32x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 48.3 μs | 444.1 μs | 66.9 μs | **9.20x** | **1.39x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **508.3 μs** | 500.3 μs | 1.28 ms | 1.22 ms | **2.56x** | **2.44x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 47.7 μs | 456.6 μs | 69.2 μs | **9.57x** | **1.45x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **780.1 μs** | 752.0 μs | 1.76 ms | 1.84 ms | **2.34x** | **2.45x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 45.0 μs | 415.0 μs | 63.4 μs | **9.23x** | **1.41x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **143.1 μs** | 130.5 μs | 567.2 μs | 247.9 μs | **4.35x** | **1.90x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 47.0 μs | 440.2 μs | 66.3 μs | **9.36x** | **1.41x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **1.05 ms** | 951.5 μs | 2.13 ms | 1.74 ms | **2.24x** | **1.83x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 46.2 μs | 451.4 μs | 69.0 μs | **9.77x** | **1.49x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **1.57 ms** | 1.47 ms | 10.70 ms | 2.68 ms | **7.28x** | **1.82x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 50.1 μs | 458.8 μs | 71.5 μs | **9.15x** | **1.43x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **59.3 μs** | 57.5 μs | 540.1 μs | 224.8 μs | **9.40x** | **3.91x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 47.9 μs | 486.7 μs | 75.7 μs | **10.17x** | **1.58x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **235.5 μs** | 172.1 μs | 777.2 μs | 1.46 ms | **4.52x** | **8.48x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 49.6 μs | 491.1 μs | 78.1 μs | **9.90x** | **1.57x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **274.1 μs** | 241.5 μs | 966.2 μs | 2.08 ms | **4.00x** | **8.61x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 48.2 μs | 476.4 μs | 74.8 μs | **9.89x** | **1.55x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **359.3 μs** | 331.6 μs | 1.40 ms | 1.90 ms | **4.21x** | **5.73x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 49.5 μs | 479.2 μs | 74.4 μs | **9.68x** | **1.50x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **651.5 μs** | 623.4 μs | 1.81 ms | 2.12 ms | **2.90x** | **3.40x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 55.3 μs | 691.8 μs | 85.1 μs | **12.50x** | **1.54x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **166.5 μs** | 170.2 μs | 974.7 μs | 1.33 ms | **5.86x** | **7.97x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 56.1 μs | 693.1 μs | 85.8 μs | **12.36x** | **1.53x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **47.0 μs** | 56.8 μs | 731.5 μs | 242.4 μs | **15.57x** | **5.16x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **44.3 μs** | 36.7 μs | 2.31 ms | 304.8 μs | **62.84x** | **8.30x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 54.9 μs | 699.6 μs | 84.7 μs | **12.75x** | **1.54x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **5.25 ms** | 5.27 ms | 6.84 ms | 15.20 ms | **1.30x** | **2.90x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **49.9 μs** | 56.5 μs | 732.1 μs | 366.4 μs | **14.66x** | **7.34x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **3.38 ms** | 3.16 ms | 52.14 ms | 4.69 ms | **16.51x** | **1.49x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 48.0 μs | 478.0 μs | 75.1 μs | **9.96x** | **1.56x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **2.54 ms** | 2.51 ms | 10.65 ms | 8.36 ms | **4.24x** | **3.33x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 50.4 μs | 496.2 μs | 74.3 μs | **9.84x** | **1.47x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **663.7 μs** | 667.1 μs | 1.48 ms | 2.21 ms | **2.23x** | **3.33x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 48.1 μs | 487.5 μs | 74.6 μs | **10.14x** | **1.55x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **81.2 μs** | 89.0 μs | 576.3 μs | 232.9 μs | **7.09x** | **2.87x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 43.7 μs | 416.8 μs | 61.7 μs | **9.53x** | **1.41x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **44.1 μs** | 48.2 μs | 421.1 μs | 107.1 μs | **9.55x** | **2.43x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 45.2 μs | 441.5 μs | 65.0 μs | **9.77x** | **1.44x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **68.1 μs** | 71.9 μs | 474.6 μs | 165.7 μs | **6.97x** | **2.43x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 48.4 μs | 454.4 μs | 67.5 μs | **9.39x** | **1.40x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **127.5 μs** | 111.3 μs | 553.3 μs | 265.5 μs | **4.97x** | **2.39x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 43.8 μs | 411.4 μs | 61.7 μs | **9.39x** | **1.41x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **48.8 μs** | 51.3 μs | 425.6 μs | 110.4 μs | **8.73x** | **2.26x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 46.8 μs | 443.9 μs | 65.1 μs | **9.49x** | **1.39x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **110.8 μs** | 97.9 μs | 520.9 μs | 195.7 μs | **5.32x** | **2.00x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 49.0 μs | 459.3 μs | 68.2 μs | **9.38x** | **1.39x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **231.2 μs** | 192.1 μs | 714.8 μs | 347.8 μs | **3.72x** | **1.81x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 43.1 μs | 408.9 μs | 63.0 μs | **9.48x** | **1.46x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **45.2 μs** | 47.0 μs | 410.1 μs | 103.6 μs | **9.07x** | **2.29x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 44.0 μs | 434.7 μs | 66.0 μs | **9.87x** | **1.50x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **63.2 μs** | 61.2 μs | 442.1 μs | 143.7 μs | **7.23x** | **2.35x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 44.9 μs | 459.8 μs | 69.1 μs | **10.25x** | **1.54x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **92.3 μs** | 83.6 μs | 496.0 μs | 220.7 μs | **5.93x** | **2.64x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 43.1 μs | 406.6 μs | 61.2 μs | **9.43x** | **1.42x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **42.8 μs** | 47.9 μs | 414.8 μs | 107.4 μs | **9.68x** | **2.51x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 43.4 μs | 436.1 μs | 65.3 μs | **10.05x** | **1.51x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **65.9 μs** | 70.9 μs | 454.0 μs | 164.8 μs | **6.89x** | **2.50x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 45.8 μs | 455.5 μs | 68.1 μs | **9.95x** | **1.49x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **132.3 μs** | 112.1 μs | 561.9 μs | 262.4 μs | **5.01x** | **2.34x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 41.5 μs | 411.4 μs | 62.8 μs | **9.91x** | **1.51x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **52.7 μs** | 51.8 μs | 435.0 μs | 110.6 μs | **8.40x** | **2.14x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 45.7 μs | 437.0 μs | 66.6 μs | **9.56x** | **1.46x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **114.7 μs** | 99.3 μs | 524.7 μs | 192.8 μs | **5.28x** | **1.94x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 48.1 μs | 457.4 μs | 69.2 μs | **9.51x** | **1.44x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **212.0 μs** | 187.3 μs | 720.7 μs | 349.3 μs | **3.85x** | **1.86x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 47.6 μs | 465.0 μs | 70.6 μs | **9.77x** | **1.48x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **37.2 μs** | 43.3 μs | 534.2 μs | 113.3 μs | **14.35x** | **3.04x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 48.4 μs | 484.8 μs | 74.4 μs | **10.02x** | **1.54x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **49.6 μs** | 54.5 μs | 559.7 μs | 186.9 μs | **11.29x** | **3.77x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 49.3 μs | 506.9 μs | 78.4 μs | **10.29x** | **1.59x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **65.0 μs** | 66.4 μs | 565.4 μs | 270.4 μs | **8.69x** | **4.16x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 48.5 μs | 480.4 μs | 75.0 μs | **9.90x** | **1.54x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **58.7 μs** | 63.8 μs | 539.2 μs | 236.7 μs | **9.18x** | **4.03x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 48.0 μs | 479.5 μs | 73.7 μs | **9.99x** | **1.53x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **77.6 μs** | 77.8 μs | 557.3 μs | 230.9 μs | **7.18x** | **2.97x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 45.8 μs | 432.1 μs | 66.2 μs | **9.43x** | **1.45x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **77.9 μs** | 77.0 μs | 468.3 μs | 166.6 μs | **6.08x** | **2.17x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 44.6 μs | 437.5 μs | 66.1 μs | **9.81x** | **1.48x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **68.6 μs** | 72.7 μs | 458.8 μs | 163.3 μs | **6.69x** | **2.38x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 45.3 μs | 444.0 μs | 65.4 μs | **9.81x** | **1.44x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **75.5 μs** | 73.2 μs | 477.4 μs | 164.6 μs | **6.52x** | **2.25x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 47.3 μs | 440.2 μs | 67.2 μs | **9.30x** | **1.42x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **73.7 μs** | 71.3 μs | 476.4 μs | 163.2 μs | **6.68x** | **2.29x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 45.4 μs | 440.0 μs | 64.8 μs | **9.70x** | **1.43x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **66.4 μs** | 72.3 μs | 473.6 μs | 168.2 μs | **7.14x** | **2.53x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 44.2 μs | 421.0 μs | 61.6 μs | **9.53x** | **1.39x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **36.7 μs** | 43.2 μs | 415.9 μs | 98.7 μs | **11.33x** | **2.69x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 45.7 μs | 441.0 μs | 65.6 μs | **9.65x** | **1.44x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **40.7 μs** | 50.5 μs | 437.1 μs | 101.2 μs | **10.75x** | **2.49x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 48.2 μs | 459.1 μs | 73.1 μs | **9.53x** | **1.52x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **41.6 μs** | 50.1 μs | 449.6 μs | 105.4 μs | **10.81x** | **2.53x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 44.1 μs | 421.0 μs | 62.0 μs | **9.56x** | **1.41x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **40.5 μs** | 53.2 μs | 417.4 μs | 98.4 μs | **10.30x** | **2.43x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 47.3 μs | 434.9 μs | 65.3 μs | **9.20x** | **1.38x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **44.3 μs** | 53.7 μs | 417.4 μs | 105.7 μs | **9.42x** | **2.38x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 47.8 μs | 458.0 μs | 69.7 μs | **9.59x** | **1.46x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **44.6 μs** | 54.4 μs | 445.9 μs | 107.1 μs | **10.00x** | **2.40x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 42.8 μs | 415.4 μs | 61.8 μs | **9.70x** | **1.44x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **36.4 μs** | 40.2 μs | 402.3 μs | 99.2 μs | **11.04x** | **2.72x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 46.1 μs | 442.4 μs | 64.0 μs | **9.59x** | **1.39x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **37.8 μs** | 42.6 μs | 427.9 μs | 100.9 μs | **11.31x** | **2.67x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 47.3 μs | 453.5 μs | 68.1 μs | **9.60x** | **1.44x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **38.5 μs** | 45.3 μs | 442.9 μs | 101.0 μs | **11.51x** | **2.63x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 42.7 μs | 412.6 μs | 60.9 μs | **9.65x** | **1.42x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **36.1 μs** | 40.7 μs | 410.0 μs | 100.9 μs | **11.35x** | **2.79x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 43.0 μs | 435.3 μs | 64.8 μs | **10.11x** | **1.51x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **40.3 μs** | 47.8 μs | 427.8 μs | 98.6 μs | **10.62x** | **2.45x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 45.6 μs | 457.6 μs | 68.7 μs | **10.03x** | **1.51x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **42.3 μs** | 49.1 μs | 430.3 μs | 105.1 μs | **10.17x** | **2.49x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 43.7 μs | 413.9 μs | 62.0 μs | **9.46x** | **1.42x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **36.8 μs** | 41.8 μs | 395.9 μs | 99.1 μs | **10.76x** | **2.69x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 45.8 μs | 431.8 μs | 65.9 μs | **9.43x** | **1.44x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **43.3 μs** | 49.7 μs | 417.4 μs | 103.3 μs | **9.65x** | **2.39x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 46.2 μs | 452.3 μs | 67.6 μs | **9.79x** | **1.46x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **46.1 μs** | 51.8 μs | 443.8 μs | 108.0 μs | **9.63x** | **2.34x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 46.7 μs | 455.3 μs | 72.1 μs | **9.75x** | **1.54x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **36.9 μs** | 40.9 μs | 526.8 μs | 101.1 μs | **14.29x** | **2.74x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 48.5 μs | 480.4 μs | 74.5 μs | **9.91x** | **1.54x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **35.1 μs** | 43.9 μs | 536.9 μs | 109.3 μs | **15.29x** | **3.11x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 49.1 μs | 496.7 μs | 78.2 μs | **10.11x** | **1.59x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **38.1 μs** | 44.4 μs | 557.6 μs | 112.3 μs | **14.65x** | **2.95x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **312.0 μs** | 131.5 μs | 3.49 ms | 6.30 ms | **26.57x** | **47.90x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **46.1 μs** | 47.2 μs | 8.42 ms | 6.26 ms | **182.69x** | **135.84x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **45.1 μs** | 48.5 μs | 1.50 ms | 6.23 ms | **33.18x** | **138.25x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **47.2 μs** | 49.4 μs | 1.84 ms | 3.37 ms | **38.91x** | **71.43x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **67.7 μs** | 50.9 μs | 2.87 ms | 3.00 ms | **56.33x** | **58.92x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **248.9 μs** | 92.2 μs | 2.11 ms | 884.8 μs | **22.89x** | **9.60x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **48.2 μs** | 50.9 μs | 2.34 ms | 905.4 μs | **48.58x** | **18.77x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **48.3 μs** | 48.9 μs | 1.39 ms | 884.5 μs | **28.82x** | **18.33x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **48.0 μs** | 49.1 μs | 1.78 ms | 576.9 μs | **37.03x** | **12.01x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **70.8 μs** | 52.7 μs | 1.65 ms | 495.3 μs | **31.21x** | **9.40x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **13.64 ms** | 6.43 ms | 94.80 ms | 333.32 ms | **14.74x** | **51.84x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **55.1 μs** | 53.7 μs | 26.54 ms | 55.04 ms | **493.96x** | **1024.61x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **49.7 μs** | 49.3 μs | 51.42 ms | 529.95 ms | **1043.33x** | **10752.17x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **52.3 μs** | 51.9 μs | 23.65 ms | 129.98 ms | **455.42x** | **2502.95x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **81.5 μs** | 54.7 μs | 21.26 ms | 121.03 ms | **388.66x** | **2212.53x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **1.60 ms** | 718.5 μs | 8.48 ms | 27.69 ms | **11.80x** | **38.54x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **49.1 μs** | 50.6 μs | 2.71 ms | 4.63 ms | **55.23x** | **94.33x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **48.3 μs** | 48.2 μs | 4.35 ms | 45.42 ms | **90.20x** | **942.50x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **48.4 μs** | 50.6 μs | 3.56 ms | 12.26 ms | **73.64x** | **253.46x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **79.8 μs** | 52.9 μs | 2.69 ms | 8.69 ms | **50.83x** | **164.26x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **420.4 μs** | 147.4 μs | 3.26 ms | 3.02 ms | **22.12x** | **20.50x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **48.4 μs** | 50.5 μs | 2.05 ms | 704.8 μs | **42.27x** | **14.56x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **47.5 μs** | 50.0 μs | 2.15 ms | 4.56 ms | **45.24x** | **96.07x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **48.6 μs** | 49.5 μs | 2.79 ms | 1.52 ms | **57.41x** | **31.28x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **76.5 μs** | 52.5 μs | 2.02 ms | 1.13 ms | **38.58x** | **21.48x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **295.2 μs** | 88.3 μs | 2.78 ms | 629.1 μs | **31.51x** | **7.13x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **48.6 μs** | 51.6 μs | 1.96 ms | 320.8 μs | **40.21x** | **6.60x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **47.0 μs** | 48.8 μs | 1.97 ms | 780.6 μs | **41.96x** | **16.61x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **48.4 μs** | 51.0 μs | 2.69 ms | 469.5 μs | **55.70x** | **9.71x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **77.0 μs** | 53.1 μs | 1.95 ms | 381.3 μs | **36.71x** | **7.17x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **11.09 ms** | 6.29 ms | 28.16 ms | 9.21 ms | **4.48x** | **1.46x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **50.5 μs** | 50.2 μs | 3.76 ms | 34.94 ms | **74.90x** | **695.67x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **49.9 μs** | 54.7 μs | 6.43 ms | 5.54 ms | **128.85x** | **111.00x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **50.8 μs** | 50.8 μs | 4.13 ms | 2.78 ms | **81.27x** | **54.72x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **70.5 μs** | 54.5 μs | 3.67 ms | 2.73 ms | **67.27x** | **50.10x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.36 ms** | 712.8 μs | 4.64 ms | 1.07 ms | **6.51x** | **1.50x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **49.1 μs** | 52.1 μs | 1.76 ms | 3.70 ms | **35.79x** | **75.39x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **48.7 μs** | 49.6 μs | 1.95 ms | 681.9 μs | **40.09x** | **13.99x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **49.9 μs** | 49.4 μs | 2.23 ms | 497.8 μs | **45.21x** | **10.08x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **67.3 μs** | 52.2 μs | 1.76 ms | 453.1 μs | **33.62x** | **8.68x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **355.4 μs** | 151.5 μs | 2.18 ms | 336.5 μs | **14.37x** | **2.22x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **49.1 μs** | 50.6 μs | 1.49 ms | 585.1 μs | **30.46x** | **11.92x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **47.9 μs** | 50.0 μs | 1.48 ms | 277.4 μs | **30.89x** | **5.79x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **49.8 μs** | 51.7 μs | 1.90 ms | 264.2 μs | **38.06x** | **5.30x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **69.7 μs** | 53.3 μs | 1.48 ms | 236.8 μs | **27.74x** | **4.45x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **262.3 μs** | 94.7 μs | 1.92 ms | 257.0 μs | **20.25x** | **2.72x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **49.6 μs** | 51.3 μs | 1.47 ms | 275.7 μs | **29.70x** | **5.55x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **48.2 μs** | 50.5 μs | 1.48 ms | 240.1 μs | **30.65x** | **4.98x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **48.0 μs** | 51.8 μs | 1.84 ms | 238.1 μs | **38.33x** | **4.96x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **69.1 μs** | 53.4 μs | 1.45 ms | 217.0 μs | **27.16x** | **4.06x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **796.8 μs** | 525.0 μs | 3.94 ms | 54.91 ms | **7.51x** | **104.59x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **47.9 μs** | 49.7 μs | 30.30 ms | 52.70 ms | **632.99x** | **1100.99x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **48.0 μs** | 50.7 μs | 2.38 ms | 54.37 ms | **49.57x** | **1133.04x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **49.6 μs** | 50.7 μs | 2.26 ms | 23.65 ms | **45.56x** | **476.92x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **68.2 μs** | 53.3 μs | 1.76 ms | 20.21 ms | **33.07x** | **379.49x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **309.1 μs** | 132.2 μs | 2.18 ms | 5.70 ms | **16.49x** | **43.13x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **52.4 μs** | 58.2 μs | 4.40 ms | 5.52 ms | **83.95x** | **105.31x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **45.8 μs** | 48.4 μs | 1.57 ms | 5.64 ms | **34.28x** | **123.24x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **54.7 μs** | 52.2 μs | 2.01 ms | 2.70 ms | **38.43x** | **51.74x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **70.5 μs** | 52.0 μs | 1.54 ms | 2.29 ms | **29.69x** | **44.02x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **770.6 μs** | 512.5 μs | 4.13 ms | 134.22 ms | **8.05x** | **261.89x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **48.6 μs** | 50.4 μs | 513.17 ms | 132.74 ms | **10554.40x** | **2730.02x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **46.9 μs** | 49.2 μs | 2.26 ms | 129.59 ms | **48.23x** | **2762.98x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **48.4 μs** | 49.3 μs | 2.11 ms | 133.31 ms | **43.65x** | **2756.65x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **68.8 μs** | 52.7 μs | 1.68 ms | 132.79 ms | **31.83x** | **2519.88x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **294.8 μs** | 130.6 μs | 2.06 ms | 12.52 ms | **15.75x** | **95.86x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **67.4 μs** | 70.2 μs | 89.41 ms | 12.53 ms | **1326.08x** | **185.88x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **67.4 μs** | 69.5 μs | 2.44 ms | 21.04 ms | **36.14x** | **312.04x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **47.3 μs** | 50.0 μs | 1.84 ms | 12.47 ms | **38.82x** | **263.50x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **67.6 μs** | 51.5 μs | 1.43 ms | 12.62 ms | **27.79x** | **245.01x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **328.3 μs** | 145.9 μs | 3.05 ms | 2.45 ms | **20.88x** | **16.83x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **47.7 μs** | 50.7 μs | 6.50 ms | 1.45 ms | **136.39x** | **30.33x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **48.1 μs** | 50.0 μs | 1.41 ms | 1.43 ms | **29.40x** | **29.70x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **49.6 μs** | 50.7 μs | 1.76 ms | 1.44 ms | **35.53x** | **29.13x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **111.8 μs** | 81.0 μs | 2.34 ms | 2.46 ms | **28.87x** | **30.32x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **2.96 ms** | 718.2 μs | 46.96 ms | 118.60 ms | **65.38x** | **165.14x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **50.0 μs** | 51.0 μs | 8.02 ms | 7.66 ms | **160.49x** | **153.26x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **51.2 μs** | 50.1 μs | 33.33 ms | 188.23 ms | **665.37x** | **3757.31x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **49.2 μs** | 50.1 μs | 11.88 ms | 55.93 ms | **241.64x** | **1137.66x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **148.6 μs** | 52.4 μs | 7.97 ms | 41.49 ms | **152.09x** | **791.56x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.01 ms** | 157.2 μs | 11.31 ms | 11.55 ms | **71.92x** | **73.49x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **48.9 μs** | 50.8 μs | 6.12 ms | 1.34 ms | **125.15x** | **27.40x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **46.8 μs** | 49.7 μs | 7.01 ms | 18.34 ms | **149.58x** | **391.62x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **48.8 μs** | 50.6 μs | 9.52 ms | 5.68 ms | **195.07x** | **116.29x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **145.1 μs** | 52.4 μs | 6.11 ms | 4.35 ms | **116.69x** | **83.09x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **822.9 μs** | 94.5 μs | 9.29 ms | 1.83 ms | **98.28x** | **19.36x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **48.8 μs** | 51.4 μs | 5.86 ms | 784.9 μs | **119.96x** | **16.07x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **47.0 μs** | 50.0 μs | 5.93 ms | 2.55 ms | **126.04x** | **54.14x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **50.0 μs** | 50.4 μs | 9.07 ms | 1.32 ms | **181.37x** | **26.40x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **146.8 μs** | 53.3 μs | 5.77 ms | 1.05 ms | **108.24x** | **19.67x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fitstable | narrow_100000 [predicate_filter] | 0.0013417843729257584 | fitsio/fitsio_torch | 1.2522065476340507 |
| fitstable | narrow_1000000 [predicate_filter] | 0.011092509143054485 | fitsio/fitsio_torch | 1.2047833896857394 |
| fitstable | narrow_10000 [predicate_filter] | 0.0003554150462150574 | fitsio/fitsio_torch | 1.0702068768245796 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `20260709_163739` | fits + fitstable (lab) | 2754 | 3 | lab bench-all + `--mmap-matrix` |
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
