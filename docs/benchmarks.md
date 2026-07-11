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

### Refreshing GPU numbers (CANFAR staging)

CUDA lab numbers come from a **headless GPU session** on `@staging`, not GitHub
Actions. From a machine with `canfar` x509 auth:

```bash
# Preflight + smoke (quick)
bash scripts/selfcheck_canfar_launcher.sh
TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest TORCHFITS_BENCH_MODE=smoke \
  pixi run bench-canfar-gpu

# Full exhaustive lab bench-all + mmap matrix (CUDA rows)
TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest TORCHFITS_BENCH_MODE=exhaustive \
  pixi run bench-canfar-gpu

# After scratch CSV is copied locally:
bash scripts/patch_canfar_exhaustive_docs.sh exhaustive_cuda_0.7.0_<stamp>
```

Launcher: `scripts/launch_canfar_gpu_bench.sh` (`--gpu 1`). Default image
`astroai/base:latest` needs Harbor registry credentials (not in `canfar image ls`).
**`astroai/notebook:latest`** works with x509 alone on staging.

```bash
canfar config set registry.url https://images.canfar.net
canfar config set registry.username <harbor-username>
canfar config set registry.secret <harbor-token>
```

Skaha passes headless `args` as URL query parameters (spaces/`$`/`&` break remote
bash). The launcher tab-encodes the clone+bench script; ops may want a proper
argv array in the API long term.

In-container work uses **pixi**; stdout/stderr + CSVs tee to
`${TMP_SCRATCH_DIR}/torchfits-gpu-bench/<run-id>/`. Platform logs land under
`benchmarks_results/canfar_<run-id>/` locally.

```bash
# Local CI + docs before push (mirrors GitHub workflows)
bash scripts/ci_local.sh
# skip release-gate for a quick pass:
CI_LOCAL_FAST=1 bash scripts/ci_local.sh

# Apple Silicon dev only (MPS transport rows — not the CUDA release gate)
pixi run bench-mps
```

## I/O Transport × Backend

> **GPU summary:** Image **`disk→CPU→GPU`** and **`disk→RAM→GPU`** rows appear only when the benchmark CSV was
> produced on CUDA or MPS hardware. **`disk→GPU`** is intentionally empty (unsupported by
> all backends). **Table GPU transports are not benchmarked.** CI weekly `bench-report`
> uses CPU PyTorch and will not update GPU cells.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/exhaustive_0.7.0_20260711_022156/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.12 ms` (n=269) | `0.68 ms` (n=269) | `0.24 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.10 ms` (n=269) | `4.68 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.24 ms` (n=223) | `0.53 ms` (n=79) | `0.30 ms` (n=79) | — |
| `disk→RAM→GPU` | `0.32 ms` (n=223) | `0.98 ms` (n=79) | `0.32 ms` (n=79) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.09 ms` (n=180) | `2.46 ms` (n=162) | `2.85 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.10 ms` (n=180) | `6.85 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **3.18 ms** | 2.76 ms | 6.38 ms | 2.98 ms | **2.31x** | **1.08x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.18 ms** | 3.35 ms | 6.42 ms | 3.53 ms | **2.02x** | **1.11x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **9.11 ms** | 8.96 ms | 20.91 ms | 7.71 ms | **2.33x** | **0.86x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **6.95 ms** | 7.10 ms | 18.52 ms | 7.12 ms | **2.66x** | **1.02x** |
| Repeated Cutouts (50x 100x100) | CPU | **6.59 ms** | 5.91 ms | 81.77 ms | 6.09 ms | **13.82x** | **1.03x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **82.5 μs** | 80.0 μs | 3.93 ms | 53.68 ms | **49.12x** | **670.99x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **81.4 μs** | 84.5 μs | 2.44 ms | 122.68 ms | **30.03x** | **1507.54x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 124.4 μs | 1.51 ms | 246.7 μs | **12.13x** | **1.98x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **14.20 ms** | 12.99 ms | 27.75 ms | 16.71 ms | **2.14x** | **1.29x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 117.1 μs | 1.50 ms | 270.5 μs | **12.84x** | **2.31x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **14.24 ms** | 12.54 ms | 47.36 ms | 17.28 ms | **3.78x** | **1.38x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 134.3 μs | 1.63 ms | 282.4 μs | **12.15x** | **2.10x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **24.64 ms** | 24.62 ms | 29.76 ms | 25.01 ms | **1.21x** | **1.02x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **698.9 μs** | 705.5 μs | 6.43 ms | 886.9 μs | **9.19x** | **1.27x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 135.7 μs | 1.51 ms | 291.5 μs | **11.16x** | **2.15x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **9.11 ms** | 8.96 ms | 20.91 ms | 7.71 ms | **2.33x** | **0.86x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 84.2 μs | 573.0 μs | 152.9 μs | **6.81x** | **1.82x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **772.8 μs** | 766.6 μs | 2.11 ms | 981.2 μs | **2.75x** | **1.28x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 91.8 μs | 493.0 μs | 144.8 μs | **5.37x** | **1.58x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **3.18 ms** | 2.76 ms | 6.38 ms | 2.98 ms | **2.31x** | **1.08x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 84.8 μs | 584.6 μs | 153.2 μs | **6.89x** | **1.81x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **1.64 ms** | 1.73 ms | 3.77 ms | 1.80 ms | **2.30x** | **1.10x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 79.6 μs | 573.0 μs | 140.4 μs | **7.20x** | **1.76x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **6.73 ms** | 6.88 ms | 12.72 ms | 7.19 ms | **1.89x** | **1.07x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 84.2 μs | 557.2 μs | 153.2 μs | **6.62x** | **1.82x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **433.4 μs** | 438.8 μs | 1.36 ms | 575.8 μs | **3.13x** | **1.33x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 88.0 μs | 565.3 μs | 174.7 μs | **6.42x** | **1.98x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **2.05 ms** | 1.68 ms | 4.89 ms | 2.24 ms | **2.91x** | **1.33x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 86.8 μs | 562.3 μs | 157.6 μs | **6.48x** | **1.82x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **771.9 μs** | 797.0 μs | 3.06 ms | 924.1 μs | **3.97x** | **1.20x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 82.7 μs | 590.8 μs | 159.2 μs | **7.14x** | **1.92x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **3.18 ms** | 3.12 ms | 6.35 ms | 2.89 ms | **2.04x** | **0.93x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 86.1 μs | 526.9 μs | 154.5 μs | **6.12x** | **1.79x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **1.68 ms** | 1.92 ms | 3.61 ms | 2.43 ms | **2.15x** | **1.44x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 81.4 μs | 567.0 μs | 145.2 μs | **6.96x** | **1.78x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **6.81 ms** | 6.65 ms | 12.11 ms | 7.18 ms | **1.82x** | **1.08x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 89.5 μs | 582.4 μs | 154.7 μs | **6.50x** | **1.73x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **628.3 μs** | 625.8 μs | 854.2 μs | 738.2 μs | **1.36x** | **1.18x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 87.9 μs | 575.5 μs | 154.2 μs | **6.55x** | **1.75x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **2.54 ms** | 2.50 ms | 3.16 ms | 5.13 ms | **1.26x** | **2.05x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 91.4 μs | 570.1 μs | 162.8 μs | **6.24x** | **1.78x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **3.35 ms** | 3.17 ms | 6.29 ms | 3.44 ms | **1.99x** | **1.08x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 82.9 μs | 610.3 μs | 159.6 μs | **7.36x** | **1.92x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **4.76 ms** | 4.54 ms | 8.36 ms | 4.89 ms | **1.84x** | **1.08x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 85.9 μs | 563.6 μs | 134.3 μs | **6.56x** | **1.56x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **156.4 μs** | 159.6 μs | 667.3 μs | 261.6 μs | **4.27x** | **1.67x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 88.7 μs | 541.0 μs | 155.4 μs | **6.10x** | **1.75x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **805.5 μs** | 800.9 μs | 2.74 ms | 934.9 μs | **3.43x** | **1.17x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 94.4 μs | 564.3 μs | 167.6 μs | **5.98x** | **1.77x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **1.14 ms** | 1.95 ms | 3.02 ms | 1.35 ms | **2.64x** | **1.18x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 79.6 μs | 570.2 μs | 146.4 μs | **7.17x** | **1.84x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **413.8 μs** | 209.7 μs | 813.7 μs | 361.8 μs | **3.88x** | **1.73x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 82.1 μs | 584.0 μs | 137.2 μs | **7.11x** | **1.67x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **2.00 ms** | 1.70 ms | 4.48 ms | 1.86 ms | **2.63x** | **1.09x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 90.7 μs | 597.1 μs | 166.2 μs | **6.59x** | **1.83x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **2.65 ms** | 2.64 ms | 4.93 ms | 2.89 ms | **1.87x** | **1.10x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 84.7 μs | 553.7 μs | 153.6 μs | **6.54x** | **1.81x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **118.9 μs** | 127.3 μs | 638.2 μs | 226.5 μs | **5.37x** | **1.91x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 85.7 μs | 574.2 μs | 137.4 μs | **6.70x** | **1.60x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **427.5 μs** | 462.4 μs | 1.49 ms | 597.1 μs | **3.49x** | **1.40x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 87.0 μs | 567.6 μs | 166.2 μs | **6.53x** | **1.91x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **920.0 μs** | 709.5 μs | 1.99 ms | 1.01 ms | **2.80x** | **1.42x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 73.7 μs | 541.5 μs | 157.3 μs | **7.35x** | **2.14x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **335.8 μs** | 141.8 μs | 735.1 μs | 257.0 μs | **5.18x** | **1.81x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 83.7 μs | 589.7 μs | 156.6 μs | **7.05x** | **1.87x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **1.07 ms** | 831.3 μs | 2.56 ms | 990.6 μs | **3.08x** | **1.19x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 88.2 μs | 593.9 μs | 169.2 μs | **6.74x** | **1.92x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **1.23 ms** | 1.17 ms | 3.24 ms | 2.21 ms | **2.77x** | **1.89x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 90.3 μs | 554.8 μs | 157.6 μs | **6.14x** | **1.75x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **195.6 μs** | 202.5 μs | 674.4 μs | 336.5 μs | **3.45x** | **1.72x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 88.2 μs | 531.6 μs | 158.6 μs | **6.03x** | **1.80x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **1.83 ms** | 1.82 ms | 4.69 ms | 1.86 ms | **2.58x** | **1.02x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 90.0 μs | 559.5 μs | 165.0 μs | **6.22x** | **1.83x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **2.64 ms** | 3.50 ms | 5.36 ms | 3.23 ms | **2.03x** | **1.23x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 89.5 μs | 588.7 μs | 168.0 μs | **6.58x** | **1.88x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **148.5 μs** | 170.0 μs | 676.8 μs | 272.6 μs | **4.56x** | **1.84x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 90.7 μs | 623.4 μs | 169.2 μs | **6.87x** | **1.86x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **767.4 μs** | 673.7 μs | 1.10 ms | 932.5 μs | **1.64x** | **1.38x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 93.3 μs | 623.2 μs | 168.5 μs | **6.68x** | **1.80x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **1.15 ms** | 1.03 ms | 1.94 ms | 1.18 ms | **1.88x** | **1.14x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 90.9 μs | 596.2 μs | 163.8 μs | **6.56x** | **1.80x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **854.7 μs** | 1.28 ms | 2.69 ms | 1.14 ms | **3.14x** | **1.34x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 88.9 μs | 599.7 μs | 151.7 μs | **6.75x** | **1.71x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **1.23 ms** | 1.36 ms | 2.92 ms | 1.31 ms | **2.36x** | **1.06x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 98.3 μs | 788.4 μs | 171.3 μs | **8.02x** | **1.74x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **701.0 μs** | 660.8 μs | 1.04 ms | 1.71 ms | **1.57x** | **2.59x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 100.7 μs | 770.7 μs | 149.2 μs | **7.66x** | **1.48x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **125.8 μs** | 153.3 μs | 895.5 μs | 274.7 μs | **7.12x** | **2.18x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **80.7 μs** | 82.0 μs | 3.51 ms | 333.9 μs | **43.45x** | **4.14x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 99.7 μs | 812.8 μs | 166.2 μs | **8.15x** | **1.67x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **7.84 ms** | 6.53 ms | 7.95 ms | 7.56 ms | **1.22x** | **1.16x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **133.5 μs** | 148.5 μs | 858.8 μs | 323.8 μs | **6.43x** | **2.42x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **6.59 ms** | 5.91 ms | 81.77 ms | 6.09 ms | **13.82x** | **1.03x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 92.9 μs | 596.6 μs | 148.5 μs | **6.42x** | **1.60x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **3.28 ms** | 4.27 ms | 9.00 ms | 3.54 ms | **2.74x** | **1.08x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 96.6 μs | 607.3 μs | 157.8 μs | **6.29x** | **1.63x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **887.5 μs** | 797.7 μs | 2.47 ms | 1.60 ms | **3.10x** | **2.00x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 92.8 μs | 564.9 μs | 140.4 μs | **6.08x** | **1.51x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **144.7 μs** | 134.3 μs | 740.7 μs | 210.0 μs | **5.51x** | **1.56x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 77.6 μs | 482.4 μs | 128.4 μs | **6.21x** | **1.65x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **107.2 μs** | 129.5 μs | 613.4 μs | 211.7 μs | **5.72x** | **1.98x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 76.5 μs | 478.8 μs | 116.4 μs | **6.26x** | **1.52x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **134.5 μs** | 145.0 μs | 727.1 μs | 235.9 μs | **5.41x** | **1.75x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 75.8 μs | 509.1 μs | 129.2 μs | **6.72x** | **1.70x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **196.5 μs** | 177.5 μs | 761.1 μs | 290.2 μs | **4.29x** | **1.64x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 70.1 μs | 450.6 μs | 113.3 μs | **6.43x** | **1.62x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **115.8 μs** | 113.7 μs | 571.6 μs | 215.3 μs | **5.03x** | **1.89x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 70.8 μs | 456.0 μs | 126.0 μs | **6.44x** | **1.78x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **239.2 μs** | 174.9 μs | 754.6 μs | 293.9 μs | **4.31x** | **1.68x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 73.8 μs | 480.8 μs | 112.9 μs | **6.51x** | **1.53x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **301.7 μs** | 608.4 μs | 932.3 μs | 436.3 μs | **3.09x** | **1.45x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 67.4 μs | 440.7 μs | 109.7 μs | **6.54x** | **1.63x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **110.1 μs** | 345.5 μs | 588.0 μs | 166.1 μs | **5.34x** | **1.51x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 68.1 μs | 469.0 μs | 109.7 μs | **6.89x** | **1.61x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **87.6 μs** | 153.3 μs | 611.3 μs | 218.3 μs | **6.98x** | **2.49x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 70.8 μs | 489.7 μs | 125.8 μs | **6.92x** | **1.78x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **132.5 μs** | 159.8 μs | 666.1 μs | 243.0 μs | **5.03x** | **1.83x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 67.5 μs | 466.6 μs | 107.5 μs | **6.91x** | **1.59x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **105.0 μs** | 114.4 μs | 633.0 μs | 387.3 μs | **6.03x** | **3.69x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 70.3 μs | 490.3 μs | 108.3 μs | **6.98x** | **1.54x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **118.2 μs** | 194.0 μs | 616.8 μs | 225.4 μs | **5.22x** | **1.91x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 74.1 μs | 506.2 μs | 120.9 μs | **6.83x** | **1.63x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **242.7 μs** | 173.2 μs | 730.4 μs | 297.0 μs | **4.22x** | **1.71x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 69.0 μs | 464.5 μs | 116.1 μs | **6.73x** | **1.68x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **114.9 μs** | 116.6 μs | 726.9 μs | 227.1 μs | **6.33x** | **1.98x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 67.7 μs | 476.2 μs | 117.7 μs | **7.04x** | **1.74x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **192.0 μs** | 168.0 μs | 695.9 μs | 277.3 μs | **4.14x** | **1.65x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 71.3 μs | 476.2 μs | 130.2 μs | **6.68x** | **1.83x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **336.7 μs** | 266.7 μs | 897.3 μs | 368.6 μs | **3.36x** | **1.38x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 74.9 μs | 471.2 μs | 134.4 μs | **6.29x** | **1.79x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **75.2 μs** | 97.0 μs | 742.8 μs | 224.0 μs | **9.88x** | **2.98x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 70.0 μs | 516.3 μs | 121.8 μs | **7.38x** | **1.74x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **158.6 μs** | 138.7 μs | 781.1 μs | 273.8 μs | **5.63x** | **1.97x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 71.4 μs | 522.7 μs | 120.0 μs | **7.32x** | **1.68x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **193.6 μs** | 258.9 μs | 736.3 μs | 287.0 μs | **3.80x** | **1.48x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 69.0 μs | 498.0 μs | 136.8 μs | **7.22x** | **1.98x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **131.8 μs** | 139.6 μs | 714.9 μs | 241.3 μs | **5.42x** | **1.83x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 73.7 μs | 502.5 μs | 115.7 μs | **6.82x** | **1.57x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **149.0 μs** | 174.0 μs | 709.2 μs | 264.2 μs | **4.76x** | **1.77x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 71.2 μs | 474.3 μs | 117.1 μs | **6.66x** | **1.65x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **130.8 μs** | 142.6 μs | 666.7 μs | 229.5 μs | **5.10x** | **1.76x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 70.2 μs | 451.4 μs | 117.0 μs | **6.43x** | **1.67x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **117.0 μs** | 135.7 μs | 574.4 μs | 209.0 μs | **4.91x** | **1.79x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 74.5 μs | 467.7 μs | 120.0 μs | **6.27x** | **1.61x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **103.0 μs** | 136.9 μs | 670.6 μs | 229.3 μs | **6.51x** | **2.23x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 71.5 μs | 467.8 μs | 116.2 μs | **6.54x** | **1.62x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **119.0 μs** | 126.0 μs | 725.0 μs | 532.5 μs | **6.09x** | **4.48x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 66.0 μs | 471.5 μs | 116.7 μs | **7.15x** | **1.77x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **125.3 μs** | 133.1 μs | 646.0 μs | 257.2 μs | **5.16x** | **2.05x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 67.7 μs | 455.7 μs | 110.5 μs | **6.73x** | **1.63x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **72.2 μs** | 81.2 μs | 544.7 μs | 183.1 μs | **7.55x** | **2.54x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 66.4 μs | 465.7 μs | 123.0 μs | **7.02x** | **1.85x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **87.8 μs** | 121.0 μs | 602.4 μs | 207.4 μs | **6.86x** | **2.36x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 72.2 μs | 479.3 μs | 112.2 μs | **6.64x** | **1.55x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **89.9 μs** | 105.3 μs | 577.0 μs | 194.4 μs | **6.42x** | **2.16x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 68.8 μs | 449.7 μs | 121.4 μs | **6.54x** | **1.76x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **72.7 μs** | 85.4 μs | 686.4 μs | 188.5 μs | **9.45x** | **2.59x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 78.8 μs | 502.5 μs | 136.0 μs | **6.38x** | **1.73x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **86.2 μs** | 89.8 μs | 603.5 μs | 187.8 μs | **7.00x** | **2.18x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 80.0 μs | 559.3 μs | 132.5 μs | **6.99x** | **1.66x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **106.8 μs** | 106.5 μs | 670.4 μs | 238.4 μs | **6.30x** | **2.24x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 82.9 μs | 546.3 μs | 144.9 μs | **6.59x** | **1.75x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **77.6 μs** | 82.5 μs | 609.8 μs | 215.5 μs | **7.86x** | **2.78x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 79.3 μs | 519.1 μs | 149.1 μs | **6.55x** | **1.88x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **77.9 μs** | 88.6 μs | 594.5 μs | 203.2 μs | **7.63x** | **2.61x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 86.8 μs | 608.5 μs | 132.5 μs | **7.01x** | **1.53x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **74.0 μs** | 88.7 μs | 589.6 μs | 214.7 μs | **7.97x** | **2.90x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 88.8 μs | 547.9 μs | 140.8 μs | **6.17x** | **1.59x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **75.2 μs** | 87.0 μs | 626.9 μs | 204.0 μs | **8.34x** | **2.71x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 89.1 μs | 559.3 μs | 135.2 μs | **6.28x** | **1.52x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **102.3 μs** | 102.5 μs | 724.2 μs | 210.2 μs | **7.08x** | **2.05x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 91.4 μs | 585.4 μs | 129.3 μs | **6.41x** | **1.42x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **90.7 μs** | 96.7 μs | 515.8 μs | 180.0 μs | **5.68x** | **1.98x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 84.6 μs | 562.8 μs | 144.8 μs | **6.65x** | **1.71x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **68.5 μs** | 83.4 μs | 616.3 μs | 211.5 μs | **9.00x** | **3.09x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 84.0 μs | 574.5 μs | 128.9 μs | **6.84x** | **1.53x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **91.2 μs** | 100.8 μs | 733.1 μs | 210.3 μs | **8.04x** | **2.31x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 81.7 μs | 599.8 μs | 147.5 μs | **7.34x** | **1.81x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **104.1 μs** | 124.1 μs | 613.4 μs | 208.1 μs | **5.89x** | **2.00x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 90.3 μs | 583.0 μs | 152.6 μs | **6.46x** | **1.69x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **75.7 μs** | 87.7 μs | 720.9 μs | 190.0 μs | **9.52x** | **2.51x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 88.2 μs | 612.0 μs | 150.3 μs | **6.94x** | **1.70x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **75.3 μs** | 89.3 μs | 752.3 μs | 229.1 μs | **9.99x** | **3.04x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 87.9 μs | 625.7 μs | 141.0 μs | **7.12x** | **1.60x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **79.8 μs** | 91.9 μs | 711.7 μs | 203.2 μs | **8.92x** | **2.55x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | **11.43 ms** | 11.53 ms | 23.93 ms | 14.26 ms | **2.09x** | **1.25x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | **12.11 ms** | 12.28 ms | 41.94 ms | 14.80 ms | **3.46x** | **1.22x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | **21.20 ms** | 21.47 ms | 25.35 ms | 21.12 ms | **1.20x** | **1.00x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | **775.7 μs** | 754.8 μs | 5.57 ms | 841.5 μs | **7.38x** | **1.11x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | **6.95 ms** | 7.10 ms | 18.52 ms | 7.12 ms | **2.66x** | **1.02x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | **710.6 μs** | 772.0 μs | 1.78 ms | 805.0 μs | **2.51x** | **1.13x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | **3.18 ms** | 3.35 ms | 6.42 ms | 3.53 ms | **2.02x** | **1.11x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | **452.6 μs** | 516.3 μs | 987.2 μs | 517.9 μs | **2.18x** | **1.14x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | **1.58 ms** | 1.78 ms | 3.24 ms | 1.92 ms | **2.05x** | **1.22x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | **709.7 μs** | 799.1 μs | 1.82 ms | 820.5 μs | **2.57x** | **1.16x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | **3.11 ms** | 3.37 ms | 6.54 ms | 3.68 ms | **2.11x** | **1.19x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | **525.5 μs** | 1.92 ms | 3.35 ms | 1.82 ms | **6.38x** | **3.46x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | **1.58 ms** | 7.91 ms | 12.14 ms | 8.36 ms | **7.68x** | **5.29x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | **255.1 μs** | 714.1 μs | 595.7 μs | 742.3 μs | **2.34x** | **2.91x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | **703.5 μs** | 2.63 ms | 2.05 ms | 2.72 ms | **2.91x** | **3.87x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | **1.52 ms** | 3.45 ms | 5.21 ms | 3.46 ms | **3.43x** | **2.28x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | **3.32 ms** | 5.13 ms | 8.15 ms | 5.28 ms | **2.46x** | **1.59x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | **210.9 μs** | 232.9 μs | 436.7 μs | 266.5 μs | **2.07x** | **1.26x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | **764.5 μs** | 998.1 μs | 2.19 ms | 822.6 μs | **2.86x** | **1.08x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | **1.31 ms** | 1.17 ms | 2.77 ms | 1.37 ms | **2.36x** | **1.16x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | **226.8 μs** | 222.8 μs | 414.3 μs | 266.7 μs | **1.86x** | **1.20x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | **456.0 μs** | 460.8 μs | 996.7 μs | 496.9 μs | **2.19x** | **1.09x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | **660.7 μs** | 600.3 μs | 1.48 ms | 689.5 μs | **2.46x** | **1.15x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | **222.5 μs** | 210.3 μs | 412.2 μs | 239.4 μs | **1.96x** | **1.14x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | **744.9 μs** | 732.2 μs | 1.66 ms | 877.3 μs | **2.27x** | **1.20x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | **1.16 ms** | 1.34 ms | 3.37 ms | 11.15 ms | **2.90x** | **9.61x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | **243.6 μs** | 277.7 μs | 602.2 μs | 335.8 μs | **2.47x** | **1.38x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | **537.1 μs** | 1.87 ms | 3.50 ms | 2.04 ms | **6.51x** | **3.80x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | **719.9 μs** | 3.00 ms | 4.36 ms | 2.63 ms | **6.06x** | **3.65x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | **171.3 μs** | 224.4 μs | 462.0 μs | 324.6 μs | **2.70x** | **1.89x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | **257.3 μs** | 727.8 μs | 708.7 μs | 766.5 μs | **2.75x** | **2.98x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | **396.0 μs** | 1.06 ms | 811.6 μs | 1.12 ms | **2.05x** | **2.84x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | **432.5 μs** | 855.8 μs | 1.64 ms | 906.0 μs | **3.80x** | **2.10x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | **749.8 μs** | 1.24 ms | 2.29 ms | 1.27 ms | **3.06x** | **1.70x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | **244.6 μs** | 744.8 μs | 761.1 μs | 811.5 μs | **3.11x** | **3.32x** |
| fits | mef_small | read_full | 0.45 MB | MPS | **165.6 μs** | 211.2 μs | 566.9 μs | 278.0 μs | **3.42x** | **1.68x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | **177.4 μs** | 196.4 μs | 2.42 ms | 316.0 μs | **13.65x** | **1.78x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | **168.5 μs** | 201.6 μs | 561.6 μs | 340.8 μs | **3.33x** | **2.02x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | **13.70 ms** | 12.92 ms | 78.55 ms | 13.27 ms | **6.08x** | **1.03x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | **1.52 ms** | 3.18 ms | 6.66 ms | 3.16 ms | **4.38x** | **2.08x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | **466.0 μs** | 835.8 μs | 2.24 ms | 883.5 μs | **4.80x** | **1.90x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | **191.2 μs** | 234.1 μs | 534.6 μs | 264.8 μs | **2.80x** | **1.38x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | **192.9 μs** | 176.1 μs | 364.2 μs | 209.3 μs | **2.07x** | **1.19x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | **223.8 μs** | 228.3 μs | 432.5 μs | 261.8 μs | **1.93x** | **1.17x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | **259.0 μs** | 285.3 μs | 457.6 μs | 294.9 μs | **1.77x** | **1.14x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | **169.7 μs** | 165.5 μs | 341.2 μs | 201.0 μs | **2.06x** | **1.21x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | **169.8 μs** | 184.3 μs | 394.2 μs | 223.0 μs | **2.32x** | **1.31x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | **220.5 μs** | 228.3 μs | 414.9 μs | 251.3 μs | **1.88x** | **1.14x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | **141.0 μs** | 174.7 μs | 361.8 μs | 190.0 μs | **2.57x** | **1.35x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | **214.0 μs** | 210.3 μs | 413.5 μs | 263.9 μs | **1.97x** | **1.26x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | **237.9 μs** | 229.8 μs | 466.4 μs | 298.1 μs | **2.03x** | **1.30x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | **183.0 μs** | 174.5 μs | 441.0 μs | 203.3 μs | **2.53x** | **1.17x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | **197.1 μs** | 243.6 μs | 429.8 μs | 259.5 μs | **2.18x** | **1.32x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | **234.3 μs** | 389.7 μs | 616.7 μs | 378.4 μs | **2.63x** | **1.61x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | **165.5 μs** | 168.8 μs | 397.0 μs | 215.5 μs | **2.40x** | **1.30x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | **166.3 μs** | 190.7 μs | 421.5 μs | 241.5 μs | **2.53x** | **1.45x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | **175.2 μs** | 240.7 μs | 508.0 μs | 358.2 μs | **2.90x** | **2.04x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | **177.4 μs** | 199.5 μs | 514.2 μs | 289.9 μs | **2.90x** | **1.63x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | **206.1 μs** | 242.7 μs | 536.2 μs | 289.2 μs | **2.60x** | **1.40x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | **193.8 μs** | 217.8 μs | 427.3 μs | 257.5 μs | **2.20x** | **1.33x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | **200.8 μs** | 218.8 μs | 419.4 μs | 218.2 μs | **2.09x** | **1.09x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | **209.1 μs** | 216.6 μs | 420.8 μs | 227.0 μs | **2.01x** | **1.09x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | **194.7 μs** | 217.4 μs | 414.3 μs | 257.9 μs | **2.13x** | **1.32x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | **193.3 μs** | 220.8 μs | 430.6 μs | 255.7 μs | **2.23x** | **1.32x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | **154.2 μs** | 184.3 μs | 400.2 μs | 193.0 μs | **2.60x** | **1.25x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | **163.8 μs** | 173.1 μs | 368.7 μs | 182.3 μs | **2.25x** | **1.11x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | **146.0 μs** | 161.2 μs | 391.6 μs | 230.8 μs | **2.68x** | **1.58x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | **144.8 μs** | 153.9 μs | 365.8 μs | 186.9 μs | **2.53x** | **1.29x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | **157.4 μs** | 174.7 μs | 374.1 μs | 186.7 μs | **2.38x** | **1.19x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | **151.5 μs** | 158.5 μs | 386.3 μs | 215.4 μs | **2.55x** | **1.42x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | **162.3 μs** | 173.9 μs | 353.5 μs | 208.8 μs | **2.18x** | **1.29x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | **171.9 μs** | 171.6 μs | 373.0 μs | 180.3 μs | **2.17x** | **1.05x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | **198.0 μs** | 165.6 μs | 370.3 μs | 209.5 μs | **2.24x** | **1.27x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | **190.0 μs** | 165.9 μs | 367.8 μs | 202.5 μs | **2.22x** | **1.22x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | **184.5 μs** | 194.9 μs | 362.0 μs | 193.3 μs | **1.96x** | **1.05x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | **191.4 μs** | 171.0 μs | 382.1 μs | 227.9 μs | **2.23x** | **1.33x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | **138.1 μs** | 146.6 μs | 425.5 μs | 200.9 μs | **3.08x** | **1.46x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | **163.9 μs** | 179.3 μs | 413.9 μs | 200.7 μs | **2.53x** | **1.22x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | **157.2 μs** | 173.9 μs | 434.9 μs | 190.3 μs | **2.77x** | **1.21x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **443.5 μs** | 218.0 μs | 4.10 ms | 7.18 ms | **18.80x** | **32.95x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **80.0 μs** | 93.5 μs | 9.47 ms | 6.78 ms | **118.49x** | **84.84x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **83.7 μs** | 84.6 μs | 1.39 ms | 6.87 ms | **16.62x** | **81.98x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **81.3 μs** | 85.1 μs | 1.59 ms | 3.01 ms | **19.54x** | **37.04x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **90.3 μs** | 90.1 μs | 2.99 ms | 2.52 ms | **33.18x** | **27.93x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **361.9 μs** | 163.1 μs | 1.87 ms | 986.6 μs | **11.47x** | **6.05x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **87.4 μs** | 88.0 μs | 2.16 ms | 1.00 ms | **24.72x** | **11.47x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **82.2 μs** | 90.6 μs | 1.25 ms | 979.9 μs | **15.20x** | **11.92x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **84.7 μs** | 90.3 μs | 1.60 ms | 602.0 μs | **18.89x** | **7.11x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **86.7 μs** | 92.7 μs | 1.53 ms | 496.4 μs | **17.60x** | **5.73x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **16.19 ms** | 2.24 ms | 58.58 ms | 325.82 ms | **26.10x** | **145.17x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **80.6 μs** | 83.0 μs | 10.20 ms | 66.21 ms | **126.57x** | **821.59x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **78.0 μs** | 72.8 μs | 25.65 ms | 544.14 ms | **352.54x** | **7479.63x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **82.4 μs** | 81.6 μs | 12.89 ms | 103.35 ms | **157.91x** | **1266.11x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **97.0 μs** | 90.7 μs | 11.00 ms | 99.95 ms | **121.27x** | **1101.33x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **1.69 ms** | 378.1 μs | 8.16 ms | 31.87 ms | **21.58x** | **84.30x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **78.8 μs** | 80.9 μs | 2.62 ms | 7.12 ms | **33.21x** | **90.44x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **82.5 μs** | 80.0 μs | 3.93 ms | 53.68 ms | **49.12x** | **670.99x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **76.7 μs** | 84.6 μs | 3.40 ms | 14.31 ms | **44.34x** | **186.60x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **90.1 μs** | 87.5 μs | 2.78 ms | 10.11 ms | **31.79x** | **115.45x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **528.8 μs** | 247.8 μs | 3.10 ms | 3.72 ms | **12.53x** | **15.00x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **89.6 μs** | 84.0 μs | 1.91 ms | 987.8 μs | **22.73x** | **11.75x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **81.7 μs** | 89.3 μs | 2.08 ms | 5.90 ms | **25.43x** | **72.22x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **81.4 μs** | 86.8 μs | 2.45 ms | 1.75 ms | **30.08x** | **21.46x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **105.2 μs** | 89.2 μs | 1.86 ms | 1.29 ms | **20.88x** | **14.44x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **473.4 μs** | 193.4 μs | 2.76 ms | 828.6 μs | **14.28x** | **4.28x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **104.7 μs** | 110.4 μs | 1.98 ms | 496.3 μs | **18.93x** | **4.74x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **103.7 μs** | 101.1 μs | 2.03 ms | 1.04 ms | **20.05x** | **10.24x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **102.7 μs** | 113.1 μs | 2.79 ms | 651.6 μs | **27.13x** | **6.34x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **124.0 μs** | 116.7 μs | 2.24 ms | 515.4 μs | **19.22x** | **4.42x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **10.66 ms** | 1.86 ms | 28.72 ms | 10.92 ms | **15.47x** | **5.88x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **72.4 μs** | 77.1 μs | 4.49 ms | 33.67 ms | **61.97x** | **464.91x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **76.7 μs** | 84.9 μs | 8.27 ms | 6.50 ms | **107.78x** | **84.68x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **78.0 μs** | 82.5 μs | 5.27 ms | 3.69 ms | **67.56x** | **47.34x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **88.5 μs** | 82.2 μs | 4.64 ms | 3.44 ms | **56.38x** | **41.83x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.35 ms** | 511.5 μs | 4.86 ms | 1.34 ms | **9.51x** | **2.62x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **94.7 μs** | 78.1 μs | 1.68 ms | 3.56 ms | **21.46x** | **45.59x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **165.0 μs** | 186.0 μs | 2.66 ms | 2.54 ms | **16.11x** | **15.39x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **82.4 μs** | 84.8 μs | 2.04 ms | 589.4 μs | **24.71x** | **7.15x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **96.2 μs** | 98.4 μs | 1.62 ms | 475.2 μs | **16.87x** | **4.94x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **458.8 μs** | 246.0 μs | 2.11 ms | 415.0 μs | **8.58x** | **1.69x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **79.2 μs** | 88.6 μs | 1.40 ms | 654.2 μs | **17.70x** | **8.25x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **83.8 μs** | 82.9 μs | 1.38 ms | 337.9 μs | **16.70x** | **4.08x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **82.1 μs** | 90.8 μs | 1.76 ms | 333.8 μs | **21.45x** | **4.07x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **90.7 μs** | 89.9 μs | 1.35 ms | 303.7 μs | **15.07x** | **3.38x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **431.5 μs** | 201.0 μs | 1.97 ms | 409.0 μs | **9.79x** | **2.03x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **107.1 μs** | 113.1 μs | 1.60 ms | 451.1 μs | **14.92x** | **4.21x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **101.0 μs** | 108.1 μs | 1.59 ms | 397.8 μs | **15.76x** | **3.94x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **101.2 μs** | 102.4 μs | 1.92 ms | 400.1 μs | **18.97x** | **3.95x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **114.0 μs** | 113.8 μs | 1.57 ms | 350.4 μs | **13.79x** | **3.08x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.16 ms** | 415.3 μs | 4.30 ms | 58.80 ms | **10.34x** | **141.58x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **85.6 μs** | 89.0 μs | 30.40 ms | 55.76 ms | **354.99x** | **651.24x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **78.2 μs** | 90.1 μs | 2.63 ms | 58.69 ms | **33.69x** | **750.47x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **83.0 μs** | 92.2 μs | 2.23 ms | 20.04 ms | **26.82x** | **241.37x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **89.4 μs** | 91.4 μs | 1.77 ms | 15.82 ms | **19.84x** | **176.93x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **422.4 μs** | 209.5 μs | 2.01 ms | 6.43 ms | **9.60x** | **30.70x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **83.1 μs** | 87.9 μs | 4.54 ms | 5.93 ms | **54.63x** | **71.37x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **83.8 μs** | 88.3 μs | 1.47 ms | 6.32 ms | **17.59x** | **75.40x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **81.5 μs** | 85.2 μs | 1.69 ms | 2.44 ms | **20.77x** | **29.96x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **89.1 μs** | 94.2 μs | 1.36 ms | 1.78 ms | **15.28x** | **20.01x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **1.60 ms** | 469.4 μs | 4.50 ms | 130.71 ms | **9.60x** | **278.48x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **89.0 μs** | 89.7 μs | 464.86 ms | 131.40 ms | **5225.64x** | **1477.05x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **81.4 μs** | 84.5 μs | 2.44 ms | 122.68 ms | **30.03x** | **1507.54x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **86.7 μs** | 92.6 μs | 2.08 ms | 125.91 ms | **23.96x** | **1452.77x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **93.1 μs** | 93.7 μs | 1.77 ms | 134.48 ms | **19.05x** | **1444.78x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **386.9 μs** | 214.9 μs | 1.85 ms | 12.42 ms | **8.63x** | **57.78x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **81.7 μs** | 79.8 μs | 47.51 ms | 12.22 ms | **595.69x** | **153.19x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **82.8 μs** | 83.6 μs | 1.36 ms | 12.92 ms | **16.43x** | **156.01x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **78.1 μs** | 86.6 μs | 1.56 ms | 12.13 ms | **19.93x** | **155.21x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **85.1 μs** | 92.5 μs | 1.28 ms | 12.10 ms | **15.04x** | **142.19x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **326.8 μs** | 144.6 μs | 1.53 ms | 1.48 ms | **10.59x** | **10.27x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **77.7 μs** | 77.7 μs | 5.78 ms | 1.51 ms | **74.47x** | **19.41x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **78.8 μs** | 79.5 μs | 1.25 ms | 1.46 ms | **15.87x** | **18.53x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **76.2 μs** | 85.5 μs | 1.53 ms | 1.47 ms | **20.03x** | **19.26x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **78.2 μs** | 91.1 μs | 1.24 ms | 1.45 ms | **15.82x** | **18.56x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.54 ms** | 407.2 μs | 38.54 ms | 138.22 ms | **94.64x** | **339.40x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **75.7 μs** | 78.5 μs | 8.37 ms | 7.91 ms | **110.55x** | **104.45x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **79.1 μs** | 80.7 μs | 23.15 ms | 228.73 ms | **292.62x** | **2890.76x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **82.4 μs** | 81.9 μs | 12.52 ms | 65.34 ms | **152.92x** | **798.02x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **166.4 μs** | 85.1 μs | 8.47 ms | 46.73 ms | **99.54x** | **548.94x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.31 ms** | 256.7 μs | 10.70 ms | 15.57 ms | **41.67x** | **60.65x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **87.4 μs** | 86.3 μs | 5.45 ms | 1.44 ms | **63.18x** | **16.72x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **85.0 μs** | 93.1 μs | 6.47 ms | 24.48 ms | **76.11x** | **287.83x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **86.8 μs** | 87.5 μs | 8.23 ms | 7.55 ms | **94.83x** | **86.98x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **187.8 μs** | 110.2 μs | 9.19 ms | 6.33 ms | **83.38x** | **57.49x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **837.5 μs** | 152.1 μs | 8.18 ms | 2.05 ms | **53.77x** | **13.50x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **80.0 μs** | 88.8 μs | 4.96 ms | 612.4 μs | **61.93x** | **7.65x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **97.7 μs** | 107.7 μs | 5.01 ms | 2.92 ms | **51.31x** | **29.88x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **84.0 μs** | 92.1 μs | 8.34 ms | 1.31 ms | **99.26x** | **15.64x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **173.7 μs** | 93.3 μs | 5.13 ms | 1.10 ms | **55.01x** | **11.76x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | tiny_int8_3d [read_full @ mps] | 0.0003546250518411398 | fitsio/fitsio_torch_device | 2.0414987313539865 |
| fits | small_int8_1d [read_full @ mps] | 0.0003517919685691595 | fitsio/fitsio_torch_device | 1.9263088244293496 |
| fits | tiny_int8_2d [read_full @ mps] | 0.0003366251476109028 | fitsio/fitsio_torch_device | 1.8606621091538058 |
| fits | tiny_int8_1d [read_full @ mps] | 0.000372790964320302 | fitsio/fitsio_torch_device | 1.8554474744244043 |
| fits | small_int8_2d [read_full @ mps] | 0.0003952921833842993 | fitsio/fitsio_torch_device | 1.7923738699791072 |
| fits | small_uint32_2d [read_full @ mps] | 0.0004392080008983612 | fitsio/fitsio_torch_device | 1.6681396247538314 |
| fits | mef_small [read_full @ mps] | 0.0004034999292343855 | fitsio/fitsio_torch_device | 1.640243544628711 |
| fits | medium_int8_1d [read_full @ mps] | 0.0003874169196933508 | fitsio/fitsio_torch_device | 1.5669031227075958 |
| fits | scaled_medium [read_full @ mps] | 0.0013065000530332327 | fitsio/fitsio_torch_device | 1.4787774956227318 |
| fits | large_int8_1d [read_full @ mps] | 0.0008683330379426479 | astropy/astropy_torch_device | 1.4576489928263943 |
| fits | medium_uint32_2d [read_full @ mps] | 0.0017070418689399958 | fitsio/fitsio_torch_device | 1.343114314465123 |
| fits | medium_int16_3d [read_full] | 0.0009199581108987331 | fitsio/fitsio_torch | 1.3327088914830472 |
| fits | small_uint16_2d [read_full @ mps] | 0.0003858329728245735 | fitsio/fitsio_torch_device | 1.3308394582648027 |
| fits | multi_mef_10ext [read_full @ mps] | 0.0003785830922424793 | fitsio/fitsio_torch_device | 1.3011607203588191 |
| fits | large_int16_2d [read_full] | 0.0020485830027610064 | fitsio/fitsio_torch | 1.2854860964198178 |
| fits | medium_uint16_2d [read_full @ mps] | 0.0011004579719156027 | fitsio/fitsio_torch_device | 1.2145787034847983 |
| fits | large_uint32_2d [read_full @ mps] | 0.0063255419954657555 | fitsio/fitsio_torch_device | 1.1985112131045714 |
| fits | scaled_small [read_full @ mps] | 0.0003173330333083868 | fitsio/fitsio_torch_device | 1.1982341248385648 |
| fits | scaled_large [read_full @ mps] | 0.003742000088095665 | fitsio/fitsio_torch_device | 1.1825867619303403 |
| fits | small_int8_3d [read_full @ mps] | 0.00039858301170170307 | fitsio/fitsio_torch_device | 1.1585294217169153 |
| fits | mef_medium [read_full @ mps] | 0.0008811249863356352 | astropy/astropy_torch_device | 1.1577250799751715 |
| fits | medium_int16_3d [read_full @ mps] | 0.0007735001854598522 | fitsio/fitsio_torch_device | 1.121827661428118 |
| fits | medium_int8_2d [read_full @ mps] | 0.0007851249538362026 | astropy/astropy_torch_device | 1.107824153498723 |
| fits | scaled_large [read_full] | 0.003123542061075568 | fitsio/fitsio_torch | 1.1042129035635375 |
| fits | large_int16_1d [read_full @ mps] | 0.0005677500739693642 | fitsio/fitsio_torch_device | 1.0962180542762057 |
| fits | large_float32_1d [read_full @ mps] | 0.0008780418429523706 | fitsio/fitsio_torch_device | 1.090790947313373 |
| fits | medium_float32_2d [read_full @ mps] | 0.0008970829658210278 | fitsio/fitsio_torch_device | 1.090567117590266 |
| fits | repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | 0.0060594589449465275 | fitsio/fitsio_torch | 1.0574049327348793 |
| fits | large_int32_1d [read_full @ mps] | 0.0008562498260289431 | fitsio/fitsio_torch_device | 1.0435706824675561 |
| fits | medium_int8_3d [read_full @ mps] | 0.0008408331777900457 | astropy/astropy_torch_device | 1.0359872320027264 |
| fits | repeated_cutouts_50x_100x100 @ mps | 0.013702667085453868 | fitsio/fitsio_torch_device | 1.0329486581094187 |
| fits | large_int8_2d [read_full @ mps] | 0.0020965419244021177 | astropy/astropy_torch_device | 1.0251828793171838 |
| fits | compressed_hcompress_1 [read_full @ mps] | 0.021516166161745787 | fitsio/fitsio_torch_device | 1.0186151563993808 |
| fits | compressed_hcompress_1 [read_full] | 0.022908917162567377 | fitsio/fitsio_torch | 1.0109606873218127 |
| fits | large_int16_2d [read_full] | 0.0016792919486761093 | fitsio/fitsio | 1.1238044635852498 |
| fits | medium_int16_3d [read_full] | 0.0007095420733094215 | fitsio/fitsio | 1.0658454565008681 |
| fits | compressed_hcompress_1 [read_full] | 0.02286570891737938 | fitsio/fitsio | 1.0173992808611765 |
| fitstable | narrow_10000 [predicate_filter] | 0.00045879208482801914 | fitsio/fitsio_torch | 1.1054111092162424 |
| fitstable | narrow_100000 [predicate_filter] | 0.0013494158629328012 | fitsio/fitsio_torch | 1.0317944976628526 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_0.7.0_20260711_022156` | fits + fitstable (lab) | 3516 | 39 | lab bench-all + `--mmap-matrix` + CUDA/MPS |
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
