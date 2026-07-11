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
   runs will **not** refresh GPU cells; published CUDA numbers come from CANFAR
   staging (`exhaustive_cuda_0.7.0_20260711_055635`, via `pixi run bench-canfar-gpu`).
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

# Download CSVs from VOSpace (if launcher did not auto-fetch via vcp):
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cuda_0.7.0_<stamp>

# Patch docs from local CSV:
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

In-container work uses **pixi**; bench CSVs upload to **`vos:sfabbro/torchfits-gpu-bench/<run-id>/`**
via `vcp` (x509 in the session). Platform logs land under
`benchmarks_results/canfar_<run-id>/` locally. Download results with
`scripts/fetch_canfar_bench_vos.sh` (needs `pip install vos` + cert locally).

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
Source: `benchmarks_results/exhaustive_cuda_0.7.0_20260711_055635/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.13 ms` (n=269) | `0.77 ms` (n=269) | `0.24 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.14 ms` (n=269) | `0.65 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.14 ms` (n=256) | `0.66 ms` (n=90) | `0.23 ms` (n=90) | — |
| `disk→RAM→GPU` | `0.15 ms` (n=256) | `0.95 ms` (n=90) | `0.24 ms` (n=90) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.10 ms` (n=180) | `3.38 ms` (n=162) | `3.20 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.09 ms` (n=180) | `3.12 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **3.93 ms** | 3.97 ms | 7.60 ms | 6.09 ms | **1.93x** | **1.55x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.19 ms** | 3.33 ms | 8.18 ms | 5.15 ms | **2.57x** | **1.61x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **8.99 ms** | 9.07 ms | 28.14 ms | 9.36 ms | **3.13x** | **1.04x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **8.89 ms** | 8.94 ms | 27.73 ms | 9.20 ms | **3.12x** | **1.04x** |
| Repeated Cutouts (50x 100x100) | CPU | **4.63 ms** | 4.38 ms | 76.04 ms | 4.76 ms | **17.38x** | **1.09x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **86.9 μs** | 89.9 μs | 6.32 ms | 59.41 ms | **72.72x** | **683.64x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **90.8 μs** | 95.4 μs | 3.49 ms | 219.88 ms | **38.49x** | **2421.88x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 144.2 μs | 2.09 ms | 261.7 μs | **14.52x** | **1.82x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **16.05 ms** | 16.11 ms | 41.07 ms | 17.84 ms | **2.56x** | **1.11x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 143.7 μs | 2.09 ms | 259.3 μs | **14.52x** | **1.80x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **15.57 ms** | 15.64 ms | 67.22 ms | 17.49 ms | **4.32x** | **1.12x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 150.2 μs | 2.22 ms | 284.0 μs | **14.76x** | **1.89x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **30.68 ms** | 30.73 ms | 38.48 ms | 29.50 ms | **1.25x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **909.1 μs** | 907.2 μs | 10.31 ms | 1.10 ms | **11.37x** | **1.21x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 155.2 μs | 2.19 ms | 282.8 μs | **14.14x** | **1.82x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **8.99 ms** | 9.07 ms | 28.14 ms | 9.36 ms | **3.13x** | **1.04x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 84.7 μs | 586.9 μs | 120.0 μs | **6.93x** | **1.42x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **976.1 μs** | 974.0 μs | 2.18 ms | 1.46 ms | **2.24x** | **1.50x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 89.4 μs | 617.9 μs | 116.7 μs | **6.91x** | **1.31x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **3.93 ms** | 3.97 ms | 7.60 ms | 6.09 ms | **1.93x** | **1.55x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 82.9 μs | 590.1 μs | 120.2 μs | **7.11x** | **1.45x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **1.92 ms** | 1.91 ms | 3.89 ms | 2.47 ms | **2.03x** | **1.29x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 86.0 μs | 616.9 μs | 123.2 μs | **7.17x** | **1.43x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **10.40 ms** | 10.52 ms | 23.49 ms | 11.09 ms | **2.26x** | **1.07x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 82.4 μs | 575.9 μs | 118.0 μs | **6.99x** | **1.43x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **598.1 μs** | 600.5 μs | 1.46 ms | 739.6 μs | **2.44x** | **1.24x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 87.6 μs | 610.5 μs | 123.2 μs | **6.97x** | **1.41x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **2.05 ms** | 2.07 ms | 4.07 ms | 2.44 ms | **1.99x** | **1.19x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 82.0 μs | 585.2 μs | 119.3 μs | **7.14x** | **1.45x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **951.7 μs** | 968.1 μs | 2.18 ms | 1.45 ms | **2.29x** | **1.53x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 84.0 μs | 612.1 μs | 123.1 μs | **7.29x** | **1.47x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **3.90 ms** | 3.92 ms | 7.53 ms | 6.04 ms | **1.93x** | **1.55x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 83.6 μs | 581.1 μs | 118.0 μs | **6.95x** | **1.41x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **1.91 ms** | 1.93 ms | 3.88 ms | 2.48 ms | **2.03x** | **1.30x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 84.5 μs | 612.1 μs | 127.3 μs | **7.25x** | **1.51x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.49 ms** | 10.47 ms | 23.58 ms | 11.15 ms | **2.25x** | **1.07x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 87.6 μs | 641.8 μs | 128.4 μs | **7.33x** | **1.47x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **308.6 μs** | 328.7 μs | 1.22 ms | 437.2 μs | **3.95x** | **1.42x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 88.9 μs | 672.9 μs | 128.8 μs | **7.57x** | **1.45x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **1.06 ms** | 1.08 ms | 2.68 ms | 1.39 ms | **2.52x** | **1.31x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 89.5 μs | 674.9 μs | 130.7 μs | **7.54x** | **1.46x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **2.42 ms** | 2.45 ms | 6.44 ms | 2.75 ms | **2.66x** | **1.13x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 89.1 μs | 675.3 μs | 128.6 μs | **7.58x** | **1.44x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **4.90 ms** | 4.90 ms | 10.26 ms | 6.80 ms | **2.10x** | **1.39x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 84.2 μs | 591.2 μs | 121.8 μs | **7.02x** | **1.45x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **178.0 μs** | 194.6 μs | 765.4 μs | 305.3 μs | **4.30x** | **1.71x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 80.9 μs | 622.1 μs | 116.8 μs | **7.69x** | **1.44x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **1.02 ms** | 1.03 ms | 2.51 ms | 1.56 ms | **2.47x** | **1.53x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 83.2 μs | 644.1 μs | 122.8 μs | **7.74x** | **1.48x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **1.51 ms** | 1.54 ms | 3.26 ms | 2.31 ms | **2.16x** | **1.53x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 81.4 μs | 591.1 μs | 114.3 μs | **7.26x** | **1.40x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **263.9 μs** | 275.4 μs | 945.5 μs | 386.4 μs | **3.58x** | **1.46x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 82.6 μs | 619.7 μs | 120.1 μs | **7.50x** | **1.45x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **1.99 ms** | 2.00 ms | 4.03 ms | 2.57 ms | **2.02x** | **1.29x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 84.2 μs | 650.0 μs | 120.0 μs | **7.72x** | **1.43x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **3.06 ms** | 3.09 ms | 5.98 ms | 4.04 ms | **1.95x** | **1.32x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 77.0 μs | 588.7 μs | 112.4 μs | **7.64x** | **1.46x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **147.8 μs** | 162.2 μs | 698.4 μs | 230.1 μs | **4.73x** | **1.56x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 84.0 μs | 620.9 μs | 119.2 μs | **7.40x** | **1.42x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **629.3 μs** | 633.3 μs | 1.53 ms | 784.2 μs | **2.43x** | **1.25x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 80.5 μs | 641.7 μs | 122.6 μs | **7.97x** | **1.52x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **919.3 μs** | 936.9 μs | 2.04 ms | 1.12 ms | **2.21x** | **1.22x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 80.0 μs | 586.3 μs | 111.4 μs | **7.33x** | **1.39x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **191.1 μs** | 203.8 μs | 783.5 μs | 319.7 μs | **4.10x** | **1.67x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 84.4 μs | 608.8 μs | 121.0 μs | **7.22x** | **1.43x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **1.12 ms** | 1.13 ms | 2.65 ms | 1.71 ms | **2.36x** | **1.53x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 85.1 μs | 642.5 μs | 122.9 μs | **7.55x** | **1.44x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **1.60 ms** | 1.62 ms | 3.39 ms | 2.46 ms | **2.12x** | **1.54x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 80.3 μs | 581.5 μs | 113.3 μs | **7.24x** | **1.41x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **282.8 μs** | 305.1 μs | 958.7 μs | 413.7 μs | **3.39x** | **1.46x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 85.1 μs | 611.5 μs | 119.6 μs | **7.18x** | **1.41x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **2.07 ms** | 2.08 ms | 4.10 ms | 2.64 ms | **1.99x** | **1.28x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 83.6 μs | 631.5 μs | 120.6 μs | **7.56x** | **1.44x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **3.10 ms** | 3.11 ms | 6.02 ms | 4.07 ms | **1.94x** | **1.31x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 85.9 μs | 637.5 μs | 123.9 μs | **7.42x** | **1.44x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **121.2 μs** | 137.6 μs | 795.8 μs | 206.6 μs | **6.57x** | **1.70x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 87.7 μs | 667.4 μs | 127.5 μs | **7.61x** | **1.45x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **357.0 μs** | 371.5 μs | 1.29 ms | 494.5 μs | **3.61x** | **1.39x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 90.8 μs | 700.0 μs | 132.0 μs | **7.71x** | **1.45x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **487.7 μs** | 505.6 μs | 1.58 ms | 673.3 μs | **3.23x** | **1.38x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 86.8 μs | 674.5 μs | 126.2 μs | **7.77x** | **1.45x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **636.8 μs** | 655.6 μs | 2.10 ms | 775.2 μs | **3.29x** | **1.22x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 86.6 μs | 672.7 μs | 132.1 μs | **7.77x** | **1.53x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **1.16 ms** | 1.19 ms | 2.95 ms | 1.68 ms | **2.54x** | **1.45x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 96.7 μs | 964.8 μs | 142.2 μs | **9.98x** | **1.47x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **322.4 μs** | 349.9 μs | 1.51 ms | 490.2 μs | **4.70x** | **1.52x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 93.2 μs | 951.7 μs | 142.1 μs | **10.21x** | **1.52x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **111.8 μs** | 134.5 μs | 1.06 ms | 243.6 μs | **9.50x** | **2.18x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **94.5 μs** | 90.4 μs | 3.35 ms | 334.0 μs | **37.02x** | **3.69x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 94.7 μs | 960.0 μs | 137.8 μs | **10.14x** | **1.45x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **6.16 ms** | 6.17 ms | 11.39 ms | 9.62 ms | **1.85x** | **1.56x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **114.8 μs** | 134.1 μs | 1.07 ms | 314.1 μs | **9.36x** | **2.74x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **4.63 ms** | 4.38 ms | 76.04 ms | 4.76 ms | **17.38x** | **1.09x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 90.9 μs | 683.9 μs | 127.1 μs | **7.53x** | **1.40x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **3.52 ms** | 3.55 ms | 7.27 ms | 4.80 ms | **2.06x** | **1.36x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 90.1 μs | 682.3 μs | 129.3 μs | **7.57x** | **1.43x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **957.0 μs** | 983.1 μs | 2.42 ms | 1.33 ms | **2.52x** | **1.39x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 87.8 μs | 691.2 μs | 126.0 μs | **7.87x** | **1.43x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **155.2 μs** | 173.7 μs | 908.7 μs | 250.1 μs | **5.85x** | **1.61x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 78.9 μs | 578.8 μs | 117.9 μs | **7.33x** | **1.49x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **110.5 μs** | 125.3 μs | 623.1 μs | 194.9 μs | **5.64x** | **1.76x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 83.0 μs | 612.2 μs | 118.0 μs | **7.37x** | **1.42x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **165.2 μs** | 179.7 μs | 738.1 μs | 273.7 μs | **4.47x** | **1.66x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 83.8 μs | 641.1 μs | 120.7 μs | **7.65x** | **1.44x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **261.2 μs** | 273.8 μs | 948.9 μs | 417.8 μs | **3.63x** | **1.60x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 79.5 μs | 578.8 μs | 114.8 μs | **7.28x** | **1.44x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **118.6 μs** | 132.8 μs | 646.3 μs | 199.2 μs | **5.45x** | **1.68x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 84.3 μs | 608.7 μs | 116.7 μs | **7.22x** | **1.39x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **226.2 μs** | 237.3 μs | 866.4 μs | 341.4 μs | **3.83x** | **1.51x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 82.9 μs | 646.2 μs | 126.8 μs | **7.79x** | **1.53x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **428.9 μs** | 434.1 μs | 1.22 ms | 577.7 μs | **2.85x** | **1.35x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 78.4 μs | 582.0 μs | 113.5 μs | **7.42x** | **1.45x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **101.9 μs** | 117.6 μs | 602.8 μs | 181.7 μs | **5.92x** | **1.78x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 81.3 μs | 611.7 μs | 119.3 μs | **7.52x** | **1.47x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **131.2 μs** | 147.7 μs | 677.3 μs | 212.4 μs | **5.16x** | **1.62x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 82.6 μs | 636.6 μs | 119.5 μs | **7.71x** | **1.45x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **177.6 μs** | 194.5 μs | 778.6 μs | 272.3 μs | **4.38x** | **1.53x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 76.9 μs | 582.2 μs | 113.7 μs | **7.57x** | **1.48x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **109.2 μs** | 121.8 μs | 613.3 μs | 183.5 μs | **5.61x** | **1.68x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 80.4 μs | 607.0 μs | 120.9 μs | **7.55x** | **1.50x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **160.0 μs** | 177.7 μs | 736.3 μs | 263.9 μs | **4.60x** | **1.65x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 84.0 μs | 640.6 μs | 122.9 μs | **7.63x** | **1.46x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **258.0 μs** | 279.9 μs | 934.4 μs | 411.5 μs | **3.62x** | **1.59x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 81.0 μs | 577.2 μs | 118.2 μs | **7.13x** | **1.46x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **118.4 μs** | 136.2 μs | 646.0 μs | 196.7 μs | **5.45x** | **1.66x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 85.7 μs | 616.5 μs | 120.0 μs | **7.19x** | **1.40x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **215.7 μs** | 239.1 μs | 859.2 μs | 332.1 μs | **3.98x** | **1.54x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 83.7 μs | 637.4 μs | 124.2 μs | **7.61x** | **1.48x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **421.4 μs** | 439.0 μs | 1.22 ms | 587.0 μs | **2.90x** | **1.39x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 89.1 μs | 643.1 μs | 118.4 μs | **7.22x** | **1.33x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **87.5 μs** | 108.4 μs | 762.3 μs | 181.9 μs | **8.71x** | **2.08x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 91.5 μs | 668.6 μs | 131.1 μs | **7.30x** | **1.43x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **116.5 μs** | 137.4 μs | 802.7 μs | 202.6 μs | **6.89x** | **1.74x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 90.4 μs | 701.8 μs | 123.4 μs | **7.77x** | **1.37x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **141.5 μs** | 165.0 μs | 866.9 μs | 238.8 μs | **6.13x** | **1.69x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 90.9 μs | 670.5 μs | 131.9 μs | **7.37x** | **1.45x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **134.1 μs** | 154.8 μs | 793.3 μs | 217.2 μs | **5.91x** | **1.62x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 87.4 μs | 677.6 μs | 131.4 μs | **7.75x** | **1.50x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **162.6 μs** | 180.8 μs | 836.0 μs | 272.0 μs | **5.14x** | **1.67x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 81.7 μs | 612.2 μs | 118.6 μs | **7.50x** | **1.45x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **164.9 μs** | 174.3 μs | 739.1 μs | 271.5 μs | **4.48x** | **1.65x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 83.3 μs | 610.9 μs | 119.6 μs | **7.34x** | **1.44x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **159.2 μs** | 172.9 μs | 737.7 μs | 268.6 μs | **4.63x** | **1.69x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 82.8 μs | 612.9 μs | 117.5 μs | **7.41x** | **1.42x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **163.5 μs** | 174.3 μs | 728.7 μs | 272.4 μs | **4.46x** | **1.67x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 83.8 μs | 609.6 μs | 119.3 μs | **7.27x** | **1.42x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **163.2 μs** | 178.2 μs | 734.8 μs | 275.7 μs | **4.50x** | **1.69x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 84.4 μs | 609.0 μs | 116.5 μs | **7.22x** | **1.38x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **159.2 μs** | 175.5 μs | 725.2 μs | 268.2 μs | **4.56x** | **1.68x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 81.7 μs | 581.9 μs | 117.3 μs | **7.12x** | **1.44x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **87.1 μs** | 100.2 μs | 604.1 μs | 171.0 μs | **6.93x** | **1.96x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 82.6 μs | 616.3 μs | 119.2 μs | **7.46x** | **1.44x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **104.2 μs** | 121.8 μs | 630.9 μs | 178.0 μs | **6.06x** | **1.71x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 83.4 μs | 635.8 μs | 121.5 μs | **7.62x** | **1.46x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **104.3 μs** | 119.9 μs | 645.9 μs | 185.4 μs | **6.19x** | **1.78x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 80.7 μs | 585.8 μs | 117.8 μs | **7.26x** | **1.46x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **86.5 μs** | 128.9 μs | 723.9 μs | 211.4 μs | **8.37x** | **2.45x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 82.9 μs | 613.2 μs | 116.4 μs | **7.40x** | **1.40x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **145.1 μs** | 177.9 μs | 641.4 μs | 179.7 μs | **4.42x** | **1.24x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 84.7 μs | 640.1 μs | 122.9 μs | **7.56x** | **1.45x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **104.4 μs** | 121.7 μs | 642.7 μs | 181.3 μs | **6.16x** | **1.74x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 80.8 μs | 578.5 μs | 115.3 μs | **7.16x** | **1.43x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **84.2 μs** | 97.9 μs | 588.4 μs | 166.4 μs | **6.99x** | **1.98x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 83.8 μs | 606.5 μs | 120.9 μs | **7.24x** | **1.44x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **89.1 μs** | 100.8 μs | 608.8 μs | 172.0 μs | **6.83x** | **1.93x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 84.1 μs | 640.6 μs | 123.8 μs | **7.62x** | **1.47x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **83.8 μs** | 102.1 μs | 633.9 μs | 170.5 μs | **7.57x** | **2.03x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 81.6 μs | 578.0 μs | 117.3 μs | **7.09x** | **1.44x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **86.0 μs** | 97.1 μs | 584.8 μs | 168.0 μs | **6.80x** | **1.95x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 83.2 μs | 605.6 μs | 114.9 μs | **7.27x** | **1.38x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **98.6 μs** | 116.7 μs | 612.2 μs | 175.2 μs | **6.21x** | **1.78x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 86.6 μs | 637.1 μs | 123.9 μs | **7.36x** | **1.43x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **100.3 μs** | 118.9 μs | 638.0 μs | 175.7 μs | **6.36x** | **1.75x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 82.4 μs | 585.9 μs | 117.3 μs | **7.11x** | **1.42x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **82.7 μs** | 100.9 μs | 585.9 μs | 169.5 μs | **7.08x** | **2.05x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 82.4 μs | 607.8 μs | 116.1 μs | **7.37x** | **1.41x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **103.3 μs** | 117.5 μs | 623.3 μs | 180.6 μs | **6.04x** | **1.75x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 85.8 μs | 635.2 μs | 124.0 μs | **7.40x** | **1.44x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **108.6 μs** | 125.9 μs | 646.5 μs | 182.4 μs | **5.95x** | **1.68x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 86.3 μs | 643.0 μs | 122.1 μs | **7.45x** | **1.41x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **80.9 μs** | 98.4 μs | 740.5 μs | 165.6 μs | **9.15x** | **2.05x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 91.1 μs | 672.7 μs | 123.3 μs | **7.38x** | **1.35x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **84.3 μs** | 100.1 μs | 763.4 μs | 177.6 μs | **9.06x** | **2.11x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 90.0 μs | 700.3 μs | 127.6 μs | **7.78x** | **1.42x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **87.5 μs** | 105.6 μs | 775.8 μs | 172.5 μs | **8.87x** | **1.97x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CUDA | **15.89 ms** | 15.95 ms | 40.37 ms | 17.61 ms | **2.54x** | **1.11x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CUDA | **15.43 ms** | 15.50 ms | 66.43 ms | 17.31 ms | **4.31x** | **1.12x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CUDA | **30.45 ms** | 30.71 ms | 37.91 ms | 29.20 ms | **1.24x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CUDA | **838.0 μs** | 836.6 μs | 9.96 ms | 967.8 μs | **11.91x** | **1.16x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CUDA | **8.89 ms** | 8.94 ms | 27.73 ms | 9.20 ms | **3.12x** | **1.04x** |
| fits | large_float32_1d | read_full | 3.82 MB | CUDA | **716.6 μs** | 734.7 μs | 1.46 ms | 1.23 ms | **2.03x** | **1.72x** |
| fits | large_float32_2d | read_full | 16.00 MB | CUDA | **3.19 ms** | 3.33 ms | 8.18 ms | 5.15 ms | **2.57x** | **1.61x** |
| fits | large_float64_1d | read_full | 7.63 MB | CUDA | **1.34 ms** | 1.36 ms | 2.87 ms | 1.86 ms | **2.15x** | **1.39x** |
| fits | large_float64_2d | read_full | 32.00 MB | CUDA | **11.19 ms** | 11.32 ms | 25.33 ms | 12.46 ms | **2.26x** | **1.11x** |
| fits | large_int16_1d | read_full | 1.91 MB | CUDA | **432.6 μs** | 453.8 μs | 994.5 μs | 564.6 μs | **2.30x** | **1.31x** |
| fits | large_int16_2d | read_full | 8.00 MB | CUDA | **1.49 ms** | 1.51 ms | 3.21 ms | 1.82 ms | **2.16x** | **1.22x** |
| fits | large_int32_1d | read_full | 3.82 MB | CUDA | **721.6 μs** | 735.4 μs | 1.49 ms | 1.27 ms | **2.06x** | **1.76x** |
| fits | large_int32_2d | read_full | 16.00 MB | CUDA | **3.33 ms** | 3.44 ms | 8.41 ms | 5.28 ms | **2.53x** | **1.59x** |
| fits | large_int64_1d | read_full | 7.63 MB | CUDA | **543.1 μs** | 1.68 ms | 2.98 ms | 2.15 ms | **5.49x** | **3.96x** |
| fits | large_int64_2d | read_full | 32.00 MB | CUDA | **2.14 ms** | 11.28 ms | 25.77 ms | 12.39 ms | **12.02x** | **5.78x** |
| fits | large_int8_1d | read_full | 0.96 MB | CUDA | **220.0 μs** | 299.3 μs | 891.6 μs | 370.3 μs | **4.05x** | **1.68x** |
| fits | large_int8_2d | read_full | 4.00 MB | CUDA | **792.9 μs** | 1.07 ms | 2.94 ms | 1.21 ms | **3.71x** | **1.52x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CUDA | **1.80 ms** | 2.11 ms | 5.65 ms | 2.34 ms | **3.14x** | **1.30x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CUDA | **3.06 ms** | 3.67 ms | 10.03 ms | 5.76 ms | **3.28x** | **1.88x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CUDA | **108.8 μs** | 130.4 μs | 534.9 μs | 220.7 μs | **4.92x** | **2.03x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CUDA | **927.9 μs** | 946.8 μs | 1.69 ms | 1.44 ms | **1.82x** | **1.55x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CUDA | **1.39 ms** | 1.39 ms | 2.38 ms | 2.16 ms | **1.71x** | **1.55x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CUDA | **194.7 μs** | 214.7 μs | 665.3 μs | 294.2 μs | **3.42x** | **1.51x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CUDA | **1.75 ms** | 1.78 ms | 4.50 ms | 2.25 ms | **2.56x** | **1.28x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CUDA | **2.29 ms** | 2.34 ms | 6.37 ms | 3.30 ms | **2.78x** | **1.44x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CUDA | **66.1 μs** | 80.8 μs | 493.5 μs | 135.7 μs | **7.46x** | **2.05x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CUDA | **535.0 μs** | 553.9 μs | 1.72 ms | 647.8 μs | **3.21x** | **1.21x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CUDA | **770.3 μs** | 790.0 μs | 1.46 ms | 915.9 μs | **1.89x** | **1.19x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CUDA | **129.5 μs** | 147.5 μs | 529.7 μs | 231.0 μs | **4.09x** | **1.78x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CUDA | **766.9 μs** | 782.2 μs | 1.70 ms | 1.33 ms | **2.21x** | **1.74x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CUDA | **1.14 ms** | 1.16 ms | 2.36 ms | 1.95 ms | **2.06x** | **1.71x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CUDA | **144.5 μs** | 217.0 μs | 671.0 μs | 302.7 μs | **4.64x** | **2.10x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CUDA | **570.6 μs** | 1.46 ms | 3.44 ms | 2.06 ms | **6.03x** | **3.60x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CUDA | **818.0 μs** | 2.32 ms | 9.72 ms | 3.34 ms | **11.89x** | **4.09x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CUDA | **45.0 μs** | 66.0 μs | 573.5 μs | 111.0 μs | **12.75x** | **2.47x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CUDA | **224.2 μs** | 301.1 μs | 930.4 μs | 377.8 μs | **4.15x** | **1.69x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CUDA | **302.1 μs** | 420.7 μs | 1.13 ms | 522.7 μs | **3.74x** | **1.73x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CUDA | **450.2 μs** | 536.0 μs | 1.66 ms | 650.7 μs | **3.68x** | **1.45x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CUDA | **755.2 μs** | 917.0 μs | 2.17 ms | 1.45 ms | **2.87x** | **1.92x** |
| fits | mef_medium | read_full | 7.02 MB | CUDA | **223.7 μs** | 301.7 μs | 1.15 ms | 414.3 μs | **5.14x** | **1.85x** |
| fits | mef_small | read_full | 0.45 MB | CUDA | **48.9 μs** | 68.1 μs | 810.7 μs | 143.0 μs | **16.57x** | **2.92x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CUDA | **42.5 μs** | 45.6 μs | 3.23 ms | 238.5 μs | **75.96x** | **5.61x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CUDA | **48.2 μs** | 64.7 μs | 816.8 μs | 195.7 μs | **16.95x** | **4.06x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | CUDA | **6.20 ms** | 7.33 ms | 82.61 ms | 6.13 ms | **13.32x** | **0.99x** |
| fits | scaled_large | read_full | 8.00 MB | CUDA | **1.48 ms** | 4.02 ms | 11.28 ms | 4.64 ms | **7.62x** | **3.14x** |
| fits | scaled_medium | read_full | 2.01 MB | CUDA | **443.4 μs** | 1.07 ms | 1.84 ms | 1.25 ms | **4.14x** | **2.81x** |
| fits | scaled_small | read_full | 0.13 MB | CUDA | **56.0 μs** | 133.4 μs | 696.7 μs | 169.5 μs | **12.44x** | **3.03x** |
| fits | small_float32_1d | read_full | 42.2 KB | CUDA | **39.6 μs** | 50.6 μs | 428.0 μs | 93.0 μs | **10.81x** | **2.35x** |
| fits | small_float32_2d | read_full | 0.26 MB | CUDA | **77.4 μs** | 94.2 μs | 519.0 μs | 176.2 μs | **6.71x** | **2.28x** |
| fits | small_float32_3d | read_full | 0.63 MB | CUDA | **168.3 μs** | 190.8 μs | 662.6 μs | 311.7 μs | **3.94x** | **1.85x** |
| fits | small_float64_1d | read_full | 0.08 MB | CUDA | **44.2 μs** | 59.1 μs | 432.8 μs | 98.8 μs | **9.79x** | **2.24x** |
| fits | small_float64_2d | read_full | 0.51 MB | CUDA | **138.4 μs** | 160.5 μs | 594.8 μs | 229.2 μs | **4.30x** | **1.66x** |
| fits | small_float64_3d | read_full | 1.26 MB | CUDA | **296.5 μs** | 319.4 μs | 845.8 μs | 439.3 μs | **2.85x** | **1.48x** |
| fits | small_int16_1d | read_full | 22.5 KB | CUDA | **36.1 μs** | 45.9 μs | 409.9 μs | 83.2 μs | **11.34x** | **2.30x** |
| fits | small_int16_2d | read_full | 0.13 MB | CUDA | **53.2 μs** | 69.7 μs | 476.2 μs | 112.2 μs | **8.95x** | **2.11x** |
| fits | small_int16_3d | read_full | 0.32 MB | CUDA | **94.0 μs** | 113.8 μs | 554.6 μs | 170.7 μs | **5.90x** | **1.82x** |
| fits | small_int32_1d | read_full | 42.2 KB | CUDA | **38.1 μs** | 51.1 μs | 421.0 μs | 92.4 μs | **11.06x** | **2.43x** |
| fits | small_int32_2d | read_full | 0.26 MB | CUDA | **75.4 μs** | 94.0 μs | 501.1 μs | 166.1 μs | **6.64x** | **2.20x** |
| fits | small_int32_3d | read_full | 0.63 MB | CUDA | **166.8 μs** | 186.8 μs | 636.5 μs | 294.0 μs | **3.82x** | **1.76x** |
| fits | small_int64_1d | read_full | 0.08 MB | CUDA | **62.1 μs** | 56.1 μs | 423.3 μs | 95.1 μs | **7.55x** | **1.70x** |
| fits | small_int64_2d | read_full | 0.51 MB | CUDA | **107.8 μs** | 158.7 μs | 587.2 μs | 227.6 μs | **5.45x** | **2.11x** |
| fits | small_int64_3d | read_full | 1.26 MB | CUDA | **185.4 μs** | 315.4 μs | 845.9 μs | 444.4 μs | **4.56x** | **2.40x** |
| fits | small_int8_1d | read_full | 14.1 KB | CUDA | **30.4 μs** | 41.3 μs | 542.9 μs | 74.4 μs | **17.84x** | **2.44x** |
| fits | small_int8_2d | read_full | 0.07 MB | CUDA | **48.8 μs** | 58.2 μs | 592.3 μs | 98.1 μs | **12.14x** | **2.01x** |
| fits | small_int8_3d | read_full | 0.16 MB | CUDA | **53.3 μs** | 80.5 μs | 634.2 μs | 129.8 μs | **11.90x** | **2.43x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CUDA | **54.9 μs** | 74.2 μs | 582.3 μs | 120.4 μs | **10.62x** | **2.20x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CUDA | **78.7 μs** | 102.2 μs | 605.7 μs | 177.6 μs | **7.70x** | **2.26x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CUDA | **72.9 μs** | 94.2 μs | 520.2 μs | 175.3 μs | **7.13x** | **2.40x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CUDA | **76.4 μs** | 96.5 μs | 522.1 μs | 170.9 μs | **6.84x** | **2.24x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CUDA | **76.6 μs** | 94.4 μs | 515.1 μs | 175.1 μs | **6.73x** | **2.29x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CUDA | **74.1 μs** | 92.7 μs | 512.2 μs | 169.5 μs | **6.91x** | **2.29x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CUDA | **72.5 μs** | 92.1 μs | 518.1 μs | 172.2 μs | **7.15x** | **2.38x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CUDA | **27.7 μs** | 39.2 μs | 409.1 μs | 80.5 μs | **14.79x** | **2.91x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CUDA | **33.9 μs** | 46.2 μs | 424.4 μs | 87.0 μs | **12.53x** | **2.57x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CUDA | **34.9 μs** | 49.6 μs | 457.0 μs | 86.7 μs | **13.09x** | **2.48x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CUDA | **29.8 μs** | 40.1 μs | 413.7 μs | 81.4 μs | **13.90x** | **2.74x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CUDA | **38.9 μs** | 51.4 μs | 434.6 μs | 83.3 μs | **11.18x** | **2.14x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CUDA | **42.7 μs** | 52.6 μs | 453.3 μs | 86.1 μs | **10.60x** | **2.01x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CUDA | **28.5 μs** | 38.0 μs | 395.2 μs | 76.4 μs | **13.85x** | **2.68x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CUDA | **34.1 μs** | 46.4 μs | 432.4 μs | 79.6 μs | **12.67x** | **2.33x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CUDA | **30.2 μs** | 41.8 μs | 449.4 μs | 81.1 μs | **14.87x** | **2.68x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CUDA | **28.4 μs** | 38.6 μs | 399.3 μs | 78.5 μs | **14.04x** | **2.76x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CUDA | **36.6 μs** | 47.0 μs | 431.8 μs | 83.7 μs | **11.80x** | **2.29x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CUDA | **36.5 μs** | 48.1 μs | 449.5 μs | 87.4 μs | **12.32x** | **2.40x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CUDA | **52.7 μs** | 39.5 μs | 403.7 μs | 75.1 μs | **10.22x** | **1.90x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CUDA | **58.2 μs** | 52.1 μs | 438.6 μs | 85.6 μs | **8.42x** | **1.64x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CUDA | **58.1 μs** | 51.5 μs | 463.2 μs | 87.1 μs | **8.99x** | **1.69x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CUDA | **28.4 μs** | 40.4 μs | 535.3 μs | 72.3 μs | **18.85x** | **2.55x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CUDA | **29.5 μs** | 40.8 μs | 551.6 μs | 78.6 μs | **18.68x** | **2.66x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CUDA | **29.8 μs** | 42.2 μs | 571.3 μs | 74.3 μs | **19.14x** | **2.49x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **621.4 μs** | 346.0 μs | 4.87 ms | 7.56 ms | **14.08x** | **21.86x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **89.6 μs** | 90.0 μs | 11.46 ms | 7.57 ms | **127.86x** | **84.52x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **87.8 μs** | 90.3 μs | 2.22 ms | 7.46 ms | **25.24x** | **85.01x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **88.7 μs** | 91.6 μs | 2.65 ms | 3.67 ms | **29.84x** | **41.32x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **130.5 μs** | 101.9 μs | 3.91 ms | 3.10 ms | **38.40x** | **30.46x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **500.3 μs** | 284.0 μs | 3.05 ms | 1.19 ms | **10.75x** | **4.19x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **93.2 μs** | 97.3 μs | 3.28 ms | 1.25 ms | **35.16x** | **13.41x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **93.6 μs** | 97.7 μs | 2.04 ms | 1.18 ms | **21.84x** | **12.62x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **92.9 μs** | 95.4 μs | 2.55 ms | 765.8 μs | **27.43x** | **8.24x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **131.7 μs** | 106.0 μs | 2.35 ms | 629.9 μs | **22.17x** | **5.94x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **18.69 ms** | 8.48 ms | 108.83 ms | 401.72 ms | **12.83x** | **47.35x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **88.2 μs** | 93.0 μs | 19.94 ms | 59.46 ms | **226.13x** | **674.35x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **87.1 μs** | 89.9 μs | 58.67 ms | 643.56 ms | **673.33x** | **7385.86x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **89.6 μs** | 92.7 μs | 21.54 ms | 133.28 ms | **240.36x** | **1487.06x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **138.3 μs** | 99.7 μs | 20.34 ms | 130.30 ms | **204.03x** | **1307.19x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **2.37 ms** | 1.11 ms | 11.70 ms | 36.57 ms | **10.58x** | **33.04x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **89.7 μs** | 91.9 μs | 4.22 ms | 5.82 ms | **47.12x** | **64.90x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **86.9 μs** | 89.9 μs | 6.32 ms | 59.41 ms | **72.72x** | **683.64x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **87.1 μs** | 90.4 μs | 5.51 ms | 16.61 ms | **63.26x** | **190.79x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **134.7 μs** | 102.0 μs | 4.19 ms | 12.25 ms | **41.04x** | **120.06x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **715.8 μs** | 328.7 μs | 4.73 ms | 4.04 ms | **14.38x** | **12.29x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **88.2 μs** | 90.8 μs | 3.00 ms | 973.8 μs | **34.02x** | **11.05x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **85.7 μs** | 89.5 μs | 3.19 ms | 6.12 ms | **37.17x** | **71.46x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **89.1 μs** | 95.0 μs | 4.07 ms | 2.10 ms | **45.67x** | **23.61x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **137.9 μs** | 99.3 μs | 2.96 ms | 1.51 ms | **29.79x** | **15.24x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **538.1 μs** | 258.2 μs | 3.92 ms | 874.0 μs | **15.19x** | **3.38x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **88.0 μs** | 93.0 μs | 2.76 ms | 478.4 μs | **31.41x** | **5.44x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **88.8 μs** | 91.3 μs | 2.80 ms | 1.06 ms | **31.58x** | **11.97x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **88.8 μs** | 91.3 μs | 3.81 ms | 659.1 μs | **42.86x** | **7.42x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **135.7 μs** | 99.9 μs | 2.74 ms | 511.3 μs | **27.40x** | **5.12x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **14.10 ms** | 8.04 ms | 36.65 ms | 13.07 ms | **4.56x** | **1.63x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **90.2 μs** | 92.4 μs | 6.26 ms | 40.52 ms | **69.43x** | **449.34x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **90.4 μs** | 89.1 μs | 10.31 ms | 8.73 ms | **115.70x** | **98.01x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **90.4 μs** | 93.4 μs | 7.00 ms | 5.16 ms | **77.37x** | **57.02x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **120.7 μs** | 102.4 μs | 6.21 ms | 5.05 ms | **60.66x** | **49.31x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **1.87 ms** | 1.09 ms | 6.25 ms | 1.63 ms | **5.73x** | **1.50x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **89.4 μs** | 93.8 μs | 2.61 ms | 4.44 ms | **29.21x** | **49.62x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **88.5 μs** | 91.0 μs | 2.99 ms | 1.23 ms | **33.78x** | **13.86x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **89.0 μs** | 93.5 μs | 3.30 ms | 881.2 μs | **37.04x** | **9.90x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **118.8 μs** | 100.0 μs | 2.58 ms | 786.9 μs | **25.86x** | **7.87x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **613.6 μs** | 354.4 μs | 3.04 ms | 521.0 μs | **8.59x** | **1.47x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **90.0 μs** | 90.6 μs | 2.13 ms | 809.4 μs | **23.65x** | **8.99x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **85.9 μs** | 92.5 μs | 2.13 ms | 458.2 μs | **24.76x** | **5.33x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **88.1 μs** | 91.7 μs | 2.70 ms | 438.0 μs | **30.68x** | **4.97x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **121.6 μs** | 100.9 μs | 2.08 ms | 382.4 μs | **20.59x** | **3.79x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **475.1 μs** | 278.5 μs | 2.69 ms | 416.3 μs | **9.65x** | **1.49x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **89.0 μs** | 93.4 μs | 2.04 ms | 446.0 μs | **22.98x** | **5.01x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **87.1 μs** | 91.7 μs | 2.07 ms | 403.4 μs | **23.74x** | **4.63x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **89.7 μs** | 92.1 μs | 2.62 ms | 394.5 μs | **29.19x** | **4.40x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **119.7 μs** | 100.9 μs | 2.02 ms | 341.3 μs | **20.04x** | **3.38x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.50 ms** | 963.1 μs | 6.08 ms | 64.08 ms | **6.32x** | **66.54x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **96.5 μs** | 101.6 μs | 39.58 ms | 60.88 ms | **410.13x** | **630.81x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **95.1 μs** | 100.5 μs | 3.64 ms | 63.00 ms | **38.31x** | **662.24x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **95.5 μs** | 101.4 μs | 3.46 ms | 22.24 ms | **36.21x** | **232.96x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **132.0 μs** | 110.6 μs | 2.75 ms | 17.75 ms | **24.90x** | **160.48x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **603.9 μs** | 361.6 μs | 3.24 ms | 6.87 ms | **8.97x** | **19.00x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **93.2 μs** | 100.3 μs | 5.88 ms | 6.52 ms | **63.06x** | **69.98x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **91.6 μs** | 97.5 μs | 2.35 ms | 6.68 ms | **25.67x** | **72.90x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **93.8 μs** | 99.2 μs | 2.93 ms | 2.81 ms | **31.20x** | **29.94x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **127.7 μs** | 109.6 μs | 2.29 ms | 2.22 ms | **20.90x** | **20.29x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **1.33 ms** | 1.01 ms | 6.04 ms | 233.71 ms | **6.01x** | **232.23x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **96.9 μs** | 100.5 μs | 770.13 ms | 232.14 ms | **7948.36x** | **2395.86x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **90.8 μs** | 95.4 μs | 3.49 ms | 219.88 ms | **38.49x** | **2421.88x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **95.9 μs** | 97.8 μs | 3.23 ms | 234.95 ms | **33.68x** | **2451.11x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **124.0 μs** | 109.9 μs | 2.58 ms | 237.74 ms | **23.49x** | **2163.93x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **573.4 μs** | 349.9 μs | 2.96 ms | 21.92 ms | **8.45x** | **62.65x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **92.4 μs** | 95.1 μs | 78.62 ms | 21.41 ms | **850.54x** | **231.57x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **89.8 μs** | 93.2 μs | 2.14 ms | 21.16 ms | **23.87x** | **235.81x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **91.6 μs** | 92.4 μs | 2.60 ms | 21.76 ms | **28.44x** | **237.63x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **119.5 μs** | 102.9 μs | 2.05 ms | 21.63 ms | **19.90x** | **210.25x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **464.4 μs** | 278.3 μs | 2.54 ms | 2.31 ms | **9.14x** | **8.32x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **92.0 μs** | 89.8 μs | 9.70 ms | 2.39 ms | **108.00x** | **26.64x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **87.9 μs** | 90.2 μs | 1.99 ms | 2.37 ms | **22.61x** | **26.94x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **89.3 μs** | 94.4 μs | 2.47 ms | 2.28 ms | **27.60x** | **25.54x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **115.9 μs** | 101.7 μs | 1.93 ms | 2.31 ms | **18.99x** | **22.68x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **4.96 ms** | 1.08 ms | 58.18 ms | 150.55 ms | **53.77x** | **139.14x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **90.8 μs** | 91.2 μs | 12.71 ms | 9.56 ms | **140.00x** | **105.26x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **88.3 μs** | 92.7 μs | 37.35 ms | 241.80 ms | **422.78x** | **2736.89x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **87.0 μs** | 91.0 μs | 18.89 ms | 69.29 ms | **217.26x** | **796.75x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **241.4 μs** | 98.9 μs | 12.70 ms | 50.99 ms | **128.40x** | **515.69x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.49 ms** | 349.6 μs | 16.30 ms | 15.13 ms | **46.61x** | **43.27x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **87.8 μs** | 92.8 μs | 8.97 ms | 1.60 ms | **102.09x** | **18.20x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **87.5 μs** | 90.3 μs | 10.39 ms | 23.50 ms | **118.78x** | **268.57x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **88.3 μs** | 93.4 μs | 13.81 ms | 7.63 ms | **156.45x** | **86.49x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **236.3 μs** | 101.5 μs | 8.91 ms | 5.62 ms | **87.81x** | **55.34x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **1.11 ms** | 280.8 μs | 13.49 ms | 2.37 ms | **48.04x** | **8.42x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **88.0 μs** | 91.9 μs | 8.36 ms | 768.6 μs | **95.02x** | **8.73x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **88.8 μs** | 89.5 μs | 8.66 ms | 3.16 ms | **97.58x** | **35.59x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **89.5 μs** | 91.6 μs | 13.21 ms | 1.62 ms | **147.63x** | **18.08x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **240.2 μs** | 101.3 μs | 8.40 ms | 1.22 ms | **82.89x** | **12.05x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | compressed_hcompress_1 [read_full @ cuda] | 0.030480971559882164 | fitsio/fitsio_torch_device | 1.0437711895920028 |
| fits | compressed_hcompress_1 [read_full] | 0.03060827497392893 | fitsio/fitsio_torch | 1.0351335616889226 |
| fits | small_int8_1d [read_full @ cuda] | 7.693935185670853e-05 | fitsio/fitsio_torch_device | 1.0348097301901444 |
| fits | tiny_int8_3d [read_full @ cuda] | 7.638335227966309e-05 | fitsio/fitsio_torch_device | 1.0278724684178866 |
| fits | repeated_cutouts_50x_100x100 @ cuda | 0.006200309842824936 | fitsio/fitsio_torch_device | 1.0114105398574356 |
| fits | tiny_int8_1d [read_full @ cuda] | 7.288437336683273e-05 | fitsio/fitsio_torch_device | 1.0075444491650896 |
| fits | compressed_hcompress_1 [read_full] | 0.030662798322737217 | fitsio/fitsio | 1.0394091417942253 |
| fitstable | narrow_10000 [predicate_filter] | 0.0006061336025595665 | fitsio/fitsio_torch | 1.1701994530459157 |
| fitstable | narrow_1000 [predicate_filter] | 0.00047508999705314636 | fitsio/fitsio_torch | 1.1472845203717197 |
| fitstable | narrow_100000 [predicate_filter] | 0.0018413299694657326 | fitsio/fitsio_torch | 1.131314212145054 |
| fitstable | narrow_1000000 [predicate_filter] | 0.013966827653348446 | fitsio/fitsio_torch | 1.097294760912315 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_cuda_0.7.0_20260711_055635` | fits + fitstable (CANFAR staging) | 3626 | 11 | lab bench-all + `--mmap-matrix` + CUDA |
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
