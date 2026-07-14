# Benchmarks

`torchfits` benchmarks cover FITS **image** and **table** I/O vs astropy and fitsio.

## How to read this page

You do **not** need every table on first visit.

| If you want… | Jump to |
|---|---|
| Headline wins (image read, cutouts, tables) | [Performance highlights](#performance-highlights) |
| Cases where torchfits is not #1 | [Performance deficits](#performance-deficits) |
| GPU transport rows (CUDA/MPS) | [I/O transport and backend](#io-transport-and-backend) |
| Reproduce numbers locally | [Reproducing](#reproducing) |
| Every measured configuration | [Exhaustive benchmark results](#exhaustive-benchmark-results) (long) |

Published GPU numbers come from a CANFAR staging run (`exhaustive_cuda_0.9.0_20260714_065950`).
GitHub Actions weekly benches use CPU-only PyTorch and do not refresh GPU cells.

## Comparison targets

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

Named **deficit-cluster** recipes (mmap on+off, no unrelated GPU matrix when scoped to tables):

```bash
pixi run bench-deficit-focus              # hcompress + tiny_int8 + narrow predicates
pixi run bench-deficit-focus hcompress
pixi run bench-deficit-focus tiny_int8
pixi run bench-deficit-focus predicate
```

Rankings and deficit scoring group by `(domain, case_id, family, mmap_target)` so
mmap-on and mmap-off peers are never cross-compared.
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
   staging (`exhaustive_cuda_0.9.0_20260714_065950`, via `pixi run bench-canfar-gpu`).
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
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cuda_0.9.0_<stamp>

# Patch docs from local CSV:
bash scripts/patch_canfar_exhaustive_docs.sh exhaustive_cuda_0.9.0_<stamp>
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

## I/O transport and backend

> **GPU summary:** Image **`disk→CPU→GPU`** and **`disk→RAM→GPU`** rows appear only when the benchmark CSV was
> produced on CUDA or MPS hardware. **`disk→GPU`** is intentionally empty (unsupported by
> all backends). **Table GPU transports are not benchmarked.** CI weekly `bench-report`
> uses CPU PyTorch and will not update GPU cells.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/exhaustive_cuda_0.9.1_20260714_202004/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.12 ms` (n=269) | `0.82 ms` (n=269) | `0.26 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.14 ms` (n=269) | `0.73 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.16 ms` (n=267) | `0.67 ms` (n=90) | `0.25 ms` (n=90) | — |
| `disk→RAM→GPU` | `0.18 ms` (n=267) | `1.00 ms` (n=90) | `0.27 ms` (n=90) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.11 ms` (n=180) | `3.48 ms` (n=162) | `3.23 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.11 ms` (n=180) | `3.25 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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

## Performance highlights

<!-- BENCH_HIGHLIGHTS_BEGIN -->
The following table showcases median wall-clock execution times of key representative FITS benchmarks.
In almost all core I/O paths, `torchfits` is significantly faster than standard astronomical tools, with extra performance wins from persistent handle caches and direct-to-device transfers.

| Benchmark Case | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **3.98 ms** | 3.93 ms | 14.94 ms | 6.04 ms | **3.80x** | **1.54x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **4.08 ms** | 3.69 ms | 15.93 ms | 5.93 ms | **4.32x** | **1.61x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **8.71 ms** | 8.79 ms | 28.05 ms | 9.00 ms | **3.22x** | **1.03x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **8.90 ms** | 8.89 ms | 27.48 ms | 9.24 ms | **3.09x** | **1.04x** |
| Repeated Cutouts (50x 100x100) | CPU | **4.65 ms** | 4.45 ms | 77.29 ms | 4.85 ms | **17.35x** | **1.09x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **102.3 μs** | 105.7 μs | 6.65 ms | 59.22 ms | **65.02x** | **578.75x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **103.1 μs** | 104.7 μs | 3.58 ms | 215.47 ms | **34.71x** | **2090.58x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

Aggregated wins across every domain and operation in the exhaustive suite
(`exhaustive_cuda_0.9.0_20260714_065950`, 3,648 rows).

### FITS image I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **1D** (float32/64, int8–int64, tiny–large) | 48 | 85 μs – 1.86 ms | 640 μs – 3.83 ms | 130 μs – 2.39 ms | **2.1–7.8×** | **1.3–2.3×** |
| **2D** (float32/64, int8–int64, uint16/32, tiny–large) | 52 | 101 μs – 10.59 ms | 660 μs – 24.0 ms | 135 μs – 11.24 ms | **2.3–7.7×** | **1.1–2.4×** |
| **3D** (float32/64, int8–int64, small–medium) | 36 | 103 μs – 3.20 ms | 689 μs – 6.12 ms | 140 μs – 4.16 ms | **1.9–7.5×** | **1.2–2.1×** |
| **Compressed** (gzip, hcompress, rice) | 6 | 9.06–30.64 ms | 27.77–40.35 ms | 9.43–29.45 ms | **1.2–4.3×** | **1.0–1.1×** |
| **Scaled** (BSCALE/BZERO, small–large) | 6 | 154 μs – 3.53 ms | 935 μs – 11.44 ms | 277 μs – 4.76 ms | **2.5–6.1×** | **1.3–1.8×** |
| **MEF** (multi-extension, small/medium) | 2 | 112–324 μs | 1.11–1.56 ms | 267–507 μs | **4.8–9.9×** | **1.6–2.4×** |
| **Multi-MEF** (10 extensions, cutouts + random reads) | 3 | 95 μs – 6.62 ms | 3.39–11.25 ms | 361 μs – 10.19 ms | **1.7–35.9×** | **1.1–3.8×** |
| **Repeated cutouts** (50× 100×100) | 1 | 4.68 ms | 75.36 ms | 4.94 ms | **16.7×** | **1.1×** |
| **Time series frames** (5 frames) | 5 | 148–160 μs | 750–763 μs | 272–278 μs | **4.7–5.1×** | **1.7–1.9×** |
| **Header read** (all fixture types) | ~55 | 88–109 μs | 615–1010 μs | 128–159 μs | **6.5–9.5×** | **1.4–1.5×** |

**GPU (CUDA) image reads** — 76 `read_full` cases:

| Category | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|
| **1D** (tiny–large) | 27–818 μs | 382–1620 μs | 74–1330 μs | **2.0–14.9×** | **1.3–3.0×** |
| **2D** (tiny–large) | 28–3.51 ms | 382–17.8 ms | 74–5.50 ms | **2.1–19.9×** | **1.1–2.9×** |
| **3D** (tiny–medium) | 33–2.62 ms | 437–10.43 ms | 87–3.51 ms | **1.9–15.1×** | **1.3–2.6×** |
| **Scaled** | 54 μs – 1.68 ms | 674 μs – 11.61 ms | 169–4880 μs | **3.9–12.5×** | **1.6–3.1×** |
| **MEF + Multi-MEF** | 42–238 μs | 795–3220 μs | 134–419 μs | **6.1–78.4×** | **1.8–5.4×** |
| **Repeated cutouts (GPU)** | 6.30 ms | 82.38 ms | 6.35 ms | **14.1×** | **1.1×** |

### FITS table I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **read_full** (mixed/narrow/wide/varlen, 1k–100k rows) | 20 | 93–184 μs | 2.25–6.74 ms | 3.25–59.84 ms | **24–115×** | **34–628×** |
| **projection** (column subset) | 20 | 93–101 μs | 2.60–13.53 ms | 219 μs – 9.94 ms | **26–147×** | **2.3–91×** |
| **row_slice** (row range) | 20 | 94–103 μs | 2.69–14.18 ms | 308 μs – 15.70 ms | **28–147×** | **3.2–162×** |
| **predicate_filter** (WHERE clause) | 20 | 643 μs – 1.06 ms | 3.10–13.53 ms | 561 μs – 7.60 ms | **3.2–57×** | **0.44–25×** |
| **scan_count** (streaming) | 20 | 134–277 μs | 3.98–11.15 ms | 490 μs – 12.44 ms | **30–85×** | **4.6–55×** |

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | mmap | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | off | **—** | 164.0 μs | 2.13 ms | 273.9 μs | **12.96x** | **1.67x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **15.72 ms** | 15.75 ms | 41.51 ms | 17.43 ms | **2.64x** | **1.11x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | off | **—** | 160.6 μs | 2.13 ms | 271.0 μs | **13.26x** | **1.69x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **15.21 ms** | 15.31 ms | 67.39 ms | 17.30 ms | **4.43x** | **1.14x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | off | **—** | 163.6 μs | 2.23 ms | 295.2 μs | **13.65x** | **1.80x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **30.21 ms** | 30.22 ms | 38.01 ms | 29.06 ms | **1.26x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | off | **927.9 μs** | 933.5 μs | 10.43 ms | 1.13 ms | **11.24x** | **1.22x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | off | **—** | 166.6 μs | 2.22 ms | 293.7 μs | **13.33x** | **1.76x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **8.71 ms** | 8.79 ms | 28.05 ms | 9.00 ms | **3.22x** | **1.03x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | off | **—** | 98.5 μs | 626.3 μs | 135.8 μs | **6.36x** | **1.38x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | off | **1.00 ms** | 988.0 μs | 2.26 ms | 1.52 ms | **2.29x** | **1.54x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | off | **—** | 98.8 μs | 666.4 μs | 138.2 μs | **6.75x** | **1.40x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | off | **3.98 ms** | 3.93 ms | 14.94 ms | 6.04 ms | **3.80x** | **1.54x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | off | **—** | 95.6 μs | 631.9 μs | 133.1 μs | **6.61x** | **1.39x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | off | **1.91 ms** | 1.87 ms | 3.88 ms | 2.44 ms | **2.07x** | **1.30x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | off | **—** | 99.0 μs | 659.0 μs | 137.1 μs | **6.65x** | **1.38x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | off | **10.58 ms** | 10.54 ms | 23.77 ms | 11.39 ms | **2.25x** | **1.08x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | off | **—** | 93.8 μs | 627.7 μs | 132.0 μs | **6.69x** | **1.41x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | off | **556.7 μs** | 532.7 μs | 1.47 ms | 718.5 μs | **2.76x** | **1.35x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | off | **—** | 96.0 μs | 659.0 μs | 136.5 μs | **6.87x** | **1.42x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | off | **2.07 ms** | 2.06 ms | 4.03 ms | 2.43 ms | **1.95x** | **1.18x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | off | **—** | 94.0 μs | 627.6 μs | 130.5 μs | **6.68x** | **1.39x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | off | **1.01 ms** | 981.6 μs | 2.24 ms | 1.52 ms | **2.28x** | **1.54x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | off | **—** | 98.4 μs | 661.2 μs | 136.5 μs | **6.72x** | **1.39x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | off | **3.96 ms** | 3.89 ms | 14.91 ms | 6.02 ms | **3.83x** | **1.55x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | off | **—** | 95.2 μs | 625.3 μs | 132.7 μs | **6.57x** | **1.39x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | off | **1.90 ms** | 1.88 ms | 3.87 ms | 2.46 ms | **2.06x** | **1.31x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | off | **—** | 96.3 μs | 657.5 μs | 139.1 μs | **6.83x** | **1.44x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | off | **10.61 ms** | 10.58 ms | 23.83 ms | 11.40 ms | **2.25x** | **1.08x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | off | **—** | 100.7 μs | 691.8 μs | 139.9 μs | **6.87x** | **1.39x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | off | **349.0 μs** | 316.1 μs | 1.28 ms | 462.1 μs | **4.06x** | **1.46x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | off | **—** | 101.8 μs | 710.3 μs | 145.0 μs | **6.98x** | **1.42x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | off | **1.17 ms** | 1.13 ms | 2.95 ms | 1.34 ms | **2.61x** | **1.18x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | off | **—** | 101.7 μs | 724.5 μs | 147.7 μs | **7.12x** | **1.45x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | off | **2.50 ms** | 2.49 ms | 6.31 ms | 2.88 ms | **2.54x** | **1.16x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | off | **—** | 101.8 μs | 716.7 μs | 146.7 μs | **7.04x** | **1.44x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | off | **4.98 ms** | 4.95 ms | 10.15 ms | 6.85 ms | **2.05x** | **1.38x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | off | **—** | 92.5 μs | 624.2 μs | 131.6 μs | **6.74x** | **1.42x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | off | **191.0 μs** | 165.4 μs | 818.5 μs | 325.4 μs | **4.95x** | **1.97x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | off | **—** | 99.8 μs | 656.5 μs | 135.9 μs | **6.58x** | **1.36x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | off | **1.06 ms** | 1.03 ms | 2.96 ms | 1.61 ms | **2.89x** | **1.57x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | off | **—** | 99.3 μs | 687.4 μs | 140.6 μs | **6.92x** | **1.42x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | off | **1.61 ms** | 1.56 ms | 3.31 ms | 2.39 ms | **2.12x** | **1.53x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | off | **—** | 95.0 μs | 623.9 μs | 134.7 μs | **6.56x** | **1.42x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | off | **273.4 μs** | 247.4 μs | 998.8 μs | 412.7 μs | **4.04x** | **1.67x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | off | **—** | 99.8 μs | 654.7 μs | 136.4 μs | **6.56x** | **1.37x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | off | **1.99 ms** | 1.97 ms | 4.04 ms | 2.54 ms | **2.06x** | **1.29x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | off | **—** | 99.2 μs | 698.6 μs | 137.4 μs | **7.04x** | **1.39x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | off | **3.09 ms** | 3.09 ms | 6.05 ms | 4.08 ms | **1.96x** | **1.32x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | off | **—** | 94.6 μs | 621.6 μs | 136.0 μs | **6.57x** | **1.44x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | off | **153.1 μs** | 128.1 μs | 746.7 μs | 246.5 μs | **5.83x** | **1.92x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | off | **—** | 97.0 μs | 658.9 μs | 137.1 μs | **6.79x** | **1.41x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | off | **586.4 μs** | 550.8 μs | 1.53 ms | 731.8 μs | **2.77x** | **1.33x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | off | **—** | 98.2 μs | 691.9 μs | 139.9 μs | **7.05x** | **1.42x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | off | **872.9 μs** | 843.9 μs | 2.01 ms | 1.05 ms | **2.39x** | **1.25x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | off | **—** | 93.6 μs | 634.3 μs | 137.7 μs | **6.78x** | **1.47x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | off | **186.2 μs** | 160.5 μs | 830.0 μs | 326.0 μs | **5.17x** | **2.03x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | off | **—** | 99.7 μs | 667.8 μs | 133.2 μs | **6.70x** | **1.34x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | off | **1.05 ms** | 1.02 ms | 2.94 ms | 1.58 ms | **2.87x** | **1.54x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | off | **—** | 97.4 μs | 693.4 μs | 136.8 μs | **7.12x** | **1.40x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | off | **1.59 ms** | 1.56 ms | 3.31 ms | 2.37 ms | **2.12x** | **1.52x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | off | **—** | 94.3 μs | 627.1 μs | 131.7 μs | **6.65x** | **1.40x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | off | **287.6 μs** | 256.8 μs | 996.7 μs | 412.3 μs | **3.88x** | **1.61x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | off | **—** | 97.4 μs | 663.0 μs | 136.8 μs | **6.81x** | **1.40x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | off | **1.99 ms** | 1.95 ms | 4.02 ms | 2.58 ms | **2.06x** | **1.32x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | off | **—** | 96.6 μs | 695.6 μs | 139.1 μs | **7.20x** | **1.44x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | off | **3.09 ms** | 3.05 ms | 6.01 ms | 4.07 ms | **1.97x** | **1.33x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | off | **—** | 99.1 μs | 697.0 μs | 142.8 μs | **7.03x** | **1.44x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | off | **126.4 μs** | 100.5 μs | 868.2 μs | 228.2 μs | **8.64x** | **2.27x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | off | **—** | 103.4 μs | 718.6 μs | 144.6 μs | **6.95x** | **1.40x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | off | **367.7 μs** | 334.8 μs | 1.33 ms | 474.8 μs | **3.97x** | **1.42x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | off | **—** | 103.4 μs | 753.8 μs | 149.3 μs | **7.29x** | **1.44x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | off | **503.6 μs** | 477.0 μs | 1.59 ms | 628.6 μs | **3.33x** | **1.32x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | off | **—** | 102.7 μs | 718.1 μs | 146.0 μs | **6.99x** | **1.42x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **668.1 μs** | 634.8 μs | 2.14 ms | 814.8 μs | **3.37x** | **1.28x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | off | **—** | 101.7 μs | 721.2 μs | 147.0 μs | **7.09x** | **1.44x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **1.23 ms** | 1.22 ms | 3.02 ms | 1.75 ms | **2.48x** | **1.43x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | off | **—** | 111.4 μs | 1.02 ms | 158.5 μs | **9.14x** | **1.42x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | off | **353.3 μs** | 325.3 μs | 1.57 ms | 527.3 μs | **4.84x** | **1.62x** |
| fits | mef_small | header_read | 0.45 MB | CPU | off | **—** | 112.3 μs | 1.02 ms | 156.3 μs | **9.06x** | **1.39x** |
| fits | mef_small | read_full | 0.45 MB | CPU | off | **121.9 μs** | 97.4 μs | 1.15 ms | 265.5 μs | **11.76x** | **2.73x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | off | **104.0 μs** | 106.1 μs | 3.39 ms | 367.6 μs | **32.59x** | **3.53x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | off | **—** | 109.7 μs | 1.02 ms | 155.1 μs | **9.30x** | **1.41x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | off | **6.80 ms** | 6.83 ms | 10.07 ms | 9.99 ms | **1.48x** | **1.47x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **120.7 μs** | 93.2 μs | 1.15 ms | 336.5 μs | **12.28x** | **3.61x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | off | **4.65 ms** | 4.45 ms | 77.29 ms | 4.85 ms | **17.35x** | **1.09x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | off | **—** | 104.7 μs | 745.3 μs | 149.3 μs | **7.12x** | **1.43x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | off | **3.60 ms** | 3.58 ms | 7.40 ms | 4.85 ms | **2.07x** | **1.35x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | off | **—** | 102.9 μs | 743.9 μs | 148.7 μs | **7.23x** | **1.45x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | off | **997.2 μs** | 973.1 μs | 2.51 ms | 1.38 ms | **2.58x** | **1.42x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | off | **—** | 104.4 μs | 733.9 μs | 146.4 μs | **7.03x** | **1.40x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | off | **166.2 μs** | 145.2 μs | 970.5 μs | 275.8 μs | **6.68x** | **1.90x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | off | **—** | 95.7 μs | 631.2 μs | 134.6 μs | **6.60x** | **1.41x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | off | **117.4 μs** | 93.7 μs | 680.5 μs | 214.7 μs | **7.26x** | **2.29x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | off | **—** | 95.8 μs | 669.6 μs | 137.7 μs | **6.99x** | **1.44x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | off | **163.7 μs** | 136.8 μs | 777.7 μs | 280.2 μs | **5.68x** | **2.05x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | off | **—** | 100.7 μs | 696.4 μs | 140.8 μs | **6.92x** | **1.40x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | off | **254.7 μs** | 228.9 μs | 969.5 μs | 403.9 μs | **4.24x** | **1.76x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | off | **—** | 95.6 μs | 632.2 μs | 136.3 μs | **6.62x** | **1.43x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | off | **127.3 μs** | 105.2 μs | 702.3 μs | 218.6 μs | **6.67x** | **2.08x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | off | **—** | 100.0 μs | 666.8 μs | 138.0 μs | **6.67x** | **1.38x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | off | **221.3 μs** | 185.5 μs | 888.3 μs | 327.8 μs | **4.79x** | **1.77x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | off | **—** | 100.6 μs | 694.4 μs | 141.4 μs | **6.90x** | **1.41x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | off | **394.5 μs** | 361.7 μs | 1.24 ms | 546.7 μs | **3.42x** | **1.51x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | off | **—** | 92.4 μs | 626.4 μs | 134.5 μs | **6.78x** | **1.46x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | off | **113.2 μs** | 89.5 μs | 669.2 μs | 205.1 μs | **7.47x** | **2.29x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | off | **—** | 96.4 μs | 668.6 μs | 139.5 μs | **6.93x** | **1.45x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | off | **140.3 μs** | 113.5 μs | 745.9 μs | 230.2 μs | **6.57x** | **2.03x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | off | **—** | 98.6 μs | 686.2 μs | 143.6 μs | **6.96x** | **1.46x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | off | **179.5 μs** | 153.7 μs | 829.7 μs | 284.7 μs | **5.40x** | **1.85x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | off | **—** | 92.1 μs | 632.1 μs | 133.8 μs | **6.86x** | **1.45x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | off | **117.7 μs** | 93.3 μs | 687.6 μs | 211.9 μs | **7.37x** | **2.27x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | off | **—** | 95.9 μs | 661.3 μs | 137.0 μs | **6.90x** | **1.43x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | off | **161.7 μs** | 138.0 μs | 788.2 μs | 280.4 μs | **5.71x** | **2.03x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | off | **—** | 100.6 μs | 688.7 μs | 140.6 μs | **6.84x** | **1.40x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | off | **237.0 μs** | 212.0 μs | 966.3 μs | 407.0 μs | **4.56x** | **1.92x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | off | **—** | 94.3 μs | 632.3 μs | 133.2 μs | **6.70x** | **1.41x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | off | **125.3 μs** | 97.4 μs | 690.2 μs | 219.7 μs | **7.09x** | **2.26x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | off | **—** | 95.8 μs | 660.8 μs | 136.9 μs | **6.90x** | **1.43x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | off | **213.1 μs** | 189.0 μs | 882.5 μs | 322.5 μs | **4.67x** | **1.71x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | off | **—** | 96.0 μs | 691.3 μs | 137.2 μs | **7.20x** | **1.43x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | off | **397.6 μs** | 367.5 μs | 1.23 ms | 541.3 μs | **3.35x** | **1.47x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | off | **—** | 98.8 μs | 695.7 μs | 140.9 μs | **7.04x** | **1.43x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | off | **102.5 μs** | 77.3 μs | 818.3 μs | 200.7 μs | **10.59x** | **2.60x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | off | **—** | 103.7 μs | 716.0 μs | 146.1 μs | **6.91x** | **1.41x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | off | **114.4 μs** | 87.8 μs | 867.5 μs | 221.8 μs | **9.88x** | **2.53x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | off | **—** | 102.4 μs | 743.4 μs | 148.1 μs | **7.26x** | **1.45x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | off | **139.4 μs** | 112.7 μs | 922.7 μs | 245.0 μs | **8.19x** | **2.17x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | off | **—** | 100.8 μs | 716.6 μs | 142.8 μs | **7.11x** | **1.42x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | off | **143.6 μs** | 114.7 μs | 849.2 μs | 238.5 μs | **7.40x** | **2.08x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | off | **—** | 98.5 μs | 715.0 μs | 143.3 μs | **7.26x** | **1.46x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | off | **169.2 μs** | 142.5 μs | 883.2 μs | 285.5 μs | **6.20x** | **2.00x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | off | **—** | 96.5 μs | 669.7 μs | 136.3 μs | **6.94x** | **1.41x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **164.6 μs** | 136.1 μs | 768.1 μs | 279.1 μs | **5.64x** | **2.05x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | off | **—** | 98.8 μs | 664.5 μs | 134.5 μs | **6.73x** | **1.36x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **159.8 μs** | 132.9 μs | 783.3 μs | 279.3 μs | **5.89x** | **2.10x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | off | **—** | 99.1 μs | 660.8 μs | 134.3 μs | **6.67x** | **1.35x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **161.1 μs** | 136.8 μs | 776.3 μs | 282.3 μs | **5.67x** | **2.06x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | off | **—** | 99.9 μs | 666.4 μs | 137.7 μs | **6.67x** | **1.38x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **160.7 μs** | 138.0 μs | 777.6 μs | 276.3 μs | **5.63x** | **2.00x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | off | **—** | 98.4 μs | 669.3 μs | 138.0 μs | **6.80x** | **1.40x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **159.6 μs** | 134.2 μs | 776.8 μs | 276.7 μs | **5.79x** | **2.06x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | off | **—** | 94.9 μs | 636.5 μs | 132.6 μs | **6.70x** | **1.40x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **96.8 μs** | 74.4 μs | 657.9 μs | 202.2 μs | **8.84x** | **2.72x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | off | **—** | 97.8 μs | 658.2 μs | 140.1 μs | **6.73x** | **1.43x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **121.6 μs** | 87.7 μs | 675.9 μs | 204.7 μs | **7.71x** | **2.33x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | off | **—** | 99.5 μs | 691.4 μs | 136.6 μs | **6.95x** | **1.37x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **110.1 μs** | 87.7 μs | 708.5 μs | 208.0 μs | **8.08x** | **2.37x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | off | **—** | 91.8 μs | 633.1 μs | 132.3 μs | **6.90x** | **1.44x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **103.8 μs** | 72.3 μs | 647.1 μs | 202.9 μs | **8.95x** | **2.81x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | off | **—** | 97.1 μs | 665.1 μs | 134.0 μs | **6.85x** | **1.38x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **117.9 μs** | 94.4 μs | 703.2 μs | 215.2 μs | **7.45x** | **2.28x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | off | **—** | 98.2 μs | 690.6 μs | 137.4 μs | **7.03x** | **1.40x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **120.6 μs** | 98.6 μs | 720.7 μs | 225.6 μs | **7.31x** | **2.29x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | off | **—** | 94.7 μs | 628.9 μs | 133.9 μs | **6.64x** | **1.41x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **102.5 μs** | 80.1 μs | 644.8 μs | 195.2 μs | **8.05x** | **2.44x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | off | **—** | 97.1 μs | 656.7 μs | 140.8 μs | **6.76x** | **1.45x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **101.8 μs** | 72.1 μs | 666.8 μs | 198.3 μs | **9.24x** | **2.75x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | off | **—** | 97.6 μs | 690.8 μs | 138.1 μs | **7.08x** | **1.42x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **115.1 μs** | 87.0 μs | 692.1 μs | 196.5 μs | **7.95x** | **2.26x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | off | **—** | 92.8 μs | 630.9 μs | 133.2 μs | **6.80x** | **1.44x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **96.5 μs** | 69.8 μs | 652.7 μs | 196.0 μs | **9.36x** | **2.81x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | off | **—** | 97.1 μs | 663.7 μs | 141.7 μs | **6.83x** | **1.46x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **112.9 μs** | 85.0 μs | 674.9 μs | 200.2 μs | **7.94x** | **2.35x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | off | **—** | 97.0 μs | 690.9 μs | 138.3 μs | **7.12x** | **1.43x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **107.8 μs** | 87.1 μs | 693.3 μs | 202.0 μs | **7.96x** | **2.32x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | off | **—** | 94.4 μs | 634.8 μs | 136.0 μs | **6.73x** | **1.44x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **97.8 μs** | 71.6 μs | 645.8 μs | 203.3 μs | **9.02x** | **2.84x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | off | **—** | 95.0 μs | 660.3 μs | 139.0 μs | **6.95x** | **1.46x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **118.4 μs** | 91.9 μs | 678.2 μs | 205.4 μs | **7.38x** | **2.24x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | off | **—** | 101.7 μs | 691.6 μs | 138.7 μs | **6.80x** | **1.36x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **119.5 μs** | 91.9 μs | 696.6 μs | 207.1 μs | **7.58x** | **2.25x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | off | **—** | 100.0 μs | 689.8 μs | 142.6 μs | **6.90x** | **1.43x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **100.9 μs** | 73.3 μs | 804.3 μs | 204.1 μs | **10.97x** | **2.78x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | off | **—** | 99.3 μs | 721.8 μs | 147.5 μs | **7.27x** | **1.49x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **99.4 μs** | 75.6 μs | 821.1 μs | 202.4 μs | **10.86x** | **2.68x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | off | **—** | 103.8 μs | 752.6 μs | 145.9 μs | **7.25x** | **1.41x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **100.9 μs** | 78.1 μs | 851.8 μs | 202.3 μs | **10.90x** | **2.59x** |
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | on | **—** | 160.0 μs | 2.12 ms | 271.4 μs | **13.24x** | **1.70x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **15.90 ms** | 15.90 ms | 40.70 ms | 17.56 ms | **2.56x** | **1.10x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | on | **—** | 159.2 μs | 2.12 ms | 276.0 μs | **13.30x** | **1.73x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **15.40 ms** | 15.44 ms | 67.11 ms | 17.36 ms | **4.36x** | **1.13x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | on | **—** | 164.6 μs | 2.22 ms | 294.8 μs | **13.48x** | **1.79x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **30.30 ms** | 30.38 ms | 38.43 ms | 29.23 ms | **1.27x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | on | **930.5 μs** | 920.3 μs | 9.13 ms | 1.12 ms | **9.92x** | **1.22x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | on | **—** | 164.6 μs | 2.22 ms | 292.7 μs | **13.51x** | **1.78x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **8.82 ms** | 8.76 ms | 27.99 ms | 9.08 ms | **3.20x** | **1.04x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | on | **—** | 93.1 μs | 634.4 μs | 129.1 μs | **6.81x** | **1.39x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | on | **971.1 μs** | 938.5 μs | 1.92 ms | 1.45 ms | **2.04x** | **1.55x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | on | **—** | 95.5 μs | 678.3 μs | 131.4 μs | **7.10x** | **1.38x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | on | **3.90 ms** | 3.79 ms | 13.25 ms | 5.94 ms | **3.50x** | **1.57x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | on | **—** | 94.9 μs | 641.2 μs | 129.6 μs | **6.75x** | **1.37x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | on | **1.87 ms** | 1.84 ms | 3.17 ms | 2.42 ms | **1.72x** | **1.31x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | on | **—** | 97.0 μs | 677.2 μs | 136.1 μs | **6.98x** | **1.40x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | on | **10.01 ms** | 9.94 ms | 17.06 ms | 10.73 ms | **1.72x** | **1.08x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | on | **—** | 96.0 μs | 636.0 μs | 132.8 μs | **6.62x** | **1.38x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | on | **551.7 μs** | 519.7 μs | 1.30 ms | 682.6 μs | **2.50x** | **1.31x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | on | **—** | 94.2 μs | 663.8 μs | 135.6 μs | **7.04x** | **1.44x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | on | **2.00 ms** | 1.97 ms | 5.16 ms | 2.37 ms | **2.61x** | **1.20x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | on | **—** | 95.2 μs | 637.1 μs | 129.3 μs | **6.70x** | **1.36x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | on | **982.8 μs** | 958.8 μs | 1.94 ms | 1.50 ms | **2.02x** | **1.56x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | on | **—** | 96.4 μs | 676.7 μs | 135.0 μs | **7.02x** | **1.40x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | on | **3.91 ms** | 3.84 ms | 13.27 ms | 5.89 ms | **3.46x** | **1.53x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | on | **—** | 92.2 μs | 642.1 μs | 130.5 μs | **6.97x** | **1.42x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | on | **1.85 ms** | 1.81 ms | 3.21 ms | 2.37 ms | **1.77x** | **1.31x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | on | **—** | 95.6 μs | 668.1 μs | 133.4 μs | **6.99x** | **1.39x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | on | **10.14 ms** | 9.97 ms | 17.08 ms | 10.71 ms | **1.71x** | **1.07x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | on | **—** | 96.6 μs | 696.4 μs | 139.6 μs | **7.21x** | **1.44x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | on | **336.4 μs** | 302.6 μs | — | 441.5 μs | **—** | **1.46x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | on | **—** | 102.8 μs | 763.2 μs | 143.1 μs | **7.43x** | **1.39x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | on | **1.14 ms** | 1.11 ms | — | 1.27 ms | **—** | **1.15x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | on | **—** | 97.8 μs | 720.6 μs | 141.8 μs | **7.37x** | **1.45x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | on | **2.36 ms** | 2.33 ms | — | 2.65 ms | **—** | **1.14x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | on | **—** | 102.1 μs | 721.2 μs | 141.2 μs | **7.06x** | **1.38x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | on | **4.79 ms** | 4.72 ms | — | 6.63 ms | **—** | **1.40x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | on | **—** | 94.1 μs | 636.0 μs | 125.0 μs | **6.76x** | **1.33x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | on | **179.9 μs** | 155.9 μs | 788.2 μs | 324.9 μs | **5.06x** | **2.08x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | on | **—** | 97.4 μs | 663.9 μs | 132.7 μs | **6.82x** | **1.36x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | on | **1.01 ms** | 982.4 μs | 2.91 ms | 1.53 ms | **2.96x** | **1.56x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | on | **—** | 97.4 μs | 687.0 μs | 133.8 μs | **7.06x** | **1.37x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | on | **1.52 ms** | 1.47 ms | 2.75 ms | 2.30 ms | **1.87x** | **1.56x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | on | **—** | 94.0 μs | 627.9 μs | 128.5 μs | **6.68x** | **1.37x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | on | **271.0 μs** | 242.6 μs | 911.4 μs | 398.4 μs | **3.76x** | **1.64x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | on | **—** | 92.5 μs | 658.3 μs | 131.0 μs | **7.12x** | **1.42x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | on | **1.92 ms** | 1.91 ms | 4.90 ms | 2.51 ms | **2.57x** | **1.31x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | on | **—** | 97.7 μs | 689.5 μs | 134.4 μs | **7.05x** | **1.38x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | on | **2.97 ms** | 2.96 ms | 4.81 ms | 3.88 ms | **1.63x** | **1.31x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | on | **—** | 90.9 μs | 628.7 μs | 127.4 μs | **6.92x** | **1.40x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | on | **145.9 μs** | 116.5 μs | 716.8 μs | 242.6 μs | **6.15x** | **2.08x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | on | **—** | 90.5 μs | 653.1 μs | 131.7 μs | **7.22x** | **1.46x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | on | **572.2 μs** | 525.7 μs | 1.97 ms | 695.0 μs | **3.75x** | **1.32x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | on | **—** | 96.3 μs | 686.1 μs | 134.4 μs | **7.13x** | **1.40x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | on | **833.2 μs** | 803.7 μs | 1.73 ms | 1.00 ms | **2.15x** | **1.25x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | on | **—** | 90.4 μs | 635.3 μs | 164.0 μs | **7.03x** | **1.81x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | on | **179.9 μs** | 153.1 μs | 792.1 μs | 308.6 μs | **5.17x** | **2.02x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | on | **—** | 109.6 μs | 768.2 μs | 144.4 μs | **7.01x** | **1.32x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | on | **991.9 μs** | 964.6 μs | 3.08 ms | 1.52 ms | **3.20x** | **1.58x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | on | **—** | 109.0 μs | 803.1 μs | 148.8 μs | **7.37x** | **1.37x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | on | **1.52 ms** | 1.48 ms | 2.72 ms | 2.31 ms | **1.84x** | **1.56x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | on | **—** | 103.0 μs | 624.6 μs | 129.7 μs | **6.06x** | **1.26x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | on | **270.3 μs** | 249.6 μs | 915.8 μs | 399.4 μs | **3.67x** | **1.60x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | on | **—** | 94.5 μs | 654.8 μs | 130.5 μs | **6.93x** | **1.38x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | on | **1.91 ms** | 1.87 ms | 4.93 ms | 2.47 ms | **2.63x** | **1.32x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | on | **—** | 96.2 μs | 682.7 μs | 137.6 μs | **7.10x** | **1.43x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | on | **2.95 ms** | 2.91 ms | 4.84 ms | 3.89 ms | **1.66x** | **1.33x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | on | **—** | 99.5 μs | 684.2 μs | 139.8 μs | **6.88x** | **1.41x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | on | **120.1 μs** | 91.2 μs | — | 230.7 μs | **—** | **2.53x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | on | **—** | 100.8 μs | 705.5 μs | 143.0 μs | **7.00x** | **1.42x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | on | **355.2 μs** | 317.6 μs | — | 453.9 μs | **—** | **1.43x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | on | **—** | 101.1 μs | 739.3 μs | 145.0 μs | **7.31x** | **1.43x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | on | **485.4 μs** | 457.5 μs | — | 619.9 μs | **—** | **1.36x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | on | **—** | 100.8 μs | 708.7 μs | 143.1 μs | **7.03x** | **1.42x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **640.4 μs** | 596.5 μs | — | 772.5 μs | **—** | **1.30x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | on | **—** | 100.4 μs | 709.4 μs | 146.5 μs | **7.06x** | **1.46x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **1.17 ms** | 1.13 ms | — | 1.67 ms | **—** | **1.47x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | on | **—** | 107.1 μs | 1.00 ms | 151.4 μs | **9.35x** | **1.41x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | on | **337.1 μs** | 313.4 μs | — | 500.2 μs | **—** | **1.60x** |
| fits | mef_small | header_read | 0.45 MB | CPU | on | **—** | 109.9 μs | 1.00 ms | 153.0 μs | **9.11x** | **1.39x** |
| fits | mef_small | read_full | 0.45 MB | CPU | on | **112.7 μs** | 86.7 μs | — | 261.6 μs | **—** | **3.02x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | on | **96.0 μs** | 97.0 μs | — | 364.3 μs | **—** | **3.79x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | on | **—** | 108.1 μs | 995.4 μs | 154.4 μs | **9.21x** | **1.43x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | on | **6.73 ms** | 6.70 ms | — | 9.64 ms | **—** | **1.44x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **114.7 μs** | 86.7 μs | — | 335.1 μs | **—** | **3.86x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | on | **4.32 ms** | 4.14 ms | 39.95 ms | 4.54 ms | **9.66x** | **1.10x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | on | **—** | 104.3 μs | 725.5 μs | 142.4 μs | **6.96x** | **1.37x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | on | **3.51 ms** | 3.48 ms | — | 4.76 ms | **—** | **1.37x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | on | **—** | 102.7 μs | 723.2 μs | 141.6 μs | **7.04x** | **1.38x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | on | **967.1 μs** | 944.1 μs | — | 1.34 ms | **—** | **1.42x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | on | **—** | 102.2 μs | 726.9 μs | 142.2 μs | **7.11x** | **1.39x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | on | **161.6 μs** | 142.0 μs | — | 273.0 μs | **—** | **1.92x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | on | **—** | 92.5 μs | 622.4 μs | 127.1 μs | **6.73x** | **1.37x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | on | **149.0 μs** | 115.5 μs | 668.6 μs | 205.2 μs | **5.79x** | **1.78x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | on | **—** | 99.0 μs | 646.9 μs | 132.5 μs | **6.53x** | **1.34x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | on | **192.2 μs** | 164.9 μs | 760.7 μs | 275.0 μs | **4.61x** | **1.67x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | on | **—** | 98.9 μs | 686.9 μs | 136.8 μs | **6.95x** | **1.38x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | on | **235.6 μs** | 206.1 μs | 922.3 μs | 407.8 μs | **4.47x** | **1.98x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | on | **—** | 93.6 μs | 622.8 μs | 129.1 μs | **6.66x** | **1.38x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | on | **152.8 μs** | 125.9 μs | 683.3 μs | 218.8 μs | **5.43x** | **1.74x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | on | **—** | 95.3 μs | 649.5 μs | 132.1 μs | **6.81x** | **1.39x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | on | **237.5 μs** | 213.8 μs | 844.1 μs | 324.3 μs | **3.95x** | **1.52x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | on | **—** | 98.7 μs | 692.8 μs | 140.0 μs | **7.02x** | **1.42x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | on | **383.7 μs** | 352.6 μs | 1.13 ms | 526.1 μs | **3.21x** | **1.49x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | on | **—** | 90.0 μs | 623.2 μs | 133.4 μs | **6.93x** | **1.48x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | on | **136.1 μs** | 112.7 μs | 658.5 μs | 204.8 μs | **5.84x** | **1.82x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | on | **—** | 96.0 μs | 649.1 μs | 137.8 μs | **6.76x** | **1.44x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | on | **187.3 μs** | 162.5 μs | 720.5 μs | 231.6 μs | **4.43x** | **1.43x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | on | **—** | 97.9 μs | 672.3 μs | 135.8 μs | **6.86x** | **1.39x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | on | **172.1 μs** | 149.5 μs | 806.2 μs | 269.1 μs | **5.39x** | **1.80x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | on | **—** | 94.2 μs | 621.3 μs | 130.5 μs | **6.59x** | **1.39x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | on | **141.7 μs** | 117.9 μs | 661.0 μs | 215.1 μs | **5.60x** | **1.82x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | on | **—** | 96.6 μs | 645.6 μs | 131.5 μs | **6.68x** | **1.36x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | on | **188.7 μs** | 164.0 μs | 754.6 μs | 276.2 μs | **4.60x** | **1.68x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | on | **—** | 97.7 μs | 676.2 μs | 137.9 μs | **6.92x** | **1.41x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | on | **230.4 μs** | 205.4 μs | 908.9 μs | 391.7 μs | **4.43x** | **1.91x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | on | **—** | 91.2 μs | 622.2 μs | 127.6 μs | **6.82x** | **1.40x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | on | **152.8 μs** | 124.4 μs | 684.1 μs | 215.4 μs | **5.50x** | **1.73x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | on | **—** | 94.7 μs | 647.5 μs | 129.1 μs | **6.84x** | **1.36x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | on | **238.5 μs** | 205.3 μs | 846.8 μs | 324.3 μs | **4.12x** | **1.58x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | on | **—** | 94.1 μs | 676.4 μs | 133.3 μs | **7.19x** | **1.42x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | on | **389.2 μs** | 355.8 μs | 1.12 ms | 550.0 μs | **3.15x** | **1.55x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | on | **—** | 94.6 μs | 669.8 μs | 137.8 μs | **7.08x** | **1.46x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | on | **102.4 μs** | 71.6 μs | — | 199.4 μs | **—** | **2.79x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | on | **—** | 97.0 μs | 700.8 μs | 138.7 μs | **7.22x** | **1.43x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | on | **115.3 μs** | 83.2 μs | — | 216.4 μs | **—** | **2.60x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | on | **—** | 97.8 μs | 730.8 μs | 143.6 μs | **7.47x** | **1.47x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | on | **134.4 μs** | 106.4 μs | — | 239.5 μs | **—** | **2.25x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | on | **—** | 97.5 μs | 698.5 μs | 142.1 μs | **7.17x** | **1.46x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | on | **197.1 μs** | 167.9 μs | — | 237.3 μs | **—** | **1.41x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | on | **—** | 97.6 μs | 694.3 μs | 136.2 μs | **7.11x** | **1.40x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | on | **192.7 μs** | 161.6 μs | — | 288.7 μs | **—** | **1.79x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | on | **—** | 92.9 μs | 643.9 μs | 130.6 μs | **6.93x** | **1.41x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **190.7 μs** | 161.8 μs | 755.4 μs | 280.3 μs | **4.67x** | **1.73x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | on | **—** | 92.9 μs | 639.0 μs | 131.0 μs | **6.88x** | **1.41x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **189.9 μs** | 166.6 μs | 761.0 μs | 274.3 μs | **4.57x** | **1.65x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | on | **—** | 92.8 μs | 640.1 μs | 129.5 μs | **6.90x** | **1.40x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **191.7 μs** | 160.1 μs | 761.2 μs | 271.4 μs | **4.75x** | **1.70x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | on | **—** | 91.9 μs | 643.1 μs | 131.7 μs | **7.00x** | **1.43x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **199.7 μs** | 165.8 μs | 762.6 μs | 274.1 μs | **4.60x** | **1.65x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | on | **—** | 91.0 μs | 648.6 μs | 133.2 μs | **7.13x** | **1.46x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **192.0 μs** | 166.5 μs | 759.6 μs | 274.0 μs | **4.56x** | **1.65x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | on | **—** | 91.7 μs | 620.3 μs | 130.3 μs | **6.76x** | **1.42x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **130.8 μs** | 103.1 μs | 639.5 μs | 203.2 μs | **6.20x** | **1.97x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | on | **—** | 96.1 μs | 647.7 μs | 133.7 μs | **6.74x** | **1.39x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **135.3 μs** | 104.9 μs | 673.5 μs | 199.9 μs | **6.42x** | **1.91x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | on | **—** | 94.1 μs | 682.2 μs | 133.7 μs | **7.25x** | **1.42x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **133.0 μs** | 107.6 μs | 699.4 μs | 201.9 μs | **6.50x** | **1.88x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | on | **—** | 94.2 μs | 618.2 μs | 127.6 μs | **6.56x** | **1.35x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **136.7 μs** | 102.5 μs | 642.6 μs | 198.6 μs | **6.27x** | **1.94x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | on | **—** | 95.6 μs | 653.7 μs | 136.0 μs | **6.84x** | **1.42x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **141.6 μs** | 110.6 μs | 689.2 μs | 201.0 μs | **6.23x** | **1.82x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | on | **—** | 100.2 μs | 676.5 μs | 140.7 μs | **6.75x** | **1.40x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **143.6 μs** | 114.5 μs | 706.6 μs | 208.2 μs | **6.17x** | **1.82x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | on | **—** | 93.4 μs | 615.4 μs | 129.8 μs | **6.59x** | **1.39x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **133.4 μs** | 107.1 μs | 651.7 μs | 193.7 μs | **6.09x** | **1.81x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | on | **—** | 98.1 μs | 648.3 μs | 137.9 μs | **6.61x** | **1.41x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **131.8 μs** | 108.6 μs | 672.4 μs | 196.2 μs | **6.19x** | **1.81x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | on | **—** | 97.1 μs | 682.3 μs | 136.3 μs | **7.03x** | **1.40x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **136.8 μs** | 107.2 μs | 689.8 μs | 200.8 μs | **6.43x** | **1.87x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | on | **—** | 94.6 μs | 616.6 μs | 127.6 μs | **6.52x** | **1.35x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **131.9 μs** | 105.1 μs | 648.5 μs | 195.0 μs | **6.17x** | **1.86x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | on | **—** | 96.6 μs | 647.1 μs | 133.6 μs | **6.70x** | **1.38x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **137.0 μs** | 107.7 μs | 671.6 μs | 198.2 μs | **6.24x** | **1.84x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | on | **—** | 97.5 μs | 675.1 μs | 136.4 μs | **6.92x** | **1.40x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **136.5 μs** | 115.1 μs | 693.0 μs | 205.6 μs | **6.02x** | **1.79x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | on | **—** | 94.8 μs | 618.9 μs | 129.5 μs | **6.53x** | **1.37x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **131.7 μs** | 100.5 μs | 644.4 μs | 194.7 μs | **6.41x** | **1.94x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | on | **—** | 93.4 μs | 654.7 μs | 134.9 μs | **7.01x** | **1.44x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **145.9 μs** | 115.3 μs | 675.6 μs | 200.5 μs | **5.86x** | **1.74x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | on | **—** | 98.8 μs | 681.1 μs | 137.5 μs | **6.90x** | **1.39x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **140.3 μs** | 113.6 μs | 705.2 μs | 212.0 μs | **6.21x** | **1.87x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | on | **—** | 96.9 μs | 685.7 μs | 139.6 μs | **7.07x** | **1.44x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **95.9 μs** | 66.5 μs | — | 201.8 μs | **—** | **3.03x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | on | **—** | 98.7 μs | 709.3 μs | 146.1 μs | **7.19x** | **1.48x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **94.7 μs** | 70.6 μs | — | 199.2 μs | **—** | **2.82x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | on | **—** | 103.5 μs | 739.3 μs | 150.9 μs | **7.14x** | **1.46x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **94.9 μs** | 72.3 μs | — | 202.4 μs | **—** | **2.80x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CUDA | off | **19.88 ms** | 19.82 ms | 49.15 ms | 21.99 ms | **2.48x** | **1.11x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CUDA | off | **19.31 ms** | 19.44 ms | 80.32 ms | 21.64 ms | **4.16x** | **1.12x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CUDA | off | **30.38 ms** | 30.43 ms | 37.85 ms | 29.24 ms | **1.25x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CUDA | off | **840.6 μs** | 831.3 μs | 10.03 ms | 966.7 μs | **12.07x** | **1.16x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CUDA | off | **8.90 ms** | 8.89 ms | 27.48 ms | 9.24 ms | **3.09x** | **1.04x** |
| fits | large_float32_1d | read_full | 3.82 MB | CUDA | off | **735.9 μs** | 726.0 μs | 1.53 ms | 1.28 ms | **2.10x** | **1.76x** |
| fits | large_float32_2d | read_full | 16.00 MB | CUDA | off | **4.08 ms** | 3.69 ms | 15.93 ms | 5.93 ms | **4.32x** | **1.61x** |
| fits | large_float64_1d | read_full | 7.63 MB | CUDA | off | **1.39 ms** | 1.35 ms | 3.03 ms | 2.05 ms | **2.24x** | **1.52x** |
| fits | large_float64_2d | read_full | 32.00 MB | CUDA | off | **12.34 ms** | 12.30 ms | 25.87 ms | 13.53 ms | **2.10x** | **1.10x** |
| fits | large_int16_1d | read_full | 1.91 MB | CUDA | off | **447.0 μs** | 435.9 μs | 1.02 ms | 581.9 μs | **2.33x** | **1.33x** |
| fits | large_int16_2d | read_full | 8.00 MB | CUDA | off | **1.51 ms** | 1.51 ms | 5.34 ms | 2.03 ms | **3.54x** | **1.35x** |
| fits | large_int32_1d | read_full | 3.82 MB | CUDA | off | **731.7 μs** | 726.0 μs | 1.50 ms | 1.28 ms | **2.07x** | **1.77x** |
| fits | large_int32_2d | read_full | 16.00 MB | CUDA | off | **3.49 ms** | 3.43 ms | 15.78 ms | 5.74 ms | **4.60x** | **1.67x** |
| fits | large_int64_1d | read_full | 7.63 MB | CUDA | off | **1.37 ms** | 1.44 ms | 3.28 ms | 2.05 ms | **2.39x** | **1.49x** |
| fits | large_int64_2d | read_full | 32.00 MB | CUDA | off | **11.99 ms** | 12.08 ms | 25.92 ms | 13.46 ms | **2.16x** | **1.12x** |
| fits | large_int8_1d | read_full | 0.96 MB | CUDA | off | **228.9 μs** | 276.4 μs | 925.8 μs | 389.7 μs | **4.04x** | **1.70x** |
| fits | large_int8_2d | read_full | 4.00 MB | CUDA | off | **631.1 μs** | 884.5 μs | 2.57 ms | 1.11 ms | **4.07x** | **1.75x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CUDA | off | **1.52 ms** | 1.77 ms | 5.51 ms | 2.17 ms | **3.61x** | **1.42x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CUDA | off | **3.62 ms** | 3.77 ms | 11.44 ms | 5.90 ms | **3.16x** | **1.63x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CUDA | off | **111.9 μs** | 123.5 μs | 564.4 μs | 242.7 μs | **5.04x** | **2.17x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CUDA | off | **759.7 μs** | 768.6 μs | 1.70 ms | 1.34 ms | **2.24x** | **1.77x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CUDA | off | **1.14 ms** | 1.13 ms | 2.50 ms | 2.03 ms | **2.21x** | **1.80x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CUDA | off | **213.6 μs** | 219.0 μs | 671.8 μs | 318.3 μs | **3.14x** | **1.49x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CUDA | off | **1.44 ms** | 1.45 ms | 4.00 ms | 2.16 ms | **2.77x** | **1.49x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CUDA | off | **2.54 ms** | 3.00 ms | 8.34 ms | 3.58 ms | **3.28x** | **1.41x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CUDA | off | **95.0 μs** | 71.7 μs | 493.9 μs | 160.9 μs | **6.89x** | **2.25x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CUDA | off | **453.4 μs** | 452.8 μs | 1.15 ms | 599.8 μs | **2.54x** | **1.32x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CUDA | off | **649.7 μs** | 644.5 μs | 1.35 ms | 842.1 μs | **2.10x** | **1.31x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CUDA | off | **124.3 μs** | 116.4 μs | 537.5 μs | 235.3 μs | **4.62x** | **2.02x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CUDA | off | **772.9 μs** | 768.4 μs | 1.70 ms | 1.35 ms | **2.22x** | **1.75x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CUDA | off | **1.16 ms** | 1.19 ms | 2.46 ms | 1.99 ms | **2.13x** | **1.72x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CUDA | off | **213.2 μs** | 204.5 μs | 680.4 μs | 310.8 μs | **3.33x** | **1.52x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CUDA | off | **1.44 ms** | 1.44 ms | 4.62 ms | 2.16 ms | **3.21x** | **1.50x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CUDA | off | **2.40 ms** | 2.46 ms | 8.36 ms | 3.96 ms | **3.48x** | **1.65x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CUDA | off | **69.3 μs** | 64.9 μs | 603.2 μs | 142.0 μs | **9.30x** | **2.19x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CUDA | off | **233.4 μs** | 293.1 μs | 937.2 μs | 400.9 μs | **4.02x** | **1.72x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CUDA | off | **302.6 μs** | 390.3 μs | 1.11 ms | 528.6 μs | **3.68x** | **1.75x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CUDA | off | **458.4 μs** | 518.3 μs | 1.66 ms | 660.0 μs | **3.62x** | **1.44x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CUDA | off | **751.6 μs** | 893.7 μs | 2.16 ms | 1.45 ms | **2.88x** | **1.92x** |
| fits | mef_medium | read_full | 7.02 MB | CUDA | off | **243.6 μs** | 295.7 μs | 1.16 ms | 438.0 μs | **4.78x** | **1.80x** |
| fits | mef_small | read_full | 0.45 MB | CUDA | off | **61.8 μs** | 66.5 μs | 830.7 μs | 180.8 μs | **13.44x** | **2.92x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CUDA | off | **41.3 μs** | 42.2 μs | 3.19 ms | 220.6 μs | **77.20x** | **5.34x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CUDA | off | **68.9 μs** | 64.7 μs | 843.4 μs | 244.3 μs | **13.04x** | **3.78x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | CUDA | off | **6.02 ms** | 5.57 ms | 83.95 ms | 6.07 ms | **15.08x** | **1.09x** |
| fits | scaled_large | read_full | 8.00 MB | CUDA | off | **2.57 ms** | 4.46 ms | 10.10 ms | 5.69 ms | **3.94x** | **2.22x** |
| fits | scaled_medium | read_full | 2.01 MB | CUDA | off | **454.2 μs** | 1.05 ms | 1.80 ms | 1.27 ms | **3.96x** | **2.81x** |
| fits | scaled_small | read_full | 0.13 MB | CUDA | off | **79.6 μs** | 126.3 μs | 679.1 μs | 199.8 μs | **8.53x** | **2.51x** |
| fits | small_float32_1d | read_full | 42.2 KB | CUDA | off | **49.1 μs** | 51.8 μs | 433.9 μs | 127.0 μs | **8.83x** | **2.58x** |
| fits | small_float32_2d | read_full | 0.26 MB | CUDA | off | **82.0 μs** | 104.5 μs | 527.3 μs | 195.2 μs | **6.43x** | **2.38x** |
| fits | small_float32_3d | read_full | 0.63 MB | CUDA | off | **172.4 μs** | 173.1 μs | 661.9 μs | 316.9 μs | **3.84x** | **1.84x** |
| fits | small_float64_1d | read_full | 0.08 MB | CUDA | off | **60.4 μs** | 56.3 μs | 455.7 μs | 130.3 μs | **8.10x** | **2.31x** |
| fits | small_float64_2d | read_full | 0.51 MB | CUDA | off | **159.8 μs** | 157.7 μs | 595.4 μs | 246.2 μs | **3.78x** | **1.56x** |
| fits | small_float64_3d | read_full | 1.26 MB | CUDA | off | **303.5 μs** | 298.6 μs | 851.0 μs | 456.8 μs | **2.85x** | **1.53x** |
| fits | small_int16_1d | read_full | 22.5 KB | CUDA | off | **41.1 μs** | 41.4 μs | 422.2 μs | 111.6 μs | **10.28x** | **2.72x** |
| fits | small_int16_2d | read_full | 0.13 MB | CUDA | off | **60.8 μs** | 74.3 μs | 487.8 μs | 148.8 μs | **8.03x** | **2.45x** |
| fits | small_int16_3d | read_full | 0.32 MB | CUDA | off | **124.0 μs** | 104.6 μs | 556.2 μs | 191.0 μs | **5.32x** | **1.83x** |
| fits | small_int32_1d | read_full | 42.2 KB | CUDA | off | **55.4 μs** | 50.2 μs | 431.2 μs | 127.0 μs | **8.59x** | **2.53x** |
| fits | small_int32_2d | read_full | 0.26 MB | CUDA | off | **107.5 μs** | 82.8 μs | 518.0 μs | 191.3 μs | **6.25x** | **2.31x** |
| fits | small_int32_3d | read_full | 0.63 MB | CUDA | off | **184.7 μs** | 168.7 μs | 655.8 μs | 322.5 μs | **3.89x** | **1.91x** |
| fits | small_int64_1d | read_full | 0.08 MB | CUDA | off | **60.9 μs** | 62.2 μs | 447.0 μs | 133.5 μs | **7.34x** | **2.19x** |
| fits | small_int64_2d | read_full | 0.51 MB | CUDA | off | **148.9 μs** | 135.7 μs | 594.1 μs | 247.7 μs | **4.38x** | **1.83x** |
| fits | small_int64_3d | read_full | 1.26 MB | CUDA | off | **305.0 μs** | 302.9 μs | 836.1 μs | 446.9 μs | **2.76x** | **1.48x** |
| fits | small_int8_1d | read_full | 14.1 KB | CUDA | off | **50.7 μs** | 41.1 μs | 558.6 μs | 113.3 μs | **13.58x** | **2.75x** |
| fits | small_int8_2d | read_full | 0.07 MB | CUDA | off | **50.4 μs** | 49.0 μs | 591.0 μs | 133.5 μs | **12.07x** | **2.72x** |
| fits | small_int8_3d | read_full | 0.16 MB | CUDA | off | **64.9 μs** | 83.5 μs | 647.3 μs | 155.5 μs | **9.98x** | **2.40x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CUDA | off | **75.1 μs** | 78.5 μs | 578.0 μs | 149.1 μs | **7.70x** | **1.99x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CUDA | off | **99.3 μs** | 98.9 μs | 606.6 μs | 205.5 μs | **6.13x** | **2.08x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CUDA | off | **79.5 μs** | 89.8 μs | 514.8 μs | 198.0 μs | **6.47x** | **2.49x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CUDA | off | **104.0 μs** | 90.4 μs | 520.9 μs | 191.3 μs | **5.76x** | **2.12x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CUDA | off | **92.7 μs** | 82.3 μs | 517.0 μs | 194.5 μs | **6.28x** | **2.36x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CUDA | off | **89.5 μs** | 92.1 μs | 520.3 μs | 200.4 μs | **5.81x** | **2.24x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CUDA | off | **83.6 μs** | 79.0 μs | 521.3 μs | 197.5 μs | **6.60x** | **2.50x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CUDA | off | **32.4 μs** | 42.1 μs | 420.9 μs | 112.8 μs | **12.98x** | **3.48x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CUDA | off | **51.2 μs** | 45.0 μs | 450.1 μs | 115.1 μs | **10.01x** | **2.56x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CUDA | off | **50.3 μs** | 43.4 μs | 463.7 μs | 122.6 μs | **10.70x** | **2.83x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CUDA | off | **43.9 μs** | 41.1 μs | 427.5 μs | 115.6 μs | **10.41x** | **2.81x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CUDA | off | **55.2 μs** | 42.3 μs | 460.1 μs | 118.1 μs | **10.88x** | **2.79x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CUDA | off | **54.8 μs** | 55.3 μs | 478.3 μs | 122.4 μs | **8.72x** | **2.23x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CUDA | off | **41.9 μs** | 38.6 μs | 420.3 μs | 108.7 μs | **10.89x** | **2.81x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CUDA | off | **35.5 μs** | 36.1 μs | 436.7 μs | 107.9 μs | **12.31x** | **3.04x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CUDA | off | **44.2 μs** | 48.3 μs | 460.3 μs | 115.9 μs | **10.42x** | **2.62x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CUDA | off | **34.3 μs** | 43.5 μs | 418.9 μs | 108.8 μs | **12.22x** | **3.17x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CUDA | off | **52.2 μs** | 47.0 μs | 443.9 μs | 115.4 μs | **9.44x** | **2.45x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CUDA | off | **43.6 μs** | 42.4 μs | 461.2 μs | 119.6 μs | **10.89x** | **2.82x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CUDA | off | **33.9 μs** | 43.0 μs | 420.6 μs | 112.5 μs | **12.42x** | **3.32x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CUDA | off | **52.5 μs** | 46.3 μs | 459.2 μs | 126.6 μs | **9.91x** | **2.73x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CUDA | off | **59.4 μs** | 52.0 μs | 476.3 μs | 122.9 μs | **9.16x** | **2.36x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CUDA | off | **40.1 μs** | 42.8 μs | 555.5 μs | 110.3 μs | **13.85x** | **2.75x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CUDA | off | **40.2 μs** | 37.4 μs | 578.8 μs | 109.1 μs | **15.47x** | **2.92x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CUDA | off | **45.5 μs** | 43.6 μs | 593.0 μs | 113.6 μs | **13.59x** | **2.60x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CUDA | on | **16.11 ms** | 16.11 ms | 40.39 ms | 17.81 ms | **2.51x** | **1.11x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CUDA | on | **15.59 ms** | 15.59 ms | 65.46 ms | 17.54 ms | **4.20x** | **1.12x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CUDA | on | **30.52 ms** | 30.49 ms | 38.07 ms | 29.38 ms | **1.25x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CUDA | on | **836.1 μs** | 830.2 μs | 8.81 ms | 977.1 μs | **10.61x** | **1.18x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CUDA | on | **8.99 ms** | 9.02 ms | 27.92 ms | 9.35 ms | **3.10x** | **1.04x** |
| fits | large_float32_1d | read_full | 3.82 MB | CUDA | on | **832.1 μs** | 844.3 μs | 1.49 ms | 1.36 ms | **1.80x** | **1.63x** |
| fits | large_float32_2d | read_full | 16.00 MB | CUDA | on | **3.82 ms** | 4.13 ms | 12.62 ms | 6.08 ms | **3.31x** | **1.59x** |
| fits | large_float64_1d | read_full | 7.63 MB | CUDA | on | **1.57 ms** | 1.57 ms | 2.83 ms | 2.10 ms | **1.81x** | **1.34x** |
| fits | large_float64_2d | read_full | 32.00 MB | CUDA | on | **12.50 ms** | 12.44 ms | 19.81 ms | 13.65 ms | **1.59x** | **1.10x** |
| fits | large_int16_1d | read_full | 1.91 MB | CUDA | on | **512.6 μs** | 504.0 μs | 1.02 ms | 654.7 μs | **2.02x** | **1.30x** |
| fits | large_int16_2d | read_full | 8.00 MB | CUDA | on | **1.77 ms** | 1.78 ms | 5.04 ms | 2.16 ms | **2.85x** | **1.22x** |
| fits | large_int32_1d | read_full | 3.82 MB | CUDA | on | **854.3 μs** | 849.0 μs | 1.50 ms | 1.37 ms | **1.77x** | **1.61x** |
| fits | large_int32_2d | read_full | 16.00 MB | CUDA | on | **3.99 ms** | 6.17 ms | 12.78 ms | 6.15 ms | **3.21x** | **1.54x** |
| fits | large_int64_1d | read_full | 7.63 MB | CUDA | on | **1.58 ms** | 1.56 ms | 2.77 ms | 2.11 ms | **1.77x** | **1.35x** |
| fits | large_int64_2d | read_full | 32.00 MB | CUDA | on | **12.54 ms** | 12.48 ms | 19.84 ms | 13.71 ms | **1.59x** | **1.10x** |
| fits | large_int8_1d | read_full | 0.96 MB | CUDA | on | **267.8 μs** | 293.9 μs | 1.36 ms | 411.7 μs | **5.09x** | **1.54x** |
| fits | large_int8_2d | read_full | 4.00 MB | CUDA | on | **739.3 μs** | 989.8 μs | 3.15 ms | 1.20 ms | **4.26x** | **1.63x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CUDA | on | **1.74 ms** | 2.06 ms | 6.73 ms | 2.50 ms | **3.87x** | **1.44x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CUDA | on | **3.91 ms** | 4.44 ms | 12.78 ms | 6.53 ms | **3.27x** | **1.67x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CUDA | on | **136.4 μs** | 123.5 μs | 579.3 μs | 274.7 μs | **4.69x** | **2.22x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CUDA | on | **881.5 μs** | 897.6 μs | 2.49 ms | 1.46 ms | **2.83x** | **1.66x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CUDA | on | **1.31 ms** | 1.32 ms | 2.33 ms | 2.18 ms | **1.78x** | **1.66x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CUDA | on | **236.1 μs** | 234.6 μs | 723.2 μs | 346.5 μs | **3.08x** | **1.48x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CUDA | on | **1.67 ms** | 1.65 ms | 4.87 ms | 2.30 ms | **2.95x** | **1.39x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CUDA | on | **2.88 ms** | 2.72 ms | 5.10 ms | 3.92 ms | **1.87x** | **1.44x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CUDA | on | **95.9 μs** | 84.5 μs | 514.0 μs | 170.4 μs | **6.09x** | **2.02x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CUDA | on | **529.2 μs** | 517.1 μs | 1.78 ms | 667.2 μs | **3.43x** | **1.29x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CUDA | on | **748.6 μs** | 742.9 μs | 1.36 ms | 951.0 μs | **1.83x** | **1.28x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CUDA | on | **145.5 μs** | 130.3 μs | 566.9 μs | 252.0 μs | **4.35x** | **1.93x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CUDA | on | **894.0 μs** | 888.9 μs | 2.87 ms | 1.45 ms | **3.23x** | **1.63x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CUDA | on | **1.33 ms** | 1.32 ms | 2.47 ms | 2.18 ms | **1.86x** | **1.64x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CUDA | on | **221.4 μs** | 218.2 μs | 708.2 μs | 341.4 μs | **3.25x** | **1.56x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CUDA | on | **1.71 ms** | 1.69 ms | 4.84 ms | 2.32 ms | **2.87x** | **1.37x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CUDA | on | **2.80 ms** | 2.75 ms | 5.10 ms | 3.81 ms | **1.85x** | **1.39x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CUDA | on | **72.0 μs** | 68.3 μs | 963.9 μs | 152.4 μs | **14.12x** | **2.23x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CUDA | on | **275.2 μs** | 312.5 μs | 1.68 ms | 427.9 μs | **6.12x** | **1.55x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CUDA | on | **349.1 μs** | 436.5 μs | 1.62 ms | 572.1 μs | **4.65x** | **1.64x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CUDA | on | **523.3 μs** | 585.2 μs | 2.21 ms | 748.6 μs | **4.22x** | **1.43x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CUDA | on | **869.9 μs** | 1.03 ms | 2.85 ms | 1.65 ms | **3.27x** | **1.90x** |
| fits | mef_medium | read_full | 7.02 MB | CUDA | on | **262.8 μs** | 307.0 μs | 2.16 ms | 459.4 μs | **8.21x** | **1.75x** |
| fits | mef_small | read_full | 0.45 MB | CUDA | on | **70.0 μs** | 70.2 μs | 1.45 ms | 190.4 μs | **20.65x** | **2.72x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CUDA | on | **41.0 μs** | 42.6 μs | 2.45 ms | 220.4 μs | **59.64x** | **5.37x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CUDA | on | **65.0 μs** | 73.3 μs | 1.47 ms | 254.1 μs | **22.54x** | **3.91x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | CUDA | on | **6.21 ms** | 5.75 ms | 43.07 ms | 6.29 ms | **7.50x** | **1.09x** |
| fits | scaled_large | read_full | 8.00 MB | CUDA | on | **2.11 ms** | 4.62 ms | 11.52 ms | 5.17 ms | **5.45x** | **2.45x** |
| fits | scaled_medium | read_full | 2.01 MB | CUDA | on | **584.6 μs** | 1.23 ms | 2.66 ms | 1.34 ms | **4.55x** | **2.29x** |
| fits | scaled_small | read_full | 0.13 MB | CUDA | on | **94.8 μs** | 120.4 μs | 1.10 ms | 207.0 μs | **11.62x** | **2.18x** |
| fits | small_float32_1d | read_full | 42.2 KB | CUDA | on | **60.3 μs** | 73.5 μs | 460.3 μs | 132.0 μs | **7.63x** | **2.19x** |
| fits | small_float32_2d | read_full | 0.26 MB | CUDA | on | **110.3 μs** | 140.7 μs | 558.1 μs | 206.8 μs | **5.06x** | **1.87x** |
| fits | small_float32_3d | read_full | 0.63 MB | CUDA | on | **192.7 μs** | 187.9 μs | 708.7 μs | 334.3 μs | **3.77x** | **1.78x** |
| fits | small_float64_1d | read_full | 0.08 MB | CUDA | on | **59.0 μs** | 81.3 μs | 477.9 μs | 142.0 μs | **8.10x** | **2.41x** |
| fits | small_float64_2d | read_full | 0.51 MB | CUDA | on | **171.5 μs** | 192.8 μs | 628.0 μs | 268.0 μs | **3.66x** | **1.56x** |
| fits | small_float64_3d | read_full | 1.26 MB | CUDA | on | **343.5 μs** | 338.4 μs | 893.1 μs | 491.0 μs | **2.64x** | **1.45x** |
| fits | small_int16_1d | read_full | 22.5 KB | CUDA | on | **49.4 μs** | 55.6 μs | 450.4 μs | 120.7 μs | **9.12x** | **2.44x** |
| fits | small_int16_2d | read_full | 0.13 MB | CUDA | on | **83.9 μs** | 125.9 μs | 512.8 μs | 152.2 μs | **6.11x** | **1.82x** |
| fits | small_int16_3d | read_full | 0.32 MB | CUDA | on | **126.7 μs** | 104.8 μs | 594.8 μs | 205.9 μs | **5.68x** | **1.97x** |
| fits | small_int32_1d | read_full | 42.2 KB | CUDA | on | **62.3 μs** | 72.0 μs | 473.9 μs | 133.5 μs | **7.60x** | **2.14x** |
| fits | small_int32_2d | read_full | 0.26 MB | CUDA | on | **103.8 μs** | 130.4 μs | 553.8 μs | 207.5 μs | **5.34x** | **2.00x** |
| fits | small_int32_3d | read_full | 0.63 MB | CUDA | on | **204.7 μs** | 177.6 μs | 684.1 μs | 335.0 μs | **3.85x** | **1.89x** |
| fits | small_int64_1d | read_full | 0.08 MB | CUDA | on | **63.0 μs** | 85.4 μs | 467.7 μs | 135.0 μs | **7.42x** | **2.14x** |
| fits | small_int64_2d | read_full | 0.51 MB | CUDA | on | **164.7 μs** | 196.3 μs | 647.3 μs | 289.6 μs | **3.93x** | **1.76x** |
| fits | small_int64_3d | read_full | 1.26 MB | CUDA | on | **343.4 μs** | 345.3 μs | 869.0 μs | 490.0 μs | **2.53x** | **1.43x** |
| fits | small_int8_1d | read_full | 14.1 KB | CUDA | on | **50.5 μs** | 50.1 μs | 969.7 μs | 119.3 μs | **19.37x** | **2.38x** |
| fits | small_int8_2d | read_full | 0.07 MB | CUDA | on | **79.1 μs** | 56.0 μs | 1.00 ms | 145.0 μs | **17.91x** | **2.59x** |
| fits | small_int8_3d | read_full | 0.16 MB | CUDA | on | **70.7 μs** | 82.6 μs | 1.09 ms | 172.0 μs | **15.47x** | **2.43x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CUDA | on | **69.5 μs** | 128.4 μs | 1.03 ms | 165.4 μs | **14.76x** | **2.38x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CUDA | on | **108.4 μs** | 147.1 μs | 1.06 ms | 219.0 μs | **9.74x** | **2.02x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CUDA | on | **93.9 μs** | 137.7 μs | 560.0 μs | 201.9 μs | **5.97x** | **2.15x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CUDA | on | **97.0 μs** | 136.4 μs | 562.4 μs | 205.7 μs | **5.80x** | **2.12x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CUDA | on | **100.7 μs** | 133.5 μs | 564.9 μs | 218.2 μs | **5.61x** | **2.17x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CUDA | on | **108.2 μs** | 139.2 μs | 561.0 μs | 206.2 μs | **5.19x** | **1.91x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CUDA | on | **111.5 μs** | 145.1 μs | 562.7 μs | 205.9 μs | **5.05x** | **1.85x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CUDA | on | **42.7 μs** | 58.8 μs | 453.3 μs | 119.7 μs | **10.61x** | **2.80x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CUDA | on | **54.4 μs** | 62.5 μs | 466.8 μs | 120.6 μs | **8.58x** | **2.22x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CUDA | on | **55.6 μs** | 64.4 μs | 497.9 μs | 124.9 μs | **8.96x** | **2.25x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CUDA | on | **40.8 μs** | 60.1 μs | 449.8 μs | 115.9 μs | **11.03x** | **2.84x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CUDA | on | **53.5 μs** | 72.1 μs | 488.1 μs | 126.1 μs | **9.12x** | **2.36x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CUDA | on | **58.1 μs** | 75.1 μs | 502.5 μs | 130.0 μs | **8.64x** | **2.24x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CUDA | on | **37.0 μs** | 52.3 μs | 454.1 μs | 120.0 μs | **12.26x** | **3.24x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CUDA | on | **54.6 μs** | 72.4 μs | 617.1 μs | 150.1 μs | **11.31x** | **2.75x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CUDA | on | **52.7 μs** | 63.2 μs | 512.8 μs | 126.3 μs | **9.74x** | **2.40x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CUDA | on | **47.7 μs** | 46.0 μs | 444.8 μs | 113.7 μs | **9.68x** | **2.47x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CUDA | on | **45.2 μs** | 51.6 μs | 474.9 μs | 119.5 μs | **10.51x** | **2.64x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CUDA | on | **49.9 μs** | 68.7 μs | 483.0 μs | 122.8 μs | **9.68x** | **2.46x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CUDA | on | **49.4 μs** | 55.1 μs | 436.4 μs | 118.4 μs | **8.83x** | **2.40x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CUDA | on | **56.1 μs** | 69.1 μs | 474.3 μs | 136.2 μs | **8.45x** | **2.43x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CUDA | on | **59.3 μs** | 76.8 μs | 503.9 μs | 132.1 μs | **8.50x** | **2.23x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CUDA | on | **53.4 μs** | 44.9 μs | 942.1 μs | 116.9 μs | **20.97x** | **2.60x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CUDA | on | **53.9 μs** | 47.7 μs | 994.3 μs | 117.4 μs | **20.86x** | **2.46x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CUDA | on | **55.5 μs** | 36.2 μs | 1.00 ms | 120.5 μs | **27.73x** | **3.33x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **919.0 μs** | 355.1 μs | 4.93 ms | 7.59 ms | **13.88x** | **21.38x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | off | **95.7 μs** | 102.7 μs | 11.44 ms | 7.59 ms | **119.63x** | **79.32x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | off | **97.6 μs** | 100.4 μs | 2.28 ms | 7.49 ms | **23.41x** | **76.79x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | off | **98.5 μs** | 101.8 μs | 2.72 ms | 3.71 ms | **27.60x** | **37.63x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | off | **132.3 μs** | 114.2 μs | 3.97 ms | 3.15 ms | **34.73x** | **27.57x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **526.4 μs** | 301.1 μs | 3.11 ms | 1.25 ms | **10.34x** | **4.14x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | off | **104.0 μs** | 108.6 μs | 3.36 ms | 1.30 ms | **32.29x** | **12.50x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | off | **103.3 μs** | 108.0 μs | 2.13 ms | 1.23 ms | **20.60x** | **11.92x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | off | **104.7 μs** | 104.6 μs | 2.62 ms | 817.5 μs | **25.03x** | **7.82x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | off | **138.6 μs** | 115.2 μs | 2.41 ms | 672.9 μs | **20.96x** | **5.84x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **25.66 ms** | 8.58 ms | 110.71 ms | 400.48 ms | **12.91x** | **46.70x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | off | **104.1 μs** | 104.4 μs | 21.00 ms | 60.27 ms | **201.76x** | **579.05x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | off | **95.2 μs** | 100.8 μs | 59.73 ms | 641.77 ms | **627.50x** | **6742.51x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **100.7 μs** | 102.2 μs | 22.64 ms | 131.49 ms | **224.85x** | **1305.84x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **155.9 μs** | 119.4 μs | 21.15 ms | 126.85 ms | **177.13x** | **1062.47x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **3.21 ms** | 1.11 ms | 12.02 ms | 36.76 ms | **10.81x** | **33.05x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | off | **103.1 μs** | 106.8 μs | 4.45 ms | 6.00 ms | **43.18x** | **58.17x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | off | **102.3 μs** | 105.7 μs | 6.65 ms | 59.22 ms | **65.02x** | **578.75x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | off | **100.2 μs** | 103.4 μs | 5.81 ms | 16.88 ms | **57.94x** | **168.43x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | off | **146.1 μs** | 112.7 μs | 4.46 ms | 12.61 ms | **39.55x** | **111.93x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **800.9 μs** | 348.1 μs | 4.81 ms | 4.07 ms | **13.82x** | **11.70x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | off | **101.3 μs** | 103.7 μs | 3.10 ms | 1.04 ms | **30.58x** | **10.26x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | off | **96.8 μs** | 102.0 μs | 3.21 ms | 6.14 ms | **33.16x** | **63.37x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | off | **99.5 μs** | 106.9 μs | 4.18 ms | 2.18 ms | **41.98x** | **21.94x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | off | **148.2 μs** | 113.2 μs | 3.07 ms | 1.60 ms | **27.14x** | **14.15x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **500.7 μs** | 272.4 μs | 4.02 ms | 933.1 μs | **14.75x** | **3.43x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | off | **97.9 μs** | 102.1 μs | 2.84 ms | 528.9 μs | **29.04x** | **5.40x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | off | **95.8 μs** | 97.1 μs | 2.90 ms | 1.12 ms | **30.27x** | **11.66x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | off | **97.6 μs** | 101.8 μs | 3.91 ms | 725.0 μs | **40.04x** | **7.43x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | off | **146.5 μs** | 112.4 μs | 2.82 ms | 564.2 μs | **25.05x** | **5.02x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **15.44 ms** | 8.11 ms | 37.17 ms | 13.55 ms | **4.58x** | **1.67x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | off | **99.5 μs** | 103.7 μs | 6.63 ms | 40.77 ms | **66.66x** | **409.68x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | off | **98.2 μs** | 102.5 μs | 10.71 ms | 9.15 ms | **109.10x** | **93.21x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **98.7 μs** | 102.8 μs | 7.35 ms | 5.58 ms | **74.47x** | **56.53x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **132.4 μs** | 117.7 μs | 6.61 ms | 5.46 ms | **56.14x** | **46.42x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **2.06 ms** | 1.11 ms | 6.43 ms | 1.71 ms | **5.77x** | **1.54x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | off | **100.3 μs** | 104.4 μs | 2.75 ms | 4.53 ms | **27.42x** | **45.17x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | off | **100.4 μs** | 103.7 μs | 3.12 ms | 1.29 ms | **31.12x** | **12.87x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | off | **100.1 μs** | 106.4 μs | 3.41 ms | 974.1 μs | **34.04x** | **9.74x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | off | **132.5 μs** | 111.1 μs | 2.70 ms | 871.3 μs | **24.33x** | **7.84x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **616.1 μs** | 366.3 μs | 3.15 ms | 586.9 μs | **8.60x** | **1.60x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | off | **100.1 μs** | 102.7 μs | 2.22 ms | 866.3 μs | **22.17x** | **8.66x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | off | **95.4 μs** | 100.7 μs | 2.23 ms | 505.9 μs | **23.34x** | **5.30x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | off | **98.7 μs** | 102.8 μs | 2.81 ms | 494.6 μs | **28.46x** | **5.01x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | off | **132.7 μs** | 115.9 μs | 2.18 ms | 434.2 μs | **18.84x** | **3.75x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **488.5 μs** | 287.3 μs | 2.79 ms | 455.6 μs | **9.72x** | **1.59x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | off | **100.5 μs** | 106.9 μs | 2.13 ms | 490.6 μs | **21.19x** | **4.88x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | off | **92.9 μs** | 101.8 μs | 2.15 ms | 446.5 μs | **23.19x** | **4.81x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | off | **98.0 μs** | 104.1 μs | 2.70 ms | 443.6 μs | **27.53x** | **4.53x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | off | **132.7 μs** | 112.0 μs | 2.10 ms | 388.1 μs | **18.75x** | **3.47x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **2.27 ms** | 982.8 μs | 6.22 ms | 63.26 ms | **6.33x** | **64.36x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | off | **107.0 μs** | 112.6 μs | 39.53 ms | 60.35 ms | **369.52x** | **564.08x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | off | **103.4 μs** | 111.4 μs | 3.80 ms | 62.61 ms | **36.78x** | **605.46x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | off | **104.7 μs** | 109.0 μs | 3.56 ms | 22.27 ms | **33.98x** | **212.80x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | off | **135.1 μs** | 119.5 μs | 2.87 ms | 17.76 ms | **24.01x** | **148.65x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **722.4 μs** | 366.9 μs | 3.31 ms | 6.81 ms | **9.03x** | **18.56x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | off | **105.0 μs** | 110.8 μs | 5.92 ms | 6.51 ms | **56.40x** | **61.95x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | off | **108.0 μs** | 108.3 μs | 2.42 ms | 6.68 ms | **22.45x** | **61.82x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | off | **104.8 μs** | 108.3 μs | 3.00 ms | 2.84 ms | **28.59x** | **27.05x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | off | **137.2 μs** | 119.7 μs | 2.36 ms | 2.27 ms | **19.75x** | **18.94x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **1.99 ms** | 979.9 μs | 6.11 ms | 228.20 ms | **6.23x** | **232.89x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | off | **107.3 μs** | 105.2 μs | 771.51 ms | 229.52 ms | **7331.29x** | **2181.03x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | off | **103.1 μs** | 104.7 μs | 3.58 ms | 215.47 ms | **34.71x** | **2090.58x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | off | **106.0 μs** | 109.7 μs | 3.31 ms | 223.18 ms | **31.20x** | **2104.57x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | off | **135.6 μs** | 121.6 μs | 2.69 ms | 226.17 ms | **22.13x** | **1859.95x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **612.9 μs** | 361.8 μs | 3.04 ms | 20.94 ms | **8.40x** | **57.88x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | off | **98.9 μs** | 102.5 μs | 78.44 ms | 20.96 ms | **793.13x** | **211.95x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | off | **94.6 μs** | 102.2 μs | 2.25 ms | 20.58 ms | **23.82x** | **217.67x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | off | **100.0 μs** | 104.6 μs | 2.67 ms | 20.81 ms | **26.73x** | **208.09x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | off | **131.3 μs** | 115.1 μs | 2.13 ms | 21.34 ms | **18.48x** | **185.51x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **499.6 μs** | 296.5 μs | 2.64 ms | 2.38 ms | **8.92x** | **8.04x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | off | **100.4 μs** | 104.1 μs | 9.75 ms | 2.45 ms | **97.07x** | **24.40x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | off | **93.5 μs** | 103.1 μs | 2.09 ms | 2.35 ms | **22.35x** | **25.12x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | off | **100.0 μs** | 101.2 μs | 2.55 ms | 2.34 ms | **25.51x** | **23.45x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | off | **134.4 μs** | 110.3 μs | 2.03 ms | 2.38 ms | **18.40x** | **21.53x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **7.20 ms** | 1.11 ms | 59.50 ms | 150.29 ms | **53.65x** | **135.52x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | off | **102.3 μs** | 101.5 μs | 13.20 ms | 9.86 ms | **130.07x** | **97.15x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | off | **99.1 μs** | 101.1 μs | 35.55 ms | 240.45 ms | **358.70x** | **2426.45x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | off | **99.2 μs** | 102.5 μs | 19.26 ms | 70.07 ms | **194.21x** | **706.43x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | off | **256.2 μs** | 112.2 μs | 13.04 ms | 51.18 ms | **116.22x** | **456.18x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **1.41 ms** | 365.7 μs | 16.62 ms | 15.10 ms | **45.46x** | **41.29x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | off | **100.3 μs** | 103.2 μs | 9.16 ms | 1.75 ms | **91.32x** | **17.45x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | off | **97.2 μs** | 103.4 μs | 10.62 ms | 23.42 ms | **109.17x** | **240.90x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | off | **97.8 μs** | 103.9 μs | 14.06 ms | 7.78 ms | **143.77x** | **79.55x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | off | **250.1 μs** | 114.1 μs | 9.08 ms | 5.77 ms | **79.60x** | **50.60x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **672.6 μs** | 292.1 μs | 13.62 ms | 2.42 ms | **46.63x** | **8.30x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | off | **100.5 μs** | 104.5 μs | 8.53 ms | 824.2 μs | **84.89x** | **8.20x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | off | **99.4 μs** | 103.9 μs | 8.71 ms | 3.23 ms | **87.57x** | **32.47x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | off | **99.4 μs** | 105.9 μs | 13.30 ms | 1.68 ms | **133.72x** | **16.94x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | off | **249.5 μs** | 111.1 μs | 8.55 ms | 1.27 ms | **76.93x** | **11.44x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **724.3 μs** | 365.6 μs | 4.93 ms | 7.67 ms | **13.48x** | **20.97x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | on | **102.0 μs** | 103.1 μs | 11.59 ms | 7.70 ms | **113.61x** | **75.43x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | on | **99.5 μs** | 103.1 μs | 2.21 ms | 7.59 ms | **22.17x** | **76.28x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | on | **101.7 μs** | 103.5 μs | 2.60 ms | 3.72 ms | **25.59x** | **36.57x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | on | **133.5 μs** | 112.5 μs | 3.94 ms | 3.16 ms | **35.03x** | **28.11x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **599.3 μs** | 304.9 μs | 3.20 ms | 1.26 ms | **10.48x** | **4.12x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | on | **103.2 μs** | 102.7 μs | 3.38 ms | 1.30 ms | **32.91x** | **12.62x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | on | **110.2 μs** | 113.3 μs | 2.15 ms | 1.22 ms | **19.54x** | **11.11x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | on | **108.3 μs** | 108.7 μs | 2.63 ms | 817.4 μs | **24.31x** | **7.55x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | on | **141.9 μs** | 122.9 μs | 2.45 ms | 668.0 μs | **19.93x** | **5.44x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **15.52 ms** | 8.53 ms | 99.78 ms | 404.33 ms | **11.70x** | **47.42x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | on | **104.0 μs** | 106.0 μs | 9.39 ms | 59.68 ms | **90.31x** | **573.91x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | on | **99.4 μs** | 105.2 μs | 50.12 ms | 638.07 ms | **504.29x** | **6420.58x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **103.8 μs** | 104.6 μs | 11.66 ms | 130.62 ms | **112.24x** | **1257.82x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **150.9 μs** | 112.2 μs | 10.27 ms | 127.86 ms | **91.53x** | **1139.86x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **2.28 ms** | 1.13 ms | 11.47 ms | 36.43 ms | **10.18x** | **32.34x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | on | **98.6 μs** | 104.1 μs | 3.69 ms | 6.04 ms | **37.40x** | **61.20x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | on | **97.1 μs** | 101.1 μs | 6.06 ms | 59.88 ms | **62.45x** | **616.73x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | on | **100.0 μs** | 102.3 μs | 5.16 ms | 16.89 ms | **51.62x** | **168.86x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | on | **146.4 μs** | 109.6 μs | 3.81 ms | 12.45 ms | **34.79x** | **113.61x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **722.0 μs** | 346.8 μs | 4.74 ms | 4.04 ms | **13.65x** | **11.65x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | on | **98.9 μs** | 101.4 μs | 2.96 ms | 1.00 ms | **29.96x** | **10.17x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | on | **96.7 μs** | 100.1 μs | 3.10 ms | 6.15 ms | **32.04x** | **63.66x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | on | **99.7 μs** | 104.1 μs | 4.08 ms | 2.12 ms | **40.94x** | **21.30x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | on | **148.7 μs** | 112.0 μs | 2.95 ms | 1.55 ms | **26.36x** | **13.88x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **594.5 μs** | 272.8 μs | 4.10 ms | 917.4 μs | **15.03x** | **3.36x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | on | **98.4 μs** | 103.4 μs | 2.87 ms | 536.6 μs | **29.19x** | **5.45x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | on | **97.4 μs** | 102.3 μs | 2.93 ms | 1.14 ms | **30.12x** | **11.69x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | on | **100.9 μs** | 103.3 μs | 3.94 ms | 726.5 μs | **39.08x** | **7.20x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | on | **146.6 μs** | 111.9 μs | 2.86 ms | 553.2 μs | **25.53x** | **4.94x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **10.46 ms** | 8.09 ms | 36.13 ms | 13.53 ms | **4.47x** | **1.67x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | on | **101.7 μs** | 105.8 μs | 4.88 ms | 40.70 ms | **47.96x** | **400.38x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | on | **100.2 μs** | 105.2 μs | 9.72 ms | 9.10 ms | **97.03x** | **90.81x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **102.6 μs** | 105.1 μs | 5.91 ms | 5.48 ms | **57.61x** | **53.45x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **133.6 μs** | 117.8 μs | 5.18 ms | 5.41 ms | **43.95x** | **45.92x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.61 ms** | 1.10 ms | 6.19 ms | 1.69 ms | **5.60x** | **1.53x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | on | **100.0 μs** | 104.6 μs | 2.47 ms | 4.62 ms | **24.74x** | **46.19x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | on | **96.5 μs** | 101.6 μs | 2.90 ms | 1.24 ms | **30.09x** | **12.81x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | on | **100.7 μs** | 103.7 μs | 3.19 ms | 945.3 μs | **31.72x** | **9.39x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | on | **130.7 μs** | 109.4 μs | 2.48 ms | 849.0 μs | **22.70x** | **7.76x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **668.5 μs** | 369.8 μs | 3.14 ms | 575.8 μs | **8.48x** | **1.56x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | on | **101.7 μs** | 107.9 μs | 2.18 ms | 872.8 μs | **21.46x** | **8.58x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | on | **101.8 μs** | 105.5 μs | 2.23 ms | 513.9 μs | **21.87x** | **5.05x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | on | **101.5 μs** | 111.1 μs | 2.84 ms | 485.7 μs | **27.99x** | **4.79x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | on | **133.9 μs** | 108.4 μs | 2.14 ms | 421.9 μs | **19.78x** | **3.89x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **583.6 μs** | 302.6 μs | 2.83 ms | 472.5 μs | **9.35x** | **1.56x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | on | **99.4 μs** | 103.1 μs | 2.16 ms | 500.8 μs | **21.73x** | **5.04x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | on | **150.7 μs** | 172.8 μs | 3.06 ms | 457.7 μs | **20.34x** | **3.04x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | on | **99.8 μs** | 104.2 μs | 2.73 ms | 448.7 μs | **27.40x** | **4.50x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | on | **138.6 μs** | 113.6 μs | 2.13 ms | 393.7 μs | **18.78x** | **3.47x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **1.73 ms** | 993.5 μs | 6.02 ms | 64.85 ms | **6.06x** | **65.28x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | on | **106.0 μs** | 108.0 μs | 39.09 ms | 61.03 ms | **368.70x** | **575.59x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | on | **104.5 μs** | 107.4 μs | 3.49 ms | 63.49 ms | **33.37x** | **607.32x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | on | **103.6 μs** | 107.5 μs | 3.04 ms | 22.79 ms | **29.37x** | **220.00x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | on | **142.8 μs** | 125.4 μs | 2.31 ms | 18.07 ms | **18.45x** | **144.06x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **708.9 μs** | 370.9 μs | 3.31 ms | 6.89 ms | **8.93x** | **18.58x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | on | **105.4 μs** | 109.6 μs | 5.90 ms | 6.58 ms | **55.98x** | **62.43x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | on | **102.7 μs** | 107.6 μs | 2.39 ms | 6.75 ms | **23.25x** | **65.71x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | on | **104.0 μs** | 109.5 μs | 2.92 ms | 2.85 ms | **28.06x** | **27.43x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | on | **140.9 μs** | 118.8 μs | 2.28 ms | 2.28 ms | **19.19x** | **19.20x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **1.54 ms** | 960.6 μs | 5.70 ms | 225.79 ms | **5.94x** | **235.05x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | on | **106.8 μs** | 108.4 μs | 757.20 ms | 224.37 ms | **7089.51x** | **2100.75x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | on | **108.8 μs** | 109.1 μs | 3.12 ms | 212.64 ms | **28.62x** | **1953.53x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | on | **103.8 μs** | 110.9 μs | 2.73 ms | 225.11 ms | **26.29x** | **2167.82x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | on | **137.9 μs** | 118.7 μs | 2.06 ms | 229.76 ms | **17.34x** | **1935.73x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **671.4 μs** | 365.1 μs | 2.97 ms | 20.78 ms | **8.12x** | **56.91x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | on | **103.6 μs** | 105.2 μs | 76.93 ms | 20.72 ms | **742.54x** | **200.02x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | on | **98.2 μs** | 103.8 μs | 2.17 ms | 20.21 ms | **22.06x** | **205.74x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | on | **102.7 μs** | 107.7 μs | 2.59 ms | 20.36 ms | **25.26x** | **198.31x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | on | **132.2 μs** | 114.9 μs | 2.04 ms | 20.99 ms | **17.71x** | **182.67x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **571.6 μs** | 297.5 μs | 2.66 ms | 2.37 ms | **8.93x** | **7.95x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | on | **101.3 μs** | 102.6 μs | 9.69 ms | 2.45 ms | **95.61x** | **24.16x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | on | **96.4 μs** | 103.4 μs | 2.10 ms | 2.33 ms | **21.74x** | **24.17x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | on | **100.2 μs** | 103.1 μs | 2.57 ms | 2.33 ms | **25.64x** | **23.30x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | on | **133.1 μs** | 112.7 μs | 2.01 ms | 2.36 ms | **17.86x** | **20.93x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **3.89 ms** | 1.10 ms | 48.33 ms | 151.91 ms | **43.93x** | **138.09x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | on | **99.7 μs** | 99.6 μs | 10.28 ms | 10.00 ms | **103.26x** | **100.35x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | on | **97.7 μs** | 102.6 μs | 26.69 ms | 242.05 ms | **273.12x** | **2477.40x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | on | **97.7 μs** | 102.5 μs | 16.61 ms | 69.85 ms | **170.06x** | **715.23x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | on | **254.0 μs** | 115.9 μs | 10.79 ms | 51.57 ms | **93.11x** | **445.15x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **1.33 ms** | 364.2 μs | 16.32 ms | 15.37 ms | **44.80x** | **42.21x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | on | **99.8 μs** | 104.2 μs | 8.72 ms | 1.68 ms | **87.37x** | **16.87x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | on | **97.6 μs** | 100.0 μs | 9.90 ms | 23.43 ms | **101.43x** | **240.00x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | on | **98.5 μs** | 104.7 μs | 13.65 ms | 7.66 ms | **138.52x** | **77.79x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | on | **254.9 μs** | 116.2 μs | 8.75 ms | 5.70 ms | **75.26x** | **49.00x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **1.11 ms** | 298.1 μs | 13.65 ms | 2.46 ms | **45.79x** | **8.26x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | on | **103.6 μs** | 105.5 μs | 8.47 ms | 814.3 μs | **81.77x** | **7.86x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | on | **97.2 μs** | 100.8 μs | 8.67 ms | 3.23 ms | **89.15x** | **33.26x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | on | **101.5 μs** | 104.0 μs | 13.31 ms | 1.70 ms | **131.06x** | **16.70x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | on | **258.3 μs** | 114.3 μs | 8.51 ms | 1.26 ms | **74.43x** | **10.98x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
_No deficits in this run — torchfits won every comparable case._

Same-mmap ranking, a 25% lag noise floor, and excluding fitsio from table
mmap-on peers (fitsio has no mmap mode) keep the scorecard honest. Focused
quiet runs closed the previous published cluster gaps (hcompress `MINDIRECT`,
signed-byte device casts, narrow WHERE tensor filters); remaining sub-25%
matrix jitter is scheduling/thermal noise, not a product claim against.

**Method notes:**

- Compressed codecs (including hcompress) share CFITSIO decode; rice/gzip win.
- Smart `where=` uses a tensor mask + Arrow of survivors; use `backend="cpp"`
  for fused mmap pushdown when you want that path explicitly.
- Multicore: ATen/OpenMP threads help large scans; CFITSIO single-file
  decompress is not the multi-core lever. CANFAR exhaustive seats use 8 CPUs.
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_cuda_0.9.1_20260714_202004` | fits + fitstable (lab) | 3648 | 0 | lab bench-all + `--mmap-matrix` + CUDA/MPS |
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
