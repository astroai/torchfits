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

Published GPU/CPU numbers come from the multi-host release scorecard
(`exhaustive_mps_*`, `exhaustive_cpu_*`, `exhaustive_cuda_*`).
GitHub Actions weekly benches use CPU-only PyTorch and do not refresh GPU cells.

## Comparison targets

| Domain | torchfits module | Compared against |
|---|---|---|
| FITS image I/O | `torchfits.read` / `torchfits.write` | `astropy.io.fits`, `fitsio` |
| FITS table I/O | `torchfits.table` | `astropy.io.fits`, `fitsio` |

## Methodology

Each case measures median wall-clock time over multiple repetitions, plus
**peak process RSS** (and peak CUDA alloc when on CUDA) sampled around the timed
call for comparative memory reporting. Deficit ranking stays **time-based**;
RSS is reported alongside times, not as a separate deficit gate.

Cases are grouped into two families:

- **smart** — the idiomatic high-level API, such as `torchfits.read()` vs
  `astropy.io.fits.getdata()` plus `torch.from_numpy()`.
- **specialized** — lower-level paths with explicit mmap, compression, or table
  streaming controls.

Fairness controls:

- Rows with mismatched mmap behavior are marked `SKIPPED` and excluded from
  rankings.
- Fitsio has no mmap toggle; under `mmap_target=on` it is marked
  non-comparable for both image and table domains.
- FITS comparators must be official released distributions.
- Warm-cache and cold-cache profiles are kept separate.

Deficit floors (same-mmap peers):

- **Images / cubes / spectra / cutouts** (`domain=fits`): any lag above float-timer
  ε counts — including rice/hcompress (no percent floor).
- **Arrow table interchange** (`domain=fitstable`): allow up to **1.05×**.

### Modular suites and release exhaustives

Named suites live in `benchmarks/suites.py` and resolve to `bench_all.py` flags
(`--scope` / `--filter` / `--operation` / GPU / mmap / profile):

```bash
pixi run bench-suite hcompress
pixi run bench-suite compressed_rice -- --no-mmap
pixi run bench-suite fitstable_predicate
pixi run bench-deficit-focus          # registry-driven deficit clusters
```

Release composition is the `release` suite (full fits + fitstable, mmap matrix,
GPU when present). Host recipes:

| Task | Host | Run ID prefix |
|---|---|---|
| `pixi run bench-exhaustive-local` | Mac CPU + MPS | `exhaustive_mps_*` |
| `pixi run bench-exhaustive-canfar-cpu` | CANFAR multicore CPU | `exhaustive_cpu_*` |
| `pixi run bench-exhaustive-canfar-cuda` | CANFAR CUDA | `exhaustive_cuda_*` |
| `pixi run bench-release-scorecard -- <run_dir>...` | meta | patches multi-host docs |

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
Source: `benchmarks_results/exhaustive_mps_20260715_002839/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.07 ms` (n=269) | `0.75 ms` (n=269) | `0.22 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.19 ms` (n=269) | `6.75 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.62 ms` (n=234) | `1.93 ms` (n=79) | `0.99 ms` (n=79) | — |
| `disk→RAM→GPU` | `0.36 ms` (n=234) | `2.55 ms` (n=79) | `0.54 ms` (n=79) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.08 ms` (n=180) | `3.00 ms` (n=162) | `3.52 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.12 ms` (n=180) | `11.32 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **2.86 ms** | 2.85 ms | 6.45 ms | 3.20 ms | **2.26x** | **1.12x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.81 ms** | 3.66 ms | 8.26 ms | 4.20 ms | **2.26x** | **1.15x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **8.62 ms** | 8.30 ms | 24.16 ms | 8.65 ms | **2.91x** | **1.04x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **10.34 ms** | 10.34 ms | 27.93 ms | 10.58 ms | **2.70x** | **1.02x** |
| Repeated Cutouts (50x 100x100) | CPU | **6.83 ms** | 6.38 ms | 105.42 ms | 7.27 ms | **16.52x** | **1.14x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **79.1 μs** | 85.3 μs | 5.93 ms | 104.92 ms | **74.99x** | **1325.97x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **61.8 μs** | 66.0 μs | 2.61 ms | 204.60 ms | **42.28x** | **3313.35x** |
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
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | off | **—** | 102.5 μs | 1.44 ms | 241.2 μs | **14.01x** | **2.35x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **19.65 ms** | 20.70 ms | 48.73 ms | 24.11 ms | **2.48x** | **1.23x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | off | **—** | 80.8 μs | 1.37 ms | 227.3 μs | **17.01x** | **2.81x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **20.56 ms** | 20.66 ms | 75.16 ms | 25.74 ms | **3.66x** | **1.25x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | off | **—** | 85.7 μs | 1.43 ms | 242.7 μs | **16.68x** | **2.83x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **33.50 ms** | 34.24 ms | 41.55 ms | 33.04 ms | **1.24x** | **0.99x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | off | **765.8 μs** | 860.9 μs | 8.38 ms | 1.12 ms | **10.94x** | **1.46x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | off | **—** | 95.9 μs | 1.47 ms | 255.7 μs | **15.38x** | **2.67x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **8.62 ms** | 8.30 ms | 24.16 ms | 8.65 ms | **2.91x** | **1.04x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | off | **—** | 48.1 μs | 465.2 μs | 112.9 μs | **9.67x** | **2.35x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | off | **790.3 μs** | 751.9 μs | 2.01 ms | 885.5 μs | **2.68x** | **1.18x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | off | **—** | 52.5 μs | 493.7 μs | 126.7 μs | **9.40x** | **2.41x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | off | **2.86 ms** | 2.85 ms | 6.45 ms | 3.20 ms | **2.26x** | **1.12x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | off | **—** | 66.1 μs | 700.3 μs | 201.8 μs | **10.59x** | **3.05x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | off | **1.77 ms** | 1.69 ms | 3.48 ms | 1.99 ms | **2.06x** | **1.18x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | off | **—** | 73.9 μs | 624.9 μs | 146.1 μs | **8.46x** | **1.98x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | off | **7.24 ms** | 7.60 ms | 13.12 ms | 7.35 ms | **1.81x** | **1.02x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | off | **—** | 60.5 μs | 567.5 μs | 132.3 μs | **9.38x** | **2.19x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | off | **375.0 μs** | 363.8 μs | 1.24 ms | 576.1 μs | **3.41x** | **1.58x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | off | **—** | 58.8 μs | 552.5 μs | 133.1 μs | **9.40x** | **2.26x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | off | **1.62 ms** | 1.63 ms | 4.61 ms | 1.91 ms | **2.86x** | **1.18x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | off | **—** | 58.0 μs | 520.3 μs | 135.5 μs | **8.98x** | **2.34x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | off | **805.6 μs** | 735.0 μs | 2.22 ms | 1.06 ms | **3.02x** | **1.45x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | off | **—** | 54.4 μs | 534.0 μs | 133.6 μs | **9.81x** | **2.45x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | off | **3.31 ms** | 3.20 ms | 7.54 ms | 3.64 ms | **2.36x** | **1.14x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | off | **—** | 52.0 μs | 499.5 μs | 132.6 μs | **9.61x** | **2.55x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | off | **1.94 ms** | 2.03 ms | 3.95 ms | 2.29 ms | **2.03x** | **1.18x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | off | **—** | 56.3 μs | 518.3 μs | 124.9 μs | **9.21x** | **2.22x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | off | **8.23 ms** | 7.71 ms | 13.25 ms | 8.14 ms | **1.72x** | **1.06x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | off | **—** | 58.1 μs | 585.3 μs | 135.0 μs | **10.07x** | **2.32x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | off | **208.8 μs** | 201.0 μs | 951.2 μs | 934.0 μs | **4.73x** | **4.65x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | off | **—** | 67.0 μs | 614.9 μs | 156.2 μs | **9.18x** | **2.33x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | off | **1.01 ms** | 1.05 ms | 2.56 ms | 3.99 ms | **2.53x** | **3.94x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | off | **—** | 61.3 μs | 563.9 μs | 146.0 μs | **9.19x** | **2.38x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | off | **4.28 ms** | 3.92 ms | 7.15 ms | 4.36 ms | **1.82x** | **1.11x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | off | **—** | 66.2 μs | 591.8 μs | 149.6 μs | **8.94x** | **2.26x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | off | **5.49 ms** | 5.52 ms | 10.95 ms | 6.38 ms | **1.99x** | **1.16x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | off | **—** | 56.2 μs | 533.4 μs | 142.0 μs | **9.49x** | **2.53x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | off | **134.8 μs** | 103.6 μs | 810.1 μs | 252.3 μs | **7.82x** | **2.43x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | off | **—** | 60.3 μs | 548.5 μs | 131.8 μs | **9.09x** | **2.18x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | off | **868.1 μs** | 828.6 μs | 2.50 ms | 1.04 ms | **3.02x** | **1.25x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | off | **—** | 65.4 μs | 601.0 μs | 148.4 μs | **9.19x** | **2.27x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | off | **1.39 ms** | 1.31 ms | 3.42 ms | 1.66 ms | **2.61x** | **1.27x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | off | **—** | 57.8 μs | 486.3 μs | 152.2 μs | **8.41x** | **2.63x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | off | **234.0 μs** | 180.5 μs | 948.7 μs | 328.8 μs | **5.26x** | **1.82x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | off | **—** | 60.3 μs | 556.5 μs | 149.4 μs | **9.23x** | **2.48x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | off | **2.14 ms** | 1.98 ms | 4.19 ms | 2.33 ms | **2.11x** | **1.17x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | off | **—** | 72.2 μs | 598.4 μs | 136.9 μs | **8.29x** | **1.90x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | off | **3.08 ms** | 3.36 ms | 5.28 ms | 3.59 ms | **1.72x** | **1.17x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | off | **—** | 53.1 μs | 476.6 μs | 128.2 μs | **8.98x** | **2.41x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | off | **107.5 μs** | 79.7 μs | 765.8 μs | 222.6 μs | **9.60x** | **2.79x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | off | **—** | 51.3 μs | 464.5 μs | 126.8 μs | **9.05x** | **2.47x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | off | **446.3 μs** | 438.2 μs | 1.46 ms | 611.1 μs | **3.34x** | **1.39x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | off | **—** | 67.6 μs | 763.3 μs | 159.4 μs | **11.29x** | **2.36x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | off | **677.0 μs** | 681.7 μs | 1.99 ms | 875.1 μs | **2.94x** | **1.29x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | off | **—** | 69.9 μs | 582.6 μs | 138.7 μs | **8.33x** | **1.98x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | off | **125.9 μs** | 109.8 μs | 708.3 μs | 232.8 μs | **6.45x** | **2.12x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | off | **—** | 62.2 μs | 517.0 μs | 131.3 μs | **8.32x** | **2.11x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | off | **787.5 μs** | 814.7 μs | 2.12 ms | 944.0 μs | **2.70x** | **1.20x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | off | **—** | 51.1 μs | 550.4 μs | 133.8 μs | **10.77x** | **2.62x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | off | **1.23 ms** | 1.17 ms | 3.05 ms | 1.47 ms | **2.61x** | **1.26x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | off | **—** | 58.3 μs | 504.0 μs | 134.4 μs | **8.64x** | **2.30x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | off | **199.9 μs** | 176.0 μs | 956.5 μs | 357.5 μs | **5.44x** | **2.03x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | off | **—** | 62.5 μs | 534.3 μs | 140.5 μs | **8.55x** | **2.25x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | off | **2.31 ms** | 2.17 ms | 5.45 ms | 3.14 ms | **2.51x** | **1.45x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | off | **—** | 60.2 μs | 561.4 μs | 141.7 μs | **9.32x** | **2.35x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | off | **3.58 ms** | 3.20 ms | 4.89 ms | 3.56 ms | **1.53x** | **1.11x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | off | **—** | 60.0 μs | 549.2 μs | 139.5 μs | **9.15x** | **2.32x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | off | **83.0 μs** | 57.9 μs | 745.4 μs | 257.5 μs | **12.87x** | **4.45x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | off | **—** | 62.9 μs | 617.4 μs | 144.4 μs | **9.82x** | **2.30x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | off | **232.8 μs** | 212.2 μs | 1.16 ms | 997.3 μs | **5.46x** | **4.70x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | off | **—** | 68.9 μs | 641.2 μs | 153.6 μs | **9.31x** | **2.23x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | off | **446.2 μs** | 367.8 μs | 1.27 ms | 1.24 ms | **3.46x** | **3.38x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | off | **—** | 57.7 μs | 590.4 μs | 142.5 μs | **10.24x** | **2.47x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **831.5 μs** | 1.02 ms | 2.27 ms | 1.12 ms | **2.73x** | **1.35x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | off | **—** | 67.0 μs | 594.7 μs | 147.9 μs | **8.88x** | **2.21x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **1.35 ms** | 1.40 ms | 3.07 ms | 1.56 ms | **2.27x** | **1.16x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | off | **—** | 69.6 μs | 794.6 μs | 160.6 μs | **11.41x** | **2.31x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | off | **233.9 μs** | 215.7 μs | 1.19 ms | 1.00 ms | **5.54x** | **4.64x** |
| fits | mef_small | header_read | 0.45 MB | CPU | off | **—** | 66.2 μs | 803.8 μs | 142.4 μs | **12.15x** | **2.15x** |
| fits | mef_small | read_full | 0.45 MB | CPU | off | **70.3 μs** | 48.6 μs | 883.8 μs | 249.4 μs | **18.18x** | **5.13x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | off | **42.3 μs** | 49.4 μs | 3.39 ms | 318.5 μs | **80.14x** | **7.52x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | off | **—** | 66.3 μs | 940.3 μs | 169.5 μs | **14.18x** | **2.56x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | off | **6.01 ms** | 6.16 ms | 8.06 ms | 7.70 ms | **1.34x** | **1.28x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **70.4 μs** | 54.8 μs | 1.04 ms | 350.3 μs | **18.88x** | **6.39x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | off | **6.83 ms** | 6.38 ms | 105.42 ms | 7.27 ms | **16.52x** | **1.14x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | off | **—** | 64.7 μs | 621.5 μs | 157.8 μs | **9.60x** | **2.44x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | off | **3.31 ms** | 3.29 ms | 8.65 ms | 3.86 ms | **2.63x** | **1.18x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | off | **—** | 58.7 μs | 611.4 μs | 151.6 μs | **10.42x** | **2.58x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | off | **844.7 μs** | 789.3 μs | 2.55 ms | 1.25 ms | **3.23x** | **1.58x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | off | **—** | 65.5 μs | 620.5 μs | 141.0 μs | **9.47x** | **2.15x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | off | **110.7 μs** | 76.5 μs | 908.9 μs | 256.1 μs | **11.89x** | **3.35x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | off | **—** | 48.5 μs | 490.8 μs | 122.1 μs | **10.13x** | **2.52x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | off | **89.9 μs** | 84.5 μs | 802.2 μs | 209.7 μs | **9.49x** | **2.48x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | off | **—** | 57.7 μs | 529.8 μs | 137.8 μs | **9.19x** | **2.39x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | off | **103.8 μs** | 76.8 μs | 753.5 μs | 233.8 μs | **9.81x** | **3.05x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | off | **—** | 57.0 μs | 491.7 μs | 128.5 μs | **8.63x** | **2.26x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | off | **169.0 μs** | 128.7 μs | 828.7 μs | 278.4 μs | **6.44x** | **2.16x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | off | **—** | 50.3 μs | 486.0 μs | 134.2 μs | **9.66x** | **2.67x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | off | **80.8 μs** | 64.3 μs | 666.1 μs | 199.6 μs | **10.36x** | **3.10x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | off | **—** | 61.0 μs | 1.11 ms | 161.8 μs | **18.18x** | **2.65x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | off | **280.9 μs** | 135.7 μs | 828.5 μs | 285.8 μs | **6.10x** | **2.11x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | off | **—** | 85.4 μs | 683.4 μs | 150.7 μs | **8.00x** | **1.77x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | off | **312.4 μs** | 293.5 μs | 1.16 ms | 469.0 μs | **3.95x** | **1.60x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | off | **—** | 65.7 μs | 618.4 μs | 146.0 μs | **9.41x** | **2.22x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | off | **84.3 μs** | 53.8 μs | 747.6 μs | 210.9 μs | **13.89x** | **3.92x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | off | **—** | 71.2 μs | 621.0 μs | 144.7 μs | **8.73x** | **2.03x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | off | **85.6 μs** | 65.5 μs | 768.5 μs | 212.0 μs | **11.73x** | **3.24x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | off | **—** | 65.1 μs | 610.7 μs | 153.9 μs | **9.38x** | **2.36x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | off | **122.0 μs** | 99.7 μs | 769.8 μs | 221.6 μs | **7.72x** | **2.22x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | off | **—** | 65.9 μs | 659.5 μs | 160.5 μs | **10.01x** | **2.43x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | off | **75.7 μs** | 56.4 μs | 633.5 μs | 196.8 μs | **11.23x** | **3.49x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | off | **—** | 75.9 μs | 695.3 μs | 156.6 μs | **9.16x** | **2.06x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | off | **103.7 μs** | 87.2 μs | 992.9 μs | 234.1 μs | **11.39x** | **2.68x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | off | **—** | 71.7 μs | 690.5 μs | 161.0 μs | **9.63x** | **2.25x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | off | **156.2 μs** | 130.7 μs | 802.5 μs | 311.0 μs | **6.14x** | **2.38x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | off | **—** | 64.0 μs | 615.6 μs | 146.5 μs | **9.61x** | **2.29x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | off | **93.0 μs** | 60.4 μs | 642.7 μs | 234.8 μs | **10.65x** | **3.89x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | off | **—** | 61.0 μs | 625.7 μs | 154.8 μs | **10.26x** | **2.54x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | off | **158.2 μs** | 141.7 μs | 736.3 μs | 293.8 μs | **5.20x** | **2.07x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | off | **—** | 61.5 μs | 594.0 μs | 145.3 μs | **9.65x** | **2.36x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | off | **285.4 μs** | 267.5 μs | 1.06 ms | 413.9 μs | **3.98x** | **1.55x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | off | **—** | 61.8 μs | 557.7 μs | 145.1 μs | **9.03x** | **2.35x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | off | **60.4 μs** | 39.0 μs | 770.5 μs | 186.9 μs | **19.78x** | **4.80x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | off | **—** | 61.1 μs | 561.3 μs | 141.3 μs | **9.19x** | **2.31x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | off | **77.3 μs** | 56.3 μs | 797.0 μs | 238.5 μs | **14.16x** | **4.24x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | off | **—** | 53.8 μs | 572.2 μs | 142.1 μs | **10.64x** | **2.64x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | off | **88.4 μs** | 70.2 μs | 851.3 μs | 312.2 μs | **12.13x** | **4.45x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | off | **—** | 70.3 μs | 624.6 μs | 152.2 μs | **8.89x** | **2.16x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | off | **124.0 μs** | 107.0 μs | 787.6 μs | 283.5 μs | **7.36x** | **2.65x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | off | **—** | 56.0 μs | 589.1 μs | 153.8 μs | **10.53x** | **2.75x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | off | **156.6 μs** | 121.4 μs | 904.1 μs | 296.8 μs | **7.45x** | **2.45x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | off | **—** | 64.5 μs | 573.0 μs | 148.5 μs | **8.88x** | **2.30x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **104.2 μs** | 81.2 μs | 668.8 μs | 222.0 μs | **8.23x** | **2.73x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | off | **—** | 59.2 μs | 595.8 μs | 148.6 μs | **10.07x** | **2.51x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **96.7 μs** | 82.9 μs | 704.0 μs | 219.3 μs | **8.49x** | **2.64x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | off | **—** | 61.2 μs | 532.2 μs | 146.0 μs | **8.70x** | **2.39x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **106.6 μs** | 79.6 μs | 683.0 μs | 208.3 μs | **8.58x** | **2.62x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | off | **—** | 57.8 μs | 559.2 μs | 139.5 μs | **9.67x** | **2.41x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **86.0 μs** | 66.2 μs | 519.4 μs | 184.6 μs | **7.84x** | **2.79x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | off | **—** | 70.3 μs | 644.4 μs | 154.7 μs | **9.17x** | **2.20x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **85.7 μs** | 60.7 μs | 514.2 μs | 181.5 μs | **8.48x** | **2.99x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | off | **—** | 71.2 μs | 648.6 μs | 155.8 μs | **9.11x** | **2.19x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **36.7 μs** | 29.4 μs | 450.5 μs | 145.4 μs | **15.32x** | **4.94x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | off | **—** | 66.2 μs | 670.9 μs | 157.4 μs | **10.14x** | **2.38x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **51.2 μs** | 38.3 μs | 423.1 μs | 157.0 μs | **11.05x** | **4.10x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | off | **—** | 66.3 μs | 633.6 μs | 145.3 μs | **9.55x** | **2.19x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **46.8 μs** | 43.8 μs | 489.3 μs | 166.5 μs | **11.17x** | **3.80x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | off | **—** | 58.9 μs | 568.0 μs | 130.0 μs | **9.64x** | **2.21x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **39.4 μs** | 30.6 μs | 544.5 μs | 169.9 μs | **17.78x** | **5.55x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | off | **—** | 56.5 μs | 580.0 μs | 145.6 μs | **10.27x** | **2.58x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **68.6 μs** | 43.8 μs | 501.0 μs | 154.0 μs | **11.44x** | **3.52x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | off | **—** | 63.0 μs | 586.9 μs | 136.8 μs | **9.32x** | **2.17x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **62.6 μs** | 54.2 μs | 792.8 μs | 193.5 μs | **14.63x** | **3.57x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | off | **—** | 56.4 μs | 525.4 μs | 138.3 μs | **9.32x** | **2.45x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **56.3 μs** | 37.9 μs | 569.4 μs | 176.2 μs | **15.03x** | **4.65x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | off | **—** | 57.4 μs | 541.0 μs | 132.3 μs | **9.42x** | **2.30x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **54.7 μs** | 36.9 μs | 541.5 μs | 151.6 μs | **14.67x** | **4.11x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | off | **—** | 60.2 μs | 608.1 μs | 140.8 μs | **10.09x** | **2.34x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **61.3 μs** | 42.1 μs | 608.6 μs | 180.6 μs | **14.46x** | **4.29x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | off | **—** | 63.8 μs | 554.5 μs | 139.8 μs | **8.69x** | **2.19x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **50.6 μs** | 40.8 μs | 581.8 μs | 178.5 μs | **14.28x** | **4.38x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | off | **—** | 59.5 μs | 563.2 μs | 143.2 μs | **9.47x** | **2.41x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **60.7 μs** | 44.9 μs | 601.1 μs | 173.9 μs | **13.40x** | **3.87x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | off | **—** | 60.2 μs | 518.0 μs | 143.5 μs | **8.60x** | **2.38x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **54.5 μs** | 45.7 μs | 549.7 μs | 168.0 μs | **12.02x** | **3.67x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | off | **—** | 57.3 μs | 527.5 μs | 142.3 μs | **9.21x** | **2.48x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **54.1 μs** | 41.2 μs | 575.5 μs | 175.0 μs | **13.97x** | **4.25x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | off | **—** | 57.0 μs | 528.4 μs | 146.2 μs | **9.27x** | **2.56x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **61.9 μs** | 44.5 μs | 509.1 μs | 159.3 μs | **11.45x** | **3.58x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | off | **—** | 64.7 μs | 589.2 μs | 145.6 μs | **9.11x** | **2.25x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **65.9 μs** | 41.4 μs | 510.0 μs | 155.7 μs | **12.31x** | **3.76x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | off | **—** | 62.6 μs | 604.3 μs | 150.0 μs | **9.65x** | **2.39x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **49.5 μs** | 39.7 μs | 816.0 μs | 225.2 μs | **20.57x** | **5.68x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | off | **—** | 66.7 μs | 600.9 μs | 155.2 μs | **9.01x** | **2.33x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **63.8 μs** | 39.8 μs | 646.2 μs | 165.5 μs | **16.22x** | **4.15x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | off | **—** | 65.0 μs | 644.8 μs | 153.9 μs | **9.93x** | **2.37x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **57.5 μs** | 40.0 μs | 703.2 μs | 176.8 μs | **17.58x** | **4.42x** |
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | on | **—** | 277.0 μs | 3.68 ms | 615.7 μs | **13.29x** | **2.22x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **22.73 ms** | 23.77 ms | 63.33 ms | 29.43 ms | **2.79x** | **1.30x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | on | **—** | 252.7 μs | 4.93 ms | 759.9 μs | **19.53x** | **3.01x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **17.08 ms** | 16.92 ms | 68.22 ms | 20.92 ms | **4.03x** | **1.24x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | on | **—** | 233.5 μs | 5.04 ms | 742.1 μs | **21.60x** | **3.18x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **34.63 ms** | 34.95 ms | 47.90 ms | 33.47 ms | **1.38x** | **0.97x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | on | **1.52 ms** | 1.47 ms | 21.26 ms | 2.29 ms | **14.49x** | **1.56x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | on | **—** | 213.0 μs | 4.84 ms | 813.2 μs | **22.72x** | **3.82x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **10.71 ms** | 10.19 ms | 35.37 ms | 10.53 ms | **3.47x** | **1.03x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | on | **—** | 123.8 μs | 1.62 ms | 385.5 μs | **13.11x** | **3.11x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | on | **1.15 ms** | 1.35 ms | 6.75 ms | 1.45 ms | **5.86x** | **1.26x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | on | **—** | 112.1 μs | 1.26 ms | 247.6 μs | **11.25x** | **2.21x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | on | **5.39 ms** | 5.07 ms | 13.25 ms | 5.61 ms | **2.61x** | **1.10x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | on | **—** | 149.3 μs | 1.13 ms | 286.4 μs | **7.59x** | **1.92x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | on | **2.22 ms** | 2.93 ms | 9.37 ms | 3.24 ms | **4.21x** | **1.46x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | on | **—** | 108.5 μs | 1.59 ms | 296.3 μs | **14.65x** | **2.73x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | on | **10.29 ms** | 10.84 ms | 28.59 ms | 7.71 ms | **2.78x** | **0.75x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | on | **—** | 110.8 μs | 1.65 ms | 292.0 μs | **14.90x** | **2.64x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | on | **468.9 μs** | 418.0 μs | 5.46 ms | 743.3 μs | **13.07x** | **1.78x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | on | **—** | 137.5 μs | 1.47 ms | 533.6 μs | **10.66x** | **3.88x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | on | **1.72 ms** | 1.76 ms | 16.32 ms | 3.25 ms | **9.46x** | **1.89x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | on | **—** | 115.0 μs | 1.38 ms | 309.6 μs | **11.98x** | **2.69x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.43 ms** | 1.04 ms | 6.37 ms | 974.0 μs | **6.11x** | **0.93x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | on | **—** | 142.5 μs | 1.53 ms | 266.8 μs | **10.73x** | **1.87x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | on | **4.26 ms** | 3.83 ms | 11.64 ms | 3.96 ms | **3.04x** | **1.03x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | on | **—** | 114.4 μs | 1.49 ms | 569.9 μs | **12.99x** | **4.98x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | on | **2.10 ms** | 1.76 ms | 4.05 ms | 2.16 ms | **2.30x** | **1.23x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | on | **—** | 122.0 μs | 2.28 ms | 429.3 μs | **18.72x** | **3.52x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | on | **7.24 ms** | 7.88 ms | 12.96 ms | 9.13 ms | **1.79x** | **1.26x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | on | **—** | 94.6 μs | 1.45 ms | 430.0 μs | **15.31x** | **4.54x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | on | **440.9 μs** | 338.6 μs | — | 1.22 ms | **—** | **3.61x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | on | **—** | 138.8 μs | 1.44 ms | 258.5 μs | **10.36x** | **1.86x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | on | **1.33 ms** | 1.06 ms | — | 3.41 ms | **—** | **3.23x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | on | **—** | 114.4 μs | 1.92 ms | 394.2 μs | **16.76x** | **3.45x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | on | **3.89 ms** | 5.21 ms | — | 4.61 ms | **—** | **1.19x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | on | **—** | 110.4 μs | 1.50 ms | 237.7 μs | **13.59x** | **2.15x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | on | **6.79 ms** | 8.40 ms | — | 7.44 ms | **—** | **1.10x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | on | **—** | 158.6 μs | 1.47 ms | 344.3 μs | **9.27x** | **2.17x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | on | **269.7 μs** | 186.0 μs | 8.01 ms | 392.5 μs | **43.07x** | **2.11x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | on | **—** | 198.6 μs | 1.61 ms | 406.9 μs | **8.09x** | **2.05x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | on | **1.38 ms** | 1.27 ms | 6.93 ms | 1.28 ms | **5.44x** | **1.00x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | on | **—** | 182.9 μs | 1.16 ms | 226.0 μs | **6.37x** | **1.24x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | on | **1.97 ms** | 2.49 ms | 9.04 ms | 2.42 ms | **4.58x** | **1.23x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | on | **—** | 75.7 μs | 971.8 μs | 228.0 μs | **12.83x** | **3.01x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | on | **346.5 μs** | 226.7 μs | 5.94 ms | 608.1 μs | **26.20x** | **2.68x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | on | **—** | 99.7 μs | 802.5 μs | 202.7 μs | **8.04x** | **2.03x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | on | **1.92 ms** | 2.90 ms | 12.70 ms | 3.90 ms | **6.60x** | **2.03x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | on | **—** | 125.9 μs | 795.7 μs | 180.3 μs | **6.32x** | **1.43x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | on | **3.62 ms** | 5.11 ms | 11.41 ms | 4.44 ms | **3.15x** | **1.22x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | on | **—** | 84.7 μs | 942.4 μs | 234.6 μs | **11.13x** | **2.77x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | on | **204.0 μs** | 143.0 μs | 7.12 ms | 256.7 μs | **49.81x** | **1.79x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | on | **—** | 93.1 μs | 1.12 ms | 338.9 μs | **12.01x** | **3.64x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | on | **883.5 μs** | 934.4 μs | 10.59 ms | 1.21 ms | **11.99x** | **1.37x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | on | **—** | 113.2 μs | 1.54 ms | 428.0 μs | **13.62x** | **3.78x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | on | **1.16 ms** | 1.08 ms | 8.11 ms | 1.31 ms | **7.50x** | **1.21x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | on | **—** | 113.3 μs | 1.03 ms | 315.5 μs | **9.09x** | **2.78x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | on | **207.4 μs** | 179.7 μs | 6.31 ms | 584.3 μs | **35.12x** | **3.25x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | on | **—** | 197.7 μs | 1.44 ms | 198.9 μs | **7.26x** | **1.01x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | on | **1.42 ms** | 1.54 ms | 12.69 ms | 1.58 ms | **8.94x** | **1.11x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | on | **—** | 122.7 μs | 1.38 ms | 300.1 μs | **11.28x** | **2.45x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | on | **1.93 ms** | 2.07 ms | 12.70 ms | 2.40 ms | **6.56x** | **1.24x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | on | **—** | 168.1 μs | 1.58 ms | 300.2 μs | **9.41x** | **1.79x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | on | **471.2 μs** | 272.1 μs | 8.42 ms | 715.7 μs | **30.96x** | **2.63x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | on | **—** | 100.0 μs | 1.64 ms | 267.8 μs | **16.43x** | **2.68x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | on | **2.67 ms** | 2.70 ms | 14.84 ms | 3.88 ms | **5.55x** | **1.45x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | on | **—** | 102.0 μs | 1.58 ms | 288.9 μs | **15.47x** | **2.83x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | on | **5.98 ms** | 4.57 ms | 20.64 ms | 5.45 ms | **4.52x** | **1.19x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | on | **—** | 166.2 μs | 1.78 ms | 289.5 μs | **10.74x** | **1.74x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | on | **171.6 μs** | 92.8 μs | — | 446.5 μs | **—** | **4.81x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | on | **—** | 140.5 μs | 1.68 ms | 259.3 μs | **11.93x** | **1.85x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | on | **650.1 μs** | 318.4 μs | — | 2.32 ms | **—** | **7.29x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | on | **—** | 120.9 μs | 1.64 ms | 297.5 μs | **13.53x** | **2.46x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | on | **725.5 μs** | 646.5 μs | — | 2.72 ms | **—** | **4.21x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | on | **—** | 116.9 μs | 1.56 ms | 313.1 μs | **13.39x** | **2.68x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **1.49 ms** | 1.45 ms | — | 2.19 ms | **—** | **1.51x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | on | **—** | 231.6 μs | 1.70 ms | 369.3 μs | **7.36x** | **1.59x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **2.44 ms** | 2.22 ms | — | 3.11 ms | **—** | **1.40x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | on | **—** | 307.0 μs | 1.77 ms | 303.5 μs | **5.76x** | **0.99x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | on | **515.0 μs** | 600.2 μs | — | 2.39 ms | **—** | **4.65x** |
| fits | mef_small | header_read | 0.45 MB | CPU | on | **—** | 112.5 μs | 1.37 ms | 303.3 μs | **12.14x** | **2.70x** |
| fits | mef_small | read_full | 0.45 MB | CPU | on | **170.3 μs** | 83.5 μs | — | 559.5 μs | **—** | **6.70x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | on | **95.3 μs** | 105.7 μs | — | 1.72 ms | **—** | **18.02x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | on | **—** | 107.4 μs | 1.38 ms | 311.8 μs | **12.87x** | **2.90x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | on | **13.95 ms** | 15.46 ms | — | 31.82 ms | **—** | **2.28x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **115.5 μs** | 96.1 μs | — | 847.5 μs | **—** | **8.82x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | on | **26.29 ms** | 22.92 ms | 82.21 ms | 13.70 ms | **3.59x** | **0.60x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | on | **—** | 112.7 μs | 933.3 μs | 273.5 μs | **8.28x** | **2.43x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | on | **8.49 ms** | 7.18 ms | — | 7.19 ms | **—** | **1.00x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | on | **—** | 86.0 μs | 1.24 ms | 285.5 μs | **14.43x** | **3.32x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | on | **2.00 ms** | 1.55 ms | — | 2.34 ms | **—** | **1.51x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | on | **—** | 81.3 μs | 1.23 ms | 366.3 μs | **15.17x** | **4.51x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | on | **279.2 μs** | 191.0 μs | — | 433.9 μs | **—** | **2.27x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | on | **—** | 120.8 μs | 990.8 μs | 186.5 μs | **8.20x** | **1.54x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | on | **236.3 μs** | 130.1 μs | 7.23 ms | 321.2 μs | **55.61x** | **2.47x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | on | **—** | 97.8 μs | 1.08 ms | 312.8 μs | **10.99x** | **3.20x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | on | **217.3 μs** | 193.8 μs | 14.04 ms | 290.5 μs | **72.45x** | **1.50x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | on | **—** | 121.8 μs | 1.57 ms | 283.4 μs | **12.91x** | **2.33x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | on | **413.7 μs** | 429.7 μs | 18.57 ms | 614.5 μs | **44.89x** | **1.49x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | on | **—** | 84.0 μs | 1.21 ms | 189.6 μs | **14.34x** | **2.26x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | on | **178.3 μs** | 141.9 μs | 8.86 ms | 365.7 μs | **62.45x** | **2.58x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | on | **—** | 122.2 μs | 1.57 ms | 366.3 μs | **12.81x** | **3.00x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | on | **316.1 μs** | 214.5 μs | 17.93 ms | 558.9 μs | **83.61x** | **2.61x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | on | **—** | 123.0 μs | 1.13 ms | 249.5 μs | **9.20x** | **2.03x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | on | **630.8 μs** | 461.3 μs | 12.39 ms | 761.3 μs | **26.86x** | **1.65x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | on | **—** | 92.5 μs | 1.08 ms | 203.6 μs | **11.69x** | **2.20x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | on | **161.5 μs** | 159.5 μs | 29.93 ms | 352.0 μs | **187.65x** | **2.21x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | on | **—** | 102.4 μs | 1.48 ms | 308.3 μs | **14.46x** | **3.01x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | on | **177.4 μs** | 124.1 μs | 8.25 ms | 365.6 μs | **66.46x** | **2.95x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | on | **—** | 84.7 μs | 853.2 μs | 258.3 μs | **10.07x** | **3.05x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | on | **334.1 μs** | 185.0 μs | 11.89 ms | 482.5 μs | **64.31x** | **2.61x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | on | **—** | 95.3 μs | 1.67 ms | 198.0 μs | **17.54x** | **2.08x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | on | **370.7 μs** | 122.2 μs | 6.53 ms | 466.2 μs | **53.43x** | **3.81x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | on | **—** | 113.5 μs | 1.46 ms | 282.6 μs | **12.83x** | **2.49x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | on | **221.1 μs** | 242.5 μs | 6.23 ms | 388.5 μs | **28.16x** | **1.76x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | on | **—** | 105.9 μs | 1.18 ms | 389.7 μs | **11.17x** | **3.68x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | on | **494.2 μs** | 317.9 μs | 6.33 ms | 791.5 μs | **19.92x** | **2.49x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | on | **—** | 82.2 μs | 1.27 ms | 232.1 μs | **15.48x** | **2.82x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | on | **292.8 μs** | 160.5 μs | 6.83 ms | 367.2 μs | **42.55x** | **2.29x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | on | **—** | 85.3 μs | 1.19 ms | 294.8 μs | **13.91x** | **3.45x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | on | **646.4 μs** | 205.7 μs | 6.77 ms | 540.6 μs | **32.92x** | **2.63x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | on | **—** | 105.4 μs | 1.62 ms | 228.1 μs | **15.39x** | **2.16x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | on | **707.2 μs** | 383.5 μs | 6.84 ms | 775.2 μs | **17.83x** | **2.02x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | on | **—** | 121.9 μs | 1.37 ms | 288.3 μs | **11.21x** | **2.37x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | on | **125.0 μs** | 82.0 μs | — | 352.5 μs | **—** | **4.30x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | on | **—** | 168.5 μs | 1.13 ms | 273.7 μs | **6.71x** | **1.62x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | on | **167.5 μs** | 107.5 μs | — | 430.8 μs | **—** | **4.01x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | on | **—** | 122.3 μs | 1.38 ms | 241.7 μs | **11.31x** | **1.98x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | on | **197.1 μs** | 115.8 μs | — | 650.8 μs | **—** | **5.62x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | on | **—** | 145.7 μs | 1.24 ms | 272.6 μs | **8.50x** | **1.87x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | on | **219.5 μs** | 187.1 μs | — | 468.8 μs | **—** | **2.51x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | on | **—** | 88.9 μs | 1.15 ms | 420.6 μs | **12.91x** | **4.73x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | on | **278.7 μs** | 211.8 μs | — | 558.3 μs | **—** | **2.64x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | on | **—** | 84.3 μs | 1.29 ms | 166.1 μs | **15.30x** | **1.97x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **269.3 μs** | 239.4 μs | 7.64 ms | 568.0 μs | **31.90x** | **2.37x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | on | **—** | 95.8 μs | 874.1 μs | 341.6 μs | **9.12x** | **3.56x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **241.9 μs** | 160.9 μs | 8.65 ms | 430.2 μs | **53.76x** | **2.67x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | on | **—** | 87.3 μs | 913.6 μs | 224.5 μs | **10.46x** | **2.57x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **201.6 μs** | 254.9 μs | 24.24 ms | 480.4 μs | **120.21x** | **2.38x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | on | **—** | 103.6 μs | 1.21 ms | 222.5 μs | **11.68x** | **2.15x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **220.1 μs** | 202.8 μs | 12.62 ms | 911.1 μs | **62.24x** | **4.49x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | on | **—** | 110.6 μs | 1.56 ms | 315.5 μs | **14.07x** | **2.85x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **234.1 μs** | 156.0 μs | 7.24 ms | 457.0 μs | **46.40x** | **2.93x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | on | **—** | 82.2 μs | 1.10 ms | 205.1 μs | **13.44x** | **2.50x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **199.5 μs** | 137.8 μs | 7.95 ms | 551.4 μs | **57.74x** | **4.00x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | on | **—** | 108.9 μs | 1.32 ms | 293.6 μs | **12.10x** | **2.70x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **158.0 μs** | 134.3 μs | 10.36 ms | 323.0 μs | **77.16x** | **2.40x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | on | **—** | 82.2 μs | 960.5 μs | 214.6 μs | **11.68x** | **2.61x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **213.1 μs** | 199.8 μs | 18.73 ms | 375.2 μs | **93.73x** | **1.88x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | on | **—** | 79.1 μs | 1.29 ms | 258.5 μs | **16.26x** | **3.27x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **136.7 μs** | 135.1 μs | 21.09 ms | 635.5 μs | **156.06x** | **4.70x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | on | **—** | 85.8 μs | 947.9 μs | 282.3 μs | **11.05x** | **3.29x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **227.0 μs** | 170.5 μs | 7.25 ms | 632.1 μs | **42.51x** | **3.71x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | on | **—** | 112.2 μs | 818.0 μs | 260.1 μs | **7.29x** | **2.32x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **244.5 μs** | 143.4 μs | 5.23 ms | 485.9 μs | **36.48x** | **3.39x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | on | **—** | 147.4 μs | 1.46 ms | 339.0 μs | **9.90x** | **2.30x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **150.5 μs** | 111.2 μs | 9.26 ms | 397.4 μs | **83.21x** | **3.57x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | on | **—** | 82.2 μs | 1.18 ms | 278.3 μs | **14.36x** | **3.39x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **140.2 μs** | 90.9 μs | 8.07 ms | 331.0 μs | **88.81x** | **3.64x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | on | **—** | 72.7 μs | 739.6 μs | 202.5 μs | **10.17x** | **2.78x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **168.3 μs** | 188.2 μs | 35.75 ms | 494.9 μs | **212.48x** | **2.94x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | on | **—** | 76.0 μs | 789.8 μs | 191.9 μs | **10.39x** | **2.52x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **140.2 μs** | 181.5 μs | 5.89 ms | 378.6 μs | **41.99x** | **2.70x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | on | **—** | 74.1 μs | 755.6 μs | 233.3 μs | **10.19x** | **3.15x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **158.7 μs** | 141.2 μs | 7.97 ms | 377.5 μs | **56.42x** | **2.67x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | on | **—** | 75.5 μs | 908.6 μs | 213.3 μs | **12.03x** | **2.83x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **270.4 μs** | 125.4 μs | 11.93 ms | 463.4 μs | **95.15x** | **3.70x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | on | **—** | 80.3 μs | 913.2 μs | 404.7 μs | **11.37x** | **5.04x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **130.5 μs** | 116.6 μs | 1.50 ms | 317.1 μs | **12.85x** | **2.72x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | on | **—** | 83.4 μs | 1.00 ms | 197.6 μs | **12.00x** | **2.37x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **231.4 μs** | 113.8 μs | 5.15 ms | 420.3 μs | **45.24x** | **3.69x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | on | **—** | 73.4 μs | 863.7 μs | 206.8 μs | **11.77x** | **2.82x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **161.2 μs** | 173.1 μs | 5.61 ms | 299.4 μs | **34.77x** | **1.86x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | on | **—** | 82.0 μs | 819.2 μs | 235.4 μs | **9.98x** | **2.87x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **142.5 μs** | 79.0 μs | — | 438.4 μs | **—** | **5.55x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | on | **—** | 122.3 μs | 1.13 ms | 249.8 μs | **9.22x** | **2.04x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **103.7 μs** | 52.8 μs | — | 342.0 μs | **—** | **6.47x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | on | **—** | 87.0 μs | 999.2 μs | 290.5 μs | **11.49x** | **3.34x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **140.8 μs** | 86.8 μs | — | 439.8 μs | **—** | **5.06x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **16.31 ms** | 16.15 ms | 36.49 ms | 20.26 ms | **2.26x** | **1.25x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **15.64 ms** | 15.62 ms | 54.10 ms | 19.14 ms | **3.46x** | **1.23x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **33.00 ms** | 33.38 ms | 38.67 ms | 34.27 ms | **1.17x** | **1.04x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | off | **1.61 ms** | 1.72 ms | 17.91 ms | 2.05 ms | **11.11x** | **1.28x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **10.34 ms** | 10.34 ms | 27.93 ms | 10.58 ms | **2.70x** | **1.02x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | off | **1.10 ms** | 1.07 ms | 2.55 ms | 1.17 ms | **2.38x** | **1.09x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | off | **3.81 ms** | 3.66 ms | 8.26 ms | 4.20 ms | **2.26x** | **1.15x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | off | **618.1 μs** | 798.5 μs | 1.70 ms | 991.3 μs | **2.75x** | **1.60x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | off | **3.19 ms** | 2.95 ms | 6.43 ms | 3.47 ms | **2.18x** | **1.17x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | off | **1.27 ms** | 1.29 ms | 3.59 ms | 1.66 ms | **2.83x** | **1.31x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | off | **5.25 ms** | 5.65 ms | 10.58 ms | 5.70 ms | **2.02x** | **1.08x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | off | **3.51 ms** | 3.37 ms | 6.99 ms | 4.57 ms | **2.07x** | **1.36x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | off | **12.32 ms** | 12.44 ms | 21.79 ms | 15.97 ms | **1.77x** | **1.30x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | off | **531.9 μs** | 888.4 μs | 2.26 ms | 2.01 ms | **4.24x** | **3.78x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | off | **1.35 ms** | 1.46 ms | 3.24 ms | 4.80 ms | **2.40x** | **3.56x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | off | **2.30 ms** | 4.80 ms | 7.69 ms | 5.62 ms | **3.34x** | **2.44x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | off | **4.44 ms** | 7.50 ms | 14.03 ms | 8.28 ms | **3.16x** | **1.86x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | off | **425.7 μs** | 430.8 μs | 1.26 ms | 850.2 μs | **2.97x** | **2.00x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | off | **1.69 ms** | 1.68 ms | 3.67 ms | 2.21 ms | **2.18x** | **1.31x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | off | **2.78 ms** | 2.89 ms | 5.80 ms | 2.79 ms | **2.09x** | **1.01x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | off | **301.4 μs** | 410.0 μs | 1.04 ms | 478.4 μs | **3.46x** | **1.59x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | off | **1.15 ms** | 1.11 ms | 3.00 ms | 1.66 ms | **2.70x** | **1.49x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | off | **1.68 ms** | 1.64 ms | 3.80 ms | 2.47 ms | **2.32x** | **1.51x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | off | **502.9 μs** | 796.5 μs | 1.87 ms | 917.2 μs | **3.72x** | **1.82x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | off | **1.57 ms** | 2.03 ms | 3.95 ms | 2.54 ms | **2.52x** | **1.62x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | off | **2.54 ms** | 2.43 ms | 6.93 ms | 3.19 ms | **2.85x** | **1.31x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | off | **659.9 μs** | 479.6 μs | 1.53 ms | 893.6 μs | **3.19x** | **1.86x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | off | **4.31 ms** | 3.70 ms | 6.95 ms | 5.11 ms | **1.88x** | **1.38x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | off | **5.79 ms** | 4.99 ms | 7.55 ms | 6.63 ms | **1.51x** | **1.33x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | off | **285.6 μs** | 1.09 ms | 1.48 ms | 827.5 μs | **5.18x** | **2.90x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | off | **811.4 μs** | 1.03 ms | 2.86 ms | 2.00 ms | **3.53x** | **2.46x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | off | **926.5 μs** | 1.25 ms | 3.48 ms | 3.37 ms | **3.75x** | **3.63x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **990.0 μs** | 1.67 ms | 4.30 ms | 1.93 ms | **4.34x** | **1.94x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **1.91 ms** | 2.63 ms | 6.28 ms | 3.87 ms | **3.29x** | **2.02x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | off | **860.5 μs** | 706.8 μs | 2.48 ms | 1.95 ms | **3.51x** | **2.76x** |
| fits | mef_small | read_full | 0.45 MB | MPS | off | **497.1 μs** | 475.8 μs | 1.97 ms | 1.06 ms | **4.15x** | **2.23x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | off | **630.0 μs** | 653.9 μs | 7.91 ms | 1.21 ms | **12.56x** | **1.92x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **311.5 μs** | 408.6 μs | 2.14 ms | 1.00 ms | **6.88x** | **3.22x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | off | **21.80 ms** | 20.71 ms | 213.10 ms | 19.62 ms | **10.29x** | **0.95x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | off | **3.54 ms** | 5.63 ms | 13.01 ms | 6.34 ms | **3.67x** | **1.79x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | off | **1.70 ms** | 2.36 ms | 5.50 ms | 2.34 ms | **3.23x** | **1.38x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | off | **416.3 μs** | 509.0 μs | 1.93 ms | 810.5 μs | **4.64x** | **1.95x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | off | **282.4 μs** | 347.2 μs | 1.53 ms | 593.5 μs | **5.42x** | **2.10x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | off | **391.2 μs** | 402.0 μs | 1.45 ms | 618.3 μs | **3.70x** | **1.58x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | off | **526.3 μs** | 722.0 μs | 1.99 ms | 1.11 ms | **3.79x** | **2.11x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | off | **246.0 μs** | 339.0 μs | 1.33 ms | 680.1 μs | **5.39x** | **2.76x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | off | **712.1 μs** | 390.1 μs | 1.39 ms | 617.8 μs | **3.56x** | **1.58x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | off | **590.5 μs** | 444.6 μs | 2.28 ms | 875.7 μs | **5.12x** | **1.97x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | off | **388.5 μs** | 641.1 μs | 1.41 ms | 639.6 μs | **3.64x** | **1.65x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | off | **390.3 μs** | 411.3 μs | 1.97 ms | 763.5 μs | **5.05x** | **1.96x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | off | **666.6 μs** | 759.1 μs | 1.52 ms | 554.7 μs | **2.29x** | **0.83x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | off | **619.3 μs** | 317.3 μs | 1.44 ms | 680.8 μs | **4.54x** | **2.15x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | off | **430.0 μs** | 386.1 μs | 1.19 ms | 750.7 μs | **3.08x** | **1.94x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | off | **800.1 μs** | 827.7 μs | 2.81 ms | 1.18 ms | **3.51x** | **1.48x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | off | **541.7 μs** | 568.5 μs | 1.72 ms | 907.8 μs | **3.17x** | **1.68x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | off | **283.9 μs** | 246.7 μs | 1.55 ms | 634.7 μs | **6.27x** | **2.57x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | off | **296.9 μs** | 487.3 μs | 1.44 ms | 701.2 μs | **4.85x** | **2.36x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | off | **353.3 μs** | 335.4 μs | 1.72 ms | 724.1 μs | **5.11x** | **2.16x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | off | **409.1 μs** | 476.7 μs | 1.72 ms | 782.3 μs | **4.21x** | **1.91x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **637.7 μs** | 442.9 μs | 1.53 ms | 653.4 μs | **3.46x** | **1.48x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **482.1 μs** | 400.8 μs | 1.72 ms | 1.09 ms | **4.30x** | **2.73x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **612.0 μs** | 370.8 μs | 1.43 ms | 1.07 ms | **3.86x** | **2.88x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **363.5 μs** | 661.0 μs | 1.68 ms | 776.9 μs | **4.61x** | **2.14x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **527.0 μs** | 701.0 μs | 1.50 ms | 702.3 μs | **2.84x** | **1.33x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **269.2 μs** | 267.4 μs | 1.46 ms | 529.0 μs | **5.46x** | **1.98x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **748.4 μs** | 374.4 μs | 1.36 ms | 569.2 μs | **3.64x** | **1.52x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **333.0 μs** | 379.1 μs | 1.51 ms | 662.2 μs | **4.55x** | **1.99x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **267.5 μs** | 316.8 μs | 1.27 ms | 737.4 μs | **4.73x** | **2.76x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **561.8 μs** | 326.1 μs | 1.33 ms | 577.0 μs | **4.07x** | **1.77x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **325.5 μs** | 335.3 μs | 1.43 ms | 659.4 μs | **4.38x** | **2.03x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **352.5 μs** | 287.0 μs | 1.54 ms | 828.7 μs | **5.35x** | **2.89x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **346.4 μs** | 316.1 μs | 1.57 ms | 613.2 μs | **4.95x** | **1.94x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **361.1 μs** | 282.7 μs | 1.43 ms | 660.5 μs | **5.07x** | **2.34x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **562.3 μs** | 320.4 μs | 1.41 ms | 620.0 μs | **4.40x** | **1.93x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **380.1 μs** | 301.9 μs | 1.87 ms | 693.2 μs | **6.19x** | **2.30x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **342.4 μs** | 514.1 μs | 1.50 ms | 529.4 μs | **4.38x** | **1.55x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **410.2 μs** | 224.9 μs | 1.35 ms | 456.6 μs | **5.98x** | **2.03x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **377.7 μs** | 328.5 μs | 1.80 ms | 710.4 μs | **5.49x** | **2.16x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **342.0 μs** | 306.2 μs | 1.77 ms | 689.3 μs | **5.77x** | **2.25x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **15.07 ms** | 15.46 ms | 36.09 ms | 18.46 ms | **2.39x** | **1.22x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **17.11 ms** | 17.03 ms | 63.07 ms | 20.98 ms | **3.70x** | **1.23x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **33.12 ms** | 32.96 ms | 46.10 ms | 31.90 ms | **1.40x** | **0.97x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | on | **1.41 ms** | 1.45 ms | 8.18 ms | 1.74 ms | **5.80x** | **1.23x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **10.64 ms** | 10.66 ms | 30.21 ms | 10.74 ms | **2.84x** | **1.01x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | on | **1.12 ms** | 1.11 ms | 2.51 ms | 1.44 ms | **2.25x** | **1.29x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | on | **3.78 ms** | 3.97 ms | 6.45 ms | 4.27 ms | **1.71x** | **1.13x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | on | **627.8 μs** | 508.3 μs | 1.60 ms | 747.1 μs | **3.15x** | **1.47x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | on | **2.16 ms** | 2.24 ms | 3.99 ms | 2.28 ms | **1.84x** | **1.06x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | on | **1.24 ms** | 1.52 ms | 6.23 ms | 1.97 ms | **5.01x** | **1.58x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | on | **3.91 ms** | 3.73 ms | 7.00 ms | 4.27 ms | **1.88x** | **1.15x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | on | **2.19 ms** | 1.95 ms | 3.49 ms | 2.66 ms | **1.79x** | **1.37x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | on | **9.42 ms** | 9.00 ms | 16.02 ms | 10.41 ms | **1.78x** | **1.16x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | on | **640.9 μs** | 578.5 μs | 3.69 ms | 1.21 ms | **6.38x** | **2.10x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | on | **1.13 ms** | 1.41 ms | 4.04 ms | 4.48 ms | **3.57x** | **3.96x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | on | **2.36 ms** | 5.32 ms | 13.99 ms | 5.62 ms | **5.93x** | **2.38x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | on | **4.18 ms** | 7.30 ms | 12.96 ms | 7.69 ms | **3.10x** | **1.84x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | on | **304.8 μs** | 323.0 μs | 987.8 μs | 438.1 μs | **3.24x** | **1.44x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | on | **1.26 ms** | 1.15 ms | 2.58 ms | 1.34 ms | **2.24x** | **1.17x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | on | **1.99 ms** | 1.87 ms | 7.38 ms | 2.24 ms | **3.95x** | **1.20x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | on | **540.7 μs** | 283.5 μs | 969.6 μs | 433.0 μs | **3.42x** | **1.53x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | on | **739.7 μs** | 647.7 μs | 5.31 ms | 906.3 μs | **8.20x** | **1.40x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | on | **846.6 μs** | 1.16 ms | 2.36 ms | 1.13 ms | **2.78x** | **1.33x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | on | **228.9 μs** | 293.3 μs | 976.9 μs | 405.7 μs | **4.27x** | **1.77x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | on | **1.12 ms** | 1.31 ms | 2.55 ms | 1.25 ms | **2.27x** | **1.11x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | on | **1.54 ms** | 1.61 ms | 3.39 ms | 1.98 ms | **2.20x** | **1.28x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | on | **319.7 μs** | 303.1 μs | 1.17 ms | 508.7 μs | **3.86x** | **1.68x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | on | **2.29 ms** | 2.10 ms | 7.23 ms | 2.88 ms | **3.44x** | **1.37x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | on | **3.59 ms** | 3.07 ms | 4.30 ms | 4.19 ms | **1.40x** | **1.36x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | on | **222.7 μs** | 226.7 μs | 1.43 ms | 465.6 μs | **6.42x** | **2.09x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | on | **368.5 μs** | 405.5 μs | 2.03 ms | 1.22 ms | **5.50x** | **3.31x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | on | **639.0 μs** | 616.5 μs | 3.33 ms | 2.00 ms | **5.40x** | **3.25x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **688.0 μs** | 1.32 ms | 8.08 ms | 1.65 ms | **11.74x** | **2.40x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **1.22 ms** | 1.75 ms | 7.53 ms | 1.80 ms | **6.20x** | **1.48x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | on | **356.2 μs** | 525.7 μs | 2.31 ms | 1.34 ms | **6.50x** | **3.77x** |
| fits | mef_small | read_full | 0.45 MB | MPS | on | **232.9 μs** | 192.2 μs | 1.81 ms | 492.5 μs | **9.41x** | **2.56x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | on | **494.1 μs** | 503.7 μs | 3.09 ms | 880.5 μs | **6.26x** | **1.78x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **280.7 μs** | 192.5 μs | 1.59 ms | 540.7 μs | **8.26x** | **2.81x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | on | **16.83 ms** | 16.04 ms | 59.54 ms | 16.42 ms | **3.71x** | **1.02x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | on | **2.44 ms** | 3.78 ms | 9.70 ms | 4.24 ms | **3.98x** | **1.74x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | on | **940.2 μs** | 1.19 ms | 3.77 ms | 1.28 ms | **4.01x** | **1.36x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | on | **229.3 μs** | 255.3 μs | 1.78 ms | 398.3 μs | **7.75x** | **1.74x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | on | **167.5 μs** | 261.8 μs | 794.0 μs | 366.5 μs | **4.74x** | **2.19x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | on | **235.7 μs** | 262.0 μs | 943.5 μs | 418.2 μs | **4.00x** | **1.77x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | on | **355.0 μs** | 317.6 μs | 4.71 ms | 488.4 μs | **14.82x** | **1.54x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | on | **176.7 μs** | 243.0 μs | 812.7 μs | 400.4 μs | **4.60x** | **2.27x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | on | **207.5 μs** | 233.9 μs | 832.2 μs | 375.1 μs | **4.01x** | **1.81x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | on | **244.9 μs** | 290.1 μs | 877.9 μs | 385.8 μs | **3.59x** | **1.58x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | on | **235.7 μs** | 217.5 μs | 958.0 μs | 355.4 μs | **4.40x** | **1.63x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | on | **292.7 μs** | 252.8 μs | 906.1 μs | 408.3 μs | **3.58x** | **1.61x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | on | **408.3 μs** | 335.1 μs | 1.06 ms | 500.7 μs | **3.18x** | **1.49x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | on | **224.3 μs** | 252.9 μs | 789.7 μs | 401.5 μs | **3.52x** | **1.79x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | on | **324.2 μs** | 311.9 μs | 1.04 ms | 435.5 μs | **3.32x** | **1.40x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | on | **590.3 μs** | 378.2 μs | 1.20 ms | 595.1 μs | **3.16x** | **1.57x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | on | **198.0 μs** | 189.7 μs | 1.75 ms | 375.6 μs | **9.23x** | **1.98x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | on | **281.7 μs** | 208.7 μs | 5.84 ms | 435.0 μs | **28.00x** | **2.08x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | on | **280.5 μs** | 333.3 μs | 1.69 ms | 650.7 μs | **6.04x** | **2.32x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | on | **243.4 μs** | 495.4 μs | 1.80 ms | 459.0 μs | **7.40x** | **1.89x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | on | **384.4 μs** | 262.4 μs | 1.80 ms | 470.6 μs | **6.86x** | **1.79x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **280.8 μs** | 314.2 μs | 918.5 μs | 463.0 μs | **3.27x** | **1.65x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **237.0 μs** | 257.9 μs | 1.47 ms | 489.5 μs | **6.21x** | **2.07x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **327.0 μs** | 270.6 μs | 7.69 ms | 403.5 μs | **28.40x** | **1.49x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **272.4 μs** | 299.6 μs | 6.53 ms | 590.8 μs | **23.98x** | **2.17x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **293.3 μs** | 238.7 μs | 6.77 ms | 428.4 μs | **28.38x** | **1.79x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **208.3 μs** | 231.3 μs | 6.13 ms | 391.0 μs | **29.41x** | **1.88x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **233.4 μs** | 230.0 μs | 7.03 ms | 333.1 μs | **30.57x** | **1.45x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **180.5 μs** | 204.7 μs | 8.54 ms | 322.6 μs | **47.32x** | **1.79x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **208.3 μs** | 182.6 μs | 827.7 μs | 326.8 μs | **4.53x** | **1.79x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **225.2 μs** | 467.4 μs | 984.8 μs | 399.9 μs | **4.37x** | **1.78x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **220.5 μs** | 240.7 μs | 1.19 ms | 468.6 μs | **5.40x** | **2.13x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **291.2 μs** | 287.7 μs | 5.95 ms | 445.2 μs | **20.67x** | **1.55x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **313.3 μs** | 346.0 μs | 2.08 ms | 590.5 μs | **6.65x** | **1.88x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **262.5 μs** | 402.2 μs | 1.02 ms | 410.6 μs | **3.89x** | **1.56x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **249.7 μs** | 263.8 μs | 1.03 ms | 427.6 μs | **4.12x** | **1.71x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **226.8 μs** | 493.8 μs | 5.54 ms | 370.9 μs | **24.45x** | **1.64x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **373.5 μs** | 277.5 μs | 1.15 ms | 462.8 μs | **4.13x** | **1.67x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **401.6 μs** | 273.1 μs | 2.51 ms | 497.1 μs | **9.20x** | **1.82x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **368.5 μs** | 275.5 μs | 3.32 ms | 745.4 μs | **12.06x** | **2.71x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **475.5 μs** | 213.2 μs | 1.70 ms | 501.0 μs | **8.00x** | **2.35x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **222.7 μs** | 203.9 μs | 3.96 ms | 7.15 ms | **19.40x** | **35.06x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | off | **68.7 μs** | 67.2 μs | 11.21 ms | 8.49 ms | **166.83x** | **126.34x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | off | **66.2 μs** | 75.3 μs | 1.97 ms | 9.48 ms | **29.78x** | **143.15x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | off | **64.5 μs** | 65.7 μs | 1.97 ms | 4.61 ms | **30.58x** | **71.58x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | off | **87.7 μs** | 84.3 μs | 4.25 ms | 3.44 ms | **50.48x** | **40.80x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **170.2 μs** | 137.1 μs | 2.27 ms | 1.32 ms | **16.58x** | **9.62x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | off | **61.8 μs** | 57.9 μs | 2.28 ms | 1.18 ms | **39.34x** | **20.43x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | off | **57.4 μs** | 55.6 μs | 1.24 ms | 958.5 μs | **22.25x** | **17.24x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | off | **67.5 μs** | 81.7 μs | 2.19 ms | 819.2 μs | **32.39x** | **12.14x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | off | **82.3 μs** | 73.1 μs | 2.28 ms | 608.8 μs | **31.17x** | **8.33x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **2.50 ms** | 2.36 ms | 66.76 ms | 382.26 ms | **28.34x** | **162.28x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | off | **71.8 μs** | 72.0 μs | 12.03 ms | 80.15 ms | **167.62x** | **1117.13x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | off | **59.1 μs** | 57.8 μs | 25.41 ms | 753.38 ms | **439.65x** | **13036.13x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **95.0 μs** | 77.9 μs | 18.66 ms | 126.18 ms | **239.57x** | **1620.24x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **91.0 μs** | 74.9 μs | 13.16 ms | 122.27 ms | **175.61x** | **1632.12x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **484.7 μs** | 485.4 μs | 11.08 ms | 47.59 ms | **22.85x** | **98.18x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | off | **80.6 μs** | 90.6 μs | 4.29 ms | 11.04 ms | **53.21x** | **137.00x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | off | **79.1 μs** | 85.3 μs | 5.93 ms | 104.92 ms | **74.99x** | **1325.97x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | off | **102.8 μs** | 85.6 μs | 5.13 ms | 19.99 ms | **59.95x** | **233.50x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | off | **116.2 μs** | 85.0 μs | 3.72 ms | 14.87 ms | **43.80x** | **174.88x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **276.8 μs** | 219.2 μs | 4.25 ms | 5.75 ms | **19.36x** | **26.23x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | off | **69.8 μs** | 72.4 μs | 2.46 ms | 1.33 ms | **35.24x** | **19.09x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | off | **70.0 μs** | 71.0 μs | 2.65 ms | 7.62 ms | **37.89x** | **108.85x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | off | **66.6 μs** | 72.5 μs | 3.21 ms | 2.33 ms | **48.14x** | **35.02x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | off | **176.9 μs** | 82.8 μs | 2.95 ms | 2.32 ms | **35.67x** | **28.06x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **147.7 μs** | 148.5 μs | 3.20 ms | 911.2 μs | **21.69x** | **6.17x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | off | **67.8 μs** | 72.2 μs | 2.34 ms | 465.1 μs | **34.55x** | **6.86x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | off | **63.9 μs** | 68.3 μs | 2.17 ms | 1.14 ms | **33.89x** | **17.89x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | off | **64.0 μs** | 70.2 μs | 3.05 ms | 635.0 μs | **47.69x** | **9.93x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | off | **106.1 μs** | 81.0 μs | 2.12 ms | 518.0 μs | **26.11x** | **6.40x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **2.00 ms** | 1.78 ms | 31.13 ms | 11.40 ms | **17.51x** | **6.41x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | off | **65.7 μs** | 65.0 μs | 4.81 ms | 39.98 ms | **74.05x** | **615.08x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | off | **64.4 μs** | 67.2 μs | 8.60 ms | 6.99 ms | **133.60x** | **108.61x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **68.5 μs** | 65.3 μs | 5.72 ms | 3.88 ms | **87.66x** | **59.43x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **74.0 μs** | 55.5 μs | 4.89 ms | 3.56 ms | **88.16x** | **64.22x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **457.3 μs** | 504.5 μs | 6.78 ms | 1.97 ms | **14.82x** | **4.31x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | off | **70.7 μs** | 68.5 μs | 2.15 ms | 6.23 ms | **31.41x** | **90.95x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | off | **66.2 μs** | 73.2 μs | 2.20 ms | 941.9 μs | **33.25x** | **14.23x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | off | **84.8 μs** | 81.8 μs | 3.22 ms | 697.4 μs | **39.35x** | **8.53x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | off | **153.3 μs** | 92.3 μs | 3.13 ms | 765.4 μs | **33.91x** | **8.30x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **262.2 μs** | 259.6 μs | 2.76 ms | 530.5 μs | **10.63x** | **2.04x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | off | **67.5 μs** | 69.2 μs | 1.77 ms | 821.7 μs | **26.26x** | **12.18x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | off | **69.5 μs** | 74.9 μs | 1.74 ms | 382.7 μs | **25.05x** | **5.51x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | off | **67.7 μs** | 79.3 μs | 2.58 ms | 502.8 μs | **38.15x** | **7.43x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | off | **98.6 μs** | 82.8 μs | 1.84 ms | 381.3 μs | **22.28x** | **4.61x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **166.0 μs** | 172.7 μs | 2.51 ms | 372.6 μs | **15.13x** | **2.25x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | off | **61.7 μs** | 62.4 μs | 1.57 ms | 407.0 μs | **25.38x** | **6.60x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | off | **67.2 μs** | 69.8 μs | 1.63 ms | 350.8 μs | **24.24x** | **5.22x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | off | **67.1 μs** | 81.0 μs | 2.26 ms | 374.0 μs | **33.60x** | **5.57x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | off | **86.6 μs** | 72.5 μs | 1.59 ms | 317.9 μs | **21.94x** | **4.38x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **298.2 μs** | 294.6 μs | 4.12 ms | 63.24 ms | **14.00x** | **214.66x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | off | **65.0 μs** | 71.6 μs | 37.78 ms | 63.87 ms | **581.55x** | **983.20x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | off | **60.8 μs** | 65.8 μs | 3.09 ms | 76.07 ms | **50.72x** | **1250.47x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | off | **58.7 μs** | 63.6 μs | 2.29 ms | 20.75 ms | **39.03x** | **353.19x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | off | **74.2 μs** | 68.4 μs | 1.77 ms | 16.36 ms | **25.86x** | **239.25x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **233.7 μs** | 206.1 μs | 2.38 ms | 7.32 ms | **11.55x** | **35.52x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | off | **70.0 μs** | 75.5 μs | 6.81 ms | 7.84 ms | **97.24x** | **112.02x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | off | **62.7 μs** | 61.4 μs | 1.53 ms | 10.61 ms | **24.95x** | **172.83x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | off | **62.7 μs** | 66.6 μs | 2.23 ms | 3.46 ms | **35.60x** | **55.21x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | off | **79.0 μs** | 70.7 μs | 1.56 ms | 2.13 ms | **22.03x** | **30.14x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **649.1 μs** | 339.4 μs | 4.62 ms | 149.92 ms | **13.62x** | **441.71x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | off | **74.9 μs** | 78.3 μs | 615.60 ms | 195.95 ms | **8217.14x** | **2615.49x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | off | **61.8 μs** | 66.0 μs | 2.61 ms | 204.60 ms | **42.28x** | **3313.35x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | off | **67.9 μs** | 67.3 μs | 2.33 ms | 154.60 ms | **34.59x** | **2296.08x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | off | **77.5 μs** | 78.9 μs | 2.19 ms | 154.88 ms | **28.30x** | **1999.54x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **190.2 μs** | 196.1 μs | 1.78 ms | 19.19 ms | **9.35x** | **100.91x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | off | **71.7 μs** | 74.5 μs | 67.58 ms | 16.00 ms | **942.49x** | **223.08x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | off | **71.3 μs** | 75.7 μs | 1.84 ms | 17.50 ms | **25.80x** | **245.41x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | off | **67.3 μs** | 66.3 μs | 2.07 ms | 13.23 ms | **31.13x** | **199.43x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | off | **112.7 μs** | 78.9 μs | 1.47 ms | 14.65 ms | **18.60x** | **185.72x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **134.6 μs** | 116.4 μs | 1.67 ms | 1.94 ms | **14.31x** | **16.64x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | off | **57.9 μs** | 62.5 μs | 6.50 ms | 1.62 ms | **112.37x** | **27.98x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | off | **61.4 μs** | 66.2 μs | 1.35 ms | 1.51 ms | **21.98x** | **24.68x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | off | **57.0 μs** | 58.6 μs | 1.56 ms | 1.62 ms | **27.33x** | **28.48x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | off | **84.4 μs** | 71.3 μs | 1.59 ms | 2.09 ms | **22.27x** | **29.33x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **434.0 μs** | 477.5 μs | 54.28 ms | 220.21 ms | **125.08x** | **507.45x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | off | **75.9 μs** | 74.5 μs | 11.75 ms | 10.51 ms | **157.68x** | **141.10x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | off | **75.3 μs** | 75.8 μs | 32.80 ms | 345.92 ms | **435.61x** | **4594.40x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | off | **71.7 μs** | 75.3 μs | 15.90 ms | 102.43 ms | **221.83x** | **1429.19x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | off | **197.7 μs** | 75.6 μs | 9.65 ms | 55.25 ms | **127.58x** | **730.61x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **277.5 μs** | 249.8 μs | 15.32 ms | 21.41 ms | **61.32x** | **85.69x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | off | **77.2 μs** | 83.1 μs | 7.73 ms | 1.83 ms | **100.18x** | **23.77x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | off | **69.8 μs** | 77.8 μs | 9.92 ms | 38.34 ms | **142.05x** | **549.02x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | off | **77.6 μs** | 73.2 μs | 10.80 ms | 10.71 ms | **147.49x** | **146.26x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | off | **225.3 μs** | 80.3 μs | 7.43 ms | 6.94 ms | **92.55x** | **86.43x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **160.9 μs** | 173.2 μs | 11.97 ms | 2.77 ms | **74.39x** | **17.24x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | off | **74.7 μs** | 80.4 μs | 8.97 ms | 837.6 μs | **120.11x** | **11.22x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | off | **75.3 μs** | 79.7 μs | 7.66 ms | 4.26 ms | **101.70x** | **56.55x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | off | **76.7 μs** | 80.1 μs | 11.68 ms | 1.87 ms | **152.32x** | **24.32x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | off | **225.5 μs** | 83.2 μs | 6.73 ms | 1.33 ms | **80.86x** | **16.00x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **194.4 μs** | 173.7 μs | 4.17 ms | 7.73 ms | **24.00x** | **44.51x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | on | **71.0 μs** | 80.2 μs | 13.65 ms | 10.01 ms | **192.17x** | **140.94x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | on | **68.5 μs** | 79.0 μs | 1.88 ms | 9.11 ms | **27.41x** | **132.96x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | on | **67.9 μs** | 63.6 μs | 1.79 ms | 3.17 ms | **28.19x** | **49.85x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | on | **68.2 μs** | 64.9 μs | 4.89 ms | 3.92 ms | **75.37x** | **60.41x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **167.8 μs** | 168.4 μs | 3.45 ms | 1.37 ms | **20.54x** | **8.14x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | on | **73.2 μs** | 77.1 μs | 3.20 ms | 1.33 ms | **43.72x** | **18.22x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | on | **70.8 μs** | 73.5 μs | 1.77 ms | 1.36 ms | **25.09x** | **19.23x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | on | **70.1 μs** | 75.5 μs | 2.27 ms | 772.3 μs | **32.44x** | **11.02x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | on | **87.3 μs** | 83.1 μs | 2.20 ms | 579.8 μs | **26.46x** | **6.98x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **4.44 ms** | 3.93 ms | 122.87 ms | 845.98 ms | **31.25x** | **215.19x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | on | **92.5 μs** | 96.3 μs | 22.49 ms | 144.78 ms | **243.09x** | **1565.14x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | on | **102.1 μs** | 151.7 μs | 70.02 ms | 1.129 s | **685.62x** | **11052.87x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **119.9 μs** | 172.6 μs | 29.24 ms | 233.90 ms | **243.87x** | **1950.54x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **226.3 μs** | 185.3 μs | 17.10 ms | 239.74 ms | **92.24x** | **1293.58x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **785.3 μs** | 855.0 μs | 20.05 ms | 81.12 ms | **25.54x** | **103.30x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | on | **251.6 μs** | 103.0 μs | 19.11 ms | 14.70 ms | **185.65x** | **142.75x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | on | **97.8 μs** | 103.7 μs | 15.88 ms | 151.70 ms | **162.44x** | **1551.31x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | on | **130.5 μs** | 112.0 μs | 25.36 ms | 33.84 ms | **226.33x** | **302.02x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | on | **188.5 μs** | 105.9 μs | 34.50 ms | 22.38 ms | **325.76x** | **211.26x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **394.0 μs** | 364.4 μs | 11.16 ms | 6.84 ms | **30.62x** | **18.77x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | on | **115.9 μs** | 99.2 μs | 13.05 ms | 2.34 ms | **131.53x** | **23.57x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | on | **148.9 μs** | 146.2 μs | 11.16 ms | 15.22 ms | **76.34x** | **104.07x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | on | **121.5 μs** | 105.5 μs | 15.29 ms | 3.88 ms | **145.02x** | **36.80x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | on | **124.2 μs** | 111.8 μs | 16.46 ms | 2.50 ms | **147.21x** | **22.37x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **328.2 μs** | 304.5 μs | 15.39 ms | 1.89 ms | **50.54x** | **6.20x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | on | **106.6 μs** | 159.0 μs | 9.44 ms | 820.8 μs | **88.53x** | **7.70x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | on | **144.7 μs** | 143.8 μs | 10.69 ms | 2.26 ms | **74.30x** | **15.69x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | on | **104.9 μs** | 117.5 μs | 9.95 ms | 1.21 ms | **94.89x** | **11.50x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | on | **200.0 μs** | 153.8 μs | 25.89 ms | 993.2 μs | **168.33x** | **6.46x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **4.00 ms** | 4.57 ms | 101.35 ms | 21.64 ms | **25.32x** | **5.40x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | on | **105.6 μs** | 116.2 μs | 13.80 ms | 94.44 ms | **130.67x** | **894.44x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | on | **116.8 μs** | 147.3 μs | 46.83 ms | 13.91 ms | **400.99x** | **119.07x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **105.1 μs** | 156.7 μs | 16.38 ms | 6.87 ms | **155.77x** | **65.31x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **217.5 μs** | 135.0 μs | 13.41 ms | 6.47 ms | **99.35x** | **47.92x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.13 ms** | 1.93 ms | 22.06 ms | 2.89 ms | **19.47x** | **2.55x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | on | **114.5 μs** | 104.8 μs | 8.52 ms | 8.26 ms | **81.30x** | **78.79x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | on | **144.5 μs** | 124.7 μs | 85.98 ms | 2.18 ms | **689.48x** | **17.52x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | on | **149.6 μs** | 100.5 μs | 12.03 ms | 996.2 μs | **119.75x** | **9.92x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | on | **302.8 μs** | 100.2 μs | 5.25 ms | 1.45 ms | **52.43x** | **14.43x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **457.2 μs** | 450.2 μs | 11.21 ms | 1.33 ms | **24.90x** | **2.96x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | on | **148.5 μs** | 123.2 μs | 10.18 ms | 1.30 ms | **82.68x** | **10.58x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | on | **97.5 μs** | 106.9 μs | 11.06 ms | 850.1 μs | **113.44x** | **8.72x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | on | **120.4 μs** | 136.5 μs | 12.39 ms | 1.28 ms | **102.89x** | **10.60x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | on | **154.8 μs** | 166.6 μs | 9.00 ms | 682.9 μs | **58.14x** | **4.41x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **300.7 μs** | 231.8 μs | 11.02 ms | 919.8 μs | **47.53x** | **3.97x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | on | **117.2 μs** | 129.7 μs | 8.22 ms | 857.0 μs | **70.15x** | **7.31x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | on | **156.5 μs** | 187.2 μs | 20.47 ms | 753.8 μs | **130.79x** | **4.82x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | on | **107.0 μs** | 98.9 μs | 14.46 ms | 688.6 μs | **146.24x** | **6.96x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | on | **207.0 μs** | 162.4 μs | 8.72 ms | 842.0 μs | **53.73x** | **5.19x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **507.6 μs** | 494.2 μs | 7.86 ms | 99.98 ms | **15.91x** | **202.29x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | on | **79.8 μs** | 87.3 μs | 59.98 ms | 102.91 ms | **751.75x** | **1289.78x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | on | **74.0 μs** | 75.5 μs | 4.13 ms | 103.15 ms | **55.74x** | **1393.96x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | on | **75.6 μs** | 82.9 μs | 8.53 ms | 32.72 ms | **112.85x** | **432.62x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | on | **98.4 μs** | 77.6 μs | 2.04 ms | 23.14 ms | **26.28x** | **298.07x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **276.1 μs** | 287.5 μs | 3.50 ms | 11.41 ms | **12.67x** | **41.32x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | on | **89.5 μs** | 98.6 μs | 12.32 ms | 11.43 ms | **137.69x** | **127.77x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | on | **87.8 μs** | 93.1 μs | 7.15 ms | 12.03 ms | **81.45x** | **137.08x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | on | **82.8 μs** | 88.5 μs | 3.09 ms | 3.81 ms | **37.33x** | **46.01x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | on | **136.7 μs** | 101.8 μs | 12.33 ms | 2.88 ms | **121.15x** | **28.33x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **432.3 μs** | 427.9 μs | 4.25 ms | 194.78 ms | **9.92x** | **455.23x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | on | **167.3 μs** | 100.3 μs | 1.350 s | 347.68 ms | **13458.81x** | **3465.24x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | on | **108.4 μs** | 146.8 μs | 9.49 ms | 341.19 ms | **87.54x** | **3147.07x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | on | **121.2 μs** | 137.8 μs | 8.57 ms | 211.70 ms | **70.72x** | **1745.99x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | on | **121.7 μs** | 113.4 μs | 7.13 ms | 282.53 ms | **62.87x** | **2491.12x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **628.3 μs** | 1.01 ms | 9.85 ms | 30.85 ms | **15.68x** | **49.11x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | on | **80.1 μs** | 93.6 μs | 102.68 ms | 25.51 ms | **1281.46x** | **318.39x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | on | **102.6 μs** | 96.2 μs | 2.72 ms | 26.00 ms | **28.26x** | **270.09x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | on | **94.5 μs** | 98.6 μs | 3.49 ms | 35.36 ms | **36.90x** | **374.22x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | on | **132.3 μs** | 108.1 μs | 20.25 ms | 33.57 ms | **187.36x** | **310.59x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **251.4 μs** | 229.5 μs | 4.79 ms | 2.71 ms | **20.88x** | **11.82x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | on | **117.3 μs** | 119.2 μs | 23.66 ms | 3.84 ms | **201.61x** | **32.77x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | on | **132.3 μs** | 125.3 μs | 3.59 ms | 3.54 ms | **28.65x** | **28.28x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | on | **119.5 μs** | 118.0 μs | 11.71 ms | 3.61 ms | **99.27x** | **30.61x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | on | **121.4 μs** | 107.1 μs | 2.74 ms | 2.94 ms | **25.54x** | **27.46x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **2.07 ms** | 1.88 ms | 111.84 ms | 361.28 ms | **59.51x** | **192.25x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | on | **96.5 μs** | 137.9 μs | 16.53 ms | 22.59 ms | **171.32x** | **234.04x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | on | **112.0 μs** | 99.8 μs | 59.33 ms | 599.87 ms | **594.51x** | **6011.25x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | on | **85.1 μs** | 92.5 μs | 31.76 ms | 208.17 ms | **373.05x** | **2445.47x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | on | **363.4 μs** | 192.4 μs | 21.98 ms | 115.57 ms | **114.22x** | **600.61x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **661.9 μs** | 440.8 μs | 39.40 ms | 35.36 ms | **89.38x** | **80.22x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | on | **105.0 μs** | 144.0 μs | 16.61 ms | 3.22 ms | **158.22x** | **30.71x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | on | **89.3 μs** | 87.8 μs | 21.41 ms | 51.74 ms | **243.89x** | **589.41x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | on | **117.3 μs** | 111.9 μs | 35.66 ms | 17.29 ms | **318.78x** | **154.52x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | on | **474.2 μs** | 150.1 μs | 27.81 ms | 22.73 ms | **185.22x** | **151.43x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **202.6 μs** | 205.6 μs | 15.20 ms | 3.42 ms | **75.02x** | **16.87x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | on | **129.5 μs** | 129.4 μs | 37.83 ms | 1.53 ms | **292.42x** | **11.79x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | on | **159.3 μs** | 177.7 μs | 21.83 ms | 11.93 ms | **137.03x** | **74.86x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | on | **126.3 μs** | 108.8 μs | 29.28 ms | 2.12 ms | **269.25x** | **19.50x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | on | **266.4 μs** | 106.1 μs | 9.06 ms | 1.91 ms | **85.40x** | **17.97x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Host | Domain | Case | mmap | torchfits (s) | TF RSS | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| NRC-054711 | fits | tiny_float32_1d [read_full @ mps] | off | 0.0006351249758154154 | 304.375 | fitsio/fitsio_torch_device | 1.2006142943707692 |
| NRC-054711 | fits | repeated_cutouts_50x_100x100 @ mps | off | 0.02180304197827354 | 194.40625 | fitsio/fitsio_torch_device | 1.1115352551874116 |
| NRC-054711 | fits | small_float32_2d [read_full @ mps] | off | 0.0006719169905409217 | 372.71875 | fitsio/fitsio_torch_device | 1.0867308167366083 |
| NRC-054711 | fits | scaled_large [read_full @ mps] | off | 0.006802125019021332 | 378.0625 | fitsio/fitsio_torch_device | 1.0725378260804201 |
| NRC-054711 | fits | medium_int64_3d [read_full] | off | 0.0035763330524787307 | 498.4375 | fitsio/fitsio_torch | 1.0326272476507206 |
| NRC-054711 | fits | large_int32_2d [read_full @ mps] | off | 0.005865000013727695 | 519.46875 | fitsio/fitsio_torch_device | 1.0295947584477227 |
| NRC-054711 | fits | large_int64_2d [read_full] | off | 0.00822691700886935 | 488.953125 | fitsio/fitsio_torch | 1.0177996129264666 |
| NRC-054711 | fits | compressed_hcompress_1 [read_full] | off | 0.03350466600386426 | 433.65625 | fitsio/fitsio_torch | 1.0065919903922145 |
| NRC-054711 | fits | compressed_rice_1 [read_full] | off | 0.008622209017630666 | 433.65625 | fitsio/fitsio_torch | 1.003564653801535 |
| NRC-054711 | fits | large_uint16_2d [read_full] | off | 0.004279457964003086 | 488.96875 | fitsio/fitsio_torch | 1.0030272733735703 |
| NRC-054711 | fits | compressed_hcompress_1 [read_full] | off | 0.034244999988004565 | 433.65625 | fitsio/fitsio | 1.0365467714078571 |
| NRC-054711 | fits | large_float64_2d [read_full] | off | 0.0075950000318698585 | 473.9375 | fitsio/fitsio | 1.0330931251204905 |
| NRC-054711 | fits | mef_medium [header_read] | on | 0.00030700000934302807 | 106.890625 | fitsio/fitsio | 1.0115320196751456 |
| torchfits-gpu-exhaustive-cpu-20260715-002944 | fits | compressed_hcompress_1 [read_full] | off | 0.030163539573550224 | 625.55859375 | fitsio/fitsio_torch | 1.0356104779900446 |
| torchfits-gpu-exhaustive-cpu-20260715-002944 | fits | compressed_hcompress_1 [read_full] | off | 0.03013010136783123 | 625.55859375 | fitsio/fitsio | 1.0322099322729237 |
| torchfits-gpu-exhaustive-cpu-20260715-002944 | fitstable | narrow_10000 [predicate_filter] | off | 0.0004640212282538414 | 690.1640625 | fitsio/fitsio | 1.079257012888552 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | small_uint16_2d [read_full @ cuda] | off | 0.0002897977828979492 | 733.1484375 | fitsio/fitsio_torch_device | 1.4251730123616244 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | tiny_int64_2d [read_full @ cuda] | off | 0.0002505742013454437 | 733.1484375 | fitsio/fitsio_torch_device | 1.3351131401349743 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | large_float64_1d [read_full @ cuda] | off | 0.002565009519457817 | 744.01171875 | fitsio/fitsio_torch_device | 1.2200635335779522 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | tiny_int8_1d [read_full @ cuda] | off | 0.00023667607456445694 | 733.1484375 | fitsio/fitsio_torch_device | 1.209521815075176 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | compressed_hcompress_1 [read_full] | off | 0.030159357003867626 | 751.48828125 | fitsio/fitsio_torch | 1.0388971727896366 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | compressed_hcompress_1 [read_full @ cuda] | off | 0.030490181408822536 | 752.87890625 | fitsio/fitsio_torch_device | 1.0285460266307558 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | large_int8_2d [read_full @ cuda] | off | 0.00119764544069767 | 745.1328125 | fitsio/fitsio_torch_device | 1.0122313591614762 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fits | compressed_hcompress_1 [read_full] | off | 0.030192077159881592 | 751.48828125 | fitsio/fitsio | 1.0433166082758798 |
| torchfits-gpu-exhaustive-cuda-20260715-003158 | fitstable | narrow_10000 [predicate_filter] | off | 0.000464046373963356 | 688.0703125 | fitsio/fitsio | 1.1207138149830072 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest lab benchmarks (one row per host):

| Run ID | Host / device | Scope | Rows | Deficits | Median peak RSS (MB) | Notes |
|---|---|---|---:|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mps_20260715_002839` | NRC-054711 / mps | fits + fitstable (lab) | 3538 | 13 | 305 | lab + mmap-matrix + GPU |
| `exhaustive_cpu_20260715_002944` | torchfits-gpu-exhaustive-cpu-20260715-002944 / cpu | fits + fitstable (lab) | 2754 | 3 | 618 | lab + mmap-matrix |
| `exhaustive_cuda_20260715_003158` | torchfits-gpu-exhaustive-cuda-20260715-003158 / cuda | fits + fitstable (lab) | 3648 | 9 | 733 | lab + mmap-matrix + GPU |
<!-- BENCH_SNAPSHOT_END -->

### Host scorecard

| Host / device | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| NRC-054711 / mps | `exhaustive_mps_20260715_002839` | 3538 | 13 | 305 | lab + mmap-matrix + GPU |
| torchfits-gpu-exhaustive-cpu-20260715-002944 / cpu | `exhaustive_cpu_20260715_002944` | 2754 | 3 | 618 | lab + mmap-matrix |
| torchfits-gpu-exhaustive-cuda-20260715-003158 / cuda | `exhaustive_cuda_20260715_003158` | 3648 | 9 | 733 | lab + mmap-matrix + GPU |
<!-- BENCH_HOSTS_END -->

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
