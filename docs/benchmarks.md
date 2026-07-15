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
| `pixi run bench-cfitsio-direct` | local C | full-suite pure vendored CFITSIO (`--profile full`) |

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

In-container work uses **pixi**; bench CSVs archive to
**`/arc/home/$USER/torchfits-gpu-bench/<run-id>/`** (persistent) and upload to
**`vos:sfabbro/torchfits-gpu-bench/<run-id>/`** via `vcp` (x509 in the session).
Either sink is required before the in-container job exits 0. Platform logs land
under `benchmarks_results/canfar_<run-id>/` locally (poller detaches by default).
Download results with `scripts/fetch_canfar_bench_vos.sh` (needs `pip install vos`
+ cert locally).

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
Source: `benchmarks_results/exhaustive_mps_20260715_190646/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.08 ms` (n=174) | `0.60 ms` (n=174) | `0.19 ms` (n=174) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.16 ms` (n=174) | `1.08 ms` (n=128) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.30 ms` (n=152) | `1.02 ms` (n=152) | `0.48 ms` (n=152) | — |
| `disk→RAM→GPU` | `0.41 ms` (n=152) | `1.82 ms` (n=152) | `34.03 ms` (n=8) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `1.58 ms` (n=180) | `2.57 ms` (n=146) | `1.40 ms` (n=164) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.88 ms` (n=180) | `4.47 ms` (n=146) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |
<!-- BENCH_IOPATH_END -->

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`,
  `disk→CPU→GPU`, `disk→RAM→GPU`).
- Columns are **backends** (`torchfits` / `astropy` / `fitsio` / `cfitsio-direct`).
- Pure-C CFITSIO (vendored): `pixi run bench-cfitsio-direct` runs the **full**
  image+table scorecard fixture set with op→API mapping in
  `benchmarks/cfitsio_direct/bench_cfitsio_direct.c`
  (`fits_read_img` / `fits_read_subset` / `fits_read_record` /
  `fits_read_tblbytes` / `fits_read_col`). CSV:
  `benchmarks_results/<run-id>/cfitsio_direct.csv`.
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **3.72 ms** | 3.70 ms | 8.51 ms | 4.25 ms | **2.30x** | **1.15x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **4.31 ms** | 4.30 ms | 8.88 ms | 4.64 ms | **2.07x** | **1.08x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **14.30 ms** | 14.78 ms | 39.91 ms | 14.30 ms | **2.79x** | **1.00x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **12.02 ms** | 13.08 ms | 34.17 ms | 11.93 ms | **2.84x** | **0.99x** |
| Repeated Cutouts (50x 100x100) | CPU | **9.76 ms** | 9.69 ms | 153.81 ms | 10.29 ms | **15.87x** | **1.06x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **4.75 ms** | 4.66 ms | 4.07 ms | 18.10 ms | **0.87x** | **3.88x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **157.91 ms** | 157.84 ms | 2.75 ms | 217.20 ms | **0.02x** | **1.38x** |
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
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 106.6 μs | 1.96 ms | 265.9 μs | **18.35x** | **2.49x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 94.8 μs | 1.74 ms | 239.5 μs | **18.37x** | **2.53x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 103.3 μs | 1.86 ms | 267.7 μs | **18.03x** | **2.59x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.06 ms** | 1.16 ms | 10.71 ms | 1.33 ms | **10.10x** | **1.26x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 104.2 μs | 1.84 ms | 269.5 μs | **17.63x** | **2.59x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 38.9 μs | 490.1 μs | 89.0 μs | **12.59x** | **2.29x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 28.3 μs | 412.8 μs | 81.5 μs | **14.59x** | **2.88x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 28.1 μs | 371.4 μs | 74.6 μs | **13.21x** | **2.65x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 29.8 μs | 404.3 μs | 75.6 μs | **13.55x** | **2.53x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 25.6 μs | 374.4 μs | 71.4 μs | **14.61x** | **2.79x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 28.8 μs | 397.2 μs | 85.0 μs | **13.78x** | **2.95x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 24.5 μs | 374.6 μs | 72.7 μs | **15.29x** | **2.97x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 27.6 μs | 399.5 μs | 85.4 μs | **14.48x** | **3.10x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 30.7 μs | 416.9 μs | 80.9 μs | **13.60x** | **2.64x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 31.1 μs | 468.9 μs | 92.3 μs | **15.07x** | **2.97x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 32.1 μs | 434.0 μs | 80.5 μs | **13.51x** | **2.50x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 41.5 μs | 538.2 μs | 102.2 μs | **12.97x** | **2.46x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 37.1 μs | 533.8 μs | 97.5 μs | **14.38x** | **2.63x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 35.7 μs | 454.3 μs | 91.5 μs | **12.72x** | **2.56x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 24.6 μs | 377.8 μs | 74.0 μs | **15.37x** | **3.01x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 35.5 μs | 476.2 μs | 94.4 μs | **13.40x** | **2.66x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 39.3 μs | 536.6 μs | 105.3 μs | **13.67x** | **2.68x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 34.8 μs | 478.2 μs | 97.5 μs | **13.76x** | **2.81x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 29.5 μs | 399.2 μs | 76.6 μs | **13.55x** | **2.60x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 32.3 μs | 480.4 μs | 92.4 μs | **14.88x** | **2.86x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 30.1 μs | 413.9 μs | 79.6 μs | **13.74x** | **2.64x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 32.0 μs | 449.6 μs | 88.0 μs | **14.05x** | **2.75x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 29.4 μs | 423.9 μs | 79.7 μs | **14.41x** | **2.71x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 22.9 μs | 366.1 μs | 73.0 μs | **15.97x** | **3.18x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 26.2 μs | 400.0 μs | 80.1 μs | **15.29x** | **3.06x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 33.7 μs | 476.0 μs | 90.8 μs | **14.14x** | **2.70x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 27.1 μs | 418.6 μs | 85.7 μs | **15.43x** | **3.16x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 33.7 μs | 445.8 μs | 88.6 μs | **13.23x** | **2.63x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 29.7 μs | 421.0 μs | 81.2 μs | **14.15x** | **2.73x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 29.0 μs | 425.0 μs | 82.0 μs | **14.68x** | **2.83x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 29.4 μs | 448.6 μs | 82.0 μs | **15.27x** | **2.79x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 32.5 μs | 493.7 μs | 95.5 μs | **15.17x** | **2.94x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 34.7 μs | 473.5 μs | 89.0 μs | **13.66x** | **2.57x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 48.2 μs | 640.7 μs | 115.5 μs | **13.30x** | **2.40x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 38.2 μs | 688.3 μs | 99.1 μs | **18.02x** | **2.59x** |
| fits | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 44.8 μs | 786.5 μs | 114.0 μs | **17.56x** | **2.55x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | n/a | **22.7 μs** | 33.9 μs | 4.66 ms | 313.8 μs | **205.17x** | **13.82x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 46.7 μs | 717.3 μs | 105.4 μs | **15.36x** | **2.26x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **9.47 ms** | 9.54 ms | 13.36 ms | 12.12 ms | **1.41x** | **1.28x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **9.76 ms** | 9.69 ms | 153.81 ms | 10.29 ms | **15.87x** | **1.06x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 30.0 μs | 453.3 μs | 85.7 μs | **15.11x** | **2.86x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 29.9 μs | 450.0 μs | 84.7 μs | **15.06x** | **2.84x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 31.5 μs | 449.7 μs | 87.7 μs | **14.26x** | **2.78x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 32.3 μs | 406.8 μs | 81.1 μs | **12.60x** | **2.51x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 25.5 μs | 393.9 μs | 78.1 μs | **15.45x** | **3.06x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 29.1 μs | 418.7 μs | 79.4 μs | **14.37x** | **2.73x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 26.4 μs | 414.0 μs | 73.9 μs | **15.67x** | **2.80x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 27.6 μs | 426.6 μs | 84.3 μs | **15.44x** | **3.05x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 33.7 μs | 496.5 μs | 92.6 μs | **14.73x** | **2.75x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 26.0 μs | 399.5 μs | 80.9 μs | **15.39x** | **3.12x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 34.4 μs | 479.4 μs | 94.9 μs | **13.93x** | **2.76x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 28.4 μs | 421.6 μs | 79.2 μs | **14.86x** | **2.79x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 29.0 μs | 422.0 μs | 88.8 μs | **14.55x** | **3.06x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 28.3 μs | 385.8 μs | 80.2 μs | **13.64x** | **2.84x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 27.5 μs | 415.0 μs | 84.2 μs | **15.09x** | **3.06x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 26.3 μs | 420.3 μs | 89.0 μs | **15.99x** | **3.38x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 30.9 μs | 457.6 μs | 87.0 μs | **14.82x** | **2.82x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 28.7 μs | 415.0 μs | 82.0 μs | **14.48x** | **2.86x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 33.1 μs | 480.0 μs | 92.2 μs | **14.51x** | **2.79x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 32.7 μs | 449.2 μs | 82.4 μs | **13.75x** | **2.52x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 35.4 μs | 468.3 μs | 86.2 μs | **13.22x** | **2.44x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 37.0 μs | 499.8 μs | 94.4 μs | **13.51x** | **2.55x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 35.9 μs | 514.7 μs | 95.0 μs | **14.35x** | **2.65x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 30.0 μs | 444.7 μs | 85.3 μs | **14.84x** | **2.85x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 28.0 μs | 402.4 μs | 77.8 μs | **14.39x** | **2.78x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 30.2 μs | 443.0 μs | 89.2 μs | **14.64x** | **2.95x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 27.1 μs | 389.5 μs | 76.4 μs | **14.38x** | **2.82x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 33.4 μs | 495.8 μs | 102.6 μs | **14.84x** | **3.07x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 29.3 μs | 407.9 μs | 88.1 μs | **13.90x** | **3.00x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 32.7 μs | 452.2 μs | 88.5 μs | **13.81x** | **2.70x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 29.3 μs | 424.0 μs | 81.4 μs | **14.50x** | **2.78x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 28.2 μs | 420.6 μs | 83.8 μs | **14.91x** | **2.97x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 28.2 μs | 383.4 μs | 77.3 μs | **13.61x** | **2.74x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 35.3 μs | 503.0 μs | 99.5 μs | **14.24x** | **2.81x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 30.6 μs | 423.0 μs | 83.7 μs | **13.81x** | **2.73x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 26.6 μs | 379.9 μs | 73.4 μs | **14.27x** | **2.76x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 27.4 μs | 419.9 μs | 78.9 μs | **15.32x** | **2.88x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 23.5 μs | 353.5 μs | 73.6 μs | **15.04x** | **3.13x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 27.7 μs | 395.2 μs | 77.5 μs | **14.24x** | **2.79x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 32.7 μs | 477.5 μs | 89.2 μs | **14.62x** | **2.73x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 28.3 μs | 412.2 μs | 86.8 μs | **14.55x** | **3.06x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 30.2 μs | 452.8 μs | 89.8 μs | **14.97x** | **2.97x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 29.1 μs | 418.3 μs | 81.5 μs | **14.38x** | **2.80x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 32.2 μs | 484.1 μs | 91.2 μs | **15.01x** | **2.83x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 33.7 μs | 488.5 μs | 90.6 μs | **14.51x** | **2.69x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 35.7 μs | 528.1 μs | 98.0 μs | **14.79x** | **2.75x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **27.47 ms** | 24.27 ms | 66.22 ms | 34.14 ms | **2.73x** | **1.41x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **32.22 ms** | 26.88 ms | 91.11 ms | 39.78 ms | **3.39x** | **1.48x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **47.16 ms** | 47.05 ms | 56.19 ms | 46.77 ms | **1.19x** | **0.99x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **14.30 ms** | 14.78 ms | 39.91 ms | 14.30 ms | **2.79x** | **1.00x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | off | **781.9 μs** | 664.2 μs | 2.36 ms | 944.5 μs | **3.55x** | **1.42x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | off | **3.72 ms** | 3.70 ms | 8.51 ms | 4.25 ms | **2.30x** | **1.15x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | off | **2.21 ms** | 2.15 ms | 4.20 ms | 2.47 ms | **1.96x** | **1.15x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | off | **10.09 ms** | 10.33 ms | 16.81 ms | 10.97 ms | **1.67x** | **1.09x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | off | **473.5 μs** | 505.8 μs | 1.69 ms | 580.2 μs | **3.56x** | **1.23x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | off | **1.91 ms** | 2.00 ms | 5.05 ms | 2.05 ms | **2.65x** | **1.07x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | off | **884.5 μs** | 878.8 μs | 2.70 ms | 1.03 ms | **3.07x** | **1.18x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | off | **4.03 ms** | 4.16 ms | 8.93 ms | 4.45 ms | **2.21x** | **1.10x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | off | **2.30 ms** | 2.41 ms | 4.42 ms | 2.53 ms | **1.92x** | **1.10x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | off | **10.59 ms** | 10.43 ms | 18.65 ms | 11.77 ms | **1.79x** | **1.13x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | off | **194.9 μs** | 195.4 μs | 1.14 ms | 1.20 ms | **5.83x** | **6.16x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | off | **1.20 ms** | 1.42 ms | 3.20 ms | 5.24 ms | **2.66x** | **4.36x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | off | **5.46 ms** | 5.41 ms | 8.48 ms | 5.74 ms | **1.57x** | **1.06x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | off | **7.60 ms** | 7.18 ms | 13.49 ms | 7.95 ms | **1.88x** | **1.11x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | off | **59.3 μs** | 56.0 μs | 513.3 μs | 147.5 μs | **9.17x** | **2.63x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | off | **818.9 μs** | 745.5 μs | 2.35 ms | 948.5 μs | **3.16x** | **1.27x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | off | **1.28 ms** | 1.22 ms | 3.35 ms | 1.38 ms | **2.74x** | **1.13x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | off | **126.5 μs** | 120.8 μs | 602.0 μs | 216.4 μs | **4.98x** | **1.79x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | off | **2.22 ms** | 2.06 ms | 4.22 ms | 2.33 ms | **2.05x** | **1.13x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | off | **3.68 ms** | 3.46 ms | 5.03 ms | 3.83 ms | **1.45x** | **1.11x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | off | **40.4 μs** | 32.6 μs | 450.2 μs | 122.8 μs | **13.80x** | **3.76x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | off | **335.4 μs** | 345.8 μs | 1.32 ms | 447.0 μs | **3.94x** | **1.33x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | off | **604.6 μs** | 556.6 μs | 1.79 ms | 679.6 μs | **3.21x** | **1.22x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | off | **58.6 μs** | 49.4 μs | 463.7 μs | 141.6 μs | **9.38x** | **2.87x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | off | **725.2 μs** | 792.4 μs | 2.25 ms | 942.8 μs | **3.11x** | **1.30x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | off | **1.18 ms** | 1.23 ms | 3.14 ms | 1.38 ms | **2.67x** | **1.17x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | off | **126.2 μs** | 122.1 μs | 588.7 μs | 213.1 μs | **4.82x** | **1.74x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | off | **2.12 ms** | 2.29 ms | 4.18 ms | 2.55 ms | **1.97x** | **1.20x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | off | **3.68 ms** | 3.43 ms | 5.01 ms | 3.77 ms | **1.46x** | **1.10x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | off | **38.2 μs** | 26.2 μs | 607.3 μs | 212.0 μs | **23.17x** | **8.09x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | off | **326.3 μs** | 151.7 μs | 1.64 ms | 1.50 ms | **10.84x** | **9.87x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | off | **413.2 μs** | 350.5 μs | 2.14 ms | 1.82 ms | **6.11x** | **5.18x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **1.10 ms** | 1.26 ms | 2.38 ms | 1.21 ms | **2.17x** | **1.10x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **1.70 ms** | 1.53 ms | 3.62 ms | 1.81 ms | **2.37x** | **1.18x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | off | **167.7 μs** | 153.0 μs | 1.06 ms | 1.13 ms | **6.91x** | **7.35x** |
| fits | mef_small | read_full | 0.45 MB | CPU | off | **32.3 μs** | 25.5 μs | 816.6 μs | 219.2 μs | **32.02x** | **8.59x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **33.6 μs** | 22.5 μs | 813.7 μs | 324.5 μs | **36.23x** | **14.45x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | off | **3.94 ms** | 3.84 ms | 10.45 ms | 4.44 ms | **2.72x** | **1.16x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | off | **968.6 μs** | 1.03 ms | 3.21 ms | 1.09 ms | **3.32x** | **1.13x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | off | **63.3 μs** | 55.8 μs | 659.5 μs | 147.3 μs | **11.81x** | **2.64x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | off | **28.6 μs** | 16.7 μs | 409.8 μs | 108.5 μs | **24.47x** | **6.48x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | off | **45.7 μs** | 37.0 μs | 492.8 μs | 133.3 μs | **13.33x** | **3.61x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | off | **81.2 μs** | 71.2 μs | 575.5 μs | 175.6 μs | **8.09x** | **2.47x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | off | **34.6 μs** | 27.3 μs | 435.2 μs | 115.8 μs | **15.95x** | **4.24x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | off | **89.2 μs** | 87.8 μs | 556.6 μs | 176.6 μs | **6.34x** | **2.01x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | off | **210.9 μs** | 190.2 μs | 778.7 μs | 296.0 μs | **4.09x** | **1.56x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | off | **26.2 μs** | 17.7 μs | 432.1 μs | 109.0 μs | **24.40x** | **6.16x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | off | **38.5 μs** | 29.0 μs | 455.6 μs | 128.8 μs | **15.73x** | **4.45x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | off | **53.1 μs** | 43.2 μs | 555.6 μs | 138.3 μs | **12.87x** | **3.20x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | off | **29.1 μs** | 21.5 μs | 413.5 μs | 104.9 μs | **19.23x** | **4.88x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | off | **45.9 μs** | 37.7 μs | 484.8 μs | 126.6 μs | **12.86x** | **3.36x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | off | **81.0 μs** | 72.0 μs | 538.2 μs | 162.5 μs | **7.48x** | **2.26x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | off | **35.7 μs** | 26.8 μs | 422.2 μs | 119.7 μs | **15.73x** | **4.46x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | off | **95.2 μs** | 86.9 μs | 520.4 μs | 185.1 μs | **5.99x** | **2.13x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | off | **199.5 μs** | 199.0 μs | 878.6 μs | 301.6 μs | **4.41x** | **1.52x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | off | **24.6 μs** | 14.7 μs | 547.0 μs | 110.5 μs | **37.19x** | **7.51x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | off | **30.6 μs** | 22.7 μs | 581.5 μs | 161.3 μs | **25.56x** | **7.09x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | off | **51.6 μs** | 35.2 μs | 699.8 μs | 266.9 μs | **19.88x** | **7.58x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | off | **86.9 μs** | 73.7 μs | 706.7 μs | 170.7 μs | **9.58x** | **2.31x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | off | **97.7 μs** | 86.1 μs | 582.5 μs | 181.4 μs | **6.77x** | **2.11x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **45.5 μs** | 37.9 μs | 526.8 μs | 130.0 μs | **13.91x** | **3.43x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **44.4 μs** | 39.0 μs | 513.5 μs | 128.4 μs | **13.15x** | **3.29x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **46.8 μs** | 35.2 μs | 537.3 μs | 142.2 μs | **15.28x** | **4.04x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **51.7 μs** | 40.5 μs | 478.9 μs | 129.2 μs | **11.84x** | **3.19x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **45.6 μs** | 34.4 μs | 485.6 μs | 137.6 μs | **14.11x** | **4.00x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **21.0 μs** | 10.0 μs | 426.8 μs | 101.6 μs | **42.68x** | **10.16x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **34.1 μs** | 15.7 μs | 529.6 μs | 115.3 μs | **33.81x** | **7.36x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **25.7 μs** | 18.0 μs | 462.9 μs | 110.8 μs | **25.72x** | **6.16x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **23.7 μs** | 11.3 μs | 507.1 μs | 108.5 μs | **44.74x** | **9.58x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **30.5 μs** | 19.4 μs | 460.9 μs | 115.5 μs | **23.79x** | **5.96x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **32.1 μs** | 22.2 μs | 454.6 μs | 110.7 μs | **20.51x** | **4.99x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **27.3 μs** | 11.5 μs | 458.2 μs | 110.0 μs | **39.85x** | **9.57x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **22.7 μs** | 15.2 μs | 433.9 μs | 106.9 μs | **28.61x** | **7.05x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **31.8 μs** | 15.5 μs | 510.0 μs | 108.6 μs | **32.81x** | **6.99x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **21.2 μs** | 13.6 μs | 425.7 μs | 103.7 μs | **31.34x** | **7.64x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **26.6 μs** | 15.4 μs | 440.7 μs | 100.0 μs | **28.59x** | **6.49x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **25.7 μs** | 16.8 μs | 470.2 μs | 105.4 μs | **28.00x** | **6.28x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **23.3 μs** | 10.9 μs | 419.1 μs | 105.1 μs | **38.39x** | **9.63x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **31.5 μs** | 24.1 μs | 448.4 μs | 116.5 μs | **18.59x** | **4.83x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **29.7 μs** | 22.7 μs | 451.2 μs | 108.3 μs | **19.91x** | **4.78x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **26.5 μs** | 13.9 μs | 534.3 μs | 114.3 μs | **38.39x** | **8.21x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **24.1 μs** | 12.3 μs | 591.1 μs | 106.8 μs | **47.93x** | **8.66x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **22.5 μs** | 11.9 μs | 583.7 μs | 108.5 μs | **49.15x** | **9.14x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **32.88 ms** | 36.87 ms | 94.44 ms | 40.49 ms | **2.87x** | **1.23x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **32.49 ms** | 32.43 ms | 151.67 ms | 40.23 ms | **4.68x** | **1.24x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **70.75 ms** | 62.46 ms | 97.23 ms | 74.52 ms | **1.56x** | **1.19x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **17.97 ms** | 18.37 ms | 59.33 ms | 18.66 ms | **3.30x** | **1.04x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | on | **1.38 ms** | 1.29 ms | 3.17 ms | — | **2.46x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | on | **5.60 ms** | 6.39 ms | 11.97 ms | — | **2.14x** | **—** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | on | **2.87 ms** | 2.80 ms | 5.26 ms | — | **1.88x** | **—** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | on | **15.37 ms** | 13.28 ms | 24.38 ms | — | **1.84x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | on | **1.82 ms** | 618.9 μs | 2.70 ms | — | **4.36x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | on | **3.39 ms** | 3.31 ms | 7.83 ms | — | **2.36x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.68 ms** | 1.90 ms | 4.75 ms | — | **2.82x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | on | **6.82 ms** | 6.15 ms | 11.57 ms | — | **1.88x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | on | **3.00 ms** | 2.90 ms | 6.49 ms | — | **2.24x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | on | **13.65 ms** | 13.31 ms | 28.15 ms | — | **2.11x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | on | **498.8 μs** | 256.5 μs | — | — | **—** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | on | **1.65 ms** | 1.51 ms | — | — | **—** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | on | **6.65 ms** | 6.77 ms | — | — | **—** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | on | **9.90 ms** | 9.46 ms | — | — | **—** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | on | **206.4 μs** | 120.4 μs | 1.37 ms | — | **11.37x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | on | **1.39 ms** | 1.57 ms | 3.63 ms | — | **2.61x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | on | **2.61 ms** | 2.28 ms | 17.07 ms | — | **7.48x** | **—** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | on | **235.6 μs** | 225.3 μs | 8.01 ms | — | **35.57x** | **—** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | on | **2.78 ms** | 2.90 ms | 15.42 ms | — | **5.54x** | **—** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | on | **4.62 ms** | 4.24 ms | 7.09 ms | — | **1.67x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | on | **77.2 μs** | 102.6 μs | 750.7 μs | — | **9.73x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | on | **614.5 μs** | 672.2 μs | 6.17 ms | — | **10.04x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | on | **1.16 ms** | 988.5 μs | 7.34 ms | — | **7.42x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | on | **130.2 μs** | 178.0 μs | 978.5 μs | — | **7.51x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | on | **1.27 ms** | 1.50 ms | 6.53 ms | — | **5.13x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | on | **2.39 ms** | 2.17 ms | 9.96 ms | — | **4.59x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | on | **176.5 μs** | 197.3 μs | 1.03 ms | — | **5.83x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | on | **2.95 ms** | 3.15 ms | 5.80 ms | — | **1.97x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | on | **4.40 ms** | 4.60 ms | 11.56 ms | — | **2.63x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | on | **93.3 μs** | 61.8 μs | — | — | **—** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | on | **477.0 μs** | 299.1 μs | — | — | **—** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | on | **639.6 μs** | 613.2 μs | — | — | **—** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **1.82 ms** | 2.03 ms | — | — | **—** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **2.71 ms** | 2.47 ms | — | — | **—** | **—** |
| fits | mef_medium | read_full | 7.02 MB | CPU | on | **356.7 μs** | 324.3 μs | — | — | **—** | **—** |
| fits | mef_small | read_full | 0.45 MB | CPU | on | **94.9 μs** | 94.5 μs | — | — | **—** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **109.5 μs** | 56.3 μs | — | — | **—** | **—** |
| fits | scaled_large | read_full | 8.00 MB | CPU | on | **7.68 ms** | 6.63 ms | — | — | **—** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | on | **1.61 ms** | 1.64 ms | — | — | **—** | **—** |
| fits | scaled_small | read_full | 0.13 MB | CPU | on | **160.0 μs** | 76.4 μs | — | — | **—** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | on | **93.2 μs** | 49.3 μs | 989.6 μs | — | **20.08x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | on | **110.4 μs** | 133.6 μs | 956.2 μs | — | **8.66x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | on | **166.9 μs** | 153.1 μs | 1.12 ms | — | **7.35x** | **—** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | on | **51.5 μs** | 44.0 μs | 549.8 μs | — | **12.51x** | **—** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | on | **152.7 μs** | 150.4 μs | 979.9 μs | — | **6.51x** | **—** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | on | **267.6 μs** | 272.0 μs | 1.27 ms | — | **4.75x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | on | **76.2 μs** | 44.4 μs | 790.5 μs | — | **17.82x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | on | **58.0 μs** | 56.9 μs | 556.8 μs | — | **9.78x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | on | **122.5 μs** | 86.0 μs | 1.32 ms | — | **15.40x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | on | **103.2 μs** | 50.1 μs | 894.0 μs | — | **17.85x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | on | **94.5 μs** | 161.3 μs | 852.1 μs | — | **9.02x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | on | **167.0 μs** | 167.5 μs | 1.22 ms | — | **7.29x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | on | **80.1 μs** | 47.1 μs | 805.1 μs | — | **17.10x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | on | **135.3 μs** | 126.9 μs | 929.1 μs | — | **7.32x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | on | **232.7 μs** | 265.3 μs | 1.57 ms | — | **6.75x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | on | **34.9 μs** | 22.5 μs | — | — | **—** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | on | **46.5 μs** | 29.2 μs | — | — | **—** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | on | **62.1 μs** | 67.5 μs | — | — | **—** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | on | **170.5 μs** | 145.0 μs | — | — | **—** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | on | **181.6 μs** | 161.1 μs | — | — | **—** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **101.3 μs** | 86.0 μs | 928.2 μs | — | **10.79x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **102.1 μs** | 76.1 μs | 777.8 μs | — | **10.22x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **92.3 μs** | 70.2 μs | 687.5 μs | — | **9.80x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **91.8 μs** | 59.1 μs | 750.9 μs | — | **12.71x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **121.7 μs** | 73.9 μs | 1.01 ms | — | **13.72x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **37.5 μs** | 79.0 μs | 477.6 μs | — | **12.72x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **45.0 μs** | 43.9 μs | 523.1 μs | — | **11.92x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **55.2 μs** | 69.1 μs | 743.4 μs | — | **13.46x** | **—** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **55.2 μs** | 60.1 μs | 793.5 μs | — | **14.37x** | **—** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **81.3 μs** | 42.7 μs | 5.81 ms | — | **135.99x** | **—** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **68.3 μs** | 56.3 μs | 725.5 μs | — | **12.89x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **47.0 μs** | 40.3 μs | 627.2 μs | — | **15.57x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **67.3 μs** | 47.4 μs | 957.5 μs | — | **20.19x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **54.7 μs** | 37.9 μs | 692.3 μs | — | **18.28x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **55.5 μs** | 44.0 μs | 629.3 μs | — | **14.32x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **65.5 μs** | 46.0 μs | 794.8 μs | — | **17.29x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **56.6 μs** | 36.7 μs | 729.8 μs | — | **19.86x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **68.7 μs** | 65.3 μs | 819.2 μs | — | **12.54x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **67.7 μs** | 62.6 μs | 880.0 μs | — | **14.05x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **79.1 μs** | 36.3 μs | 916.0 μs | — | **25.24x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **77.4 μs** | 64.5 μs | — | — | **—** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **87.9 μs** | 31.6 μs | — | — | **—** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **79.6 μs** | 52.2 μs | — | — | **—** | **—** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | n/a | **1.36 ms** | 1.35 ms | 11.43 ms | 1.73 ms | **8.45x** | **1.28x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | n/a | **215.8 μs** | 206.1 μs | 5.19 ms | 542.0 μs | **25.17x** | **2.63x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | n/a | **16.51 ms** | 16.85 ms | 169.79 ms | 17.28 ms | **10.29x** | **1.05x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **18.03 ms** | 17.27 ms | 35.55 ms | 22.34 ms | **2.06x** | **1.29x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **21.72 ms** | 21.98 ms | 80.15 ms | 25.69 ms | **3.69x** | **1.18x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **41.45 ms** | 41.24 ms | 49.58 ms | 41.45 ms | **1.20x** | **1.00x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **11.86 ms** | 13.08 ms | 34.62 ms | 13.19 ms | **2.92x** | **1.11x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | off | **1.15 ms** | 1.08 ms | 2.62 ms | 1.38 ms | **2.42x** | **1.28x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | off | **4.31 ms** | 4.30 ms | 8.88 ms | 4.72 ms | **2.07x** | **1.10x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | off | **690.9 μs** | 616.3 μs | 1.64 ms | 807.5 μs | **2.66x** | **1.31x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | off | **2.22 ms** | 2.33 ms | 4.44 ms | 2.48 ms | **2.00x** | **1.12x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | off | **1.20 ms** | 1.14 ms | 2.74 ms | 1.41 ms | **2.41x** | **1.24x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | off | **4.40 ms** | 4.42 ms | 8.92 ms | 4.80 ms | **2.03x** | **1.09x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | off | **2.75 ms** | 2.71 ms | 4.72 ms | 2.75 ms | **1.74x** | **1.02x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | off | **11.03 ms** | 11.02 ms | 17.78 ms | 11.59 ms | **1.61x** | **1.05x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | off | **523.5 μs** | 437.2 μs | 1.18 ms | 1.30 ms | **2.71x** | **2.98x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | off | **1.59 ms** | 1.50 ms | 3.25 ms | 4.89 ms | **2.17x** | **3.26x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | off | **5.47 ms** | 5.61 ms | 8.76 ms | 5.98 ms | **1.60x** | **1.09x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | off | **7.32 ms** | 7.34 ms | 13.42 ms | 8.48 ms | **1.83x** | **1.16x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | off | **256.9 μs** | 295.5 μs | 920.7 μs | 356.0 μs | **3.58x** | **1.39x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | off | **1.27 ms** | 1.24 ms | 3.10 ms | 1.49 ms | **2.51x** | **1.20x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | off | **1.88 ms** | 1.86 ms | 4.20 ms | 2.05 ms | **2.26x** | **1.10x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | off | **258.2 μs** | 227.9 μs | 813.9 μs | 404.9 μs | **3.57x** | **1.78x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | off | **666.5 μs** | 676.7 μs | 1.96 ms | 774.3 μs | **2.94x** | **1.16x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | off | **1.06 ms** | 993.7 μs | 2.48 ms | 1.21 ms | **2.50x** | **1.22x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | off | **284.0 μs** | 253.2 μs | 920.5 μs | 360.9 μs | **3.64x** | **1.43x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | off | **1.25 ms** | 1.10 ms | 2.94 ms | 1.49 ms | **2.66x** | **1.35x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | off | **1.79 ms** | 1.78 ms | 4.13 ms | 2.03 ms | **2.32x** | **1.14x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | off | **445.3 μs** | 387.3 μs | 1.05 ms | 544.8 μs | **2.72x** | **1.41x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | off | **3.10 ms** | 2.90 ms | 5.23 ms | 3.32 ms | **1.80x** | **1.14x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | off | **4.15 ms** | 4.12 ms | 5.83 ms | 4.88 ms | **1.41x** | **1.18x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | off | **230.7 μs** | 228.6 μs | 955.8 μs | 511.5 μs | **4.18x** | **2.24x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | off | **541.9 μs** | 451.6 μs | 1.32 ms | 1.47 ms | **2.93x** | **3.26x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | off | **729.2 μs** | 691.2 μs | 2.04 ms | 2.16 ms | **2.95x** | **3.12x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **1.34 ms** | 1.40 ms | 2.78 ms | 1.52 ms | **2.07x** | **1.13x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **2.11 ms** | 1.92 ms | 4.10 ms | 2.24 ms | **2.14x** | **1.17x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | off | **469.9 μs** | 526.2 μs | 1.38 ms | 1.44 ms | **2.94x** | **3.07x** |
| fits | mef_small | read_full | 0.45 MB | MPS | off | **228.5 μs** | 215.0 μs | 1.23 ms | 451.6 μs | **5.70x** | **2.10x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **222.8 μs** | 202.0 μs | 1.20 ms | 528.9 μs | **5.96x** | **2.62x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | off | **4.44 ms** | 5.26 ms | 10.89 ms | 4.54 ms | **2.45x** | **1.02x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | off | **1.32 ms** | 1.51 ms | 4.08 ms | 1.60 ms | **3.08x** | **1.21x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | off | **296.7 μs** | 265.7 μs | 1.15 ms | 425.7 μs | **4.35x** | **1.60x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | off | **236.2 μs** | 239.8 μs | 748.3 μs | 337.0 μs | **3.17x** | **1.43x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | off | **289.4 μs** | 344.2 μs | 932.0 μs | 380.9 μs | **3.22x** | **1.32x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | off | **510.0 μs** | 467.6 μs | 1.02 ms | 650.0 μs | **2.19x** | **1.39x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | off | **232.7 μs** | 211.8 μs | 764.2 μs | 338.9 μs | **3.61x** | **1.60x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | off | **248.9 μs** | 227.7 μs | 764.6 μs | 393.7 μs | **3.36x** | **1.73x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | off | **259.4 μs** | 259.9 μs | 917.7 μs | 374.4 μs | **3.54x** | **1.44x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | off | **259.9 μs** | 205.2 μs | 774.5 μs | 347.3 μs | **3.77x** | **1.69x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | off | **247.8 μs** | 256.1 μs | 865.4 μs | 347.4 μs | **3.49x** | **1.40x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | off | **314.3 μs** | 317.5 μs | 1.10 ms | 472.9 μs | **3.51x** | **1.50x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | off | **249.8 μs** | 215.0 μs | 792.2 μs | 362.0 μs | **3.68x** | **1.68x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | off | **338.8 μs** | 315.7 μs | 934.5 μs | 489.7 μs | **2.96x** | **1.55x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | off | **633.0 μs** | 604.4 μs | 1.49 ms | 619.6 μs | **2.46x** | **1.03x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | off | **215.5 μs** | 193.2 μs | 931.7 μs | 353.0 μs | **4.82x** | **1.83x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | off | **221.4 μs** | 211.0 μs | 924.9 μs | 433.2 μs | **4.38x** | **2.05x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | off | **272.1 μs** | 270.0 μs | 1.02 ms | 549.9 μs | **3.77x** | **2.04x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | off | **287.5 μs** | 272.0 μs | 910.8 μs | 424.7 μs | **3.35x** | **1.56x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | off | **329.5 μs** | 279.5 μs | 998.4 μs | 420.5 μs | **3.57x** | **1.50x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **274.0 μs** | 260.2 μs | 813.3 μs | 415.2 μs | **3.13x** | **1.60x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **261.1 μs** | 248.5 μs | 861.8 μs | 358.5 μs | **3.47x** | **1.44x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **252.6 μs** | 235.5 μs | 912.8 μs | 369.9 μs | **3.88x** | **1.57x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **301.0 μs** | 271.0 μs | 903.3 μs | 364.4 μs | **3.33x** | **1.34x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **255.1 μs** | 258.0 μs | 848.4 μs | 383.3 μs | **3.33x** | **1.50x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **197.7 μs** | 193.5 μs | 710.6 μs | 323.6 μs | **3.67x** | **1.67x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **195.1 μs** | 192.4 μs | 611.9 μs | 286.9 μs | **3.18x** | **1.49x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **238.1 μs** | 201.7 μs | 824.6 μs | 348.8 μs | **4.09x** | **1.73x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **199.5 μs** | 200.5 μs | 612.8 μs | 328.1 μs | **3.07x** | **1.64x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **210.1 μs** | 162.0 μs | 783.9 μs | 335.4 μs | **4.84x** | **2.07x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **199.7 μs** | 213.9 μs | 801.8 μs | 325.0 μs | **4.01x** | **1.63x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **196.3 μs** | 226.7 μs | 721.7 μs | 335.3 μs | **3.68x** | **1.71x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **227.6 μs** | 212.8 μs | 747.1 μs | 297.9 μs | **3.51x** | **1.40x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **209.4 μs** | 211.7 μs | 783.6 μs | 350.8 μs | **3.74x** | **1.68x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **220.7 μs** | 198.0 μs | 686.5 μs | 316.3 μs | **3.47x** | **1.60x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **229.5 μs** | 215.0 μs | 717.0 μs | 332.3 μs | **3.33x** | **1.55x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **224.5 μs** | 204.3 μs | 740.5 μs | 362.3 μs | **3.62x** | **1.77x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **196.1 μs** | 188.4 μs | 804.3 μs | 328.4 μs | **4.27x** | **1.74x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **204.5 μs** | 193.5 μs | 889.9 μs | 354.1 μs | **4.60x** | **1.83x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **212.1 μs** | 191.6 μs | 960.2 μs | 338.9 μs | **5.01x** | **1.77x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **25.81 ms** | 24.06 ms | 52.42 ms | 31.36 ms | **2.18x** | **1.30x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **30.68 ms** | 28.01 ms | 114.08 ms | 36.73 ms | **4.07x** | **1.31x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **54.17 ms** | 51.50 ms | 67.86 ms | 55.21 ms | **1.32x** | **1.07x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **16.30 ms** | 17.44 ms | 48.05 ms | 19.69 ms | **2.95x** | **1.21x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | on | **1.50 ms** | 1.73 ms | 8.02 ms | — | **5.35x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | on | **5.62 ms** | 5.32 ms | 8.68 ms | — | **1.63x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | on | **849.5 μs** | 786.5 μs | 1.87 ms | — | **2.38x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | on | **3.38 ms** | 3.03 ms | 5.40 ms | — | **1.78x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | on | **1.91 ms** | 1.65 ms | 3.24 ms | — | **1.96x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | on | **5.99 ms** | 6.28 ms | 9.81 ms | — | **1.64x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | on | **3.94 ms** | 3.33 ms | 6.53 ms | — | **1.96x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | on | **12.69 ms** | 12.94 ms | 17.71 ms | — | **1.40x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | on | **639.4 μs** | 642.2 μs | 2.27 ms | — | **3.54x** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | on | **1.94 ms** | 1.78 ms | 4.91 ms | — | **2.76x** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | on | **6.50 ms** | 6.46 ms | 10.82 ms | — | **1.68x** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | on | **8.97 ms** | 10.99 ms | 18.88 ms | — | **2.11x** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | on | **551.3 μs** | 453.5 μs | 1.43 ms | — | **3.15x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | on | **1.88 ms** | 1.80 ms | 3.64 ms | — | **2.03x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | on | **2.54 ms** | 2.42 ms | 5.00 ms | — | **2.06x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | on | **324.6 μs** | 334.8 μs | 1.02 ms | — | **3.13x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | on | **923.4 μs** | 879.3 μs | 2.10 ms | — | **2.38x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | on | **1.58 ms** | 1.29 ms | 2.67 ms | — | **2.07x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | on | **443.3 μs** | 385.7 μs | 998.7 μs | — | **2.59x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | on | **1.65 ms** | 1.62 ms | 3.40 ms | — | **2.10x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | on | **2.37 ms** | 2.53 ms | 4.42 ms | — | **1.87x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | on | **513.0 μs** | 547.2 μs | 1.37 ms | — | **2.66x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | on | **3.06 ms** | 3.03 ms | 5.24 ms | — | **1.73x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | on | **4.43 ms** | 4.27 ms | 5.63 ms | — | **1.32x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | on | **305.5 μs** | 268.9 μs | 1.73 ms | — | **6.43x** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | on | **625.8 μs** | 543.6 μs | 2.14 ms | — | **3.94x** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | on | **905.5 μs** | 944.1 μs | 2.80 ms | — | **3.09x** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **1.82 ms** | 1.78 ms | 4.73 ms | — | **2.65x** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **2.83 ms** | 3.12 ms | 5.62 ms | — | **1.99x** | **—** |
| fits | mef_medium | read_full | 7.02 MB | MPS | on | **586.9 μs** | 523.0 μs | 2.82 ms | — | **5.40x** | **—** |
| fits | mef_small | read_full | 0.45 MB | MPS | on | **286.0 μs** | 255.4 μs | 2.30 ms | — | **9.00x** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **283.1 μs** | 261.6 μs | 2.41 ms | — | **9.21x** | **—** |
| fits | scaled_large | read_full | 8.00 MB | MPS | on | **5.68 ms** | 5.54 ms | 14.97 ms | — | **2.70x** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | on | **1.74 ms** | 1.58 ms | 5.00 ms | — | **3.17x** | **—** |
| fits | scaled_small | read_full | 0.13 MB | MPS | on | **322.4 μs** | 307.0 μs | 2.21 ms | — | **7.21x** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | on | **274.3 μs** | 308.0 μs | 1.30 ms | — | **4.74x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | on | **378.8 μs** | 336.4 μs | 1.40 ms | — | **4.17x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | on | **570.3 μs** | 502.8 μs | 1.26 ms | — | **2.50x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | on | **354.1 μs** | 243.6 μs | 941.8 μs | — | **3.87x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | on | **311.4 μs** | 328.5 μs | 1.15 ms | — | **3.68x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | on | **365.2 μs** | 376.0 μs | 1.25 ms | — | **3.41x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | on | **305.2 μs** | 242.7 μs | 1.27 ms | — | **5.25x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | on | **336.7 μs** | 338.2 μs | 1.25 ms | — | **3.71x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | on | **424.0 μs** | 458.7 μs | 1.42 ms | — | **3.36x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | on | **260.0 μs** | 263.0 μs | 934.4 μs | — | **3.59x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | on | **429.5 μs** | 389.2 μs | 1.29 ms | — | **3.31x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | on | **577.9 μs** | 643.7 μs | 1.43 ms | — | **2.47x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | on | **242.7 μs** | 286.3 μs | 1.69 ms | — | **6.96x** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | on | **271.4 μs** | 254.8 μs | 1.86 ms | — | **7.30x** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | on | **288.3 μs** | 278.7 μs | 1.83 ms | — | **6.57x** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | on | **433.4 μs** | 511.7 μs | 1.85 ms | — | **4.27x** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | on | **403.5 μs** | 398.8 μs | 1.66 ms | — | **4.17x** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **317.3 μs** | 321.3 μs | 1.08 ms | — | **3.39x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **366.1 μs** | 304.0 μs | 1.04 ms | — | **3.43x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **290.0 μs** | 377.0 μs | 919.5 μs | — | **3.17x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **317.6 μs** | 346.5 μs | 1.02 ms | — | **3.23x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **314.0 μs** | 342.3 μs | 1.51 ms | — | **4.82x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **256.5 μs** | 246.6 μs | 1.03 ms | — | **4.19x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **226.1 μs** | 257.2 μs | 976.0 μs | — | **4.32x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **260.8 μs** | 226.8 μs | 934.0 μs | — | **4.12x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **242.9 μs** | 256.2 μs | 904.6 μs | — | **3.72x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **247.7 μs** | 260.1 μs | 837.7 μs | — | **3.38x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **275.8 μs** | 226.7 μs | 975.6 μs | — | **4.30x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **242.5 μs** | 246.6 μs | 780.9 μs | — | **3.22x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **341.3 μs** | 331.6 μs | 1.13 ms | — | **3.41x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **327.8 μs** | 340.3 μs | 1.12 ms | — | **3.42x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **272.6 μs** | 312.1 μs | 996.4 μs | — | **3.65x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **349.3 μs** | 310.6 μs | 1.24 ms | — | **4.00x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **340.2 μs** | 252.8 μs | 1.06 ms | — | **4.19x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **239.3 μs** | 280.0 μs | 1.82 ms | — | **7.61x** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **260.0 μs** | 265.5 μs | 1.80 ms | — | **6.92x** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **245.9 μs** | 232.3 μs | 1.89 ms | — | **8.14x** | **—** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **748.8 μs** | 2.26 ms | 6.05 ms | 4.61 ms | **8.08x** | **6.16x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | off | **1.81 ms** | 1.88 ms | 16.31 ms | 4.60 ms | **9.00x** | **2.54x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.69 ms** | 1.86 ms | 1.80 ms | 4.46 ms | **1.06x** | **2.64x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | off | **298.9 μs** | 286.0 μs | 2.22 ms | 1.05 ms | **7.76x** | **3.66x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | off | **54.2 μs** | 474.8 μs | 535.0 μs | 123.2 μs | **9.87x** | **2.27x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **203.0 μs** | 393.9 μs | 2.58 ms | 599.7 μs | **12.69x** | **2.95x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | off | **197.0 μs** | 235.9 μs | 2.56 ms | 523.8 μs | **13.00x** | **2.66x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | off | **236.2 μs** | 239.0 μs | 1.36 ms | 473.8 μs | **5.75x** | **2.01x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | off | **163.3 μs** | 170.7 μs | 1.90 ms | 292.2 μs | **11.64x** | **1.79x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | off | **52.5 μs** | 151.0 μs | 540.8 μs | 123.3 μs | **10.30x** | **2.35x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **28.55 ms** | 54.45 ms | 89.81 ms | 196.82 ms | **3.15x** | **6.89x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | off | **74.83 ms** | 64.76 ms | 14.10 ms | 93.44 ms | **0.22x** | **1.44x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | off | **28.03 ms** | 26.90 ms | 15.08 ms | 176.04 ms | **0.56x** | **6.54x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **2.36 ms** | 2.05 ms | 16.21 ms | 3.15 ms | **7.91x** | **1.54x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **74.1 μs** | 25.37 ms | 994.5 μs | 236.3 μs | **13.42x** | **3.19x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **3.06 ms** | 8.01 ms | 13.15 ms | 19.22 ms | **4.29x** | **6.27x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | off | **7.49 ms** | 8.02 ms | 4.10 ms | 11.71 ms | **0.55x** | **1.56x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | off | **4.75 ms** | 4.66 ms | 4.07 ms | 18.10 ms | **0.87x** | **3.88x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | off | **2.09 ms** | 2.32 ms | 4.97 ms | 3.07 ms | **2.38x** | **1.47x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | off | **68.0 μs** | 2.39 ms | 681.5 μs | 190.8 μs | **10.02x** | **2.81x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **573.0 μs** | 3.36 ms | 4.69 ms | 2.38 ms | **8.19x** | **4.15x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | off | **931.1 μs** | 1.23 ms | 2.67 ms | 1.41 ms | **2.86x** | **1.51x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | off | **2.10 ms** | 1.78 ms | 2.58 ms | 2.01 ms | **1.44x** | **1.13x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | off | **1.68 ms** | 1.65 ms | 3.74 ms | 601.0 μs | **2.27x** | **0.36x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | off | **72.3 μs** | 608.4 μs | 675.0 μs | 178.1 μs | **9.34x** | **2.47x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **210.8 μs** | 1.83 ms | 3.91 ms | 522.4 μs | **18.53x** | **2.48x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | off | **216.8 μs** | 280.4 μs | 2.51 ms | 377.9 μs | **11.58x** | **1.74x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | off | **1.59 ms** | 1.53 ms | 2.51 ms | 458.3 μs | **1.64x** | **0.30x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | off | **1.53 ms** | 1.58 ms | 3.58 ms | 323.3 μs | **2.34x** | **0.21x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | off | **81.6 μs** | 210.2 μs | 590.8 μs | 145.0 μs | **7.24x** | **1.78x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **16.22 ms** | 22.05 ms | 48.85 ms | 15.41 ms | **3.01x** | **0.95x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | off | **38.55 ms** | 38.99 ms | 7.47 ms | 58.09 ms | **0.19x** | **1.51x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | off | **9.16 ms** | 9.72 ms | 7.04 ms | 4.95 ms | **0.77x** | **0.54x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **1.56 ms** | 1.43 ms | 7.03 ms | 1.06 ms | **4.91x** | **0.74x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **70.7 μs** | 11.50 ms | 669.5 μs | 246.0 μs | **9.47x** | **3.48x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.65 ms** | 4.02 ms | 7.02 ms | 1.73 ms | **4.26x** | **1.05x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | off | **3.87 ms** | 4.01 ms | 2.50 ms | 5.99 ms | **0.65x** | **1.55x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | off | **2.33 ms** | 2.34 ms | 2.32 ms | 547.3 μs | **1.00x** | **0.23x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | off | **1.57 ms** | 1.55 ms | 2.63 ms | 1.07 ms | **1.70x** | **0.69x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | off | **53.2 μs** | 1.30 ms | 656.0 μs | 136.7 μs | **12.33x** | **2.57x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **416.2 μs** | 2.14 ms | 3.09 ms | 442.2 μs | **7.43x** | **1.06x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | off | **472.2 μs** | 468.0 μs | 1.97 ms | 809.1 μs | **4.21x** | **1.73x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | off | **1.50 ms** | 1.61 ms | 1.75 ms | 258.7 μs | **1.16x** | **0.17x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | off | **1.47 ms** | 1.47 ms | 2.57 ms | 325.4 μs | **1.74x** | **0.22x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | off | **51.8 μs** | 215.3 μs | 552.6 μs | 125.3 μs | **10.66x** | **2.42x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **150.8 μs** | 1.53 ms | 2.39 ms | 258.3 μs | **15.83x** | **1.71x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | off | **177.1 μs** | 193.6 μs | 1.68 ms | 262.3 μs | **9.50x** | **1.48x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | off | **1.45 ms** | 1.46 ms | 1.82 ms | 247.3 μs | **1.25x** | **0.17x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | off | **1.43 ms** | 1.50 ms | 2.34 ms | 265.5 μs | **1.64x** | **0.19x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | off | **51.7 μs** | 168.4 μs | 644.6 μs | 136.6 μs | **12.47x** | **2.64x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **2.09 ms** | 13.50 ms | 5.80 ms | 30.06 ms | **2.78x** | **14.40x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | off | **7.41 ms** | 8.37 ms | 55.92 ms | 26.12 ms | **7.54x** | **3.52x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | off | **11.03 ms** | 10.85 ms | 2.87 ms | 29.11 ms | **0.26x** | **2.68x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | off | **1.24 ms** | 1.13 ms | 2.76 ms | 3.45 ms | **2.44x** | **3.04x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | off | **66.5 μs** | 1.46 ms | 588.5 μs | 177.8 μs | **8.85x** | **2.67x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **468.9 μs** | 1.63 ms | 2.59 ms | 2.92 ms | **5.53x** | **6.22x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | off | **1.02 ms** | 1.08 ms | 7.38 ms | 2.86 ms | **7.27x** | **2.82x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | off | **1.13 ms** | 1.29 ms | 2.02 ms | 3.11 ms | **1.78x** | **2.74x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | off | **269.8 μs** | 247.8 μs | 2.19 ms | 619.1 μs | **8.82x** | **2.50x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | off | **53.4 μs** | 290.2 μs | 566.9 μs | 129.9 μs | **10.62x** | **2.43x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **1.72 ms** | 155.45 ms | 6.28 ms | 210.40 ms | **3.65x** | **122.23x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | off | **160.50 ms** | 157.32 ms | 889.79 ms | 215.65 ms | **5.66x** | **1.37x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | off | **157.91 ms** | 157.84 ms | 2.75 ms | 217.20 ms | **0.02x** | **1.38x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | off | **13.76 ms** | 14.63 ms | 3.19 ms | 23.00 ms | **0.23x** | **1.67x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | off | **57.0 μs** | 1.31 ms | 622.4 μs | 133.8 μs | **10.93x** | **2.35x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **356.1 μs** | 15.14 ms | 2.93 ms | 22.17 ms | **8.23x** | **62.26x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | off | **14.60 ms** | 14.63 ms | 92.56 ms | 22.41 ms | **6.34x** | **1.53x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | off | **14.86 ms** | 14.70 ms | 2.06 ms | 22.21 ms | **0.14x** | **1.51x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | off | **1.50 ms** | 1.53 ms | 2.33 ms | 2.49 ms | **1.56x** | **1.66x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | off | **57.9 μs** | 215.1 μs | 591.3 μs | 127.8 μs | **10.22x** | **2.21x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **129.6 μs** | 1.62 ms | 2.35 ms | 2.49 ms | **18.11x** | **19.21x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | off | **1.28 ms** | 1.38 ms | 9.78 ms | 2.20 ms | **7.62x** | **1.71x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | off | **1.53 ms** | 1.50 ms | 1.68 ms | 2.39 ms | **1.12x** | **1.59x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | off | **329.7 μs** | 292.0 μs | 2.00 ms | 475.0 μs | **6.86x** | **1.63x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | off | **53.2 μs** | 144.7 μs | 550.9 μs | 123.8 μs | **10.35x** | **2.32x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **10.20 ms** | 39.82 ms | 54.77 ms | 87.67 ms | **5.37x** | **8.60x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | off | **26.25 ms** | 26.56 ms | 13.62 ms | 13.52 ms | **0.52x** | **0.52x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | off | **23.80 ms** | 23.11 ms | 12.81 ms | 83.80 ms | **0.55x** | **3.63x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | off | **4.53 ms** | 4.33 ms | 18.52 ms | 10.62 ms | **4.28x** | **2.45x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | off | **188.4 μs** | 9.66 ms | 1.04 ms | 589.8 μs | **5.50x** | **3.13x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **2.10 ms** | 9.51 ms | 17.32 ms | 10.05 ms | **8.23x** | **4.77x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | off | **3.52 ms** | 3.82 ms | 9.27 ms | 2.12 ms | **2.63x** | **0.60x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | off | **4.75 ms** | 4.61 ms | 9.58 ms | 9.11 ms | **2.08x** | **1.98x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | off | **2.99 ms** | 2.96 ms | 14.16 ms | 1.81 ms | **4.79x** | **0.61x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | off | **185.1 μs** | 1.93 ms | 835.7 μs | 494.2 μs | **4.52x** | **2.67x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **1.02 ms** | 3.68 ms | 13.95 ms | 1.76 ms | **13.71x** | **1.73x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | off | **1.04 ms** | 1.29 ms | 8.67 ms | 794.8 μs | **8.31x** | **0.76x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | off | **2.93 ms** | 2.89 ms | 8.78 ms | 1.63 ms | **3.04x** | **0.57x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | off | **2.86 ms** | 2.76 ms | 13.37 ms | 940.0 μs | **4.84x** | **0.34x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | off | **202.7 μs** | 1.16 ms | 885.6 μs | 497.3 μs | **4.37x** | **2.45x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **550.5 μs** | 996.8 μs | 6.73 ms | — | **12.22x** | **—** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | on | **447.5 μs** | 621.1 μs | 17.28 ms | — | **38.63x** | **—** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | on | **382.4 μs** | 412.6 μs | 1.97 ms | — | **5.14x** | **—** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | on | **286.4 μs** | 264.3 μs | 2.54 ms | — | **9.62x** | **—** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | on | **79.5 μs** | 424.1 μs | 1.24 ms | — | **15.55x** | **—** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **317.2 μs** | 400.5 μs | 3.18 ms | — | **10.03x** | **—** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | on | **288.5 μs** | 237.3 μs | 3.14 ms | — | **13.22x** | **—** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | on | **261.5 μs** | 212.4 μs | 1.62 ms | — | **7.64x** | **—** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | on | **236.2 μs** | 241.7 μs | 2.22 ms | — | **9.41x** | **—** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | on | **62.3 μs** | 204.3 μs | 618.9 μs | — | **9.94x** | **—** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **12.73 ms** | 83.44 ms | 130.80 ms | — | **10.27x** | **—** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | on | **20.12 ms** | 20.42 ms | 15.65 ms | — | **0.78x** | **—** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | on | **31.92 ms** | 35.58 ms | 18.19 ms | — | **0.57x** | **—** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **638.3 μs** | 818.2 μs | 15.00 ms | — | **23.50x** | **—** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **205.2 μs** | 9.73 ms | 1.31 ms | — | **6.37x** | **—** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **2.24 ms** | 10.26 ms | 20.99 ms | — | **9.37x** | **—** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | on | **2.35 ms** | 2.74 ms | 8.24 ms | — | **3.50x** | **—** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | on | **7.54 ms** | 4.93 ms | 15.38 ms | — | **3.12x** | **—** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | on | **1.19 ms** | 1.27 ms | 8.37 ms | — | **7.06x** | **—** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | on | **149.9 μs** | 1.46 ms | 1.17 ms | — | **7.81x** | **—** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **929.3 μs** | 3.37 ms | 9.13 ms | — | **9.83x** | **—** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | on | **680.2 μs** | 663.4 μs | 4.45 ms | — | **6.70x** | **—** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | on | **926.8 μs** | 808.9 μs | 4.60 ms | — | **5.68x** | **—** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | on | **673.7 μs** | 575.0 μs | 6.79 ms | — | **11.81x** | **—** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | on | **124.8 μs** | 763.9 μs | 1.60 ms | — | **12.79x** | **—** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **423.9 μs** | 987.6 μs | 6.33 ms | — | **14.94x** | **—** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | on | **641.7 μs** | 566.6 μs | 5.18 ms | — | **9.15x** | **—** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | on | **535.1 μs** | 546.1 μs | 4.61 ms | — | **8.62x** | **—** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | on | **469.7 μs** | 537.9 μs | 6.09 ms | — | **12.96x** | **—** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | on | **126.2 μs** | 391.9 μs | 912.5 μs | — | **7.23x** | **—** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **6.99 ms** | 22.35 ms | 57.77 ms | — | **8.27x** | **—** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | on | **7.77 ms** | 7.25 ms | 8.91 ms | — | **1.23x** | **—** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | on | **9.49 ms** | 9.95 ms | 8.84 ms | — | **0.93x** | **—** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **597.3 μs** | 454.3 μs | 8.42 ms | — | **18.54x** | **—** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **86.9 μs** | 2.59 ms | 741.8 μs | — | **8.54x** | **—** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.42 ms** | 4.19 ms | 11.32 ms | — | **7.95x** | **—** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | on | **1.11 ms** | 1.74 ms | 5.45 ms | — | **4.90x** | **—** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | on | **1.35 ms** | 1.68 ms | 4.72 ms | — | **3.49x** | **—** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | on | **607.9 μs** | 551.0 μs | 6.07 ms | — | **11.02x** | **—** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | on | **114.8 μs** | 966.1 μs | 1.43 ms | — | **12.46x** | **—** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **761.5 μs** | 1.25 ms | 4.77 ms | — | **6.26x** | **—** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | on | **792.7 μs** | 646.3 μs | 4.71 ms | — | **7.29x** | **—** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | on | **507.9 μs** | 532.6 μs | 3.38 ms | — | **6.65x** | **—** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | on | **514.5 μs** | 425.9 μs | 3.98 ms | — | **9.34x** | **—** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | on | **98.5 μs** | 464.6 μs | 1.05 ms | — | **10.66x** | **—** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **247.1 μs** | 577.3 μs | 4.71 ms | — | **19.05x** | **—** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | on | **316.1 μs** | 593.0 μs | 4.30 ms | — | **13.59x** | **—** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | on | **385.9 μs** | 330.3 μs | 2.71 ms | — | **8.20x** | **—** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | on | **391.9 μs** | 378.0 μs | 3.51 ms | — | **9.29x** | **—** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | on | **74.0 μs** | 260.5 μs | 818.8 μs | — | **11.06x** | **—** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **1.01 ms** | 4.60 ms | 5.50 ms | — | **5.41x** | **—** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | on | **2.25 ms** | 2.96 ms | 70.88 ms | — | **31.49x** | **—** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | on | **2.66 ms** | 2.61 ms | 2.44 ms | — | **0.94x** | **—** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | on | **468.5 μs** | 501.7 μs | 2.93 ms | — | **6.25x** | **—** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | on | **56.4 μs** | 749.2 μs | 928.8 μs | — | **16.48x** | **—** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **413.0 μs** | 1.35 ms | 3.19 ms | — | **7.72x** | **—** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | on | **456.5 μs** | 841.9 μs | 7.80 ms | — | **17.10x** | **—** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | on | **505.8 μs** | 682.6 μs | 2.12 ms | — | **4.18x** | **—** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | on | **306.8 μs** | 312.0 μs | 2.67 ms | — | **8.71x** | **—** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | on | **66.1 μs** | 263.8 μs | 720.8 μs | — | **10.90x** | **—** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **1.10 ms** | 170.36 ms | 10.02 ms | — | **9.13x** | **—** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | on | **188.67 ms** | 183.67 ms | 1.076 s | — | **5.86x** | **—** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | on | **194.35 ms** | 219.55 ms | 7.63 ms | — | **0.04x** | **—** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | on | **18.95 ms** | 17.29 ms | 7.02 ms | — | **0.41x** | **—** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | on | **74.0 μs** | 431.8 μs | 786.3 μs | — | **10.63x** | **—** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **375.5 μs** | 18.41 ms | 3.73 ms | — | **9.92x** | **—** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | on | **17.36 ms** | 17.68 ms | 108.69 ms | — | **6.26x** | **—** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | on | **15.15 ms** | 15.66 ms | 2.36 ms | — | **0.16x** | **—** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | on | **1.94 ms** | 1.88 ms | 2.84 ms | — | **1.51x** | **—** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | on | **72.2 μs** | 274.4 μs | 812.1 μs | — | **11.25x** | **—** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **198.0 μs** | 1.84 ms | 2.75 ms | — | **13.89x** | **—** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | on | **2.16 ms** | 2.23 ms | 14.47 ms | — | **6.71x** | **—** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | on | **2.11 ms** | 2.34 ms | 3.75 ms | — | **1.78x** | **—** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | on | **544.4 μs** | 497.4 μs | 2.77 ms | — | **5.56x** | **—** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | on | **59.1 μs** | 220.5 μs | 686.0 μs | — | **11.60x** | **—** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **6.10 ms** | 74.43 ms | 97.61 ms | — | **16.00x** | **—** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | on | **8.90 ms** | 9.67 ms | 20.24 ms | — | **2.27x** | **—** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | on | **39.49 ms** | 61.18 ms | 28.12 ms | — | **0.71x** | **—** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | on | **5.31 ms** | 6.13 ms | 38.60 ms | — | **7.27x** | **—** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | on | **320.4 μs** | 5.14 ms | 1.85 ms | — | **5.77x** | **—** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **2.81 ms** | 17.41 ms | 39.30 ms | — | **13.98x** | **—** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | on | **3.01 ms** | 2.57 ms | 15.69 ms | — | **6.10x** | **—** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | on | **5.35 ms** | 7.33 ms | 21.96 ms | — | **4.11x** | **—** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | on | **2.47 ms** | 2.52 ms | 35.88 ms | — | **14.55x** | **—** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | on | **402.8 μs** | 2.26 ms | 2.17 ms | — | **5.38x** | **—** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **2.12 ms** | 4.36 ms | 31.04 ms | — | **14.67x** | **—** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | on | **2.02 ms** | 2.02 ms | 14.82 ms | — | **7.35x** | **—** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | on | **2.14 ms** | 2.15 ms | 13.85 ms | — | **6.47x** | **—** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | on | **2.37 ms** | 1.93 ms | 24.12 ms | — | **12.52x** | **—** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | on | **334.8 μs** | 1.90 ms | 1.57 ms | — | **4.69x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Host | Domain | Case | mmap | torchfits (s) | TF RSS | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| NRC-054711 | fits | compressed_hcompress_1 [read_full] | off | 0.0471648329985328 | 325.859375 | fitsio/fitsio_torch | 1.0083432064589932 |
| NRC-054711 | fits | compressed_rice_1 [read_full @ mps] | off | 0.01307562500005588 | 311.78125 | fitsio/fitsio_torch_device_specialized | 1.0962471582795557 |
| NRC-054711 | fits | compressed_rice_1 [read_full @ mps] | on | 0.017443790973629802 | 458.6875 | fitsio/fitsio_torch_device_specialized | 1.0628086995674286 |
| NRC-054711 | fits | medium_uint16_2d [read_full] | off | 0.0012560419854708016 | 512.765625 | fitsio/fitsio_torch | 1.0400204219318014 |
| NRC-054711 | fits | compressed_rice_1 [read_full] | off | 0.014776125026401132 | 334.75 | fitsio/fitsio_torch | 1.0333104860192202 |
| NRC-054711 | fits | compressed_hcompress_1 [read_full @ mps] | on | 0.05149650003295392 | 458.671875 | fitsio/fitsio_torch_device_specialized | 1.0246080234424852 |
| NRC-054711 | fits | scaled_large [read_full @ mps] | off | 0.005257375014480203 | 520.453125 | fitsio/fitsio_torch_device_specialized | 1.007747237607526 |
| NRC-054711 | fits | compressed_hcompress_1 [read_full] | off | 0.04705062502762303 | 329.421875 | fitsio/fitsio_torch | 1.005901539134656 |
| NRC-054711 | fitstable | varlen_100000 [read_full] | off | 0.15790583298075944 | 1024.15625 | astropy/astropy_torch | 41.55416650972298 |
| NRC-054711 | fitstable | varlen_100000 [read_full] | on | 0.1943470829864964 | 713.484375 | astropy/astropy_torch | 23.761234686746306 |
| NRC-054711 | fitstable | varlen_10000 [read_full] | off | 0.014855290995910764 | 982.78125 | astropy/astropy_torch | 6.5907577784536375 |
| NRC-054711 | fitstable | varlen_10000 [read_full] | on | 0.015153541986364871 | 676.5625 | astropy/astropy_torch | 6.331014056611165 |
| NRC-054711 | fitstable | narrow_1000 [read_full] | off | 0.0014535000082105398 | 762.703125 | fitsio/fitsio_torch | 5.721517618380204 |
| NRC-054711 | fitstable | narrow_1000 [row_slice] | off | 0.001432958000805229 | 762.890625 | fitsio/fitsio_torch | 5.615764954823922 |
| NRC-054711 | fitstable | varlen_100000 [row_slice] | on | 0.01895416696788743 | 693.859375 | astropy/astropy_torch | 5.277998009004249 |
| NRC-054711 | fitstable | narrow_10000 [read_full] | off | 0.0015029169735498726 | 765.59375 | fitsio/fitsio_torch | 4.312530346064135 |
| NRC-054711 | fitstable | varlen_100000 [row_slice] | off | 0.013756792002823204 | 988.5625 | astropy/astropy_torch | 4.168882748590832 |
| NRC-054711 | fitstable | narrow_10000 [row_slice] | off | 0.0014749589608982205 | 765.609375 | fitsio/fitsio_torch | 4.051150495250535 |
| NRC-054711 | fitstable | mixed_1000 [row_slice] | off | 0.0015301249804906547 | 764.46875 | fitsio/fitsio_torch | 3.457909612248839 |
| NRC-054711 | fitstable | typed_100000 [read_full] | off | 0.011032541980966926 | 989.421875 | astropy/astropy_torch | 2.7179880980630946 |
| NRC-054711 | fitstable | wide_1000 [row_slice] | off | 0.002862041990738362 | 765.546875 | fitsio/fitsio_torch | 1.8855584181152862 |
| NRC-054711 | fitstable | wide_100000 [projection] | off | 0.02625012496719137 | 778.46875 | fitsio/fitsio_torch | 1.8233526946401555 |
| NRC-054711 | fitstable | narrow_100000 [read_full] | off | 0.002332791977096349 | 767.640625 | fitsio/fitsio_torch | 1.7870665131793335 |
| NRC-054711 | fitstable | wide_10000 [projection] | off | 0.003517041972372681 | 766.09375 | fitsio/fitsio_torch | 1.774528135205274 |
| NRC-054711 | fitstable | wide_1000 [projection] | off | 0.00104320899117738 | 765.515625 | fitsio/fitsio_torch | 1.4977874821187989 |
| NRC-054711 | fitstable | narrow_1000000 [row_slice] | off | 0.0015568750095553696 | 790.40625 | fitsio/fitsio_torch | 1.281466095723117 |
| NRC-054711 | fitstable | narrow_100000 [row_slice] | off | 0.0015732910251244903 | 767.640625 | fitsio/fitsio_torch | 1.2225272303464665 |
| NRC-054711 | fitstable | mixed_1000 [read_full] | off | 0.0015851669595576823 | 764.390625 | fitsio/fitsio_torch | 1.1994447390489344 |
| NRC-054711 | fitstable | mixed_10000 [row_slice] | off | 0.0016787920030765235 | 765.921875 | fitsio/fitsio_torch | 1.0772708625834568 |
| NRC-054711 | fitstable | ascii_10000 [predicate_filter] | off | 0.0007487910334020853 | 989.46875 | fitsio/fitsio_torch | 1.071231768157979 |
| NRC-054711 | fitstable | mixed_1000000 [scan_count] | off | 0.025373084004968405 | 975.5625 | fitsio/fitsio | 107.38019828223712 |
| NRC-054711 | fitstable | varlen_100000 [read_full] | off | 0.15783641702728346 | 1018.96875 | astropy/astropy | 57.39679225895189 |
| NRC-054711 | fitstable | narrow_1000000 [scan_count] | off | 0.011500749969854951 | 790.40625 | fitsio/fitsio | 46.743039752598136 |
| NRC-054711 | fitstable | varlen_100000 [read_full] | on | 0.2195492500322871 | 716.3125 | astropy/astropy | 28.758144352088014 |
| NRC-054711 | fitstable | varlen_100000 [predicate_filter] | off | 0.15544524998404086 | 1019.15625 | astropy/astropy | 24.764585846181944 |
| NRC-054711 | fitstable | varlen_100000 [predicate_filter] | on | 0.1703586250077933 | 725.328125 | astropy/astropy | 16.99373155809504 |
| NRC-054711 | fitstable | wide_100000 [scan_count] | off | 0.009660584037192166 | 790.40625 | fitsio/fitsio | 16.38081203115568 |
| NRC-054711 | fitstable | mixed_100000 [scan_count] | off | 0.002387500018812716 | 778.234375 | fitsio/fitsio | 12.510874017806298 |
| NRC-054711 | fitstable | varlen_100000 [scan_count] | off | 0.0013087500119581819 | 989.078125 | fitsio/fitsio | 9.78204768998378 |
| NRC-054711 | fitstable | narrow_100000 [scan_count] | off | 0.0013049159897491336 | 767.640625 | fitsio/fitsio | 9.542347724708184 |

_…and 61 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest lab benchmarks (one row per host):

| Run ID | Host / device | Scope | Rows | Deficits | Median peak RSS (MB) | Notes |
|---|---|---|---:|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mps_20260715_190646` | NRC-054711 / mps | fits + fitstable (lab) | 3569 | 101 | 513 | lab + mmap-matrix + GPU |
<!-- BENCH_SNAPSHOT_END -->

### Host scorecard

| Host / device | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| NRC-054711 / mps | `exhaustive_mps_20260715_190646` | 3569 | 101 | 513 | lab + mmap-matrix + GPU |
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
