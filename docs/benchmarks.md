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
   staging (`exhaustive_cuda_20260717_042840`, via `pixi run bench-exhaustive-canfar-cuda`).
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
Source: `benchmarks_results/exhaustive_mps_20260717_040150/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.44 ms` (n=174) | `1.54 ms` (n=253) | `0.73 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.89 ms` (n=174) | `6.35 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.63 ms` (n=152) | `2.45 ms` (n=152) | `1.07 ms` (n=152) | — |
| `disk→RAM→GPU` | `0.97 ms` (n=152) | `4.61 ms` (n=152) | `41.43 ms` (n=8) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.86 ms` (n=180) | `7.82 ms` (n=164) | `2.27 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.59 ms` (n=180) | `10.26 ms` (n=164) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **6.51 ms** | 6.88 ms | 13.95 ms | 8.83 ms | **2.14x** | **1.36x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **9.00 ms** | 8.95 ms | 14.44 ms | 11.30 ms | **1.61x** | **1.26x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **15.17 ms** | 14.47 ms | 75.41 ms | 18.45 ms | **5.21x** | **1.28x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **13.24 ms** | 16.56 ms | 53.56 ms | 14.18 ms | **4.04x** | **1.07x** |
| Repeated Cutouts (50x 100x100) | CPU | **21.75 ms** | 18.34 ms | 335.26 ms | 21.03 ms | **18.28x** | **1.15x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **6.97 ms** | 7.87 ms | 95.60 ms | 30.20 ms | **13.72x** | **4.33x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **258.11 ms** | 252.70 ms | 1.624 s | 337.40 ms | **6.43x** | **1.34x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

Aggregated wins across every domain and operation in the CANFAR CUDA exhaustive
(`exhaustive_cuda_20260717_042840`, 4,079 rows, **0** TorchFits deficits).
Category ranges below are the last regenerated aggregation shape; for this
run’s absolute times prefer [Performance highlights](#performance-highlights)
and the full table.

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
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 170.5 μs | 3.61 ms | 439.9 μs | **21.16x** | **2.58x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 186.0 μs | 4.50 ms | 575.0 μs | **24.21x** | **3.09x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 217.5 μs | 5.16 ms | 591.2 μs | **23.73x** | **2.72x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.10 ms** | 1.21 ms | 21.26 ms | 2.32 ms | **19.33x** | **2.11x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 207.7 μs | 4.51 ms | 555.9 μs | **21.72x** | **2.68x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 63.2 μs | 1.00 ms | 214.9 μs | **15.90x** | **3.40x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 85.1 μs | 940.2 μs | 203.0 μs | **11.05x** | **2.38x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 102.5 μs | 759.7 μs | 236.1 μs | **7.41x** | **2.30x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 77.7 μs | 1.11 ms | 197.2 μs | **14.24x** | **2.54x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 41.9 μs | 593.8 μs | 117.5 μs | **14.17x** | **2.80x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 60.8 μs | 868.3 μs | 178.5 μs | **14.27x** | **2.93x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 62.9 μs | 950.4 μs | 184.9 μs | **15.12x** | **2.94x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 61.2 μs | 939.7 μs | 190.5 μs | **15.36x** | **3.12x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 51.8 μs | 801.2 μs | 151.2 μs | **15.47x** | **2.92x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 46.1 μs | 680.1 μs | 140.4 μs | **14.76x** | **3.05x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 53.4 μs | 790.6 μs | 134.3 μs | **14.81x** | **2.52x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 68.6 μs | 1.09 ms | 189.0 μs | **15.91x** | **2.76x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 67.1 μs | 1.12 ms | 183.9 μs | **16.74x** | **2.74x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 58.9 μs | 834.1 μs | 165.6 μs | **14.17x** | **2.81x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 41.0 μs | 569.2 μs | 122.7 μs | **13.90x** | **3.00x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 64.7 μs | 1.05 ms | 169.7 μs | **16.28x** | **2.62x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 78.2 μs | 1.15 ms | 201.1 μs | **14.69x** | **2.57x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 57.5 μs | 630.0 μs | 153.1 μs | **10.97x** | **2.66x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 60.1 μs | 762.8 μs | 160.0 μs | **12.69x** | **2.66x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 75.8 μs | 1.04 ms | 224.2 μs | **13.71x** | **2.96x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 63.0 μs | 953.5 μs | 189.7 μs | **15.12x** | **3.01x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 49.5 μs | 982.5 μs | 178.8 μs | **19.85x** | **3.61x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 67.9 μs | 897.8 μs | 177.0 μs | **13.23x** | **2.61x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 55.9 μs | 785.7 μs | 144.2 μs | **14.06x** | **2.58x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 72.6 μs | 861.7 μs | 174.5 μs | **11.87x** | **2.40x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 58.6 μs | 870.2 μs | 174.5 μs | **14.85x** | **2.98x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 67.0 μs | 942.1 μs | 177.0 μs | **14.05x** | **2.64x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 58.1 μs | 849.5 μs | 165.8 μs | **14.63x** | **2.85x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 45.2 μs | 691.6 μs | 139.8 μs | **15.30x** | **3.09x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 50.4 μs | 711.0 μs | 148.7 μs | **14.10x** | **2.95x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 60.5 μs | 878.2 μs | 167.2 μs | **14.52x** | **2.76x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 72.1 μs | 997.5 μs | 161.4 μs | **13.84x** | **2.24x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 70.4 μs | 1.05 ms | 166.2 μs | **14.91x** | **2.36x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 79.0 μs | 1.09 ms | 196.7 μs | **13.83x** | **2.49x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 100.7 μs | 1.53 ms | 204.9 μs | **15.19x** | **2.03x** |
| fits | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 88.7 μs | 1.48 ms | 230.9 μs | **16.73x** | **2.60x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | n/a | **46.8 μs** | 35.3 μs | 8.43 ms | 674.0 μs | **239.08x** | **19.12x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 88.5 μs | 1.26 ms | 231.5 μs | **14.27x** | **2.62x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **35.04 ms** | 33.54 ms | 65.87 ms | 54.57 ms | **1.96x** | **1.63x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **21.75 ms** | 18.34 ms | 335.26 ms | 21.03 ms | **18.28x** | **1.15x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 54.0 μs | 742.9 μs | 144.0 μs | **13.75x** | **2.67x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 75.1 μs | 1.34 ms | 222.6 μs | **17.89x** | **2.97x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 56.5 μs | 1.08 ms | 158.9 μs | **19.20x** | **2.81x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 54.0 μs | 739.2 μs | 136.9 μs | **13.70x** | **2.54x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 61.3 μs | 868.5 μs | 163.4 μs | **14.17x** | **2.67x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 70.8 μs | 830.4 μs | 209.4 μs | **11.72x** | **2.96x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 55.6 μs | 703.9 μs | 128.2 μs | **12.65x** | **2.30x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 62.7 μs | 935.1 μs | 313.4 μs | **14.92x** | **5.00x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 59.5 μs | 928.5 μs | 146.2 μs | **15.60x** | **2.45x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 62.7 μs | 834.6 μs | 161.1 μs | **13.31x** | **2.57x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 70.0 μs | 985.4 μs | 207.1 μs | **14.07x** | **2.96x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 56.9 μs | 755.8 μs | 199.0 μs | **13.28x** | **3.50x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 51.9 μs | 727.0 μs | 147.2 μs | **14.00x** | **2.84x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 52.1 μs | 711.8 μs | 153.3 μs | **13.67x** | **2.94x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 56.1 μs | 967.7 μs | 161.0 μs | **17.25x** | **2.87x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 52.3 μs | 663.8 μs | 135.5 μs | **12.70x** | **2.59x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 43.7 μs | 660.2 μs | 131.4 μs | **15.11x** | **3.01x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 58.5 μs | 940.0 μs | 155.4 μs | **16.07x** | **2.66x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 75.2 μs | 1.19 ms | 223.9 μs | **15.79x** | **2.98x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 56.4 μs | 732.3 μs | 144.2 μs | **12.99x** | **2.56x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 72.0 μs | 1.19 ms | 172.5 μs | **16.51x** | **2.40x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 82.1 μs | 1.70 ms | 344.5 μs | **20.77x** | **4.20x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 73.7 μs | 1.06 ms | 196.7 μs | **14.41x** | **2.67x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 54.6 μs | 845.8 μs | 143.0 μs | **15.48x** | **2.62x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 82.4 μs | 930.7 μs | 170.9 μs | **11.29x** | **2.07x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 55.0 μs | 820.6 μs | 158.0 μs | **14.92x** | **2.87x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 62.7 μs | 1.06 ms | 202.2 μs | **16.92x** | **3.22x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 93.6 μs | 780.8 μs | 215.6 μs | **8.34x** | **2.30x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 65.4 μs | 821.6 μs | 149.5 μs | **12.56x** | **2.29x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 55.8 μs | 712.6 μs | 150.1 μs | **12.77x** | **2.69x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 49.6 μs | 700.9 μs | 130.6 μs | **14.14x** | **2.63x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 55.9 μs | 787.4 μs | 170.1 μs | **14.08x** | **3.04x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 48.0 μs | 736.1 μs | 149.8 μs | **15.34x** | **3.12x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 57.8 μs | 980.0 μs | 169.5 μs | **16.96x** | **2.93x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 63.2 μs | 850.5 μs | 186.3 μs | **13.46x** | **2.95x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 52.3 μs | 735.4 μs | 141.5 μs | **14.06x** | **2.71x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 87.3 μs | 1.25 ms | 233.9 μs | **14.37x** | **2.68x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 64.8 μs | 726.7 μs | 160.8 μs | **11.21x** | **2.48x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 48.5 μs | 1.09 ms | 134.9 μs | **22.37x** | **2.78x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 44.8 μs | 694.1 μs | 136.4 μs | **15.50x** | **3.05x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 59.1 μs | 757.9 μs | 148.1 μs | **12.82x** | **2.51x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 49.2 μs | 764.0 μs | 145.5 μs | **15.53x** | **2.96x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 47.4 μs | 713.2 μs | 174.7 μs | **15.04x** | **3.68x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 59.1 μs | 790.9 μs | 175.2 μs | **13.38x** | **2.96x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 61.3 μs | 982.9 μs | 182.3 μs | **16.05x** | **2.98x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 75.1 μs | 1.01 ms | 271.6 μs | **13.42x** | **3.62x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **25.67 ms** | 26.11 ms | 229.89 ms | 55.13 ms | **8.95x** | **2.15x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **27.91 ms** | 26.97 ms | 118.25 ms | 32.82 ms | **4.39x** | **1.22x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **61.27 ms** | 63.41 ms | 92.15 ms | 53.54 ms | **1.50x** | **0.87x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **15.17 ms** | 14.47 ms | 75.41 ms | 18.45 ms | **5.21x** | **1.28x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | off | **1.80 ms** | 3.14 ms | 4.37 ms | 2.41 ms | **2.43x** | **1.34x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | off | **6.51 ms** | 6.88 ms | 13.95 ms | 8.83 ms | **2.14x** | **1.36x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | off | **5.51 ms** | 6.27 ms | 10.24 ms | 5.88 ms | **1.86x** | **1.07x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | off | **13.51 ms** | 13.50 ms | 28.23 ms | 21.19 ms | **2.09x** | **1.57x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | off | **1.13 ms** | 1.39 ms | 2.86 ms | 843.1 μs | **2.52x** | **0.74x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | off | **3.16 ms** | 3.37 ms | 8.02 ms | 4.44 ms | **2.54x** | **1.41x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | off | **2.11 ms** | 1.58 ms | 6.13 ms | 3.03 ms | **3.88x** | **1.92x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | off | **7.33 ms** | 7.82 ms | 15.11 ms | 8.77 ms | **2.06x** | **1.20x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | off | **4.34 ms** | 3.79 ms | 7.12 ms | 4.54 ms | **1.88x** | **1.20x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | off | **14.17 ms** | 14.80 ms | 37.63 ms | 24.80 ms | **2.66x** | **1.75x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | off | **962.6 μs** | 1.16 ms | 1.86 ms | 1.66 ms | **1.93x** | **1.73x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | off | **1.59 ms** | 1.62 ms | 6.25 ms | 6.71 ms | **3.93x** | **4.22x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | off | **6.57 ms** | 6.53 ms | 14.31 ms | 8.52 ms | **2.19x** | **1.31x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | off | **10.10 ms** | 10.20 ms | 23.68 ms | 14.67 ms | **2.34x** | **1.45x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | off | **329.7 μs** | 329.6 μs | 966.9 μs | 228.1 μs | **2.93x** | **0.69x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | off | **2.04 ms** | 1.76 ms | 3.76 ms | 1.80 ms | **2.14x** | **1.02x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | off | **3.19 ms** | 3.19 ms | 6.59 ms | 3.53 ms | **2.07x** | **1.11x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | off | **496.8 μs** | 284.9 μs | 1.62 ms | 609.8 μs | **5.67x** | **2.14x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | off | **4.20 ms** | 3.67 ms | 11.47 ms | 9.05 ms | **3.12x** | **2.46x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | off | **5.78 ms** | 6.06 ms | 10.76 ms | 7.38 ms | **1.86x** | **1.28x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | off | **341.9 μs** | 327.0 μs | 1.29 ms | 523.5 μs | **3.93x** | **1.60x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | off | **957.1 μs** | 1.06 ms | 3.76 ms | 1.34 ms | **3.93x** | **1.40x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | off | **1.26 ms** | 1.44 ms | 4.75 ms | 2.91 ms | **3.78x** | **2.32x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | off | **437.7 μs** | 710.9 μs | 2.00 ms | 496.7 μs | **4.57x** | **1.13x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | off | **2.09 ms** | 2.40 ms | 4.73 ms | 2.63 ms | **2.26x** | **1.26x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | off | **3.95 ms** | 4.25 ms | 8.25 ms | 5.01 ms | **2.09x** | **1.27x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | off | **748.9 μs** | 1.25 ms | 1.28 ms | 646.0 μs | **1.70x** | **0.86x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | off | **3.62 ms** | 3.95 ms | 9.27 ms | 4.90 ms | **2.56x** | **1.36x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | off | **6.04 ms** | 6.66 ms | 10.93 ms | 7.25 ms | **1.81x** | **1.20x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | off | **220.5 μs** | 276.0 μs | 1.14 ms | 314.3 μs | **5.16x** | **1.42x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | off | **496.2 μs** | 765.1 μs | 2.02 ms | 2.03 ms | **4.07x** | **4.09x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | off | **1.06 ms** | 786.5 μs | 2.34 ms | 2.48 ms | **2.97x** | **3.16x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **2.15 ms** | 1.77 ms | 5.02 ms | 2.11 ms | **2.83x** | **1.19x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **3.10 ms** | 2.79 ms | 8.55 ms | 3.53 ms | **3.06x** | **1.26x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | off | **800.5 μs** | 503.4 μs | 2.38 ms | 1.47 ms | **4.73x** | **2.92x** |
| fits | mef_small | read_full | 0.45 MB | CPU | off | **412.9 μs** | 1.31 ms | 2.91 ms | 707.8 μs | **7.04x** | **1.71x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **387.4 μs** | 377.5 μs | 3.40 ms | 1.11 ms | **9.00x** | **2.95x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | off | **8.17 ms** | 7.61 ms | 19.15 ms | 7.16 ms | **2.52x** | **0.94x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | off | **1.54 ms** | 1.54 ms | 6.05 ms | 2.05 ms | **3.93x** | **1.33x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | off | **202.8 μs** | 198.6 μs | 1.25 ms | 261.6 μs | **6.28x** | **1.32x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | off | **738.0 μs** | 301.8 μs | 637.0 μs | 191.4 μs | **2.11x** | **0.63x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | off | **234.1 μs** | 205.3 μs | 1.13 ms | 310.8 μs | **5.53x** | **1.51x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | off | **658.2 μs** | 529.2 μs | 1.16 ms | 339.5 μs | **2.19x** | **0.64x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | off | **718.8 μs** | 369.0 μs | 723.0 μs | 164.6 μs | **1.96x** | **0.45x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | off | **215.5 μs** | 279.0 μs | 1.26 ms | 299.6 μs | **5.85x** | **1.39x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | off | **985.0 μs** | 1.24 ms | 2.48 ms | 729.1 μs | **2.52x** | **0.74x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | off | **195.0 μs** | 255.0 μs | 669.5 μs | 168.4 μs | **3.43x** | **0.86x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | off | **242.7 μs** | 210.7 μs | 1.23 ms | 377.5 μs | **5.86x** | **1.79x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | off | **277.6 μs** | 341.6 μs | 1.71 ms | 396.7 μs | **6.17x** | **1.43x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | off | **193.8 μs** | 222.6 μs | 786.9 μs | 224.2 μs | **4.06x** | **1.16x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | off | **279.4 μs** | 285.7 μs | 1.47 ms | 349.6 μs | **5.27x** | **1.25x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | off | **425.7 μs** | 449.9 μs | 3.00 ms | 1.58 ms | **7.05x** | **3.71x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | off | **636.1 μs** | 435.9 μs | 918.7 μs | 428.2 μs | **2.11x** | **0.98x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | off | **421.0 μs** | 461.2 μs | 1.22 ms | 419.5 μs | **2.91x** | **1.00x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | off | **467.7 μs** | 708.1 μs | 1.86 ms | 710.8 μs | **3.97x** | **1.52x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | off | **129.3 μs** | 146.5 μs | 946.3 μs | 164.8 μs | **7.32x** | **1.27x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | off | **358.0 μs** | 180.9 μs | 1.43 ms | 387.8 μs | **7.92x** | **2.14x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | off | **351.2 μs** | 351.2 μs | 1.11 ms | 375.3 μs | **3.17x** | **1.07x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | off | **397.7 μs** | 248.8 μs | 2.04 ms | 504.4 μs | **8.19x** | **2.03x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | off | **347.0 μs** | 1.04 ms | 1.31 ms | 253.9 μs | **3.78x** | **0.73x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **272.5 μs** | 260.5 μs | 1.46 ms | 279.2 μs | **5.60x** | **1.07x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **283.3 μs** | 412.2 μs | 1.21 ms | 465.5 μs | **4.26x** | **1.64x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **301.3 μs** | 327.3 μs | 1.17 ms | 233.8 μs | **3.89x** | **0.78x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **262.1 μs** | 389.1 μs | 886.1 μs | 274.7 μs | **3.38x** | **1.05x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **168.7 μs** | 201.8 μs | 1.01 ms | 231.6 μs | **5.96x** | **1.37x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **108.5 μs** | 112.6 μs | 641.4 μs | 180.6 μs | **5.91x** | **1.66x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **160.7 μs** | 139.0 μs | 868.9 μs | 229.0 μs | **6.25x** | **1.65x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **136.0 μs** | 155.4 μs | 606.9 μs | 137.5 μs | **4.46x** | **1.01x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **133.9 μs** | 152.0 μs | 744.7 μs | 164.3 μs | **5.56x** | **1.23x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **135.5 μs** | 143.2 μs | 738.5 μs | 162.8 μs | **5.45x** | **1.20x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **214.8 μs** | 170.1 μs | 771.7 μs | 227.0 μs | **4.54x** | **1.33x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **128.3 μs** | 123.4 μs | 697.2 μs | 289.1 μs | **5.65x** | **2.34x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **105.7 μs** | 134.7 μs | 805.7 μs | 283.6 μs | **7.62x** | **2.68x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **134.8 μs** | 170.6 μs | 669.0 μs | 139.5 μs | **4.96x** | **1.03x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **126.8 μs** | 151.3 μs | 594.4 μs | 154.2 μs | **4.69x** | **1.22x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **281.6 μs** | 132.0 μs | 667.1 μs | 169.0 μs | **5.05x** | **1.28x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **177.8 μs** | 126.2 μs | 1.14 ms | 266.5 μs | **9.00x** | **2.11x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **140.5 μs** | 245.5 μs | 654.1 μs | 140.7 μs | **4.66x** | **1.00x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **122.9 μs** | 117.7 μs | 744.8 μs | 169.0 μs | **6.33x** | **1.44x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **181.8 μs** | 263.4 μs | 877.1 μs | 440.5 μs | **4.82x** | **2.42x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **106.9 μs** | 107.1 μs | 1.17 ms | 221.2 μs | **10.97x** | **2.07x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **174.5 μs** | 127.5 μs | 781.5 μs | 160.5 μs | **6.13x** | **1.26x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **125.6 μs** | 128.5 μs | 1.18 ms | 249.7 μs | **9.40x** | **1.99x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **49.34 ms** | 53.51 ms | 113.29 ms | 49.86 ms | **2.30x** | **1.01x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **38.71 ms** | 41.56 ms | 156.19 ms | 39.77 ms | **4.03x** | **1.03x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **72.29 ms** | 68.58 ms | 92.84 ms | 65.46 ms | **1.35x** | **0.95x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **16.12 ms** | 16.55 ms | 77.93 ms | 16.27 ms | **4.83x** | **1.01x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | on | **2.91 ms** | 3.46 ms | 11.10 ms | — | **3.81x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | on | **8.53 ms** | 6.40 ms | 20.62 ms | — | **3.22x** | **—** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | on | **3.74 ms** | 3.50 ms | 11.21 ms | — | **3.20x** | **—** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | on | **15.69 ms** | 13.78 ms | 35.99 ms | — | **2.61x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | on | **1.09 ms** | 1.16 ms | 7.26 ms | — | **6.64x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | on | **3.01 ms** | 3.27 ms | 12.78 ms | — | **4.24x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.77 ms** | 1.82 ms | 7.34 ms | — | **4.14x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | on | **9.16 ms** | 13.88 ms | 20.66 ms | — | **2.26x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | on | **3.58 ms** | 3.90 ms | 26.18 ms | — | **7.31x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | on | **18.08 ms** | 16.10 ms | 60.05 ms | — | **3.73x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | on | **909.7 μs** | 972.7 μs | — | — | **—** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | on | **3.09 ms** | 4.82 ms | — | — | **—** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | on | **10.54 ms** | 6.60 ms | — | — | **—** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | on | **10.43 ms** | 11.02 ms | — | — | **—** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | on | **369.4 μs** | 485.0 μs | 6.30 ms | — | **17.04x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | on | **2.02 ms** | 2.06 ms | 11.49 ms | — | **5.69x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | on | **3.09 ms** | 2.93 ms | 10.38 ms | — | **3.54x** | **—** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | on | **861.0 μs** | 671.8 μs | 6.25 ms | — | **9.30x** | **—** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | on | **4.83 ms** | 4.32 ms | 12.07 ms | — | **2.79x** | **—** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | on | **7.78 ms** | 7.43 ms | 15.92 ms | — | **2.14x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | on | **345.3 μs** | 563.3 μs | 14.03 ms | — | **40.64x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | on | **3.05 ms** | 1.94 ms | 8.44 ms | — | **4.35x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | on | **1.67 ms** | 1.64 ms | 11.86 ms | — | **7.21x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | on | **718.0 μs** | 420.4 μs | 7.63 ms | — | **18.15x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | on | **2.24 ms** | 3.89 ms | 12.06 ms | — | **5.38x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | on | **3.74 ms** | 4.00 ms | 11.84 ms | — | **3.16x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | on | **502.5 μs** | 830.0 μs | 11.44 ms | — | **22.76x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | on | **6.46 ms** | 4.76 ms | 11.91 ms | — | **2.50x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | on | **6.63 ms** | 5.81 ms | 8.65 ms | — | **1.49x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | on | **423.6 μs** | 1.05 ms | — | — | **—** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | on | **1.48 ms** | 1.59 ms | — | — | **—** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | on | **1.99 ms** | 1.34 ms | — | — | **—** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **2.75 ms** | 2.46 ms | — | — | **—** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **5.03 ms** | 2.79 ms | — | — | **—** | **—** |
| fits | mef_medium | read_full | 7.02 MB | CPU | on | **2.29 ms** | 1.96 ms | — | — | **—** | **—** |
| fits | mef_small | read_full | 0.45 MB | CPU | on | **1.05 ms** | 1.12 ms | — | — | **—** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **1.40 ms** | 1.47 ms | — | — | **—** | **—** |
| fits | scaled_large | read_full | 8.00 MB | CPU | on | **15.16 ms** | 19.11 ms | — | — | **—** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | on | **3.24 ms** | 4.16 ms | — | — | **—** | **—** |
| fits | scaled_small | read_full | 0.13 MB | CPU | on | **805.2 μs** | 385.2 μs | — | — | **—** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | on | **1.82 ms** | 1.33 ms | 1.46 ms | — | **1.10x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | on | **751.0 μs** | 954.7 μs | 2.26 ms | — | **3.01x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | on | **1.41 ms** | 1.08 ms | 4.66 ms | — | **4.29x** | **—** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | on | **1.87 ms** | 664.2 μs | 2.07 ms | — | **3.12x** | **—** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | on | **2.94 ms** | 3.13 ms | 16.41 ms | — | **5.57x** | **—** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | on | **1.13 ms** | 1.54 ms | 7.30 ms | — | **6.47x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | on | **206.0 μs** | 294.0 μs | 5.89 ms | — | **28.58x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | on | **306.0 μs** | 519.3 μs | 6.45 ms | — | **21.06x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | on | **267.0 μs** | 489.3 μs | 961.0 μs | — | **3.60x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | on | **451.3 μs** | 336.9 μs | 2.05 ms | — | **6.09x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | on | **296.0 μs** | 190.9 μs | 6.80 ms | — | **35.62x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | on | **556.3 μs** | 373.0 μs | 9.32 ms | — | **24.99x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | on | **269.9 μs** | 323.4 μs | 1.12 ms | — | **4.14x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | on | **1.99 ms** | 1.07 ms | 5.98 ms | — | **5.57x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | on | **856.8 μs** | 563.6 μs | 1.67 ms | — | **2.96x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | on | **329.5 μs** | 149.2 μs | — | — | **—** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | on | **519.5 μs** | 346.6 μs | — | — | **—** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | on | **375.0 μs** | 357.5 μs | — | — | **—** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | on | **437.0 μs** | 594.8 μs | — | — | **—** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | on | **374.5 μs** | 564.7 μs | — | — | **—** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **329.7 μs** | 243.9 μs | 2.06 ms | — | **8.43x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **501.7 μs** | 466.2 μs | 1.12 ms | — | **2.39x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **421.2 μs** | 618.1 μs | 1.41 ms | — | **3.34x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **439.6 μs** | 345.5 μs | 1.87 ms | — | **5.40x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **521.5 μs** | 588.2 μs | 9.24 ms | — | **17.73x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **167.9 μs** | 361.2 μs | 2.55 ms | — | **15.20x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **269.2 μs** | 361.5 μs | 996.2 μs | — | **3.70x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **339.0 μs** | 213.2 μs | 951.7 μs | — | **4.46x** | **—** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **228.5 μs** | 252.8 μs | 6.99 ms | — | **30.58x** | **—** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **312.0 μs** | 184.1 μs | 962.2 μs | — | **5.23x** | **—** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **235.8 μs** | 184.4 μs | 1.16 ms | — | **6.28x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **129.5 μs** | 121.9 μs | 715.2 μs | — | **5.87x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **174.8 μs** | 356.9 μs | 934.9 μs | — | **5.35x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **182.7 μs** | 256.2 μs | 1.26 ms | — | **6.89x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **134.0 μs** | 196.1 μs | 723.5 μs | — | **5.40x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **337.3 μs** | 202.5 μs | 1.01 ms | — | **4.99x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **300.0 μs** | 260.2 μs | 1.19 ms | — | **4.55x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **177.6 μs** | 226.0 μs | 813.3 μs | — | **4.58x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **193.3 μs** | 150.9 μs | 1.17 ms | — | **7.74x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **276.4 μs** | 417.7 μs | 8.03 ms | — | **29.05x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **236.3 μs** | 230.0 μs | — | — | **—** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **143.7 μs** | 156.3 μs | — | — | **—** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **162.7 μs** | 289.5 μs | — | — | **—** | **—** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | n/a | **1.73 ms** | 1.61 ms | 20.06 ms | 2.58 ms | **12.49x** | **1.61x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | n/a | **360.2 μs** | 629.2 μs | 9.29 ms | 1.15 ms | **25.80x** | **3.19x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | n/a | **33.64 ms** | 33.69 ms | 289.12 ms | 33.93 ms | **8.59x** | **1.01x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **36.77 ms** | 26.53 ms | 73.54 ms | 40.54 ms | **2.77x** | **1.53x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **27.38 ms** | 25.81 ms | 135.72 ms | 36.68 ms | **5.26x** | **1.42x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **55.61 ms** | 55.08 ms | 88.37 ms | 52.02 ms | **1.60x** | **0.94x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **16.20 ms** | 16.56 ms | 55.77 ms | 16.23 ms | **3.44x** | **1.00x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | off | **3.55 ms** | 2.09 ms | 4.36 ms | 4.22 ms | **2.08x** | **2.02x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | off | **8.71 ms** | 8.95 ms | 17.74 ms | 11.84 ms | **2.04x** | **1.36x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | off | **1.64 ms** | 1.44 ms | 4.22 ms | 4.43 ms | **2.93x** | **3.08x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | off | **4.90 ms** | 4.74 ms | 10.32 ms | 7.32 ms | **2.18x** | **1.54x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | off | **2.13 ms** | 2.23 ms | 5.78 ms | 2.77 ms | **2.71x** | **1.30x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | off | **8.69 ms** | 9.85 ms | 18.06 ms | 11.63 ms | **2.08x** | **1.34x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | off | **4.72 ms** | 5.27 ms | 8.30 ms | 7.92 ms | **1.76x** | **1.68x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | off | **18.19 ms** | 19.20 ms | 32.62 ms | 21.66 ms | **1.79x** | **1.19x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | off | **1.31 ms** | 1.36 ms | 3.26 ms | 2.54 ms | **2.49x** | **1.94x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | off | **2.91 ms** | 3.05 ms | 6.97 ms | 8.55 ms | **2.40x** | **2.94x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | off | **8.09 ms** | 8.13 ms | 20.25 ms | 12.08 ms | **2.50x** | **1.49x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | off | **11.79 ms** | 11.35 ms | 23.18 ms | 13.23 ms | **2.04x** | **1.17x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | off | **617.7 μs** | 633.3 μs | 2.16 ms | 1.04 ms | **3.49x** | **1.69x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | off | **3.24 ms** | 3.05 ms | 6.45 ms | 5.27 ms | **2.12x** | **1.73x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | off | **3.69 ms** | 3.62 ms | 9.81 ms | 5.21 ms | **2.71x** | **1.44x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | off | **663.0 μs** | 443.1 μs | 1.77 ms | 1.38 ms | **4.00x** | **3.11x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | off | **1.13 ms** | 1.55 ms | 4.52 ms | 2.50 ms | **3.99x** | **2.21x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | off | **2.11 ms** | 2.01 ms | 5.44 ms | 4.15 ms | **2.71x** | **2.07x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | off | **773.5 μs** | 576.1 μs | 1.99 ms | 1.16 ms | **3.45x** | **2.01x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | off | **2.58 ms** | 3.14 ms | 7.41 ms | 4.64 ms | **2.87x** | **1.80x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | off | **3.83 ms** | 3.60 ms | 6.21 ms | 6.10 ms | **1.73x** | **1.70x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | off | **537.1 μs** | 576.2 μs | 7.86 ms | 959.7 μs | **14.63x** | **1.79x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | off | **5.21 ms** | 4.92 ms | 7.92 ms | 8.42 ms | **1.61x** | **1.71x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | off | **7.13 ms** | 5.61 ms | 12.77 ms | 12.83 ms | **2.28x** | **2.29x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | off | **433.1 μs** | 402.9 μs | 1.86 ms | 1.01 ms | **4.62x** | **2.51x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | off | **1.13 ms** | 1.10 ms | 3.30 ms | 2.98 ms | **2.99x** | **2.70x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | off | **1.09 ms** | 2.91 ms | 5.22 ms | 2.88 ms | **4.79x** | **2.64x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **2.55 ms** | 2.13 ms | 6.39 ms | 4.00 ms | **2.99x** | **1.87x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **3.70 ms** | 3.60 ms | 8.48 ms | 4.51 ms | **2.36x** | **1.25x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | off | **1.23 ms** | 1.01 ms | 3.47 ms | 2.66 ms | **3.42x** | **2.62x** |
| fits | mef_small | read_full | 0.45 MB | MPS | off | **451.7 μs** | 368.6 μs | 2.99 ms | 826.0 μs | **8.12x** | **2.24x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **541.2 μs** | 385.3 μs | 2.01 ms | 1.34 ms | **5.20x** | **3.47x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | off | **8.24 ms** | 9.67 ms | 21.15 ms | 8.25 ms | **2.57x** | **1.00x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | off | **2.02 ms** | 2.67 ms | 7.49 ms | 2.23 ms | **3.70x** | **1.10x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | off | **433.0 μs** | 349.9 μs | 2.39 ms | 601.9 μs | **6.83x** | **1.72x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | off | **308.5 μs** | 377.9 μs | 1.30 ms | 641.7 μs | **4.20x** | **2.08x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | off | **380.1 μs** | 511.1 μs | 2.15 ms | 551.0 μs | **5.67x** | **1.45x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | off | **866.8 μs** | 1.16 ms | 2.91 ms | 1.59 ms | **3.35x** | **1.84x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | off | **403.8 μs** | 301.8 μs | 1.44 ms | 779.3 μs | **4.77x** | **2.58x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | off | **533.5 μs** | 448.1 μs | 1.66 ms | 733.9 μs | **3.71x** | **1.64x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | off | **561.8 μs** | 633.6 μs | 2.22 ms | 953.2 μs | **3.95x** | **1.70x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | off | **397.0 μs** | 437.1 μs | 1.55 ms | 681.5 μs | **3.91x** | **1.72x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | off | **561.4 μs** | 544.2 μs | 2.03 ms | 972.0 μs | **3.72x** | **1.79x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | off | **916.3 μs** | 560.1 μs | 2.38 ms | 1.19 ms | **4.25x** | **2.13x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | off | **279.2 μs** | 905.6 μs | 1.36 ms | 473.4 μs | **4.88x** | **1.70x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | off | **1.73 ms** | 727.3 μs | 2.47 ms | 1.39 ms | **3.40x** | **1.91x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | off | **1.13 ms** | 1.84 ms | 5.76 ms | 2.19 ms | **5.09x** | **1.93x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | off | **563.0 μs** | 343.8 μs | 1.99 ms | 1.39 ms | **5.77x** | **4.03x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | off | **517.4 μs** | 442.1 μs | 2.86 ms | 1.13 ms | **6.46x** | **2.56x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | off | **399.7 μs** | 472.9 μs | 2.15 ms | 977.3 μs | **5.37x** | **2.45x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | off | **538.2 μs** | 464.1 μs | 2.11 ms | 819.0 μs | **4.54x** | **1.76x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | off | **594.4 μs** | 550.8 μs | 3.08 ms | 1.01 ms | **5.59x** | **1.83x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **659.8 μs** | 521.5 μs | 2.32 ms | 1.01 ms | **4.45x** | **1.94x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **591.4 μs** | 550.4 μs | 2.40 ms | 969.4 μs | **4.37x** | **1.76x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **486.8 μs** | 499.0 μs | 1.79 ms | 795.3 μs | **3.68x** | **1.63x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **561.8 μs** | 531.2 μs | 2.09 ms | 1.03 ms | **3.93x** | **1.94x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **598.1 μs** | 653.2 μs | 1.84 ms | 911.3 μs | **3.08x** | **1.52x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **358.3 μs** | 352.7 μs | 1.70 ms | 572.7 μs | **4.82x** | **1.62x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **413.7 μs** | 376.5 μs | 2.13 ms | 550.5 μs | **5.66x** | **1.46x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **432.2 μs** | 309.7 μs | 1.84 ms | 819.6 μs | **5.93x** | **2.65x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **281.6 μs** | 284.5 μs | 1.88 ms | 508.8 μs | **6.68x** | **1.81x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **326.7 μs** | 293.2 μs | 1.47 ms | 561.2 μs | **5.03x** | **1.91x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **316.7 μs** | 304.7 μs | 1.22 ms | 660.3 μs | **4.02x** | **2.17x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **368.2 μs** | 418.1 μs | 1.82 ms | 650.3 μs | **4.96x** | **1.77x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **328.0 μs** | 304.9 μs | 1.63 ms | 577.3 μs | **5.34x** | **1.89x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **457.1 μs** | 434.8 μs | 1.38 ms | 690.8 μs | **3.17x** | **1.59x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **385.4 μs** | 404.5 μs | 2.21 ms | 646.6 μs | **5.73x** | **1.68x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **726.1 μs** | 255.0 μs | 2.05 ms | 1.09 ms | **8.04x** | **4.28x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **383.5 μs** | 467.1 μs | 1.08 ms | 678.7 μs | **2.81x** | **1.77x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **412.4 μs** | 400.7 μs | 2.09 ms | 649.4 μs | **5.21x** | **1.62x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **228.2 μs** | 431.3 μs | 2.48 ms | 388.1 μs | **10.87x** | **1.70x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **387.0 μs** | 462.7 μs | 2.12 ms | 657.4 μs | **5.48x** | **1.70x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **32.61 ms** | 29.30 ms | 159.57 ms | 35.00 ms | **5.45x** | **1.19x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **37.42 ms** | 36.81 ms | 170.22 ms | 42.73 ms | **4.62x** | **1.16x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **58.19 ms** | 69.94 ms | 96.35 ms | 63.84 ms | **1.66x** | **1.10x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **32.66 ms** | 27.87 ms | 73.72 ms | 33.11 ms | **2.65x** | **1.19x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | on | **4.10 ms** | 3.71 ms | 7.44 ms | — | **2.01x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | on | **13.66 ms** | 9.74 ms | 20.60 ms | — | **2.11x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | on | **2.39 ms** | 988.9 μs | 5.83 ms | — | **5.90x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | on | **7.02 ms** | 7.14 ms | 12.94 ms | — | **1.84x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | on | **2.69 ms** | 45.76 ms | 9.87 ms | — | **3.67x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | on | **12.42 ms** | 12.06 ms | 35.68 ms | — | **2.96x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | on | **9.75 ms** | 8.83 ms | 12.78 ms | — | **1.45x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | on | **24.38 ms** | 22.20 ms | 35.23 ms | — | **1.59x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | on | **1.13 ms** | 783.9 μs | 3.55 ms | — | **4.53x** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | on | **3.68 ms** | 2.17 ms | 13.23 ms | — | **6.10x** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | on | **10.86 ms** | 8.56 ms | 22.63 ms | — | **2.64x** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | on | **15.52 ms** | 15.34 ms | 25.56 ms | — | **1.67x** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | on | **1.04 ms** | 1.07 ms | 2.16 ms | — | **2.07x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | on | **5.13 ms** | 2.98 ms | 7.04 ms | — | **2.36x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | on | **5.43 ms** | 6.08 ms | 9.80 ms | — | **1.81x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | on | **684.9 μs** | 641.1 μs | 1.75 ms | — | **2.73x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | on | **2.94 ms** | 2.69 ms | 9.16 ms | — | **3.40x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | on | **2.21 ms** | 3.05 ms | 6.73 ms | — | **3.05x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | on | **3.44 ms** | 598.3 μs | 2.45 ms | — | **4.10x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | on | **3.95 ms** | 6.42 ms | 41.41 ms | — | **10.48x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | on | **5.46 ms** | 7.75 ms | 14.57 ms | — | **2.67x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | on | **1.39 ms** | 1.16 ms | 5.13 ms | — | **4.43x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | on | **13.27 ms** | 6.88 ms | 24.45 ms | — | **3.55x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | on | **7.84 ms** | 8.23 ms | 21.61 ms | — | **2.76x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | on | **642.3 μs** | 1.36 ms | 4.65 ms | — | **7.24x** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | on | **3.06 ms** | 5.02 ms | 6.14 ms | — | **2.01x** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | on | **2.70 ms** | 1.23 ms | 9.25 ms | — | **7.54x** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **4.32 ms** | 3.68 ms | 9.24 ms | — | **2.51x** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **3.55 ms** | 4.74 ms | 13.76 ms | — | **3.88x** | **—** |
| fits | mef_medium | read_full | 7.02 MB | MPS | on | **1.75 ms** | 1.19 ms | 6.88 ms | — | **5.81x** | **—** |
| fits | mef_small | read_full | 0.45 MB | MPS | on | **570.5 μs** | 466.8 μs | 4.35 ms | — | **9.32x** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **466.9 μs** | 412.1 μs | 3.35 ms | — | **8.13x** | **—** |
| fits | scaled_large | read_full | 8.00 MB | MPS | on | **10.71 ms** | 13.00 ms | 24.88 ms | — | **2.32x** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | on | **5.84 ms** | 5.79 ms | 9.60 ms | — | **1.66x** | **—** |
| fits | scaled_small | read_full | 0.13 MB | MPS | on | **835.3 μs** | 1.10 ms | 4.53 ms | — | **5.42x** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | on | **345.2 μs** | 326.7 μs | 3.08 ms | — | **9.41x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | on | **611.2 μs** | 501.0 μs | 1.95 ms | — | **3.89x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | on | **942.7 μs** | 842.0 μs | 3.39 ms | — | **4.02x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | on | **391.0 μs** | 480.2 μs | 1.78 ms | — | **4.55x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | on | **388.9 μs** | 1.61 ms | 2.55 ms | — | **6.57x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | on | **737.2 μs** | 598.8 μs | 2.49 ms | — | **4.17x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | on | **380.1 μs** | 401.9 μs | 1.55 ms | — | **4.09x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | on | **628.6 μs** | 496.7 μs | 2.18 ms | — | **4.39x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | on | **755.8 μs** | 732.5 μs | 2.88 ms | — | **3.93x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | on | **464.2 μs** | 516.0 μs | 1.68 ms | — | **3.63x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | on | **2.43 ms** | 1.17 ms | 2.00 ms | — | **1.71x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | on | **3.78 ms** | 1.34 ms | 13.33 ms | — | **9.96x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | on | **379.3 μs** | 406.9 μs | 3.31 ms | — | **8.74x** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | on | **285.0 μs** | 408.5 μs | 3.18 ms | — | **11.17x** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | on | **673.4 μs** | 1.76 ms | 4.04 ms | — | **6.01x** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | on | **765.5 μs** | 575.3 μs | 4.64 ms | — | **8.07x** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | on | **1.56 ms** | 384.2 μs | 3.24 ms | — | **8.42x** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **683.7 μs** | 585.1 μs | 2.55 ms | — | **4.37x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **632.2 μs** | 341.1 μs | 2.41 ms | — | **7.08x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **758.2 μs** | 575.4 μs | 2.79 ms | — | **4.86x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **631.0 μs** | 597.0 μs | 2.11 ms | — | **3.54x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **611.3 μs** | 659.0 μs | 2.05 ms | — | **3.36x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **324.7 μs** | 584.1 μs | 6.79 ms | — | **20.91x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **469.5 μs** | 253.6 μs | 1.62 ms | — | **6.38x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **544.5 μs** | 367.1 μs | 2.15 ms | — | **5.86x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **330.4 μs** | 404.0 μs | 1.83 ms | — | **5.53x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **398.6 μs** | 624.7 μs | 1.17 ms | — | **2.94x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **1.39 ms** | 375.1 μs | 2.19 ms | — | **5.83x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **555.6 μs** | 391.8 μs | 1.98 ms | — | **5.05x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **405.7 μs** | 491.6 μs | 1.75 ms | — | **4.31x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **501.4 μs** | 777.3 μs | 1.73 ms | — | **3.45x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **817.5 μs** | 487.4 μs | 1.85 ms | — | **3.80x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **683.5 μs** | 316.2 μs | 2.33 ms | — | **7.37x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **254.6 μs** | 489.8 μs | 1.78 ms | — | **7.01x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **432.5 μs** | 259.2 μs | 3.14 ms | — | **12.10x** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **446.2 μs** | 254.1 μs | 4.61 ms | — | **18.16x** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **400.7 μs** | 377.9 μs | 2.98 ms | — | **7.89x** | **—** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **987.8 μs** | 1.11 ms | 8.18 ms | 1.54 ms | **8.28x** | **1.56x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | off | **3.61 ms** | 4.06 ms | 28.16 ms | 7.43 ms | **7.81x** | **2.06x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | off | **3.51 ms** | 3.64 ms | 28.61 ms | 7.88 ms | **8.14x** | **2.24x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | off | **326.6 μs** | 316.3 μs | 7.41 ms | 1.66 ms | **23.42x** | **5.25x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | off | **102.0 μs** | 111.3 μs | 1.27 ms | 341.9 μs | **12.41x** | **3.35x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **283.4 μs** | 363.6 μs | 4.56 ms | 656.0 μs | **16.08x** | **2.31x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | off | **319.7 μs** | 354.5 μs | 6.94 ms | 1.33 ms | **21.71x** | **4.17x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | off | **372.1 μs** | 315.6 μs | 6.40 ms | 1.01 ms | **20.27x** | **3.21x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | off | **203.0 μs** | 244.4 μs | 6.07 ms | 659.7 μs | **29.90x** | **3.25x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | off | **118.2 μs** | 109.6 μs | 1.04 ms | 290.3 μs | **9.52x** | **2.65x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **50.27 ms** | 45.97 ms | 48.73 ms | 58.14 ms | **1.06x** | **1.26x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | off | **41.62 ms** | 34.17 ms | 52.77 ms | 116.18 ms | **1.54x** | **3.40x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | off | **58.71 ms** | 76.60 ms | 1.161 s | 351.39 ms | **19.78x** | **5.99x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **1.10 ms** | 1.60 ms | 39.82 ms | 5.87 ms | **36.32x** | **5.36x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **192.9 μs** | 158.0 μs | 2.08 ms | 1.07 ms | **13.17x** | **6.75x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **4.67 ms** | 6.34 ms | 9.32 ms | 8.65 ms | **1.99x** | **1.85x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | off | **5.45 ms** | 5.06 ms | 11.14 ms | 11.50 ms | **2.20x** | **2.27x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | off | **6.97 ms** | 7.87 ms | 95.60 ms | 30.20 ms | **13.72x** | **4.33x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | off | **710.5 μs** | 1.14 ms | 21.70 ms | 9.33 ms | **30.54x** | **13.14x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | off | **113.8 μs** | 113.1 μs | 1.54 ms | 308.3 μs | **13.57x** | **2.73x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **514.8 μs** | 1.24 ms | 6.44 ms | 1.72 ms | **12.51x** | **3.33x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | off | **1.30 ms** | 589.3 μs | 5.65 ms | 2.15 ms | **9.59x** | **3.65x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | off | **1.80 ms** | 1.56 ms | 15.24 ms | 3.66 ms | **9.78x** | **2.34x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | off | **407.4 μs** | 349.9 μs | 9.40 ms | 1.01 ms | **26.87x** | **2.87x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | off | **143.2 μs** | 134.9 μs | 1.13 ms | 323.3 μs | **8.38x** | **2.40x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **159.3 μs** | 345.0 μs | 4.69 ms | 662.0 μs | **29.43x** | **4.16x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | off | **247.9 μs** | 278.3 μs | 5.28 ms | 680.1 μs | **21.30x** | **2.74x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | off | **331.7 μs** | 282.9 μs | 5.75 ms | 790.6 μs | **20.32x** | **2.79x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | off | **266.1 μs** | 249.3 μs | 7.23 ms | 586.0 μs | **29.02x** | **2.35x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | off | **129.5 μs** | 124.4 μs | 1.12 ms | 262.5 μs | **8.98x** | **2.11x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **25.10 ms** | 27.76 ms | 23.05 ms | 45.88 ms | **0.92x** | **1.83x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | off | **13.19 ms** | 13.51 ms | 23.28 ms | 83.51 ms | **1.77x** | **6.33x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | off | **15.02 ms** | 15.22 ms | 24.03 ms | 18.51 ms | **1.60x** | **1.23x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **1.00 ms** | 414.8 μs | 12.79 ms | 2.41 ms | **30.84x** | **5.81x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **140.2 μs** | 117.3 μs | 989.6 μs | 268.0 μs | **8.43x** | **2.28x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **3.04 ms** | 2.91 ms | 6.66 ms | 5.90 ms | **2.29x** | **2.02x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | off | **1.66 ms** | 1.56 ms | 4.93 ms | 8.28 ms | **3.17x** | **5.31x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | off | **2.47 ms** | 2.02 ms | 5.63 ms | 2.05 ms | **2.78x** | **1.01x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | off | **370.9 μs** | 406.8 μs | 7.18 ms | 2.50 ms | **19.37x** | **6.73x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | off | **145.9 μs** | 125.8 μs | 1.49 ms | 396.5 μs | **11.87x** | **3.15x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **452.3 μs** | 683.5 μs | 4.08 ms | 985.2 μs | **9.03x** | **2.18x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | off | **418.5 μs** | 1.28 ms | 4.31 ms | 1.45 ms | **10.30x** | **3.46x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | off | **324.1 μs** | 516.0 μs | 3.87 ms | 1.14 ms | **11.93x** | **3.51x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | off | **305.5 μs** | 271.1 μs | 4.46 ms | 643.7 μs | **16.45x** | **2.37x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | off | **91.4 μs** | 103.5 μs | 1.07 ms | 269.1 μs | **11.68x** | **2.94x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **152.7 μs** | 215.3 μs | 3.10 ms | 445.8 μs | **20.29x** | **2.92x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | off | **220.2 μs** | 214.2 μs | 3.09 ms | 534.2 μs | **14.44x** | **2.49x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | off | **161.7 μs** | 206.2 μs | 3.16 ms | 468.5 μs | **19.53x** | **2.90x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | off | **281.2 μs** | 287.8 μs | 5.65 ms | 616.8 μs | **20.08x** | **2.19x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | off | **122.6 μs** | 96.7 μs | 995.4 μs | 218.8 μs | **10.29x** | **2.26x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **3.40 ms** | 2.81 ms | 7.32 ms | 5.36 ms | **2.61x** | **1.91x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | off | **11.25 ms** | 13.65 ms | 92.29 ms | 39.15 ms | **8.20x** | **3.48x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | off | **18.29 ms** | 16.81 ms | 103.24 ms | 42.44 ms | **6.14x** | **2.53x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | off | **2.20 ms** | 3.41 ms | 18.85 ms | 9.42 ms | **8.55x** | **4.27x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | off | **136.3 μs** | 110.6 μs | 1.87 ms | 376.5 μs | **16.93x** | **3.40x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **728.6 μs** | 686.0 μs | 4.84 ms | 1.98 ms | **7.05x** | **2.88x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | off | **1.64 ms** | 2.36 ms | 14.68 ms | 5.28 ms | **8.97x** | **3.23x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | off | **2.78 ms** | 2.18 ms | 15.06 ms | 5.39 ms | **6.92x** | **2.48x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | off | **240.0 μs** | 281.3 μs | 6.36 ms | 1.20 ms | **26.49x** | **5.00x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | off | **128.4 μs** | 95.5 μs | 1.08 ms | 270.2 μs | **11.26x** | **2.83x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **2.48 ms** | 2.37 ms | 5.84 ms | 4.48 ms | **2.46x** | **1.89x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | off | **260.07 ms** | 260.98 ms | 1.572 s | 332.15 ms | **6.05x** | **1.28x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | off | **258.11 ms** | 252.70 ms | 1.624 s | 337.40 ms | **6.43x** | **1.34x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | off | **26.49 ms** | 24.53 ms | 164.12 ms | 37.91 ms | **6.69x** | **1.55x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | off | **84.0 μs** | 146.1 μs | 1.37 ms | 323.2 μs | **16.37x** | **3.85x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **436.5 μs** | 2.14 ms | 5.08 ms | 1.09 ms | **11.64x** | **2.51x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | off | **27.97 ms** | 26.13 ms | 166.25 ms | 36.58 ms | **6.36x** | **1.40x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | off | **25.68 ms** | 24.90 ms | 166.88 ms | 36.38 ms | **6.70x** | **1.46x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | off | **1.82 ms** | 1.88 ms | 23.27 ms | 5.51 ms | **12.80x** | **3.03x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | off | **109.0 μs** | 121.9 μs | 1.49 ms | 332.9 μs | **13.70x** | **3.05x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **132.0 μs** | 286.5 μs | 4.40 ms | 663.7 μs | **33.32x** | **5.03x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | off | **1.79 ms** | 2.19 ms | 22.52 ms | 5.66 ms | **12.56x** | **3.15x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | off | **2.01 ms** | 2.02 ms | 26.41 ms | 4.12 ms | **13.13x** | **2.05x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | off | **456.4 μs** | 371.1 μs | 7.14 ms | 969.9 μs | **19.25x** | **2.61x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | off | **107.1 μs** | 107.9 μs | 1.30 ms | 293.2 μs | **12.10x** | **2.74x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **13.19 ms** | 14.14 ms | 30.36 ms | 16.88 ms | **2.30x** | **1.28x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | off | **14.49 ms** | 14.98 ms | 32.94 ms | 19.54 ms | **2.27x** | **1.35x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | off | **79.56 ms** | 91.26 ms | 472.25 ms | 182.11 ms | **5.94x** | **2.29x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | off | **7.07 ms** | 6.58 ms | 77.86 ms | 24.00 ms | **11.83x** | **3.65x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | off | **349.5 μs** | 391.6 μs | 1.86 ms | 778.5 μs | **5.31x** | **2.23x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **1.60 ms** | 2.35 ms | 19.07 ms | 2.95 ms | **11.88x** | **1.84x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | off | **2.13 ms** | 2.43 ms | 17.60 ms | 2.76 ms | **8.27x** | **1.30x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | off | **6.35 ms** | 6.62 ms | 59.74 ms | 20.21 ms | **9.41x** | **3.18x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | off | **1.59 ms** | 1.80 ms | 35.31 ms | 3.50 ms | **22.21x** | **2.20x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | off | **349.6 μs** | 358.9 μs | 2.24 ms | 781.9 μs | **6.41x** | **2.24x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **168.5 μs** | 655.3 μs | 16.18 ms | 1.17 ms | **96.07x** | **6.96x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | off | **434.2 μs** | 394.3 μs | 15.26 ms | 1.21 ms | **38.69x** | **3.06x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | off | **1.46 ms** | 1.27 ms | 21.41 ms | 3.24 ms | **16.86x** | **2.55x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | off | **1.05 ms** | 1.28 ms | 27.10 ms | 1.93 ms | **25.85x** | **1.85x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | off | **312.0 μs** | 337.6 μs | 1.73 ms | 834.6 μs | **5.55x** | **2.68x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **3.45 ms** | 598.2 μs | 8.93 ms | — | **14.93x** | **—** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | on | **722.0 μs** | 710.1 μs | 28.33 ms | — | **39.89x** | **—** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | on | **781.1 μs** | 591.4 μs | 34.81 ms | — | **58.86x** | **—** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | on | **400.0 μs** | 381.8 μs | 9.12 ms | — | **23.88x** | **—** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | on | **121.9 μs** | 118.5 μs | 1.21 ms | — | **10.24x** | **—** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **504.3 μs** | 321.0 μs | 5.73 ms | — | **17.84x** | **—** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | on | **361.7 μs** | 342.6 μs | 8.03 ms | — | **23.45x** | **—** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | on | **447.9 μs** | 517.6 μs | 6.97 ms | — | **15.56x** | **—** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | on | **253.7 μs** | 452.0 μs | 6.88 ms | — | **27.10x** | **—** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | on | **106.0 μs** | 128.6 μs | 1.38 ms | — | **13.07x** | **—** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **26.35 ms** | 28.19 ms | 50.56 ms | — | **1.92x** | **—** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | on | **34.94 ms** | 33.77 ms | 57.38 ms | — | **1.70x** | **—** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | on | **67.00 ms** | 62.42 ms | 1.303 s | — | **20.87x** | **—** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **2.32 ms** | 1.43 ms | 46.12 ms | — | **32.36x** | **—** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **156.7 μs** | 137.3 μs | 1.77 ms | — | **12.92x** | **—** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **9.03 ms** | 2.13 ms | 13.33 ms | — | **6.25x** | **—** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | on | **2.01 ms** | 4.36 ms | 17.04 ms | — | **8.46x** | **—** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | on | **9.77 ms** | 8.01 ms | 144.28 ms | — | **18.00x** | **—** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | on | **2.04 ms** | 1.03 ms | 24.58 ms | — | **23.92x** | **—** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | on | **174.5 μs** | 162.6 μs | 1.80 ms | — | **11.05x** | **—** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **784.0 μs** | 625.1 μs | 9.09 ms | — | **14.55x** | **—** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | on | **663.7 μs** | 487.7 μs | 6.19 ms | — | **12.68x** | **—** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | on | **1.45 ms** | 868.6 μs | 19.95 ms | — | **22.96x** | **—** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | on | **390.3 μs** | 468.5 μs | 9.69 ms | — | **24.82x** | **—** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | on | **119.4 μs** | 120.8 μs | 1.03 ms | — | **8.63x** | **—** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **303.9 μs** | 220.3 μs | 7.41 ms | — | **33.63x** | **—** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | on | **348.0 μs** | 361.5 μs | 7.00 ms | — | **20.13x** | **—** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | on | **432.3 μs** | 415.3 μs | 11.10 ms | — | **26.72x** | **—** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | on | **425.0 μs** | 409.8 μs | 11.56 ms | — | **28.20x** | **—** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | on | **169.6 μs** | 133.5 μs | 1.27 ms | — | **9.49x** | **—** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **15.16 ms** | 25.36 ms | 44.68 ms | — | **2.95x** | **—** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | on | **13.42 ms** | 17.27 ms | 52.51 ms | — | **3.91x** | **—** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | on | **13.35 ms** | 23.16 ms | 48.13 ms | — | **3.61x** | **—** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **361.5 μs** | 332.4 μs | 13.11 ms | — | **39.45x** | **—** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **119.0 μs** | 114.0 μs | 1.21 ms | — | **10.65x** | **—** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **6.95 ms** | 2.11 ms | 6.76 ms | — | **3.20x** | **—** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | on | **2.05 ms** | 1.32 ms | 6.94 ms | — | **5.27x** | **—** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | on | **1.06 ms** | 1.51 ms | 6.64 ms | — | **6.26x** | **—** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | on | **517.9 μs** | 503.0 μs | 8.36 ms | — | **16.61x** | **—** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | on | **145.0 μs** | 133.7 μs | 1.21 ms | — | **9.04x** | **—** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **2.50 ms** | 473.3 μs | 5.63 ms | — | **11.91x** | **—** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | on | **399.1 μs** | 404.8 μs | 4.95 ms | — | **12.41x** | **—** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | on | **322.5 μs** | 451.0 μs | 6.04 ms | — | **18.73x** | **—** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | on | **318.0 μs** | 389.9 μs | 9.94 ms | — | **31.27x** | **—** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | on | **158.6 μs** | 113.3 μs | 1.26 ms | — | **11.08x** | **—** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **178.8 μs** | 281.8 μs | 4.70 ms | — | **26.27x** | **—** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | on | **455.2 μs** | 279.1 μs | 3.82 ms | — | **13.67x** | **—** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | on | **327.8 μs** | 261.7 μs | 4.21 ms | — | **16.10x** | **—** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | on | **378.6 μs** | 301.6 μs | 8.18 ms | — | **27.12x** | **—** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | on | **107.4 μs** | 140.4 μs | 1.64 ms | — | **15.31x** | **—** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **1.93 ms** | 1.07 ms | 5.92 ms | — | **5.56x** | **—** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | on | **2.87 ms** | 3.71 ms | 98.06 ms | — | **34.21x** | **—** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | on | **4.77 ms** | 4.49 ms | 99.48 ms | — | **22.14x** | **—** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | on | **706.5 μs** | 1.37 ms | 17.40 ms | — | **24.63x** | **—** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | on | **118.7 μs** | 118.5 μs | 1.16 ms | — | **9.82x** | **—** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **589.1 μs** | 225.1 μs | 3.98 ms | — | **17.68x** | **—** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | on | **666.8 μs** | 671.8 μs | 13.81 ms | — | **20.70x** | **—** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | on | **606.3 μs** | 794.0 μs | 14.46 ms | — | **23.84x** | **—** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | on | **344.5 μs** | 319.5 μs | 5.88 ms | — | **18.41x** | **—** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | on | **138.5 μs** | 99.9 μs | 1.23 ms | — | **12.31x** | **—** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **4.28 ms** | 1.03 ms | 7.31 ms | — | **7.08x** | **—** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | on | **310.02 ms** | 316.01 ms | 1.992 s | — | **6.42x** | **—** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | on | **307.03 ms** | 363.99 ms | 2.144 s | — | **6.98x** | **—** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | on | **22.77 ms** | 26.03 ms | 173.61 ms | — | **7.63x** | **—** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | on | **129.2 μs** | 109.5 μs | 1.34 ms | — | **12.20x** | **—** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **2.04 ms** | 274.8 μs | 3.32 ms | — | **12.08x** | **—** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | on | **31.61 ms** | 31.57 ms | 227.48 ms | — | **7.21x** | **—** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | on | **28.85 ms** | 55.41 ms | 372.91 ms | — | **12.93x** | **—** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | on | **2.15 ms** | 2.12 ms | 31.10 ms | — | **14.64x** | **—** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | on | **93.7 μs** | 116.8 μs | 1.61 ms | — | **17.20x** | **—** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **168.7 μs** | 235.1 μs | 3.94 ms | — | **23.38x** | **—** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | on | **2.07 ms** | 2.06 ms | 26.19 ms | — | **12.73x** | **—** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | on | **2.78 ms** | 2.34 ms | 23.79 ms | — | **10.16x** | **—** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | on | **568.7 μs** | 488.0 μs | 12.16 ms | — | **24.91x** | **—** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | on | **87.3 μs** | 96.7 μs | 1.25 ms | — | **14.34x** | **—** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **8.97 ms** | 7.52 ms | 30.43 ms | — | **4.05x** | **—** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | on | **11.92 ms** | 11.31 ms | 34.06 ms | — | **3.01x** | **—** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | on | **108.12 ms** | 104.98 ms | 555.69 ms | — | **5.29x** | **—** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | on | **5.38 ms** | 7.94 ms | 98.73 ms | — | **18.36x** | **—** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | on | **398.6 μs** | 413.6 μs | 2.35 ms | — | **5.89x** | **—** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **1.55 ms** | 901.5 μs | 23.98 ms | — | **26.60x** | **—** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | on | **944.4 μs** | 1.16 ms | 29.15 ms | — | **30.87x** | **—** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | on | **5.72 ms** | 8.58 ms | 84.19 ms | — | **14.73x** | **—** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | on | **1.52 ms** | 1.45 ms | 48.96 ms | — | **33.84x** | **—** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | on | **423.3 μs** | 346.9 μs | 2.14 ms | — | **6.18x** | **—** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **381.5 μs** | 428.9 μs | 23.02 ms | — | **60.34x** | **—** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | on | **441.7 μs** | 480.1 μs | 19.79 ms | — | **44.81x** | **—** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | on | **1.54 ms** | 1.35 ms | 29.28 ms | — | **21.65x** | **—** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | on | **1.16 ms** | 1.08 ms | 30.15 ms | — | **27.97x** | **—** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | on | **346.9 μs** | 363.9 μs | 2.39 ms | — | **6.88x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Host | Domain | Case | mmap | torchfits (s) | TF RSS | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| NRC-054711 | fits | small_float32_1d [read_full] | off | 0.0007379999879049137 | 548.984375 | fitsio/fitsio_torch | 1.9664687336539501 |
| NRC-054711 | fits | small_int64_1d [read_full] | off | 0.000636124997981824 | 549.0 | fitsio/fitsio_torch | 1.933263838011597 |
| NRC-054711 | fits | small_float64_1d [read_full] | off | 0.0007188340096035972 | 549.0 | fitsio/fitsio_torch | 1.8415916577565625 |
| NRC-054711 | fits | medium_int32_1d [read_full @ mps] | on | 0.003436958009842783 | 403.671875 | astropy/astropy_torch_device | 1.4019582068859426 |
| NRC-054711 | fits | small_int64_2d [read_full @ mps] | off | 0.0017285420035477728 | 574.078125 | fitsio/fitsio_torch_device | 1.2443387065504534 |
| NRC-054711 | fits | small_int64_2d [read_full @ mps] | on | 0.0024331669992534444 | 413.3125 | astropy/astropy_torch_device | 1.2188175911209331 |
| NRC-054711 | fits | compressed_hcompress_1 [read_full @ mps] | off | 0.055612333002500236 | 497.171875 | fitsio/fitsio_torch_device | 1.0690079604346643 |
| NRC-054711 | fits | repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | n/a | 0.021746541999164037 | 549.0625 | fitsio/fitsio_torch | 1.0233505976117088 |
| NRC-054711 | fits | mef_small [read_full] | off | 0.0013126670091878623 | 548.984375 | fitsio/fitsio_torch | 2.1210535476152774 |
| NRC-054711 | fits | small_uint32_2d [read_full] | off | 0.001036499990732409 | 549.015625 | fitsio/fitsio_torch | 2.108674494480074 |
| NRC-054711 | fits | large_int32_1d [read_full @ mps] | on | 0.04576354099845048 | 254.125 | astropy/astropy_torch_device_specialized | 1.9523834058612721 |
| NRC-054711 | fits | compressed_rice_1 [read_full @ mps] | on | 0.027866875010658987 | 184.1875 | fitsio/fitsio_torch_device_specialized | 1.3553542549193847 |
| NRC-054711 | fits | compressed_rice_1 [read_full @ mps] | off | 0.01656150000053458 | 497.15625 | fitsio/fitsio_torch_device_specialized | 1.167577286014861 |
| NRC-054711 | fits | compressed_hcompress_1 [read_full @ mps] | on | 0.06994470801146235 | 214.765625 | fitsio/fitsio_torch_device_specialized | 1.0238052061883094 |
| NRC-054711 | fitstable | varlen_10000 [predicate_filter] | off | 0.0021439169941004366 | 1095.078125 | fitsio/fitsio | 1.9594808709196134 |
| NRC-054711 | fitstable | narrow_1000000 [predicate_filter] | off | 0.027760040989960544 | 846.15625 | astropy/astropy | 1.2041660520896842 |
| torchfits-gpu-exhaustive-cpu-20260717-040146 | fitstable | narrow_1000000 [predicate_filter] | off | 0.009568384848535061 | 393.83203125 | astropy/astropy | 1.1524116971081395 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest lab benchmarks (one row per host):

| Run ID | Host / device | Scope | Rows | Deficits | Median peak RSS (MB) | Notes |
|---|---|---|---:|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mps_20260717_040150` | NRC-054711 / mps | fits + fitstable (lab) | 3925 | 16 | 548 | lab + mmap-matrix + GPU |
| `exhaustive_cpu_20260717_040146` | torchfits-gpu-exhaustive-cpu-20260717-040146 / cpu | fits + fitstable (lab) | 2825 | 1 | 289 | lab + mmap-matrix |
| `exhaustive_cuda_20260717_042840` | torchfits-gpu-exhaustive-cuda-20260717-042840 / cuda | fits + fitstable (lab) | 4079 | 0 | 731 | lab + mmap-matrix + GPU |
<!-- BENCH_SNAPSHOT_END -->

### Host scorecard

| Host / device | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| NRC-054711 / mps | `exhaustive_mps_20260717_040150` | 3925 | 16 | 548 | lab + mmap-matrix + GPU |
| torchfits-gpu-exhaustive-cpu-20260717-040146 / cpu | `exhaustive_cpu_20260717_040146` | 2825 | 1 | 289 | lab + mmap-matrix |
| torchfits-gpu-exhaustive-cuda-20260717-042840 / cuda | `exhaustive_cuda_20260717_042840` | 4079 | 0 | 731 | lab + mmap-matrix + GPU |
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
