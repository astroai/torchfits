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
   staging (`exhaustive_cuda_20260716_191255`, via `pixi run bench-exhaustive-canfar-cuda`).
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
Source: `benchmarks_results/exhaustive_mps_20260717_000853/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.13 ms` (n=174) | `0.40 ms` (n=253) | `0.17 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.27 ms` (n=174) | `1.37 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.24 ms` (n=152) | `0.68 ms` (n=152) | `0.33 ms` (n=152) | — |
| `disk→RAM→GPU` | `0.23 ms` (n=152) | `0.87 ms` (n=152) | `16.42 ms` (n=8) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.19 ms` (n=180) | `2.36 ms` (n=164) | `0.79 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.17 ms` (n=180) | `2.83 ms` (n=164) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **2.42 ms** | 2.59 ms | 5.82 ms | 2.92 ms | **2.40x** | **1.20x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.52 ms** | 3.67 ms | 8.29 ms | 3.87 ms | **2.36x** | **1.10x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **7.15 ms** | 7.25 ms | 19.46 ms | 7.04 ms | **2.72x** | **0.98x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **7.85 ms** | 7.59 ms | 20.60 ms | 7.91 ms | **2.71x** | **1.04x** |
| Repeated Cutouts (50x 100x100) | CPU | **5.59 ms** | 5.81 ms | 86.30 ms | 5.83 ms | **15.42x** | **1.04x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **2.49 ms** | 2.45 ms | 36.71 ms | 12.34 ms | **15.00x** | **5.04x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **82.12 ms** | 87.49 ms | 528.28 ms | 122.72 ms | **6.43x** | **1.49x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

Aggregated wins across every domain and operation in the CANFAR CUDA exhaustive
(`exhaustive_cuda_20260716_191255`, 4,079 rows, **0** TorchFits deficits).

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
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 55.5 μs | 1.16 ms | 149.0 μs | **20.85x** | **2.69x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 66.8 μs | 1.26 ms | 155.9 μs | **18.89x** | **2.33x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 78.2 μs | 1.33 ms | 196.8 μs | **16.95x** | **2.52x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | n/a | **686.4 μs** | 639.9 μs | 6.31 ms | 819.7 μs | **9.86x** | **1.28x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 61.0 μs | 1.17 ms | 188.4 μs | **19.20x** | **3.09x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 16.1 μs | 227.7 μs | 45.5 μs | **14.12x** | **2.82x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 16.5 μs | 256.4 μs | 49.3 μs | **15.58x** | **3.00x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 17.3 μs | 242.6 μs | 50.9 μs | **14.07x** | **2.95x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 19.5 μs | 276.8 μs | 52.3 μs | **14.23x** | **2.69x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 15.8 μs | 218.8 μs | 50.7 μs | **13.89x** | **3.22x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 16.5 μs | 224.3 μs | 46.5 μs | **13.56x** | **2.81x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 15.2 μs | 223.2 μs | 46.5 μs | **14.68x** | **3.06x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 21.7 μs | 240.4 μs | 56.7 μs | **11.08x** | **2.61x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 15.8 μs | 236.7 μs | 43.5 μs | **14.99x** | **2.75x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 15.5 μs | 257.4 μs | 44.3 μs | **16.56x** | **2.85x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 19.0 μs | 244.9 μs | 47.9 μs | **12.86x** | **2.52x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 18.5 μs | 274.4 μs | 50.2 μs | **14.83x** | **2.71x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 20.1 μs | 325.7 μs | 57.6 μs | **16.18x** | **2.86x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 19.5 μs | 290.5 μs | 56.0 μs | **14.93x** | **2.88x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 17.6 μs | 226.9 μs | 48.8 μs | **12.87x** | **2.77x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 20.6 μs | 259.6 μs | 54.3 μs | **12.61x** | **2.64x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 17.3 μs | 251.7 μs | 49.5 μs | **14.56x** | **2.87x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 15.2 μs | 262.6 μs | 56.3 μs | **17.32x** | **3.71x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 20.7 μs | 281.4 μs | 56.8 μs | **13.56x** | **2.74x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 18.2 μs | 275.1 μs | 54.2 μs | **15.14x** | **2.98x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 18.8 μs | 237.6 μs | 44.2 μs | **12.64x** | **2.35x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 16.6 μs | 247.2 μs | 51.2 μs | **14.87x** | **3.08x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 18.9 μs | 258.8 μs | 51.1 μs | **13.71x** | **2.71x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 15.6 μs | 209.9 μs | 43.7 μs | **13.47x** | **2.80x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 16.8 μs | 240.5 μs | 46.2 μs | **14.32x** | **2.75x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 17.3 μs | 255.3 μs | 45.6 μs | **14.76x** | **2.64x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 19.1 μs | 242.2 μs | 51.0 μs | **12.66x** | **2.66x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 20.7 μs | 305.3 μs | 61.0 μs | **14.71x** | **2.94x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 16.4 μs | 236.0 μs | 46.3 μs | **14.41x** | **2.83x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 24.6 μs | 288.9 μs | 60.0 μs | **11.75x** | **2.44x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 20.6 μs | 269.6 μs | 55.7 μs | **13.10x** | **2.71x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 20.3 μs | 275.1 μs | 54.3 μs | **13.56x** | **2.68x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 24.3 μs | 300.4 μs | 58.3 μs | **12.34x** | **2.39x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 22.3 μs | 281.2 μs | 64.9 μs | **12.59x** | **2.91x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 28.2 μs | 465.4 μs | 61.5 μs | **16.47x** | **2.18x** |
| fits | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 33.8 μs | 492.6 μs | 80.9 μs | **14.58x** | **2.39x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | n/a | **12.6 μs** | 9.9 μs | 2.56 ms | 189.5 μs | **259.57x** | **19.19x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 31.7 μs | 446.9 μs | 69.0 μs | **14.09x** | **2.18x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **5.83 ms** | 5.37 ms | 9.16 ms | 6.69 ms | **1.71x** | **1.25x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **5.59 ms** | 5.81 ms | 86.30 ms | 5.83 ms | **15.42x** | **1.04x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 23.2 μs | 321.1 μs | 60.9 μs | **13.86x** | **2.63x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 21.7 μs | 310.2 μs | 65.8 μs | **14.29x** | **3.03x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 22.5 μs | 294.5 μs | 58.8 μs | **13.07x** | **2.61x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 19.0 μs | 262.5 μs | 55.9 μs | **13.82x** | **2.94x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 17.3 μs | 247.5 μs | 55.1 μs | **14.32x** | **3.19x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 18.7 μs | 280.8 μs | 62.8 μs | **15.04x** | **3.36x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 15.0 μs | 227.4 μs | 55.9 μs | **15.16x** | **3.73x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 15.2 μs | 227.8 μs | 50.5 μs | **14.98x** | **3.32x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 16.9 μs | 282.5 μs | 56.5 μs | **16.74x** | **3.35x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 13.9 μs | 214.7 μs | 50.5 μs | **15.43x** | **3.63x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 15.5 μs | 237.5 μs | 56.4 μs | **15.36x** | **3.65x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 20.0 μs | 284.8 μs | 53.8 μs | **14.24x** | **2.69x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 17.1 μs | 210.7 μs | 48.2 μs | **12.34x** | **2.82x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 16.5 μs | 244.7 μs | 51.1 μs | **14.87x** | **3.10x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 22.5 μs | 335.0 μs | 62.6 μs | **14.86x** | **2.78x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 20.4 μs | 251.4 μs | 58.7 μs | **12.34x** | **2.88x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 16.4 μs | 252.4 μs | 50.4 μs | **15.38x** | **3.07x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 21.1 μs | 255.3 μs | 60.6 μs | **12.11x** | **2.88x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 20.4 μs | 262.3 μs | 63.1 μs | **12.85x** | **3.09x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 19.9 μs | 298.5 μs | 58.5 μs | **15.02x** | **2.94x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 20.5 μs | 324.1 μs | 61.4 μs | **15.84x** | **3.00x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 20.7 μs | 276.3 μs | 54.1 μs | **13.34x** | **2.61x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 20.1 μs | 312.4 μs | 58.4 μs | **15.55x** | **2.91x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 18.0 μs | 270.1 μs | 57.2 μs | **15.04x** | **3.19x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 15.9 μs | 222.8 μs | 45.5 μs | **14.04x** | **2.87x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 20.0 μs | 276.2 μs | 55.2 μs | **13.84x** | **2.77x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 22.3 μs | 289.8 μs | 67.0 μs | **12.98x** | **3.00x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 18.5 μs | 291.0 μs | 59.3 μs | **15.73x** | **3.21x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 15.8 μs | 239.2 μs | 45.2 μs | **15.14x** | **2.86x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 16.4 μs | 249.2 μs | 51.6 μs | **15.22x** | **3.15x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 17.4 μs | 238.7 μs | 50.2 μs | **13.71x** | **2.88x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 15.6 μs | 213.1 μs | 43.1 μs | **13.64x** | **2.76x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 15.6 μs | 242.1 μs | 52.0 μs | **15.54x** | **3.33x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 17.1 μs | 244.7 μs | 46.7 μs | **14.29x** | **2.73x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 14.3 μs | 221.0 μs | 42.5 μs | **15.51x** | **2.99x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 19.0 μs | 276.3 μs | 51.6 μs | **14.57x** | **2.72x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 20.6 μs | 290.0 μs | 62.9 μs | **14.09x** | **3.05x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 16.8 μs | 224.7 μs | 53.7 μs | **13.38x** | **3.20x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 16.6 μs | 272.3 μs | 56.8 μs | **16.38x** | **3.42x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 18.2 μs | 266.9 μs | 52.4 μs | **14.69x** | **2.88x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 14.4 μs | 205.3 μs | 43.0 μs | **14.24x** | **2.98x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 16.8 μs | 234.9 μs | 49.4 μs | **13.99x** | **2.94x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 17.1 μs | 251.6 μs | 51.0 μs | **14.69x** | **2.98x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 22.0 μs | 273.8 μs | 58.4 μs | **12.42x** | **2.65x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 20.4 μs | 266.8 μs | 56.3 μs | **13.07x** | **2.76x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 24.0 μs | 312.6 μs | 56.0 μs | **13.02x** | **2.34x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **12.00 ms** | 12.03 ms | 25.33 ms | 14.92 ms | **2.11x** | **1.24x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **12.61 ms** | 12.61 ms | 44.65 ms | 15.51 ms | **3.54x** | **1.23x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **23.53 ms** | 24.23 ms | 28.85 ms | 23.32 ms | **1.23x** | **0.99x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **7.15 ms** | 7.25 ms | 19.46 ms | 7.04 ms | **2.72x** | **0.98x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | off | **613.9 μs** | 687.0 μs | 1.26 ms | 611.7 μs | **2.05x** | **1.00x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | off | **2.42 ms** | 2.59 ms | 5.82 ms | 2.92 ms | **2.40x** | **1.20x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | off | **1.46 ms** | 1.51 ms | 2.75 ms | 1.43 ms | **1.88x** | **0.98x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | off | **6.41 ms** | 6.44 ms | 11.02 ms | 6.83 ms | **1.72x** | **1.07x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | off | **294.5 μs** | 258.9 μs | 839.0 μs | 290.7 μs | **3.24x** | **1.12x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | off | **1.20 ms** | 1.25 ms | 2.80 ms | 1.38 ms | **2.34x** | **1.15x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | off | **516.9 μs** | 582.4 μs | 1.39 ms | 607.8 μs | **2.69x** | **1.18x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | off | **2.61 ms** | 2.66 ms | 5.51 ms | 2.84 ms | **2.11x** | **1.09x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | off | **1.49 ms** | 1.50 ms | 2.82 ms | 1.57 ms | **1.90x** | **1.05x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | off | **6.45 ms** | 6.54 ms | 12.11 ms | 7.02 ms | **1.88x** | **1.09x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | off | **164.8 μs** | 133.5 μs | 707.9 μs | 606.6 μs | **5.30x** | **4.55x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | off | **667.8 μs** | 700.8 μs | 1.45 ms | 2.52 ms | **2.17x** | **3.77x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | off | **3.20 ms** | 2.96 ms | 5.16 ms | 3.25 ms | **1.74x** | **1.10x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | off | **4.63 ms** | 4.79 ms | 8.03 ms | 4.92 ms | **1.74x** | **1.06x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | off | **82.1 μs** | 128.0 μs | 287.6 μs | 86.1 μs | **3.50x** | **1.05x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | off | **587.1 μs** | 600.7 μs | 1.34 ms | 662.1 μs | **2.29x** | **1.13x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | off | **1.06 ms** | 1.04 ms | 2.10 ms | 1.05 ms | **2.02x** | **1.01x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | off | **120.7 μs** | 139.3 μs | 418.2 μs | 144.9 μs | **3.46x** | **1.20x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | off | **1.56 ms** | 1.63 ms | 2.96 ms | 1.61 ms | **1.90x** | **1.03x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | off | **2.45 ms** | 2.37 ms | 4.01 ms | 2.50 ms | **1.69x** | **1.05x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | off | **56.5 μs** | 57.0 μs | 242.8 μs | 68.6 μs | **4.29x** | **1.21x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | off | **294.8 μs** | 332.7 μs | 730.2 μs | 282.7 μs | **2.48x** | **0.96x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | off | **501.5 μs** | 500.5 μs | 1.10 ms | 495.0 μs | **2.20x** | **0.99x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | off | **72.9 μs** | 90.1 μs | 290.7 μs | 95.5 μs | **3.99x** | **1.31x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | off | **622.4 μs** | 579.5 μs | 1.39 ms | 577.0 μs | **2.40x** | **1.00x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | off | **1.06 ms** | 1.02 ms | 2.10 ms | 1.01 ms | **2.06x** | **0.99x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | off | **147.7 μs** | 167.2 μs | 351.2 μs | 125.2 μs | **2.38x** | **0.85x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | off | **1.63 ms** | 1.55 ms | 2.86 ms | 1.53 ms | **1.85x** | **0.99x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | off | **2.57 ms** | 2.51 ms | 3.84 ms | 2.59 ms | **1.53x** | **1.03x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | off | **74.3 μs** | 84.8 μs | 287.5 μs | 100.2 μs | **3.87x** | **1.35x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | off | **136.2 μs** | 179.5 μs | 423.9 μs | 547.0 μs | **3.11x** | **4.02x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | off | **247.0 μs** | 278.2 μs | 778.1 μs | 961.5 μs | **3.15x** | **3.89x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **708.8 μs** | 701.8 μs | 1.36 ms | 730.7 μs | **1.94x** | **1.04x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **1.15 ms** | 1.09 ms | 2.04 ms | 1.07 ms | **1.88x** | **0.98x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | off | **171.6 μs** | 181.8 μs | 681.1 μs | 628.1 μs | **3.97x** | **3.66x** |
| fits | mef_small | read_full | 0.45 MB | CPU | off | **208.9 μs** | 73.2 μs | 431.0 μs | 112.2 μs | **5.89x** | **1.53x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **70.4 μs** | 91.3 μs | 448.3 μs | 181.6 μs | **6.37x** | **2.58x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | off | **2.54 ms** | 2.43 ms | 6.88 ms | 3.19 ms | **2.84x** | **1.32x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | off | **637.7 μs** | 607.7 μs | 1.71 ms | 596.5 μs | **2.81x** | **0.98x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | off | **92.2 μs** | 125.9 μs | 356.3 μs | 72.8 μs | **3.87x** | **0.79x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | off | **128.9 μs** | 55.2 μs | 211.9 μs | 105.0 μs | **3.84x** | **1.90x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | off | **71.1 μs** | 182.3 μs | 329.6 μs | 75.2 μs | **4.63x** | **1.06x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | off | **77.4 μs** | 104.1 μs | 339.9 μs | 99.1 μs | **4.39x** | **1.28x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | off | **49.9 μs** | 46.6 μs | 243.9 μs | 65.9 μs | **5.23x** | **1.41x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | off | **100.8 μs** | 134.3 μs | 353.1 μs | 119.3 μs | **3.50x** | **1.18x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | off | **213.2 μs** | 210.3 μs | 571.8 μs | 202.2 μs | **2.72x** | **0.96x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | off | **49.6 μs** | 44.7 μs | 218.0 μs | 55.2 μs | **4.88x** | **1.24x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | off | **50.4 μs** | 63.0 μs | 352.1 μs | 82.0 μs | **6.98x** | **1.63x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | off | **68.7 μs** | 75.3 μs | 345.7 μs | 82.0 μs | **5.03x** | **1.19x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | off | **142.2 μs** | 58.2 μs | 318.4 μs | 70.3 μs | **5.47x** | **1.21x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | off | **142.4 μs** | 182.9 μs | 246.2 μs | 67.3 μs | **1.73x** | **0.47x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | off | **105.9 μs** | 96.9 μs | 385.4 μs | 112.5 μs | **3.98x** | **1.16x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | off | **145.3 μs** | 81.7 μs | 320.4 μs | 76.3 μs | **3.92x** | **0.93x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | off | **80.8 μs** | 95.3 μs | 402.8 μs | 119.2 μs | **4.99x** | **1.48x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | off | **186.3 μs** | 175.3 μs | 473.9 μs | 182.5 μs | **2.70x** | **1.04x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | off | **46.3 μs** | 49.9 μs | 307.7 μs | 68.7 μs | **6.64x** | **1.48x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | off | **124.8 μs** | 105.3 μs | 324.0 μs | 90.6 μs | **3.08x** | **0.86x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | off | **136.5 μs** | 87.6 μs | 310.5 μs | 128.8 μs | **3.54x** | **1.47x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | off | **78.7 μs** | 98.0 μs | 410.2 μs | 84.9 μs | **5.21x** | **1.08x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | off | **90.7 μs** | 95.4 μs | 313.1 μs | 90.1 μs | **3.45x** | **0.99x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **73.5 μs** | 61.6 μs | 307.6 μs | 79.6 μs | **4.99x** | **1.29x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **99.5 μs** | 64.2 μs | 323.0 μs | 90.6 μs | **5.03x** | **1.41x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **93.7 μs** | 59.3 μs | 276.2 μs | 73.0 μs | **4.66x** | **1.23x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **63.7 μs** | 108.6 μs | 278.0 μs | 81.3 μs | **4.37x** | **1.28x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **81.4 μs** | 70.8 μs | 241.7 μs | 68.7 μs | **3.42x** | **0.97x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **38.7 μs** | 36.0 μs | 240.5 μs | 58.4 μs | **6.69x** | **1.62x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **39.4 μs** | 42.0 μs | 210.3 μs | 52.2 μs | **5.34x** | **1.32x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **43.4 μs** | 48.6 μs | 244.5 μs | 56.0 μs | **5.63x** | **1.29x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **40.3 μs** | 39.1 μs | 231.0 μs | 57.5 μs | **5.91x** | **1.47x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **55.0 μs** | 50.3 μs | 233.1 μs | 62.7 μs | **4.64x** | **1.25x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **68.3 μs** | 91.7 μs | 244.9 μs | 79.6 μs | **3.58x** | **1.17x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **42.6 μs** | 54.7 μs | 232.6 μs | 59.8 μs | **5.46x** | **1.41x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **43.3 μs** | 56.1 μs | 223.0 μs | 55.3 μs | **5.15x** | **1.28x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **54.4 μs** | 44.0 μs | 242.5 μs | 61.5 μs | **5.51x** | **1.40x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **40.5 μs** | 48.3 μs | 225.0 μs | 64.4 μs | **5.56x** | **1.59x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **49.5 μs** | 50.1 μs | 236.8 μs | 52.8 μs | **4.78x** | **1.07x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **52.7 μs** | 45.1 μs | 272.2 μs | 67.6 μs | **6.04x** | **1.50x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **47.0 μs** | 44.8 μs | 313.2 μs | 60.1 μs | **6.99x** | **1.34x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **46.9 μs** | 51.5 μs | 241.1 μs | 60.6 μs | **5.14x** | **1.29x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **119.8 μs** | 100.7 μs | 239.0 μs | 83.3 μs | **2.37x** | **0.83x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **49.8 μs** | 40.3 μs | 280.7 μs | 51.8 μs | **6.97x** | **1.29x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **41.7 μs** | 42.9 μs | 303.9 μs | 60.0 μs | **7.28x** | **1.44x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **42.1 μs** | 41.4 μs | 312.3 μs | 67.4 μs | **7.55x** | **1.63x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **18.49 ms** | 18.98 ms | 53.03 ms | 23.62 ms | **2.87x** | **1.28x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **21.34 ms** | 20.91 ms | 84.29 ms | 25.45 ms | **4.03x** | **1.22x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **39.74 ms** | 40.21 ms | 59.06 ms | 40.49 ms | **1.49x** | **1.02x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **11.39 ms** | 11.59 ms | 37.59 ms | 11.34 ms | **3.30x** | **1.00x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | on | **942.3 μs** | 904.9 μs | 2.03 ms | — | **2.25x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | on | **3.68 ms** | 4.37 ms | 7.86 ms | — | **2.13x** | **—** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | on | **2.26 ms** | 2.18 ms | 8.42 ms | — | **3.87x** | **—** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | on | **9.00 ms** | 8.75 ms | 13.77 ms | — | **1.57x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | on | **517.7 μs** | 504.0 μs | 1.26 ms | — | **2.49x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | on | **1.79 ms** | 1.78 ms | 8.56 ms | — | **4.80x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.20 ms** | 1.43 ms | 1.82 ms | — | **1.51x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | on | **6.34 ms** | 4.99 ms | 8.61 ms | — | **1.73x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | on | **2.24 ms** | 1.98 ms | 4.96 ms | — | **2.51x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | on | **9.29 ms** | 9.55 ms | 12.98 ms | — | **1.40x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | on | **277.1 μs** | 239.3 μs | — | — | **—** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | on | **1.10 ms** | 1.20 ms | — | — | **—** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | on | **4.71 ms** | 4.24 ms | — | — | **—** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | on | **6.40 ms** | 6.31 ms | — | — | **—** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | on | **100.6 μs** | 124.0 μs | 441.5 μs | — | **4.39x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | on | **973.8 μs** | 1.23 ms | 7.45 ms | — | **7.65x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | on | **1.38 ms** | 1.34 ms | 7.30 ms | — | **5.45x** | **—** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | on | **239.2 μs** | 419.5 μs | 5.82 ms | — | **24.32x** | **—** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | on | **2.86 ms** | 2.56 ms | 7.87 ms | — | **3.08x** | **—** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | on | **4.15 ms** | 3.45 ms | 5.12 ms | — | **1.49x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | on | **493.9 μs** | 600.5 μs | 450.2 μs | — | **0.91x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | on | **571.8 μs** | 514.3 μs | 1.29 ms | — | **2.51x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | on | **903.4 μs** | 741.5 μs | 1.78 ms | — | **2.40x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | on | **243.7 μs** | 169.6 μs | 446.4 μs | — | **2.63x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | on | **1.27 ms** | 1.39 ms | 2.60 ms | — | **2.05x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | on | **1.89 ms** | 1.63 ms | 6.83 ms | — | **4.19x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | on | **230.9 μs** | 173.2 μs | 761.1 μs | — | **4.39x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | on | **1.64 ms** | 1.92 ms | 7.52 ms | — | **4.59x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | on | **2.92 ms** | 3.41 ms | 8.29 ms | — | **2.84x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | on | **197.8 μs** | 99.2 μs | — | — | **—** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | on | **232.7 μs** | 232.6 μs | — | — | **—** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | on | **383.7 μs** | 384.7 μs | — | — | **—** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **914.9 μs** | 995.5 μs | — | — | **—** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **1.61 ms** | 1.45 ms | — | — | **—** | **—** |
| fits | mef_medium | read_full | 7.02 MB | CPU | on | **355.1 μs** | 478.6 μs | — | — | **—** | **—** |
| fits | mef_small | read_full | 0.45 MB | CPU | on | **398.2 μs** | 214.3 μs | — | — | **—** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **107.5 μs** | 385.9 μs | — | — | **—** | **—** |
| fits | scaled_large | read_full | 8.00 MB | CPU | on | **3.64 ms** | 3.81 ms | — | — | **—** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | on | **924.7 μs** | 1.02 ms | — | — | **—** | **—** |
| fits | scaled_small | read_full | 0.13 MB | CPU | on | **125.6 μs** | 113.5 μs | — | — | **—** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | on | **141.5 μs** | 150.1 μs | 304.9 μs | — | **2.15x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | on | **77.2 μs** | 79.4 μs | 380.6 μs | — | **4.93x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | on | **162.1 μs** | 172.8 μs | 5.52 ms | — | **34.02x** | **—** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | on | **277.9 μs** | 146.2 μs | 4.47 ms | — | **30.58x** | **—** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | on | **161.5 μs** | 145.3 μs | 551.9 μs | — | **3.80x** | **—** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | on | **281.6 μs** | 383.0 μs | 1.24 ms | — | **4.42x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | on | **116.2 μs** | 171.7 μs | 699.0 μs | — | **6.01x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | on | **264.4 μs** | 163.9 μs | 365.1 μs | — | **2.23x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | on | **121.0 μs** | 107.9 μs | 449.5 μs | — | **4.17x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | on | **129.2 μs** | 156.5 μs | 291.7 μs | — | **2.26x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | on | **138.8 μs** | 203.2 μs | 4.71 ms | — | **33.93x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | on | **271.2 μs** | 175.5 μs | 5.66 ms | — | **32.22x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | on | **239.8 μs** | 134.3 μs | 898.0 μs | — | **6.69x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | on | **256.3 μs** | 373.8 μs | 5.13 ms | — | **20.03x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | on | **433.7 μs** | 336.3 μs | 5.50 ms | — | **16.34x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | on | **73.9 μs** | 57.7 μs | — | — | **—** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | on | **218.5 μs** | 191.5 μs | — | — | **—** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | on | **260.6 μs** | 214.4 μs | — | — | **—** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | on | **153.3 μs** | 246.7 μs | — | — | **—** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | on | **331.4 μs** | 439.7 μs | — | — | **—** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **235.1 μs** | 346.0 μs | 408.4 μs | — | **1.74x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **325.4 μs** | 258.8 μs | 757.7 μs | — | **2.93x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **134.3 μs** | 304.5 μs | 356.2 μs | — | **2.65x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **274.7 μs** | 258.0 μs | 376.2 μs | — | **1.46x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **425.8 μs** | 275.6 μs | 722.7 μs | — | **2.62x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **173.7 μs** | 182.7 μs | 693.0 μs | — | **3.99x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **56.6 μs** | 59.3 μs | 949.9 μs | — | **16.77x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **116.7 μs** | 85.7 μs | 1.67 ms | — | **19.46x** | **—** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **104.6 μs** | 74.4 μs | 971.5 μs | — | **13.06x** | **—** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **54.1 μs** | 60.5 μs | 996.4 μs | — | **18.41x** | **—** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **214.6 μs** | 274.4 μs | 747.0 μs | — | **3.48x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **145.6 μs** | 113.6 μs | 412.1 μs | — | **3.63x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **56.4 μs** | 55.8 μs | 543.0 μs | — | **9.72x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **65.7 μs** | 82.7 μs | 881.7 μs | — | **13.43x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **65.1 μs** | 112.8 μs | 5.11 ms | — | **78.53x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **94.5 μs** | 111.3 μs | 669.3 μs | — | **7.09x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **93.0 μs** | 69.8 μs | 339.1 μs | — | **4.86x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **75.8 μs** | 68.9 μs | 289.8 μs | — | **4.20x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **66.2 μs** | 74.3 μs | 295.2 μs | — | **4.46x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **134.5 μs** | 133.6 μs | 323.8 μs | — | **2.42x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **57.2 μs** | 67.8 μs | — | — | **—** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **103.8 μs** | 86.6 μs | — | — | **—** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **65.6 μs** | 58.7 μs | — | — | **—** | **—** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | n/a | **855.4 μs** | 839.5 μs | 6.89 ms | 1.02 ms | **8.20x** | **1.22x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | n/a | **153.5 μs** | 171.0 μs | 3.42 ms | 363.0 μs | **22.30x** | **2.37x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | n/a | **12.70 ms** | 13.04 ms | 107.36 ms | 13.23 ms | **8.45x** | **1.04x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **12.78 ms** | 12.84 ms | 26.69 ms | 15.88 ms | **2.09x** | **1.24x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **13.51 ms** | 13.33 ms | 46.96 ms | 16.31 ms | **3.52x** | **1.22x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **23.81 ms** | 24.28 ms | 30.66 ms | 24.53 ms | **1.29x** | **1.03x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **7.75 ms** | 7.59 ms | 20.27 ms | 7.91 ms | **2.67x** | **1.04x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | off | **923.8 μs** | 924.6 μs | 2.19 ms | 969.3 μs | **2.37x** | **1.05x** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | off | **3.38 ms** | 3.67 ms | 6.41 ms | 3.75 ms | **1.90x** | **1.11x** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | off | **551.4 μs** | 567.0 μs | 1.22 ms | 606.4 μs | **2.22x** | **1.10x** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | off | **1.87 ms** | 1.76 ms | 3.75 ms | 2.16 ms | **2.13x** | **1.22x** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | off | **988.0 μs** | 921.2 μs | 2.19 ms | 1.11 ms | **2.38x** | **1.20x** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | off | **3.73 ms** | 4.09 ms | 7.40 ms | 3.85 ms | **1.98x** | **1.03x** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | off | **2.12 ms** | 1.98 ms | 3.62 ms | 2.24 ms | **1.83x** | **1.13x** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | off | **8.54 ms** | 8.46 ms | 13.34 ms | 9.31 ms | **1.58x** | **1.10x** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | off | **355.0 μs** | 311.5 μs | 733.4 μs | 804.9 μs | **2.35x** | **2.58x** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | off | **1.10 ms** | 1.10 ms | 2.43 ms | 3.00 ms | **2.22x** | **2.73x** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | off | **3.44 ms** | 3.57 ms | 5.74 ms | 3.66 ms | **1.67x** | **1.07x** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | off | **5.60 ms** | 5.14 ms | 9.32 ms | 5.78 ms | **1.81x** | **1.12x** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | off | **266.3 μs** | 215.7 μs | 734.7 μs | 314.9 μs | **3.41x** | **1.46x** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | off | **1.06 ms** | 1.05 ms | 2.25 ms | 1.20 ms | **2.15x** | **1.14x** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | off | **1.29 ms** | 1.49 ms | 3.02 ms | 1.49 ms | **2.34x** | **1.16x** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | off | **211.1 μs** | 193.6 μs | 548.6 μs | 279.5 μs | **2.83x** | **1.44x** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | off | **493.0 μs** | 535.6 μs | 1.37 ms | 531.4 μs | **2.77x** | **1.08x** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | off | **840.5 μs** | 758.3 μs | 1.86 ms | 911.8 μs | **2.46x** | **1.20x** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | off | **248.4 μs** | 197.8 μs | 589.1 μs | 293.7 μs | **2.98x** | **1.49x** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | off | **954.5 μs** | 879.2 μs | 2.32 ms | 1.14 ms | **2.64x** | **1.30x** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | off | **1.53 ms** | 1.29 ms | 3.07 ms | 1.50 ms | **2.38x** | **1.16x** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | off | **313.9 μs** | 288.4 μs | 771.0 μs | 391.2 μs | **2.67x** | **1.36x** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | off | **1.92 ms** | 2.05 ms | 3.60 ms | 2.36 ms | **1.88x** | **1.23x** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | off | **3.32 ms** | 3.40 ms | 4.72 ms | 3.49 ms | **1.42x** | **1.05x** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | off | **194.6 μs** | 170.9 μs | 653.2 μs | 343.3 μs | **3.82x** | **2.01x** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | off | **321.7 μs** | 323.1 μs | 889.7 μs | 768.7 μs | **2.77x** | **2.39x** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | off | **537.8 μs** | 474.8 μs | 1.51 ms | 1.30 ms | **3.18x** | **2.75x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **997.6 μs** | 977.1 μs | 1.96 ms | 1.07 ms | **2.00x** | **1.10x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **1.44 ms** | 1.37 ms | 2.84 ms | 1.55 ms | **2.08x** | **1.13x** |
| fits | mef_medium | read_full | 7.02 MB | MPS | off | **411.8 μs** | 356.3 μs | 1.27 ms | 979.5 μs | **3.58x** | **2.75x** |
| fits | mef_small | read_full | 0.45 MB | MPS | off | **184.3 μs** | 184.0 μs | 722.9 μs | 330.3 μs | **3.93x** | **1.79x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **180.8 μs** | 175.9 μs | 823.7 μs | 404.6 μs | **4.68x** | **2.30x** |
| fits | scaled_large | read_full | 8.00 MB | MPS | off | **3.39 ms** | 3.41 ms | 8.19 ms | 3.44 ms | **2.42x** | **1.01x** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | off | **936.1 μs** | 943.2 μs | 2.60 ms | 888.6 μs | **2.78x** | **0.95x** |
| fits | scaled_small | read_full | 0.13 MB | MPS | off | **245.9 μs** | 209.0 μs | 717.1 μs | 301.2 μs | **3.43x** | **1.44x** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | off | **170.8 μs** | 161.0 μs | 542.6 μs | 228.3 μs | **3.37x** | **1.42x** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | off | **213.9 μs** | 196.7 μs | 552.6 μs | 302.3 μs | **2.81x** | **1.54x** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | off | **275.9 μs** | 232.8 μs | 811.2 μs | 361.7 μs | **3.48x** | **1.55x** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | off | **156.7 μs** | 163.9 μs | 528.9 μs | 231.9 μs | **3.37x** | **1.48x** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | off | **170.0 μs** | 183.5 μs | 428.8 μs | 238.6 μs | **2.52x** | **1.40x** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | off | **209.5 μs** | 207.5 μs | 552.8 μs | 275.4 μs | **2.66x** | **1.33x** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | off | **187.3 μs** | 156.2 μs | 415.0 μs | 254.8 μs | **2.66x** | **1.63x** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | off | **235.1 μs** | 212.3 μs | 660.5 μs | 313.3 μs | **3.11x** | **1.48x** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | off | **256.5 μs** | 289.3 μs | 765.3 μs | 324.9 μs | **2.98x** | **1.27x** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | off | **188.2 μs** | 180.7 μs | 549.6 μs | 236.5 μs | **3.04x** | **1.31x** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | off | **249.7 μs** | 250.7 μs | 584.7 μs | 371.5 μs | **2.34x** | **1.49x** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | off | **400.8 μs** | 388.8 μs | 922.1 μs | 460.2 μs | **2.37x** | **1.18x** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | off | **164.1 μs** | 170.6 μs | 530.0 μs | 240.3 μs | **3.23x** | **1.46x** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | off | **181.9 μs** | 160.5 μs | 531.2 μs | 255.1 μs | **3.31x** | **1.59x** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | off | **212.5 μs** | 176.8 μs | 646.9 μs | 340.0 μs | **3.66x** | **1.92x** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | off | **240.0 μs** | 222.1 μs | 592.9 μs | 319.2 μs | **2.67x** | **1.44x** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | off | **230.9 μs** | 237.3 μs | 554.8 μs | 309.8 μs | **2.40x** | **1.34x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **233.0 μs** | 215.4 μs | 528.6 μs | 288.9 μs | **2.45x** | **1.34x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **247.1 μs** | 197.9 μs | 580.3 μs | 264.6 μs | **2.93x** | **1.34x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **223.5 μs** | 198.6 μs | 619.1 μs | 264.7 μs | **3.12x** | **1.33x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **229.6 μs** | 207.7 μs | 550.4 μs | 285.7 μs | **2.65x** | **1.38x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **205.4 μs** | 201.5 μs | 615.8 μs | 287.9 μs | **3.06x** | **1.43x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **162.5 μs** | 146.4 μs | 436.1 μs | 213.4 μs | **2.98x** | **1.46x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **160.6 μs** | 164.5 μs | 458.8 μs | 204.6 μs | **2.86x** | **1.27x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **164.5 μs** | 176.4 μs | 522.3 μs | 216.4 μs | **3.18x** | **1.32x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **153.1 μs** | 151.8 μs | 510.1 μs | 212.6 μs | **3.36x** | **1.40x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **174.6 μs** | 176.6 μs | 503.1 μs | 258.9 μs | **2.88x** | **1.48x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **165.4 μs** | 164.5 μs | 473.8 μs | 232.2 μs | **2.88x** | **1.41x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **170.0 μs** | 163.9 μs | 508.3 μs | 256.2 μs | **3.10x** | **1.56x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **158.3 μs** | 155.6 μs | 487.2 μs | 245.5 μs | **3.13x** | **1.58x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **165.5 μs** | 154.6 μs | 475.3 μs | 237.8 μs | **3.07x** | **1.54x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **157.1 μs** | 157.3 μs | 511.8 μs | 242.1 μs | **3.26x** | **1.54x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **206.5 μs** | 170.3 μs | 477.8 μs | 250.3 μs | **2.81x** | **1.47x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **176.8 μs** | 161.7 μs | 529.7 μs | 257.7 μs | **3.28x** | **1.59x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **172.0 μs** | 156.2 μs | 536.5 μs | 302.3 μs | **3.43x** | **1.93x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **181.0 μs** | 156.2 μs | 541.7 μs | 247.2 μs | **3.47x** | **1.58x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **156.5 μs** | 157.4 μs | 494.5 μs | 231.8 μs | **3.16x** | **1.48x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **12.27 ms** | 12.71 ms | 26.68 ms | 15.96 ms | **2.17x** | **1.30x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **13.65 ms** | 14.90 ms | 46.95 ms | 16.88 ms | **3.44x** | **1.24x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **24.49 ms** | 25.22 ms | 29.86 ms | 25.26 ms | **1.22x** | **1.03x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **8.08 ms** | 7.65 ms | 20.22 ms | 7.73 ms | **2.64x** | **1.01x** |
| fits | large_float32_1d | read_full | 3.82 MB | MPS | on | **931.3 μs** | 842.5 μs | 2.13 ms | — | **2.53x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | MPS | on | **3.74 ms** | 3.59 ms | 5.30 ms | — | **1.48x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | MPS | on | **463.9 μs** | 500.4 μs | 1.21 ms | — | **2.61x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | MPS | on | **1.76 ms** | 1.79 ms | 3.30 ms | — | **1.88x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | MPS | on | **832.0 μs** | 952.7 μs | 1.86 ms | — | **2.23x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | MPS | on | **3.51 ms** | 3.98 ms | 5.36 ms | — | **1.53x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | MPS | on | **1.78 ms** | 1.70 ms | 2.96 ms | — | **1.75x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | MPS | on | **7.52 ms** | 7.25 ms | 10.51 ms | — | **1.45x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | MPS | on | **321.6 μs** | 331.3 μs | 1.22 ms | — | **3.79x** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | MPS | on | **1.12 ms** | 1.17 ms | 2.53 ms | — | **2.25x** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | MPS | on | **3.56 ms** | 3.61 ms | 5.99 ms | — | **1.68x** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | MPS | on | **5.58 ms** | 5.03 ms | 10.19 ms | — | **2.02x** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | MPS | on | **221.4 μs** | 219.3 μs | 622.2 μs | — | **2.84x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | MPS | on | **995.3 μs** | 942.3 μs | 1.79 ms | — | **1.90x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | MPS | on | **1.48 ms** | 1.53 ms | 2.68 ms | — | **1.81x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | MPS | on | **201.3 μs** | 213.9 μs | 606.2 μs | — | **3.01x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | MPS | on | **562.1 μs** | 520.8 μs | 1.23 ms | — | **2.37x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | MPS | on | **746.5 μs** | 729.3 μs | 1.59 ms | — | **2.18x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | MPS | on | **236.8 μs** | 223.4 μs | 544.5 μs | — | **2.44x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | MPS | on | **937.0 μs** | 996.7 μs | 1.84 ms | — | **1.97x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | MPS | on | **1.39 ms** | 1.32 ms | 2.50 ms | — | **1.90x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | MPS | on | **284.8 μs** | 259.1 μs | 714.1 μs | — | **2.76x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | MPS | on | **1.75 ms** | 1.71 ms | 2.89 ms | — | **1.69x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | MPS | on | **2.72 ms** | 2.63 ms | 3.51 ms | — | **1.33x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | MPS | on | **202.8 μs** | 168.2 μs | 707.1 μs | — | **4.20x** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | MPS | on | **346.2 μs** | 336.3 μs | 1.27 ms | — | **3.77x** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | MPS | on | **537.5 μs** | 490.5 μs | 1.77 ms | — | **3.61x** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **975.5 μs** | 1.01 ms | 2.20 ms | — | **2.26x** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **1.43 ms** | 1.45 ms | 3.16 ms | — | **2.21x** | **—** |
| fits | mef_medium | read_full | 7.02 MB | MPS | on | **349.7 μs** | 322.8 μs | 1.45 ms | — | **4.50x** | **—** |
| fits | mef_small | read_full | 0.45 MB | MPS | on | **191.7 μs** | 157.7 μs | 1.10 ms | — | **6.98x** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **173.8 μs** | 162.9 μs | 1.08 ms | — | **6.62x** | **—** |
| fits | scaled_large | read_full | 8.00 MB | MPS | on | **3.20 ms** | 3.26 ms | 8.66 ms | — | **2.70x** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | MPS | on | **953.2 μs** | 1.01 ms | 3.11 ms | — | **3.26x** | **—** |
| fits | scaled_small | read_full | 0.13 MB | MPS | on | **232.8 μs** | 208.2 μs | 942.1 μs | — | **4.53x** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | MPS | on | **173.3 μs** | 151.7 μs | 563.8 μs | — | **3.72x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | MPS | on | **210.7 μs** | 205.5 μs | 531.0 μs | — | **2.58x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | MPS | on | **240.4 μs** | 231.5 μs | 731.9 μs | — | **3.16x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | MPS | on | **177.7 μs** | 199.5 μs | 493.9 μs | — | **2.78x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | MPS | on | **224.5 μs** | 191.2 μs | 486.2 μs | — | **2.54x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | MPS | on | **229.0 μs** | 225.7 μs | 543.6 μs | — | **2.41x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | MPS | on | **173.3 μs** | 172.1 μs | 518.7 μs | — | **3.01x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | MPS | on | **239.1 μs** | 210.8 μs | 535.9 μs | — | **2.54x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | MPS | on | **265.1 μs** | 274.5 μs | 668.7 μs | — | **2.52x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | MPS | on | **194.5 μs** | 182.5 μs | 453.8 μs | — | **2.49x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | MPS | on | **232.1 μs** | 226.8 μs | 705.7 μs | — | **3.11x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | MPS | on | **328.2 μs** | 313.7 μs | 1.04 ms | — | **3.33x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | MPS | on | **168.2 μs** | 156.4 μs | 757.4 μs | — | **4.84x** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | MPS | on | **197.6 μs** | 163.3 μs | 1.31 ms | — | **8.05x** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | MPS | on | **202.4 μs** | 190.5 μs | 891.7 μs | — | **4.68x** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | MPS | on | **236.1 μs** | 216.8 μs | 744.5 μs | — | **3.43x** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | MPS | on | **248.3 μs** | 237.3 μs | 1.00 ms | — | **4.22x** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **208.2 μs** | 213.7 μs | 552.1 μs | — | **2.65x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **215.2 μs** | 211.2 μs | 569.3 μs | — | **2.70x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **231.0 μs** | 200.0 μs | 542.5 μs | — | **2.71x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **209.2 μs** | 209.9 μs | 585.4 μs | — | **2.80x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **225.8 μs** | 199.8 μs | 551.9 μs | — | **2.76x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **161.1 μs** | 153.4 μs | 372.5 μs | — | **2.43x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **160.3 μs** | 157.3 μs | 481.2 μs | — | **3.06x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **161.9 μs** | 180.1 μs | 530.3 μs | — | **3.28x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **186.3 μs** | 159.5 μs | 482.2 μs | — | **3.02x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **169.2 μs** | 171.1 μs | 448.3 μs | — | **2.65x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **171.1 μs** | 160.4 μs | 536.6 μs | — | **3.35x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **183.0 μs** | 173.4 μs | 396.5 μs | — | **2.29x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **192.3 μs** | 174.1 μs | 477.7 μs | — | **2.74x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **169.4 μs** | 169.5 μs | 494.6 μs | — | **2.92x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **177.4 μs** | 177.5 μs | 421.0 μs | — | **2.37x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **177.5 μs** | 166.6 μs | 491.0 μs | — | **2.95x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **169.7 μs** | 179.6 μs | 536.1 μs | — | **3.16x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **158.9 μs** | 159.7 μs | 753.3 μs | — | **4.74x** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **174.2 μs** | 144.2 μs | 790.2 μs | — | **5.48x** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **154.5 μs** | 153.5 μs | 837.7 μs | — | **5.46x** | **—** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **404.5 μs** | 457.9 μs | 3.19 ms | 486.3 μs | **7.89x** | **1.20x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | off | **943.0 μs** | 983.0 μs | 9.73 ms | 2.86 ms | **10.31x** | **3.03x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.06 ms** | 1.05 ms | 10.22 ms | 2.71 ms | **9.74x** | **2.59x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | off | **98.7 μs** | 100.3 μs | 2.56 ms | 650.5 μs | **25.98x** | **6.59x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | off | **30.4 μs** | 32.3 μs | 331.1 μs | 83.0 μs | **10.90x** | **2.73x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **63.7 μs** | 92.7 μs | 1.17 ms | 148.3 μs | **18.41x** | **2.33x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | off | **116.1 μs** | 105.5 μs | 1.98 ms | 374.5 μs | **18.75x** | **3.55x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | off | **107.7 μs** | 116.3 μs | 1.97 ms | 376.8 μs | **18.29x** | **3.50x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | off | **40.7 μs** | 58.3 μs | 1.57 ms | 218.1 μs | **38.60x** | **5.36x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | off | **31.5 μs** | 40.5 μs | 371.8 μs | 89.5 μs | **11.80x** | **2.84x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **16.56 ms** | 20.74 ms | 18.00 ms | 38.26 ms | **1.09x** | **2.31x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | off | **11.25 ms** | 11.47 ms | 18.41 ms | 69.17 ms | **1.64x** | **6.15x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | off | **20.12 ms** | 21.18 ms | 344.85 ms | 124.95 ms | **17.14x** | **6.21x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **252.5 μs** | 764.9 μs | 16.11 ms | 2.64 ms | **63.78x** | **10.46x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **56.0 μs** | 61.2 μs | 465.1 μs | 138.7 μs | **8.31x** | **2.48x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.89 ms** | 1.98 ms | 3.29 ms | 3.87 ms | **1.74x** | **2.05x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | off | **1.19 ms** | 1.44 ms | 3.06 ms | 7.25 ms | **2.57x** | **6.11x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | off | **2.49 ms** | 2.45 ms | 36.71 ms | 12.34 ms | **15.00x** | **5.04x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | off | **380.3 μs** | 264.9 μs | 6.49 ms | 2.07 ms | **24.51x** | **7.80x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | off | **49.8 μs** | 55.8 μs | 437.8 μs | 116.7 μs | **8.80x** | **2.35x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **276.8 μs** | 268.4 μs | 1.61 ms | 494.5 μs | **5.98x** | **1.84x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | off | **109.8 μs** | 117.9 μs | 1.45 ms | 793.5 μs | **13.22x** | **7.23x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | off | **308.3 μs** | 345.2 μs | 4.86 ms | 1.39 ms | **15.76x** | **4.49x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | off | **83.5 μs** | 101.6 μs | 2.68 ms | 347.3 μs | **32.06x** | **4.16x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | off | **46.6 μs** | 49.4 μs | 385.1 μs | 94.5 μs | **8.26x** | **2.03x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **34.2 μs** | 81.6 μs | 1.34 ms | 185.9 μs | **39.21x** | **5.43x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | off | **61.1 μs** | 57.0 μs | 1.39 ms | 202.2 μs | **24.45x** | **3.55x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | off | **94.2 μs** | 95.2 μs | 1.75 ms | 311.6 μs | **18.58x** | **3.31x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | off | **76.7 μs** | 75.6 μs | 2.03 ms | 183.7 μs | **26.87x** | **2.43x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | off | **42.0 μs** | 42.7 μs | 367.2 μs | 88.7 μs | **8.75x** | **2.11x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **9.66 ms** | 12.82 ms | 9.98 ms | 19.05 ms | **1.03x** | **1.97x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | off | **5.19 ms** | 5.36 ms | 7.95 ms | 35.05 ms | **1.53x** | **6.76x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | off | **5.91 ms** | 6.15 ms | 8.96 ms | 7.61 ms | **1.52x** | **1.29x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **154.5 μs** | 132.0 μs | 4.80 ms | 678.3 μs | **36.39x** | **5.14x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **53.2 μs** | 40.3 μs | 357.0 μs | 96.3 μs | **8.85x** | **2.39x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.08 ms** | 1.25 ms | 2.09 ms | 2.07 ms | **1.94x** | **1.92x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | off | **446.8 μs** | 567.4 μs | 1.87 ms | 3.62 ms | **4.17x** | **8.11x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | off | **573.3 μs** | 620.3 μs | 1.71 ms | 778.2 μs | **2.98x** | **1.36x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | off | **100.6 μs** | 132.3 μs | 1.92 ms | 822.5 μs | **19.10x** | **8.18x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | off | **39.6 μs** | 37.5 μs | 387.9 μs | 93.0 μs | **10.33x** | **2.48x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **137.1 μs** | 218.1 μs | 1.10 ms | 299.5 μs | **8.02x** | **2.18x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | off | **94.6 μs** | 91.7 μs | 1.11 ms | 505.8 μs | **12.07x** | **5.52x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | off | **152.5 μs** | 87.8 μs | 1.08 ms | 288.7 μs | **12.24x** | **3.29x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | off | **60.7 μs** | 57.4 μs | 1.29 ms | 162.9 μs | **22.40x** | **2.84x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | off | **37.2 μs** | 29.4 μs | 330.6 μs | 77.9 μs | **11.25x** | **2.65x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **47.7 μs** | 86.7 μs | 936.7 μs | 132.9 μs | **19.63x** | **2.79x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | off | **51.8 μs** | 61.7 μs | 1.07 ms | 181.5 μs | **20.68x** | **3.50x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | off | **60.1 μs** | 52.0 μs | 997.6 μs | 144.2 μs | **19.20x** | **2.77x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | off | **59.9 μs** | 59.5 μs | 1.29 ms | 136.0 μs | **21.76x** | **2.29x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | off | **30.3 μs** | 31.8 μs | 349.2 μs | 92.1 μs | **11.53x** | **3.04x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **1.42 ms** | 1.14 ms | 1.89 ms | 1.77 ms | **1.66x** | **1.55x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | off | **5.04 ms** | 4.91 ms | 34.33 ms | 16.70 ms | **7.00x** | **3.41x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | off | **6.35 ms** | 6.28 ms | 35.15 ms | 18.98 ms | **5.60x** | **3.02x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | off | **758.9 μs** | 634.1 μs | 6.41 ms | 3.19 ms | **10.12x** | **5.04x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | off | **38.5 μs** | 38.3 μs | 377.7 μs | 84.5 μs | **9.86x** | **2.21x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **177.8 μs** | 145.3 μs | 960.8 μs | 260.6 μs | **6.61x** | **1.79x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | off | **507.5 μs** | 671.7 μs | 4.93 ms | 1.73 ms | **9.72x** | **3.40x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | off | **873.2 μs** | 699.2 μs | 4.53 ms | 2.04 ms | **6.47x** | **2.92x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | off | **70.2 μs** | 82.6 μs | 1.71 ms | 392.6 μs | **24.38x** | **5.59x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | off | **37.2 μs** | 35.6 μs | 375.0 μs | 81.5 μs | **10.53x** | **2.29x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **1.06 ms** | 1.01 ms | 2.20 ms | 1.55 ms | **2.17x** | **1.53x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | off | **86.79 ms** | 82.22 ms | 499.29 ms | 113.29 ms | **6.07x** | **1.38x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | off | **82.12 ms** | 87.49 ms | 528.28 ms | 122.72 ms | **6.43x** | **1.49x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | off | **8.78 ms** | 8.31 ms | 57.55 ms | 13.98 ms | **6.92x** | **1.68x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | off | **32.7 μs** | 28.2 μs | 327.3 μs | 76.0 μs | **11.62x** | **2.70x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **131.5 μs** | 201.8 μs | 979.1 μs | 237.4 μs | **7.44x** | **1.80x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | off | **7.34 ms** | 8.07 ms | 54.78 ms | 12.12 ms | **7.47x** | **1.65x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | off | **7.33 ms** | 8.21 ms | 53.37 ms | 12.20 ms | **7.28x** | **1.66x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | off | **631.6 μs** | 656.2 μs | 6.23 ms | 1.62 ms | **9.87x** | **2.56x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | off | **28.4 μs** | 27.0 μs | 320.1 μs | 68.9 μs | **11.84x** | **2.55x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **39.7 μs** | 90.0 μs | 993.2 μs | 181.2 μs | **25.01x** | **4.56x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | off | **725.6 μs** | 725.8 μs | 6.54 ms | 1.45 ms | **9.01x** | **2.00x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | off | **695.9 μs** | 718.0 μs | 6.24 ms | 1.29 ms | **8.96x** | **1.85x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | off | **158.6 μs** | 97.2 μs | 1.64 ms | 248.8 μs | **16.91x** | **2.56x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | off | **29.0 μs** | 38.5 μs | 356.4 μs | 87.1 μs | **12.29x** | **3.00x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **5.44 ms** | 5.57 ms | 10.43 ms | 6.45 ms | **1.92x** | **1.19x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | off | **4.37 ms** | 4.43 ms | 9.90 ms | 8.85 ms | **2.26x** | **2.02x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | off | **19.03 ms** | 19.40 ms | 145.80 ms | 61.12 ms | **7.66x** | **3.21x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.96 ms** | 1.82 ms | 25.25 ms | 7.05 ms | **13.87x** | **3.87x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | off | **112.0 μs** | 116.2 μs | 530.5 μs | 320.0 μs | **4.74x** | **2.86x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **610.0 μs** | 744.5 μs | 5.07 ms | 1.01 ms | **8.32x** | **1.66x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | off | **452.6 μs** | 433.8 μs | 5.42 ms | 1.12 ms | **12.50x** | **2.58x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | off | **1.83 ms** | 1.95 ms | 18.80 ms | 6.24 ms | **10.27x** | **3.41x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | off | **489.1 μs** | 507.3 μs | 10.01 ms | 1.13 ms | **20.47x** | **2.32x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | off | **113.6 μs** | 120.5 μs | 501.0 μs | 287.0 μs | **4.41x** | **2.53x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **53.5 μs** | 149.9 μs | 4.94 ms | 418.1 μs | **92.18x** | **7.81x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | off | **106.8 μs** | 107.3 μs | 4.77 ms | 564.0 μs | **44.69x** | **5.28x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | off | **468.3 μs** | 418.1 μs | 6.49 ms | 925.7 μs | **15.53x** | **2.21x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | off | **416.5 μs** | 380.3 μs | 8.13 ms | 552.2 μs | **21.37x** | **1.45x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | off | **118.8 μs** | 108.8 μs | 473.0 μs | 259.6 μs | **4.35x** | **2.39x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **204.4 μs** | 80.0 μs | 3.12 ms | — | **39.00x** | **—** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | on | **142.0 μs** | 148.8 μs | 10.12 ms | — | **71.28x** | **—** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | on | **166.0 μs** | 163.5 μs | 10.24 ms | — | **62.64x** | **—** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | on | **61.0 μs** | 64.3 μs | 2.05 ms | — | **33.59x** | **—** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | on | **31.5 μs** | 37.0 μs | 383.8 μs | — | **12.18x** | **—** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **84.8 μs** | 43.2 μs | 1.20 ms | — | **27.75x** | **—** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | on | **70.1 μs** | 68.0 μs | 1.77 ms | — | **25.98x** | **—** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | on | **76.3 μs** | 73.1 μs | 1.97 ms | — | **26.94x** | **—** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | on | **64.2 μs** | 69.2 μs | 1.60 ms | — | **24.94x** | **—** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | on | **35.6 μs** | 30.9 μs | 304.4 μs | — | **9.85x** | **—** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **7.89 ms** | 11.11 ms | 18.16 ms | — | **2.30x** | **—** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | on | **12.68 ms** | 11.74 ms | 25.33 ms | — | **2.16x** | **—** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | on | **23.56 ms** | 22.98 ms | 366.29 ms | — | **15.94x** | **—** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **289.8 μs** | 439.1 μs | 13.01 ms | — | **44.89x** | **—** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **57.0 μs** | 81.1 μs | 545.6 μs | — | **9.58x** | **—** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **929.6 μs** | 1.13 ms | 3.20 ms | — | **3.44x** | **—** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | on | **1.62 ms** | 847.5 μs | 3.66 ms | — | **4.32x** | **—** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | on | **2.70 ms** | 2.62 ms | 56.79 ms | — | **21.69x** | **—** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | on | **332.6 μs** | 298.6 μs | 13.84 ms | — | **46.34x** | **—** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | on | **51.2 μs** | 52.0 μs | 433.2 μs | — | **8.47x** | **—** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **228.5 μs** | 172.3 μs | 1.86 ms | — | **10.79x** | **—** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | on | **358.1 μs** | 357.2 μs | 2.21 ms | — | **6.18x** | **—** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | on | **322.8 μs** | 519.2 μs | 6.54 ms | — | **20.27x** | **—** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | on | **123.5 μs** | 129.7 μs | 3.06 ms | — | **24.79x** | **—** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | on | **52.2 μs** | 49.7 μs | 439.6 μs | — | **8.85x** | **—** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **71.4 μs** | 66.0 μs | 1.77 ms | — | **26.78x** | **—** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | on | **92.5 μs** | 87.3 μs | 1.86 ms | — | **21.28x** | **—** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | on | **127.9 μs** | 133.7 μs | 2.35 ms | — | **18.35x** | **—** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | on | **106.9 μs** | 96.7 μs | 2.60 ms | — | **26.92x** | **—** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | on | **53.5 μs** | 53.8 μs | 468.3 μs | — | **8.75x** | **—** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **6.07 ms** | 7.59 ms | 15.93 ms | — | **2.62x** | **—** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | on | **3.68 ms** | 3.59 ms | 8.49 ms | — | **2.36x** | **—** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.88 ms** | 4.53 ms | 14.75 ms | — | **3.80x** | **—** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **145.3 μs** | 145.0 μs | 4.11 ms | — | **28.33x** | **—** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **45.2 μs** | 49.1 μs | 432.6 μs | — | **9.56x** | **—** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.04 ms** | 1.02 ms | 2.34 ms | — | **2.29x** | **—** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | on | **622.5 μs** | 640.5 μs | 3.11 ms | — | **5.00x** | **—** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | on | **512.3 μs** | 522.6 μs | 2.77 ms | — | **5.40x** | **—** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | on | **167.4 μs** | 128.3 μs | 2.58 ms | — | **20.14x** | **—** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | on | **54.9 μs** | 55.5 μs | 557.4 μs | — | **10.16x** | **—** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **193.1 μs** | 122.7 μs | 1.34 ms | — | **10.93x** | **—** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | on | **163.2 μs** | 144.7 μs | 1.63 ms | — | **11.26x** | **—** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | on | **155.1 μs** | 140.1 μs | 1.80 ms | — | **12.86x** | **—** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | on | **127.5 μs** | 90.0 μs | 1.68 ms | — | **18.63x** | **—** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | on | **45.3 μs** | 37.2 μs | 404.5 μs | — | **10.87x** | **—** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **120.2 μs** | 112.2 μs | 2.42 ms | — | **21.62x** | **—** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | on | **77.0 μs** | 105.8 μs | 1.39 ms | — | **18.09x** | **—** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | on | **76.8 μs** | 90.1 μs | 1.31 ms | — | **17.08x** | **—** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | on | **79.1 μs** | 80.9 μs | 1.67 ms | — | **21.17x** | **—** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | on | **39.6 μs** | 42.5 μs | 460.2 μs | — | **11.63x** | **—** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **1.29 ms** | 485.8 μs | 1.42 ms | — | **2.91x** | **—** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | on | **1.81 ms** | 2.35 ms | 32.37 ms | — | **17.86x** | **—** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | on | **1.33 ms** | 1.64 ms | 36.38 ms | — | **27.36x** | **—** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | on | **219.0 μs** | 409.0 μs | 5.03 ms | — | **22.97x** | **—** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | on | **41.0 μs** | 36.6 μs | 371.8 μs | — | **10.15x** | **—** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **172.2 μs** | 65.8 μs | 1.11 ms | — | **16.93x** | **—** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | on | **198.2 μs** | 350.9 μs | 5.02 ms | — | **25.31x** | **—** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | on | **232.0 μs** | 286.3 μs | 5.50 ms | — | **23.69x** | **—** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | on | **86.9 μs** | 77.0 μs | 1.77 ms | — | **23.03x** | **—** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | on | **38.5 μs** | 36.2 μs | 365.2 μs | — | **10.07x** | **—** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **648.5 μs** | 303.2 μs | 1.33 ms | — | **4.38x** | **—** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | on | **109.74 ms** | 99.90 ms | 617.43 ms | — | **6.18x** | **—** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | on | **108.65 ms** | 117.21 ms | 664.28 ms | — | **6.11x** | **—** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | on | **10.23 ms** | 9.84 ms | 59.11 ms | — | **6.01x** | **—** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | on | **48.3 μs** | 42.5 μs | 410.5 μs | — | **9.66x** | **—** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **142.4 μs** | 63.1 μs | 968.4 μs | — | **15.35x** | **—** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | on | **11.02 ms** | 10.73 ms | 69.41 ms | — | **6.47x** | **—** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | on | **10.88 ms** | 9.80 ms | 61.78 ms | — | **6.30x** | **—** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | on | **869.9 μs** | 827.0 μs | 6.99 ms | — | **8.45x** | **—** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | on | **31.2 μs** | 38.7 μs | 376.5 μs | — | **12.08x** | **—** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **83.5 μs** | 87.7 μs | 1.20 ms | — | **14.36x** | **—** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | on | **812.8 μs** | 796.0 μs | 6.41 ms | — | **8.05x** | **—** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | on | **828.5 μs** | 760.1 μs | 6.62 ms | — | **8.70x** | **—** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | on | **183.7 μs** | 217.6 μs | 2.54 ms | — | **13.81x** | **—** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | on | **43.4 μs** | 44.3 μs | 495.6 μs | — | **11.42x** | **—** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.94 ms** | 2.06 ms | 8.96 ms | — | **4.63x** | **—** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | on | **3.30 ms** | 3.99 ms | 14.75 ms | — | **4.48x** | **—** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | on | **23.90 ms** | 23.07 ms | 181.64 ms | — | **7.87x** | **—** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.49 ms** | 1.92 ms | 32.59 ms | — | **21.90x** | **—** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | on | **131.3 μs** | 134.4 μs | 611.3 μs | — | **4.66x** | **—** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **340.0 μs** | 276.7 μs | 7.28 ms | — | **26.29x** | **—** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | on | **303.3 μs** | 336.5 μs | 6.47 ms | — | **21.33x** | **—** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | on | **1.68 ms** | 1.72 ms | 25.14 ms | — | **14.97x** | **—** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | on | **622.8 μs** | 608.3 μs | 12.88 ms | — | **21.17x** | **—** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | on | **148.4 μs** | 150.8 μs | 668.0 μs | — | **4.50x** | **—** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **113.6 μs** | 144.7 μs | 7.67 ms | — | **67.54x** | **—** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | on | **168.5 μs** | 169.5 μs | 7.20 ms | — | **42.73x** | **—** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | on | **605.5 μs** | 650.5 μs | 9.97 ms | — | **16.47x** | **—** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | on | **503.5 μs** | 536.6 μs | 11.16 ms | — | **22.16x** | **—** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | on | **159.3 μs** | 159.4 μs | 725.1 μs | — | **4.55x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Host | Domain | Case | mmap | torchfits (s) | TF RSS | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| NRC-054711 | fits | compressed_rice_1 [read_full @ mps] | on | 0.008075458012172021 | 568.40625 | fitsio/fitsio_torch_device | 1.0441445775240032 |
| NRC-054711 | fits | repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | n/a | 0.005808749992866069 | 642.984375 | fitsio/fitsio_torch | 1.0437067657218018 |
| NRC-054711 | fitstable | narrow_1000000 [predicate_filter] | off | 0.0128183749911841 | 862.015625 | astropy/astropy | 1.28382743251798 |
| NRC-054711 | fitstable | mixed_1000000 [predicate_filter] | off | 0.020739041006891057 | 1146.140625 | astropy/astropy | 1.1518596709517401 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest lab benchmarks (one row per host):

| Run ID | Host / device | Scope | Rows | Deficits | Median peak RSS (MB) | Notes |
|---|---|---|---:|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mps_20260717_000853` | NRC-054711 / mps | fits + fitstable (lab) | 3925 | 4 | 643 | lab + mmap-matrix + GPU |
| `exhaustive_cpu_20260716_191252` | torchfits-gpu-exhaustive-cpu-20260716-191252 / cpu | fits + fitstable (lab) | 2825 | 0 | 289 | lab + mmap-matrix |
| `exhaustive_cuda_20260716_191255` | torchfits-gpu-exhaustive-cuda-20260716-191255 / cuda | fits + fitstable (lab) | 4079 | 0 | 730 | lab + mmap-matrix + GPU |
<!-- BENCH_SNAPSHOT_END -->

### Host scorecard

| Host / device | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| NRC-054711 / mps | `exhaustive_mps_20260717_000853` | 3925 | 4 | 643 | lab + mmap-matrix + GPU |
| torchfits-gpu-exhaustive-cpu-20260716-191252 / cpu | `exhaustive_cpu_20260716_191252` | 2825 | 0 | 289 | lab + mmap-matrix |
| torchfits-gpu-exhaustive-cuda-20260716-191255 / cuda | `exhaustive_cuda_20260716_191255` | 4079 | 0 | 730 | lab + mmap-matrix + GPU |
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
