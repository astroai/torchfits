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
Source: `benchmarks_results/exhaustive_cuda_0.9.0_20260714_065950/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.13 ms` (n=269) | `0.80 ms` (n=269) | `0.25 ms` (n=269) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.15 ms` (n=269) | `0.72 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.14 ms` (n=267) | `0.68 ms` (n=90) | `0.22 ms` (n=90) | — |
| `disk→RAM→GPU` | `0.16 ms` (n=267) | `0.94 ms` (n=90) | `0.23 ms` (n=90) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.10 ms` (n=180) | `3.45 ms` (n=162) | `3.24 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.10 ms` (n=180) | `3.29 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **3.85 ms** | 3.86 ms | 16.67 ms | 5.89 ms | **4.33x** | **1.53x** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.42 ms** | 3.46 ms | 17.67 ms | 5.50 ms | **5.17x** | **1.61x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **9.06 ms** | 9.18 ms | 27.77 ms | 9.43 ms | **3.06x** | **1.04x** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **8.93 ms** | 8.98 ms | 27.65 ms | 9.28 ms | **3.09x** | **1.04x** |
| Repeated Cutouts (50x 100x100) | CPU | **4.68 ms** | 4.53 ms | 75.36 ms | 4.94 ms | **16.65x** | **1.09x** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **95.3 μs** | 97.8 μs | 6.74 ms | 59.84 ms | **70.64x** | **627.60x** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **93.9 μs** | 98.2 μs | 3.52 ms | 288.81 ms | **37.54x** | **3076.78x** |
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

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 150.7 μs | 2.11 ms | 272.7 μs | **13.98x** | **1.81x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **16.25 ms** | 16.30 ms | 40.35 ms | 17.93 ms | **2.48x** | **1.10x** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 148.7 μs | 2.12 ms | 273.0 μs | **14.28x** | **1.84x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **15.71 ms** | 15.79 ms | 67.08 ms | 17.58 ms | **4.27x** | **1.12x** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 155.1 μs | 2.21 ms | 297.1 μs | **14.28x** | **1.92x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **30.64 ms** | 30.71 ms | 38.28 ms | 29.45 ms | **1.25x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **911.5 μs** | 928.2 μs | 10.24 ms | 1.14 ms | **11.23x** | **1.25x** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 162.5 μs | 2.20 ms | 296.4 μs | **13.56x** | **1.82x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **9.06 ms** | 9.18 ms | 27.77 ms | 9.43 ms | **3.06x** | **1.04x** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 89.0 μs | 629.7 μs | 132.2 μs | **7.08x** | **1.49x** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **969.3 μs** | 987.0 μs | 2.20 ms | 1.48 ms | **2.27x** | **1.53x** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 97.1 μs | 662.6 μs | 134.1 μs | **6.83x** | **1.38x** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **3.85 ms** | 3.86 ms | 16.67 ms | 5.89 ms | **4.33x** | **1.53x** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 87.5 μs | 640.5 μs | 130.4 μs | **7.32x** | **1.49x** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **1.84 ms** | 1.86 ms | 3.80 ms | 2.37 ms | **2.06x** | **1.29x** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 92.7 μs | 660.6 μs | 136.4 μs | **7.12x** | **1.47x** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **10.45 ms** | 10.46 ms | 23.97 ms | 11.20 ms | **2.29x** | **1.07x** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 89.1 μs | 630.7 μs | 131.0 μs | **7.08x** | **1.47x** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **554.1 μs** | 575.4 μs | 1.42 ms | 697.2 μs | **2.56x** | **1.26x** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 92.9 μs | 658.8 μs | 134.5 μs | **7.09x** | **1.45x** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **2.04 ms** | 2.05 ms | 5.88 ms | 2.41 ms | **2.88x** | **1.18x** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 89.8 μs | 618.0 μs | 132.9 μs | **6.88x** | **1.48x** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **1.00 ms** | 1.00 ms | 2.22 ms | 1.50 ms | **2.22x** | **1.50x** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 92.7 μs | 652.4 μs | 136.9 μs | **7.04x** | **1.48x** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **3.91 ms** | 3.92 ms | 16.92 ms | 5.94 ms | **4.33x** | **1.52x** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 92.9 μs | 622.8 μs | 136.5 μs | **6.70x** | **1.47x** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **1.86 ms** | 1.86 ms | 3.83 ms | 2.39 ms | **2.06x** | **1.28x** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 96.7 μs | 640.9 μs | 137.7 μs | **6.63x** | **1.42x** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.59 ms** | 10.55 ms | 23.87 ms | 11.24 ms | **2.26x** | **1.07x** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 100.3 μs | 673.1 μs | 146.6 μs | **6.71x** | **1.46x** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **311.5 μs** | 337.0 μs | 1.25 ms | 458.6 μs | **4.02x** | **1.47x** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 103.8 μs | 706.9 μs | 148.2 μs | **6.81x** | **1.43x** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **1.10 ms** | 1.13 ms | 2.73 ms | 1.46 ms | **2.48x** | **1.33x** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 100.0 μs | 701.1 μs | 150.3 μs | **7.01x** | **1.50x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **2.49 ms** | 2.50 ms | 6.38 ms | 2.80 ms | **2.56x** | **1.13x** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 102.6 μs | 709.2 μs | 146.5 μs | **6.91x** | **1.43x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **4.98 ms** | 4.96 ms | 13.36 ms | 6.77 ms | **2.70x** | **1.37x** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 92.8 μs | 615.2 μs | 137.6 μs | **6.63x** | **1.48x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **184.3 μs** | 198.7 μs | 811.9 μs | 327.1 μs | **4.41x** | **1.78x** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 99.2 μs | 641.2 μs | 142.5 μs | **6.47x** | **1.44x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **1.05 ms** | 1.07 ms | 3.46 ms | 1.60 ms | **3.30x** | **1.52x** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 100.3 μs | 675.0 μs | 143.9 μs | **6.73x** | **1.44x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **1.58 ms** | 1.60 ms | 3.36 ms | 2.36 ms | **2.12x** | **1.50x** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 95.5 μs | 619.8 μs | 139.4 μs | **6.49x** | **1.46x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **271.5 μs** | 289.4 μs | 995.2 μs | 411.1 μs | **3.67x** | **1.51x** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 97.6 μs | 642.5 μs | 140.5 μs | **6.58x** | **1.44x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **2.01 ms** | 2.02 ms | 5.71 ms | 2.58 ms | **2.84x** | **1.28x** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 94.6 μs | 683.6 μs | 137.6 μs | **7.22x** | **1.45x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **3.06 ms** | 3.10 ms | 6.01 ms | 4.00 ms | **1.96x** | **1.31x** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 88.2 μs | 623.9 μs | 130.5 μs | **7.07x** | **1.48x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **150.4 μs** | 165.3 μs | 734.6 μs | 254.6 μs | **4.88x** | **1.69x** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 99.3 μs | 653.7 μs | 137.4 μs | **6.58x** | **1.38x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **637.8 μs** | 660.7 μs | 1.57 ms | 810.6 μs | **2.46x** | **1.27x** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 91.1 μs | 696.1 μs | 139.0 μs | **7.64x** | **1.52x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **938.8 μs** | 958.2 μs | 2.06 ms | 1.14 ms | **2.19x** | **1.22x** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 89.7 μs | 626.6 μs | 132.6 μs | **6.99x** | **1.48x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **187.0 μs** | 209.1 μs | 821.7 μs | 342.8 μs | **4.39x** | **1.83x** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 91.0 μs | 655.7 μs | 133.3 μs | **7.20x** | **1.46x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **1.13 ms** | 1.16 ms | 3.54 ms | 1.67 ms | **3.14x** | **1.48x** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 92.3 μs | 690.5 μs | 136.1 μs | **7.48x** | **1.47x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **1.62 ms** | 1.64 ms | 3.41 ms | 2.42 ms | **2.11x** | **1.50x** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 89.9 μs | 625.2 μs | 130.1 μs | **6.95x** | **1.45x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **282.1 μs** | 298.0 μs | 994.1 μs | 416.4 μs | **3.52x** | **1.48x** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 91.5 μs | 653.2 μs | 141.5 μs | **7.14x** | **1.55x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **2.07 ms** | 2.09 ms | 5.78 ms | 2.65 ms | **2.79x** | **1.28x** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 97.1 μs | 679.8 μs | 142.7 μs | **7.00x** | **1.47x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **3.20 ms** | 3.22 ms | 6.12 ms | 4.16 ms | **1.91x** | **1.30x** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 98.8 μs | 683.6 μs | 146.4 μs | **6.92x** | **1.48x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **119.0 μs** | 141.0 μs | 851.5 μs | 231.9 μs | **7.16x** | **1.95x** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 104.6 μs | 707.3 μs | 148.1 μs | **6.76x** | **1.42x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **361.2 μs** | 376.5 μs | 1.37 ms | 512.2 μs | **3.79x** | **1.42x** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 104.6 μs | 735.9 μs | 142.9 μs | **7.04x** | **1.37x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **509.8 μs** | 519.1 μs | 1.63 ms | 687.2 μs | **3.20x** | **1.35x** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 100.3 μs | 720.9 μs | 143.9 μs | **7.19x** | **1.43x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **652.6 μs** | 680.8 μs | 2.11 ms | 796.4 μs | **3.23x** | **1.22x** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 96.9 μs | 721.6 μs | 144.8 μs | **7.45x** | **1.50x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **1.19 ms** | 1.21 ms | 2.96 ms | 1.73 ms | **2.48x** | **1.45x** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 106.7 μs | 1.01 ms | 157.7 μs | **9.48x** | **1.48x** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **324.1 μs** | 346.1 μs | 1.56 ms | 507.0 μs | **4.82x** | **1.56x** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 106.1 μs | 1.00 ms | 158.8 μs | **9.47x** | **1.50x** |
| fits | mef_small | read_full | 0.45 MB | CPU | **111.6 μs** | 144.9 μs | 1.11 ms | 266.5 μs | **9.92x** | **2.39x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **94.6 μs** | 96.8 μs | 3.39 ms | 361.1 μs | **35.85x** | **3.82x** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 108.9 μs | 983.6 μs | 157.7 μs | **9.03x** | **1.45x** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **6.62 ms** | 6.62 ms | 11.25 ms | 10.19 ms | **1.70x** | **1.54x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **118.5 μs** | 142.1 μs | 1.10 ms | 329.8 μs | **9.32x** | **2.78x** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **4.68 ms** | 4.53 ms | 75.36 ms | 4.94 ms | **16.65x** | **1.09x** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 105.3 μs | 730.6 μs | 146.3 μs | **6.94x** | **1.39x** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **3.53 ms** | 3.56 ms | 11.44 ms | 4.76 ms | **3.24x** | **1.35x** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 100.6 μs | 729.1 μs | 145.7 μs | **7.25x** | **1.45x** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **1.00 ms** | 1.03 ms | 2.46 ms | 1.38 ms | **2.45x** | **1.37x** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 97.6 μs | 726.8 μs | 150.2 μs | **7.44x** | **1.54x** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **154.0 μs** | 178.5 μs | 935.1 μs | 276.6 μs | **6.07x** | **1.80x** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 94.6 μs | 619.8 μs | 134.8 μs | **6.55x** | **1.42x** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **103.9 μs** | 123.6 μs | 667.5 μs | 214.4 μs | **6.43x** | **2.06x** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 96.4 μs | 652.0 μs | 137.4 μs | **6.77x** | **1.43x** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **155.8 μs** | 179.1 μs | 779.2 μs | 296.2 μs | **5.00x** | **1.90x** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 98.5 μs | 683.3 μs | 139.5 μs | **6.94x** | **1.42x** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **260.2 μs** | 278.6 μs | 978.4 μs | 437.7 μs | **3.76x** | **1.68x** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 94.7 μs | 621.1 μs | 134.0 μs | **6.56x** | **1.41x** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **113.6 μs** | 127.5 μs | 673.4 μs | 220.5 μs | **5.93x** | **1.94x** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 99.9 μs | 657.8 μs | 138.0 μs | **6.59x** | **1.38x** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **220.4 μs** | 236.7 μs | 887.6 μs | 350.3 μs | **4.03x** | **1.59x** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 97.6 μs | 679.7 μs | 142.0 μs | **6.96x** | **1.45x** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **431.6 μs** | 453.9 μs | 1.29 ms | 612.9 μs | **2.98x** | **1.42x** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 94.3 μs | 631.8 μs | 131.6 μs | **6.70x** | **1.40x** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **98.1 μs** | 116.7 μs | 659.3 μs | 199.7 μs | **6.72x** | **2.04x** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 91.5 μs | 658.5 μs | 141.8 μs | **7.20x** | **1.55x** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **131.3 μs** | 149.8 μs | 718.0 μs | 236.8 μs | **5.47x** | **1.80x** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 97.1 μs | 678.8 μs | 143.2 μs | **6.99x** | **1.47x** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **180.9 μs** | 206.2 μs | 817.9 μs | 293.5 μs | **4.52x** | **1.62x** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 94.5 μs | 623.4 μs | 134.2 μs | **6.60x** | **1.42x** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **109.4 μs** | 126.3 μs | 669.4 μs | 212.9 μs | **6.12x** | **1.95x** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 96.5 μs | 655.0 μs | 138.5 μs | **6.79x** | **1.44x** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **164.6 μs** | 178.5 μs | 774.5 μs | 299.7 μs | **4.71x** | **1.82x** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 97.9 μs | 683.7 μs | 144.9 μs | **6.98x** | **1.48x** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **256.3 μs** | 276.6 μs | 978.6 μs | 439.8 μs | **3.82x** | **1.72x** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 94.0 μs | 621.3 μs | 134.3 μs | **6.61x** | **1.43x** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **115.1 μs** | 134.0 μs | 680.7 μs | 222.3 μs | **5.91x** | **1.93x** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 96.1 μs | 646.9 μs | 140.2 μs | **6.73x** | **1.46x** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **231.5 μs** | 241.8 μs | 901.4 μs | 359.9 μs | **3.89x** | **1.55x** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 96.5 μs | 679.3 μs | 139.8 μs | **7.04x** | **1.45x** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **423.4 μs** | 450.6 μs | 1.28 ms | 607.4 μs | **3.02x** | **1.43x** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 97.9 μs | 678.4 μs | 144.9 μs | **6.93x** | **1.48x** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **88.2 μs** | 107.4 μs | 809.5 μs | 203.1 μs | **9.18x** | **2.30x** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 97.4 μs | 720.3 μs | 146.1 μs | **7.39x** | **1.50x** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **114.9 μs** | 136.2 μs | 847.9 μs | 219.1 μs | **7.38x** | **1.91x** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 100.2 μs | 745.4 μs | 147.4 μs | **7.44x** | **1.47x** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **142.1 μs** | 166.4 μs | 916.2 μs | 254.1 μs | **6.45x** | **1.79x** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 95.0 μs | 711.0 μs | 144.5 μs | **7.49x** | **1.52x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **129.1 μs** | 151.1 μs | 828.6 μs | 240.8 μs | **6.42x** | **1.87x** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 97.5 μs | 708.6 μs | 143.2 μs | **7.27x** | **1.47x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **162.2 μs** | 181.2 μs | 872.7 μs | 284.8 μs | **5.38x** | **1.76x** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 89.8 μs | 664.3 μs | 134.0 μs | **7.40x** | **1.49x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **158.8 μs** | 173.0 μs | 759.0 μs | 273.6 μs | **4.78x** | **1.72x** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 91.8 μs | 666.4 μs | 135.0 μs | **7.26x** | **1.47x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **159.7 μs** | 175.0 μs | 750.5 μs | 274.7 μs | **4.70x** | **1.72x** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 92.4 μs | 658.6 μs | 136.5 μs | **7.12x** | **1.48x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **156.1 μs** | 171.5 μs | 760.7 μs | 272.6 μs | **4.87x** | **1.75x** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 91.3 μs | 656.6 μs | 133.8 μs | **7.19x** | **1.47x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **150.9 μs** | 170.5 μs | 762.6 μs | 272.0 μs | **5.05x** | **1.80x** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 93.5 μs | 663.8 μs | 134.7 μs | **7.10x** | **1.44x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **147.9 μs** | 167.0 μs | 758.4 μs | 277.5 μs | **5.13x** | **1.88x** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 87.9 μs | 638.8 μs | 130.8 μs | **7.27x** | **1.49x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **85.6 μs** | 101.3 μs | 645.5 μs | 199.4 μs | **7.54x** | **2.33x** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 90.2 μs | 660.6 μs | 134.5 μs | **7.33x** | **1.49x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **101.5 μs** | 121.5 μs | 672.2 μs | 201.6 μs | **6.62x** | **1.99x** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 94.1 μs | 689.4 μs | 140.9 μs | **7.33x** | **1.50x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **103.0 μs** | 122.8 μs | 696.1 μs | 212.1 μs | **6.76x** | **2.06x** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 92.9 μs | 630.7 μs | 131.4 μs | **6.79x** | **1.42x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **97.4 μs** | 109.0 μs | 645.7 μs | 197.0 μs | **6.63x** | **2.02x** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 92.4 μs | 659.7 μs | 135.3 μs | **7.14x** | **1.46x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **115.2 μs** | 133.9 μs | 686.4 μs | 210.4 μs | **5.96x** | **1.83x** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 93.3 μs | 693.1 μs | 135.5 μs | **7.43x** | **1.45x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **107.8 μs** | 137.1 μs | 709.2 μs | 212.3 μs | **6.58x** | **1.97x** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 85.6 μs | 630.8 μs | 132.0 μs | **7.37x** | **1.54x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **85.0 μs** | 101.0 μs | 642.1 μs | 195.1 μs | **7.56x** | **2.30x** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 89.0 μs | 657.7 μs | 134.0 μs | **7.39x** | **1.51x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **87.6 μs** | 106.8 μs | 671.9 μs | 196.5 μs | **7.67x** | **2.24x** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 93.4 μs | 685.6 μs | 136.4 μs | **7.34x** | **1.46x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **85.5 μs** | 106.8 μs | 693.9 μs | 200.6 μs | **8.12x** | **2.35x** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 87.7 μs | 630.5 μs | 130.2 μs | **7.19x** | **1.48x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **84.1 μs** | 100.5 μs | 655.2 μs | 194.8 μs | **7.79x** | **2.32x** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 89.8 μs | 656.5 μs | 131.9 μs | **7.31x** | **1.47x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **100.5 μs** | 118.6 μs | 673.2 μs | 204.9 μs | **6.70x** | **2.04x** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 92.0 μs | 690.2 μs | 137.6 μs | **7.51x** | **1.50x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **101.0 μs** | 122.6 μs | 688.5 μs | 202.3 μs | **6.82x** | **2.00x** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 89.4 μs | 626.0 μs | 128.3 μs | **7.00x** | **1.43x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **84.5 μs** | 102.9 μs | 640.4 μs | 193.5 μs | **7.58x** | **2.29x** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 89.1 μs | 655.8 μs | 133.6 μs | **7.36x** | **1.50x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **106.3 μs** | 124.1 μs | 678.7 μs | 206.9 μs | **6.39x** | **1.95x** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 97.3 μs | 691.0 μs | 135.8 μs | **7.10x** | **1.40x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **106.4 μs** | 125.1 μs | 706.5 μs | 207.3 μs | **6.64x** | **1.95x** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 92.8 μs | 683.1 μs | 146.0 μs | **7.36x** | **1.57x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **80.4 μs** | 106.0 μs | 799.9 μs | 192.3 μs | **9.95x** | **2.39x** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 96.4 μs | 715.8 μs | 141.8 μs | **7.42x** | **1.47x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **86.1 μs** | 106.3 μs | 814.4 μs | 201.8 μs | **9.46x** | **2.34x** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 99.1 μs | 744.4 μs | 146.6 μs | **7.51x** | **1.48x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **91.0 μs** | 112.8 μs | 843.0 μs | 197.0 μs | **9.26x** | **2.16x** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CUDA | **16.04 ms** | 16.16 ms | 40.43 ms | 17.68 ms | **2.52x** | **1.10x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CUDA | **15.63 ms** | 15.74 ms | 66.18 ms | 17.34 ms | **4.23x** | **1.11x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CUDA | **30.48 ms** | 30.54 ms | 37.96 ms | 29.23 ms | **1.25x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CUDA | **826.3 μs** | 825.4 μs | 10.08 ms | 970.7 μs | **12.22x** | **1.18x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CUDA | **8.93 ms** | 8.98 ms | 27.65 ms | 9.28 ms | **3.09x** | **1.04x** |
| fits | large_float32_1d | read_full | 3.82 MB | CUDA | **817.3 μs** | 838.7 μs | 1.61 ms | 1.33 ms | **1.97x** | **1.63x** |
| fits | large_float32_2d | read_full | 16.00 MB | CUDA | **3.42 ms** | 3.46 ms | 17.67 ms | 5.50 ms | **5.17x** | **1.61x** |
| fits | large_float64_1d | read_full | 7.63 MB | CUDA | **1.52 ms** | 1.54 ms | 3.09 ms | 2.02 ms | **2.03x** | **1.33x** |
| fits | large_float64_2d | read_full | 32.00 MB | CUDA | **11.65 ms** | 11.79 ms | 26.07 ms | 12.69 ms | **2.24x** | **1.09x** |
| fits | large_int16_1d | read_full | 1.91 MB | CUDA | **475.1 μs** | 491.4 μs | 1.06 ms | 616.7 μs | **2.23x** | **1.30x** |
| fits | large_int16_2d | read_full | 8.00 MB | CUDA | **1.71 ms** | 1.72 ms | 5.46 ms | 2.06 ms | **3.20x** | **1.21x** |
| fits | large_int32_1d | read_full | 3.82 MB | CUDA | **819.5 μs** | 839.8 μs | 1.62 ms | 1.33 ms | **1.97x** | **1.63x** |
| fits | large_int32_2d | read_full | 16.00 MB | CUDA | **3.51 ms** | 3.83 ms | 17.80 ms | 5.48 ms | **5.08x** | **1.56x** |
| fits | large_int64_1d | read_full | 7.63 MB | CUDA | **1.54 ms** | 1.57 ms | 3.13 ms | 2.05 ms | **2.03x** | **1.33x** |
| fits | large_int64_2d | read_full | 32.00 MB | CUDA | **11.88 ms** | 11.93 ms | 26.59 ms | 13.06 ms | **2.24x** | **1.10x** |
| fits | large_int8_1d | read_full | 0.96 MB | CUDA | **233.7 μs** | 303.1 μs | 930.2 μs | 380.0 μs | **3.98x** | **1.63x** |
| fits | large_int8_2d | read_full | 4.00 MB | CUDA | **698.1 μs** | 960.4 μs | 3.15 ms | 1.15 ms | **4.52x** | **1.65x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CUDA | **1.66 ms** | 1.99 ms | 5.38 ms | 2.33 ms | **3.25x** | **1.41x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CUDA | **3.32 ms** | 3.96 ms | 12.69 ms | 5.95 ms | **3.82x** | **1.79x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CUDA | **106.5 μs** | 126.5 μs | 549.7 μs | 219.3 μs | **5.16x** | **2.06x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CUDA | **889.2 μs** | 909.8 μs | 3.13 ms | 1.42 ms | **3.52x** | **1.60x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CUDA | **1.32 ms** | 1.35 ms | 2.54 ms | 2.13 ms | **1.92x** | **1.61x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CUDA | **204.6 μs** | 224.9 μs | 706.6 μs | 311.6 μs | **3.45x** | **1.52x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CUDA | **1.65 ms** | 1.67 ms | 5.28 ms | 2.18 ms | **3.20x** | **1.32x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CUDA | **2.60 ms** | 2.65 ms | 10.43 ms | 3.51 ms | **4.00x** | **1.35x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CUDA | **63.7 μs** | 79.5 μs | 484.8 μs | 135.5 μs | **7.61x** | **2.13x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CUDA | **517.4 μs** | 538.5 μs | 1.89 ms | 656.9 μs | **3.66x** | **1.27x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CUDA | **733.4 μs** | 752.1 μs | 1.45 ms | 913.3 μs | **1.97x** | **1.25x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CUDA | **105.3 μs** | 124.3 μs | 534.0 μs | 215.4 μs | **5.07x** | **2.05x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CUDA | **881.3 μs** | 900.5 μs | 1.83 ms | 1.42 ms | **2.08x** | **1.61x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CUDA | **1.31 ms** | 1.33 ms | 2.55 ms | 2.14 ms | **1.94x** | **1.63x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CUDA | **205.8 μs** | 224.3 μs | 695.6 μs | 305.0 μs | **3.38x** | **1.48x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CUDA | **1.65 ms** | 1.68 ms | 4.70 ms | 2.23 ms | **2.84x** | **1.35x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CUDA | **2.62 ms** | 2.66 ms | 10.34 ms | 3.45 ms | **3.95x** | **1.32x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CUDA | **44.9 μs** | 61.8 μs | 575.3 μs | 108.8 μs | **12.82x** | **2.42x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CUDA | **249.5 μs** | 327.3 μs | 1.27 ms | 401.5 μs | **5.09x** | **1.61x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CUDA | **327.8 μs** | 443.0 μs | 1.19 ms | 549.7 μs | **3.63x** | **1.68x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CUDA | **493.2 μs** | 586.3 μs | 1.72 ms | 705.1 μs | **3.49x** | **1.43x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CUDA | **841.4 μs** | 1.02 ms | 2.34 ms | 1.53 ms | **2.78x** | **1.82x** |
| fits | mef_medium | read_full | 7.02 MB | CUDA | **238.0 μs** | 314.4 μs | 1.46 ms | 418.9 μs | **6.13x** | **1.76x** |
| fits | mef_small | read_full | 0.45 MB | CUDA | **41.8 μs** | 57.5 μs | 794.9 μs | 134.0 μs | **19.01x** | **3.20x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CUDA | **41.1 μs** | 41.7 μs | 3.22 ms | 219.6 μs | **78.40x** | **5.35x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CUDA | **47.4 μs** | 65.2 μs | 807.0 μs | 198.7 μs | **17.01x** | **4.19x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | CUDA | **6.30 ms** | 5.85 ms | 82.38 ms | 6.35 ms | **14.09x** | **1.09x** |
| fits | scaled_large | read_full | 8.00 MB | CUDA | **1.68 ms** | 4.26 ms | 11.61 ms | 4.88 ms | **6.91x** | **2.91x** |
| fits | scaled_medium | read_full | 2.01 MB | CUDA | **494.5 μs** | 1.10 ms | 1.95 ms | 1.29 ms | **3.95x** | **2.61x** |
| fits | scaled_small | read_full | 0.13 MB | CUDA | **54.1 μs** | 129.5 μs | 674.3 μs | 168.7 μs | **12.45x** | **3.12x** |
| fits | small_float32_1d | read_full | 42.2 KB | CUDA | **39.6 μs** | 49.1 μs | 421.6 μs | 91.7 μs | **10.64x** | **2.32x** |
| fits | small_float32_2d | read_full | 0.26 MB | CUDA | **75.0 μs** | 94.0 μs | 529.8 μs | 174.0 μs | **7.06x** | **2.32x** |
| fits | small_float32_3d | read_full | 0.63 MB | CUDA | **169.3 μs** | 194.0 μs | 679.9 μs | 309.2 μs | **4.02x** | **1.83x** |
| fits | small_float64_1d | read_full | 0.08 MB | CUDA | **42.4 μs** | 53.7 μs | 431.9 μs | 94.0 μs | **10.18x** | **2.22x** |
| fits | small_float64_2d | read_full | 0.51 MB | CUDA | **139.3 μs** | 159.9 μs | 592.9 μs | 232.8 μs | **4.26x** | **1.67x** |
| fits | small_float64_3d | read_full | 1.26 MB | CUDA | **327.0 μs** | 347.4 μs | 889.9 μs | 465.4 μs | **2.72x** | **1.42x** |
| fits | small_int16_1d | read_full | 22.5 KB | CUDA | **34.0 μs** | 46.4 μs | 400.2 μs | 79.9 μs | **11.78x** | **2.35x** |
| fits | small_int16_2d | read_full | 0.13 MB | CUDA | **54.1 μs** | 64.9 μs | 469.8 μs | 108.7 μs | **8.69x** | **2.01x** |
| fits | small_int16_3d | read_full | 0.32 MB | CUDA | **96.1 μs** | 113.5 μs | 555.1 μs | 179.4 μs | **5.78x** | **1.87x** |
| fits | small_int32_1d | read_full | 42.2 KB | CUDA | **37.3 μs** | 50.8 μs | 416.5 μs | 90.1 μs | **11.15x** | **2.41x** |
| fits | small_int32_2d | read_full | 0.26 MB | CUDA | **73.8 μs** | 90.5 μs | 514.6 μs | 171.7 μs | **6.98x** | **2.33x** |
| fits | small_int32_3d | read_full | 0.63 MB | CUDA | **171.8 μs** | 192.5 μs | 659.1 μs | 303.0 μs | **3.84x** | **1.76x** |
| fits | small_int64_1d | read_full | 0.08 MB | CUDA | **42.8 μs** | 54.6 μs | 432.6 μs | 93.6 μs | **10.11x** | **2.19x** |
| fits | small_int64_2d | read_full | 0.51 MB | CUDA | **140.4 μs** | 161.8 μs | 592.8 μs | 228.8 μs | **4.22x** | **1.63x** |
| fits | small_int64_3d | read_full | 1.26 MB | CUDA | **330.9 μs** | 347.8 μs | 895.9 μs | 474.4 μs | **2.71x** | **1.43x** |
| fits | small_int8_1d | read_full | 14.1 KB | CUDA | **29.4 μs** | 39.0 μs | 536.9 μs | 77.3 μs | **18.27x** | **2.63x** |
| fits | small_int8_2d | read_full | 0.07 MB | CUDA | **47.7 μs** | 62.2 μs | 578.3 μs | 100.1 μs | **12.12x** | **2.10x** |
| fits | small_int8_3d | read_full | 0.16 MB | CUDA | **51.3 μs** | 79.0 μs | 629.1 μs | 134.2 μs | **12.27x** | **2.62x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CUDA | **52.5 μs** | 71.9 μs | 565.1 μs | 121.6 μs | **10.76x** | **2.31x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CUDA | **74.6 μs** | 99.4 μs | 608.6 μs | 176.6 μs | **8.16x** | **2.37x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CUDA | **69.8 μs** | 89.2 μs | 508.4 μs | 169.8 μs | **7.28x** | **2.43x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CUDA | **70.3 μs** | 88.7 μs | 514.1 μs | 164.7 μs | **7.31x** | **2.34x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CUDA | **69.8 μs** | 88.1 μs | 513.0 μs | 165.1 μs | **7.35x** | **2.36x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CUDA | **68.9 μs** | 89.7 μs | 503.3 μs | 165.0 μs | **7.30x** | **2.39x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CUDA | **68.5 μs** | 90.9 μs | 513.3 μs | 171.3 μs | **7.50x** | **2.50x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CUDA | **26.7 μs** | 37.4 μs | 396.8 μs | 79.1 μs | **14.85x** | **2.96x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CUDA | **33.0 μs** | 45.2 μs | 421.4 μs | 83.9 μs | **12.78x** | **2.54x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CUDA | **33.3 μs** | 48.3 μs | 437.2 μs | 86.7 μs | **13.11x** | **2.60x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CUDA | **27.7 μs** | 39.3 μs | 400.9 μs | 78.2 μs | **14.46x** | **2.82x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CUDA | **34.5 μs** | 48.0 μs | 436.3 μs | 85.5 μs | **12.63x** | **2.47x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CUDA | **37.7 μs** | 50.8 μs | 458.2 μs | 94.0 μs | **12.14x** | **2.49x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CUDA | **26.5 μs** | 35.8 μs | 381.4 μs | 73.6 μs | **14.40x** | **2.78x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CUDA | **27.9 μs** | 38.0 μs | 403.9 μs | 76.1 μs | **14.48x** | **2.73x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CUDA | **28.8 μs** | 39.7 μs | 435.0 μs | 76.2 μs | **15.12x** | **2.65x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CUDA | **27.1 μs** | 36.1 μs | 386.7 μs | 76.0 μs | **14.27x** | **2.80x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CUDA | **33.5 μs** | 44.1 μs | 417.6 μs | 81.8 μs | **12.46x** | **2.44x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CUDA | **34.3 μs** | 45.0 μs | 446.8 μs | 86.6 μs | **13.03x** | **2.53x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CUDA | **27.6 μs** | 38.0 μs | 392.5 μs | 75.1 μs | **14.23x** | **2.72x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CUDA | **36.4 μs** | 49.2 μs | 428.4 μs | 82.3 μs | **11.76x** | **2.26x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CUDA | **38.5 μs** | 52.8 μs | 448.2 μs | 86.8 μs | **11.63x** | **2.25x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CUDA | **26.9 μs** | 37.5 μs | 533.2 μs | 74.0 μs | **19.86x** | **2.75x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CUDA | **27.7 μs** | 40.3 μs | 543.1 μs | 73.7 μs | **19.60x** | **2.66x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CUDA | **27.7 μs** | 40.2 μs | 566.6 μs | 76.9 μs | **20.44x** | **2.78x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **1.01 ms** | 307.9 μs | 4.92 ms | 7.60 ms | **15.98x** | **24.67x** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **96.4 μs** | 99.3 μs | 11.53 ms | 7.58 ms | **119.63x** | **78.61x** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **93.2 μs** | 97.8 μs | 2.25 ms | 7.51 ms | **24.20x** | **80.55x** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **94.5 μs** | 96.7 μs | 2.69 ms | 3.71 ms | **28.44x** | **39.26x** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **134.2 μs** | 102.4 μs | 3.98 ms | 3.15 ms | **38.83x** | **30.71x** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **652.5 μs** | 245.3 μs | 3.10 ms | 1.23 ms | **12.63x** | **5.01x** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **99.6 μs** | 103.6 μs | 3.34 ms | 1.27 ms | **33.51x** | **12.78x** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **102.4 μs** | 102.5 μs | 2.10 ms | 1.22 ms | **20.47x** | **11.88x** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **100.5 μs** | 103.2 μs | 2.59 ms | 809.7 μs | **25.79x** | **8.06x** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **130.0 μs** | 109.4 μs | 2.43 ms | 664.3 μs | **22.16x** | **6.07x** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **22.24 ms** | 8.44 ms | 107.99 ms | 400.23 ms | **12.80x** | **47.44x** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **93.1 μs** | 94.7 μs | 19.59 ms | 59.43 ms | **210.46x** | **638.51x** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **90.4 μs** | 94.5 μs | 57.49 ms | 630.64 ms | **635.76x** | **6973.81x** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **100.6 μs** | 93.8 μs | 21.06 ms | 133.05 ms | **224.42x** | **1417.91x** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **146.9 μs** | 106.8 μs | 19.84 ms | 125.84 ms | **185.68x** | **1177.97x** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **4.00 ms** | 1.39 ms | 15.89 ms | 35.93 ms | **11.41x** | **25.80x** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **95.3 μs** | 97.3 μs | 4.38 ms | 6.04 ms | **45.93x** | **63.40x** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **95.3 μs** | 97.8 μs | 6.74 ms | 59.84 ms | **70.64x** | **627.60x** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **99.9 μs** | 100.7 μs | 5.62 ms | 20.91 ms | **56.27x** | **209.24x** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **138.7 μs** | 102.6 μs | 4.30 ms | 12.24 ms | **41.93x** | **119.32x** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **993.8 μs** | 302.4 μs | 4.82 ms | 4.08 ms | **15.94x** | **13.50x** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **93.7 μs** | 98.5 μs | 3.10 ms | 1.07 ms | **33.08x** | **11.44x** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **92.2 μs** | 92.8 μs | 3.27 ms | 6.13 ms | **35.52x** | **66.50x** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **95.7 μs** | 101.0 μs | 4.24 ms | 2.19 ms | **44.35x** | **22.91x** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **145.5 μs** | 109.4 μs | 3.18 ms | 1.65 ms | **29.04x** | **15.09x** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **670.3 μs** | 222.3 μs | 4.01 ms | 915.8 μs | **18.02x** | **4.12x** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **95.3 μs** | 99.1 μs | 2.83 ms | 520.8 μs | **29.73x** | **5.47x** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **90.5 μs** | 97.0 μs | 2.87 ms | 1.11 ms | **31.69x** | **12.25x** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **96.2 μs** | 99.8 μs | 3.89 ms | 717.8 μs | **40.39x** | **7.46x** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **144.0 μs** | 106.8 μs | 2.81 ms | 564.4 μs | **26.35x** | **5.28x** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **13.42 ms** | 7.98 ms | 36.72 ms | 13.05 ms | **4.60x** | **1.63x** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **98.4 μs** | 102.4 μs | 6.33 ms | 40.30 ms | **64.36x** | **409.52x** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **95.5 μs** | 102.4 μs | 10.31 ms | 8.65 ms | **107.95x** | **90.58x** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **91.4 μs** | 96.7 μs | 7.01 ms | 5.17 ms | **76.68x** | **56.54x** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **125.9 μs** | 106.7 μs | 6.30 ms | 5.02 ms | **59.02x** | **47.07x** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **2.11 ms** | 1.07 ms | 6.48 ms | 1.77 ms | **6.06x** | **1.66x** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **96.4 μs** | 99.4 μs | 2.76 ms | 4.65 ms | **28.67x** | **48.24x** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **96.0 μs** | 100.2 μs | 3.13 ms | 1.33 ms | **32.58x** | **13.85x** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **96.0 μs** | 99.5 μs | 3.41 ms | 1.01 ms | **35.52x** | **10.50x** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **128.9 μs** | 110.1 μs | 2.78 ms | 918.9 μs | **25.29x** | **8.35x** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **723.1 μs** | 312.7 μs | 3.13 ms | 585.3 μs | **10.01x** | **1.87x** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **96.8 μs** | 98.7 μs | 2.19 ms | 865.3 μs | **22.62x** | **8.94x** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **92.6 μs** | 97.6 μs | 2.20 ms | 504.8 μs | **23.73x** | **5.45x** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **96.2 μs** | 97.4 μs | 2.75 ms | 496.9 μs | **28.62x** | **5.17x** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **129.3 μs** | 104.8 μs | 2.17 ms | 435.2 μs | **20.69x** | **4.15x** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **643.1 μs** | 234.1 μs | 2.76 ms | 447.0 μs | **11.81x** | **1.91x** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **94.9 μs** | 100.7 μs | 2.16 ms | 483.8 μs | **22.81x** | **5.10x** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **94.2 μs** | 94.1 μs | 2.17 ms | 440.0 μs | **23.04x** | **4.67x** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **95.6 μs** | 102.0 μs | 2.73 ms | 428.9 μs | **28.57x** | **4.49x** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **129.8 μs** | 107.9 μs | 2.09 ms | 388.9 μs | **19.40x** | **3.60x** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **2.02 ms** | 960.0 μs | 6.07 ms | 63.94 ms | **6.32x** | **66.61x** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **98.7 μs** | 104.2 μs | 39.17 ms | 60.57 ms | **396.70x** | **613.49x** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **101.6 μs** | 102.0 μs | 3.71 ms | 62.82 ms | **36.52x** | **618.25x** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **100.5 μs** | 102.7 μs | 3.49 ms | 22.37 ms | **34.70x** | **222.55x** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **130.9 μs** | 113.4 μs | 2.79 ms | 17.90 ms | **24.61x** | **157.80x** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **810.9 μs** | 312.5 μs | 3.29 ms | 6.81 ms | **10.52x** | **21.79x** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **100.6 μs** | 106.7 μs | 5.93 ms | 6.52 ms | **59.00x** | **64.80x** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **98.3 μs** | 104.6 μs | 2.39 ms | 6.67 ms | **24.33x** | **67.91x** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **103.0 μs** | 107.7 μs | 2.95 ms | 2.83 ms | **28.60x** | **27.47x** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **135.1 μs** | 114.8 μs | 2.33 ms | 2.27 ms | **20.33x** | **19.74x** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **1.74 ms** | 953.0 μs | 6.12 ms | 223.44 ms | **6.42x** | **234.45x** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **102.3 μs** | 104.2 μs | 775.15 ms | 229.53 ms | **7577.39x** | **2243.78x** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **93.9 μs** | 98.2 μs | 3.52 ms | 288.81 ms | **37.54x** | **3076.78x** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **100.7 μs** | 106.6 μs | 3.31 ms | 225.10 ms | **32.91x** | **2235.66x** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **126.1 μs** | 115.8 μs | 2.70 ms | 228.04 ms | **23.28x** | **1969.22x** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **683.9 μs** | 309.6 μs | 2.98 ms | 21.42 ms | **9.62x** | **69.20x** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **97.0 μs** | 98.0 μs | 78.90 ms | 21.23 ms | **813.57x** | **218.94x** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **92.2 μs** | 95.6 μs | 2.20 ms | 20.92 ms | **23.85x** | **226.77x** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **91.9 μs** | 96.8 μs | 2.63 ms | 21.37 ms | **28.64x** | **232.39x** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **122.8 μs** | 101.0 μs | 2.09 ms | 21.21 ms | **20.72x** | **210.01x** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **617.9 μs** | 236.0 μs | 2.61 ms | 2.41 ms | **11.04x** | **10.20x** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **97.4 μs** | 95.3 μs | 9.80 ms | 2.45 ms | **102.83x** | **25.66x** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **91.2 μs** | 96.4 μs | 2.03 ms | 2.36 ms | **22.29x** | **25.87x** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **94.2 μs** | 97.0 μs | 2.50 ms | 2.36 ms | **26.58x** | **25.03x** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **123.5 μs** | 104.9 μs | 1.99 ms | 2.41 ms | **18.99x** | **22.97x** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **7.12 ms** | 1.04 ms | 57.63 ms | 149.02 ms | **55.43x** | **143.32x** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **96.6 μs** | 96.7 μs | 12.62 ms | 9.68 ms | **130.55x** | **100.19x** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **93.3 μs** | 93.9 μs | 80.69 ms | 238.90 ms | **864.65x** | **2560.06x** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **98.0 μs** | 96.0 μs | 18.73 ms | 68.52 ms | **195.21x** | **714.10x** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **249.3 μs** | 102.6 μs | 12.61 ms | 50.85 ms | **122.97x** | **495.79x** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.97 ms** | 310.9 μs | 16.91 ms | 15.54 ms | **54.40x** | **49.97x** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **96.8 μs** | 98.6 μs | 9.31 ms | 1.91 ms | **96.20x** | **19.71x** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **95.9 μs** | 97.5 μs | 10.99 ms | 23.77 ms | **114.56x** | **247.88x** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **96.7 μs** | 100.6 μs | 14.18 ms | 7.96 ms | **146.66x** | **82.34x** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **251.3 μs** | 107.6 μs | 9.16 ms | 5.93 ms | **85.13x** | **55.06x** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **1.06 ms** | 237.4 μs | 13.53 ms | 2.41 ms | **56.99x** | **10.16x** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **94.5 μs** | 98.1 μs | 8.56 ms | 826.8 μs | **90.62x** | **8.75x** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **94.2 μs** | 96.5 μs | 8.68 ms | 3.25 ms | **92.14x** | **34.45x** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **94.2 μs** | 97.5 μs | 13.17 ms | 1.68 ms | **139.83x** | **17.80x** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **243.5 μs** | 104.0 μs | 8.50 ms | 1.29 ms | **81.80x** | **12.37x** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family. All 7 are
small-N or niche compression cases; no large-N (≥100k rows) deficits exist.

| Domain | Case | torchfits | Winner | Lag | Explanation |
|---|---|---:|---|---:|---|
| fits | `compressed_hcompress_1` read_full (CPU, mmap-on) | 30.62 ms | fitsio | 1.04× | hcompress decode is fitsio-native; torchfits delegates to the same CFITSIO path but pays Arrow conversion overhead on this rarely-used compression. |
| fits | `compressed_hcompress_1` read_full (CPU, mmap-off) | 30.56 ms | fitsio | 1.04× | Same as above, buffered path. |
| fits | `compressed_hcompress_1` read_full (CUDA) | 30.51 ms | fitsio | 1.04× | Same CFITSIO decode + H2D copy; fitsio's torch integration avoids one intermediate copy. |
| fits | `tiny_int8_1d` read_full (CUDA) | 72.6 μs | fitsio | 1.01× | Sub-100 μs reads are dominated by launch overhead; fitsio's `fitsio_torch_device` path has marginally lower Python-side latency for tiny payloads. |
| fits | `tiny_int8_2d` read_full (CUDA) | 75.3 μs | fitsio | 1.03× | Same sub-100 μs regime; difference is within measurement noise. |
| fitstable | `narrow_1000` predicate_filter (CPU, mmap-off) | 643 μs | fitsio | 1.44× | Narrow 4-column table with 1k rows; fitsio's native C predicate pushdown avoids Arrow filter overhead. torchfits' automatic predicate path uses fast full-read + Arrow filter, which has fixed dispatch cost that dominates at this tiny size. |
| fitstable | `narrow_10000` predicate_filter (CPU, mmap-off) | 723 μs | fitsio | 1.29× | Same mechanism as narrow_1000; the fixed overhead shrinks as row count grows and disappears by 100k rows. |

**Key takeaways:**

- **hcompress** is a niche compression algorithm; gzip and rice (the common
  choices) are always faster or equal in torchfits.
- **Sub-100 μs CUDA reads** for tiny int8 images are measurement-noise-level
  deficits; fitsio's `fitsio_torch_device` path has slightly lower Python
  dispatch latency for payloads under ~10 KB.
- **Narrow table predicate_filter** is the only meaningful deficit. At 1k–10k
  rows the fixed Arrow filter overhead dominates; at 100k+ rows torchfits is
  always first. Users can bypass with `backend="cpp"` for native pushdown.
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_cuda_0.9.0_20260714_065950` | fits + fitstable (lab) | 3648 | 7 | lab bench-all + `--mmap-matrix` + CUDA/MPS |
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
