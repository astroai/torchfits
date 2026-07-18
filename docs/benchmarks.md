# Benchmarks

`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio. CPU↔GPU comparisons are published
when hardware was available; GPU deficits are listed, not hidden.

**Honesty:** torchfits is a **1.0.0rc** prerelease. Headline ratios below are
lab medians from named scorecard runs — not guarantees on your filesystem,
file mix, or PyTorch version. Check [Performance deficits](#performance-deficits)
before assuming torchfits wins every case.

## How to read this page

| If you want… | Jump to |
|---|---|
| Headline wins | [Performance highlights](#performance-highlights) |
| Cases where torchfits is not #1 (CPU and GPU) | [Performance deficits](#performance-deficits) |
| GPU transport rows | [I/O transport and backend](#io-transport-and-backend) |
| Reproduce numbers | [Reproducing](#reproducing) |
| Every measured configuration | [Exhaustive benchmark results](#exhaustive-benchmark-results) |
| Raw CSV | [Published CSVs](#published-csvs) |

Published GPU/CPU numbers come from the multi-host release scorecard
(`exhaustive_mps_*`, `exhaustive_cpu_*`, `exhaustive_cuda_*`). Manual
`workflow_dispatch` on `.github/workflows/bench-report.yml` is CPU-only and
does not refresh GPU cells.

## Comparison targets

| Domain | torchfits API | Compared against |
|---|---|---|
| Tensor (IMAGE HDU) | `read` / `read_tensor` / `write` | `astropy.io.fits`, `fitsio` |
| Table (dataframe) | `torchfits.table` | `astropy.io.fits`, `fitsio` |

## Methodology

Each case measures median wall-clock time over multiple repetitions, plus
**peak process RSS** (and peak CUDA alloc when on CUDA). Deficit ranking is
**time-based**; RSS is reported alongside times.

Cases are grouped into two families:

- **default** — high-level API (`torchfits.read` / `table.read`, etc.).
- **specialized** — `torchfits_specialized` methods (open-once handle /
  `open_subset_reader` paths). Empty specialized cells mean that path was not
  measured for the case.

Fairness controls:

- Rows with mismatched mmap behavior are marked `SKIPPED` and excluded from
  rankings.
- **Why fitsio has no mmap rows:** fitsio does not expose a comparable mmap
  toggle. Under `mmap_target=on` / `strict_mmap_fairness`, fitsio rows are
  non-comparable and show as skipped in transport tables (see
  `scripts/render_bench_iopath_table.py`).
- Warm-cache and cold-cache profiles are kept separate.

### Disk to GPU

True **disk→GPU** (GPUDirect Storage / cuFile, or a CFITSIO path that never
touches host RAM) is **not** implemented. Every Python FITS stack here
decodes on the host, then copies with `.to(device)`. Exploring a direct path
is a **1.1** candidate (see [Roadmap](roadmap.md)) — not a 1.0 claim.

### Tables on GPU transports

Table GPU transport rows compare `table.read_torch(..., device=cpu)` against
`device=cuda` / `device=mps` on a medium mixed catalog case
(`mixed_100000`). Decode still happens on the host; the GPU column measures
host decode plus H2D copy into tensor columns.

## Published CSVs

Exhaustive `results.csv` / `torchfits_deficits.csv` for the scorecard runs are
linked from GitHub Release assets when published, and mirrored under
`docs/assets/bench/<run-id>/` when size allows. Example local paths used to
build this page:

- `docs/assets/bench/exhaustive_mps_20260718_180230/results.csv`
- `docs/assets/bench/exhaustive_cpu_20260717_040146/results.csv`
- `docs/assets/bench/exhaustive_cuda_20260717_042840/results.csv`

(also under `benchmarks_results/<run-id>/` locally)


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
| `pixi run bench-megacam` | local | CFHT MegaCam MEF cutouts (requires fetched sample data) |
| `pixi run bench-ml` | local | PyTorch DataLoader throughput vs fitsio |

### CFHT MegaCam cutout suite

Public CFHT MegaCam MEF samples (CADC Direct Data Service) exercise **Rice
`.fz`** repeated cutouts with peer ranking:

| Method | Role |
|---|---|
| `torchfits_cached` / `fitsio_cached` | Open once + N× subset (comparable family) |
| `torchfits_materialize` | Decompress plane once, then host slices (isolates Rice vs cutout API) |
| `torchfits_naive` | Re-open per cutout (pathological baseline; not ranked) |

Uses `ZNAXIS*` for tile-compressed sizes; throughput is cutout **payload** MB/s.

```bash
bash scripts/fetch_cfht_megacam_sample.sh   # once; idempotent
pixi run bench-megacam
```

Outputs land in `benchmarks_results/<run-id>/megacam_results.csv`. Sample
FITS files are gitignored under `benchmarks_data/cfht_megacam/`.

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
pixi run bench-ml
bash scripts/fetch_cfht_megacam_sample.sh && pixi run bench-megacam
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
| CPU vs GPU device | Partial | CPU: full matrix; GPU: tensor reads | GPU requires CUDA/MPS (`pixi run -e bench-gpu`); manual CI bench is CPU-only |
| I/O transport `disk→RAM→CPU` | Yes | `bench-all` mmap-on pass | Median mixes many ops/sizes — coarse aggregate |
| I/O transport `disk→CPU` (non-mmap) | Yes | `bench-all --mmap-matrix` mmap-off pass | Buffered host decode |
| I/O transport `disk→RAM→GPU` | Partial | `bench_gpu_transports.py` (mmap on) | Tensor `read_full`, cutouts, repeated cutouts; tables until suite lands |
| I/O transport `disk→CPU→GPU` | Partial | `bench_gpu_transports.py` (mmap off) | Same with buffered host decode + H2D |
| I/O transport `disk→GPU` | No | — | No host-bypass path yet (see Methodology); 1.1 candidate |
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D fixtures; unsigned via BZERO in `scaled_*` |
| Tensor dimensions / sizes | Yes | tiny → large; 1D–3D (4D where fixtures exist) | Large 3D cubes may hit size caps |
| Compression (read) | Yes | gzip, rice, hcompress, plio | Write→compress cases are being added to the suite |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` | MEF random ext reads on selected fixtures |
| Multi-extension (MEF) | Yes | `mef_*`, `multi_mef_10ext`, MegaCam suite | — |
| Table full read / projection / slice | Yes | `bench_fitstable_io.py` | — |
| Table predicate / scan | Yes | `predicate_filter`, `scan_count` | — |
| Table schemas | Partial | mixed / narrow / wide / varlen | typed / ascii at selected row counts |
| Table GPU vs CPU | Partial | GPU transports / fitstable | Expanding into published tables |
| Writes / write→compress | Partial | suite expansion | Read-heavy historically; write parity also in tests |
| ML DataLoader | Yes | `bench_ml_loader.py` | Reported in highlights / dedicated section |

### Why the I/O transport table looks sparse on GPU

1. **`disk→GPU` is always empty** — backends decode on the host first, then
   `.to(device)`. See [Disk to GPU](#disk-to-gpu).
2. **`disk→CPU→GPU` vs `disk→RAM→GPU`** — mmap-off vs mmap-on host decode + H2D.
3. **GPU rows need CUDA/MPS hardware** — published CUDA numbers come from
   CANFAR staging (`exhaustive_cuda_20260717_042840`).
4. **Tables** — see [Tables on GPU transports](#tables-on-gpu-transports).

### GPU integer dtype comparisons

The **deficit table** compares default
`torchfits.read(..., scale_on_device=True)` against
`torch.from_numpy(fitsio.read(...)).to(cuda)`. That pairing is not
dtype-equivalent for every scaled integer FITS file.

| FITS convention | fitsio @ CUDA | default `read` @ CUDA |
|---|---|---|
| Signed byte (BITPIX=8, BZERO=-128) | native `int8` H2D | narrow `int8` H2D + offset on device |
| Unsigned uint16/uint32 (BZERO) | native uint H2D | narrow storage H2D, offset on device |
| Generic BSCALE/BZERO | often native storage | `float32` on device (ML-friendly) |

For apples-to-apples integer GPU timing, the suite also records
`torchfits_dtype_fair_device` (`read_tensor(..., raw_scale=True)`).

**Training loops:** call
`torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=…)` before
`DataLoader` epochs so handle caches stay warm.

### Refreshing GPU numbers (CANFAR staging)

CUDA lab numbers come from a headless GPU session on `@staging`. From a
machine with `canfar` x509 auth:

```bash
bash scripts/selfcheck_canfar_launcher.sh
TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest TORCHFITS_BENCH_MODE=exhaustive \
  pixi run bench-canfar-gpu
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cuda_<stamp>
bash scripts/patch_canfar_exhaustive_docs.sh exhaustive_cuda_<stamp>
```

```bash
# Local CI + docs before push
bash scripts/ci_local.sh
# Apple Silicon (MPS transport rows)
pixi run bench-mps
```

## I/O transport and backend

> **GPU summary:** Tensor **`disk→CPU→GPU`** / **`disk→RAM→GPU`** rows appear
> only when the CSV was produced on CUDA or MPS. **`disk→GPU`** stays empty
> (unsupported). Table GPU cells stay empty until the table-GPU suite lands.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/exhaustive_mps_20260718_180230/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.29 ms` (n=174) | `1.20 ms` (n=253) | `0.44 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.46 ms` (n=174) | `1.73 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.52 ms` (n=152) | `1.84 ms` (n=152) | `0.85 ms` (n=152) | — |
| `disk→RAM→GPU` | `0.46 ms` (n=152) | `1.91 ms` (n=152) | `39.52 ms` (n=8) | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.81 ms` (n=182) | `8.51 ms` (n=164) | `2.69 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.45 ms` (n=180) | `10.24 ms` (n=164) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
The following table showcases median wall-clock times for key FITS tensor and table cases. The **specialized** column is `torchfits_specialized` (open-once / subset-reader paths); it is empty when that path was not measured.

| Benchmark Case | Device | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Large tensor read (Float32 2D, 16.0 MB) | CPU | **3.85 ms** | 3.64 ms | 11.35 ms | 5.25 ms | **3.12x** | **1.44x** |
| Large tensor read (Float32 2D @ CUDA) | CUDA | **8.37 ms** | 9.50 ms | 15.94 ms | 12.22 ms | **1.90x** | **1.46x** |
| Compressed tensor read (Rice, 1.1 MB) | CPU | **18.16 ms** | 16.39 ms | 75.33 ms | 18.03 ms | **4.60x** | **1.10x** |
| Compressed tensor read (Rice @ CUDA) | CUDA | **12.08 ms** | 16.88 ms | 62.53 ms | 17.92 ms | **5.17x** | **1.48x** |
| Repeated cutouts (50x 100x100) | CPU | **13.60 ms** | 15.64 ms | 269.20 ms | 18.16 ms | **19.79x** | **1.33x** |
| Table read (100k rows, 8 cols, mixed) | CPU | **5.65 ms** | 7.12 ms | 98.48 ms | 31.45 ms | **17.43x** | **5.57x** |
| Varlen table read (100k rows, 3 cols) | CPU | **236.16 ms** | 251.56 ms | 1.530 s | 324.38 ms | **6.48x** | **1.37x** |
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
The complete, un-cherrypicked list of all measured configurations. Empty cells mean that method was not run for the case (for example `torchfits_specialized` is only used for open-once / subset-reader paths). Domain `tensor` = IMAGE HDU payloads (1D–4D); `table` = binary/ASCII tables.

| Domain | Benchmark Case | Operation | Size | Device | mmap | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | cfitsio (direct) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| tensor | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 151.2 μs | 2.69 ms | 388.1 μs | — | **17.82x** | **2.57x** |
| tensor | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 137.6 μs | 2.47 ms | 345.0 μs | — | **17.96x** | **2.51x** |
| tensor | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 200.2 μs | 3.61 ms | 499.9 μs | — | **18.04x** | **2.50x** |
| tensor | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.47 ms** | 1.48 ms | 15.33 ms | 1.74 ms | — | **10.45x** | **1.19x** |
| tensor | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 157.4 μs | 2.79 ms | 415.0 μs | — | **17.71x** | **2.64x** |
| tensor | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 49.0 μs | 539.5 μs | 109.6 μs | — | **11.02x** | **2.24x** |
| tensor | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 83.5 μs | 894.9 μs | 181.1 μs | — | **10.71x** | **2.17x** |
| tensor | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 37.8 μs | 511.4 μs | 102.0 μs | — | **13.52x** | **2.69x** |
| tensor | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 45.4 μs | 552.5 μs | 109.8 μs | — | **12.18x** | **2.42x** |
| tensor | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 60.2 μs | 660.6 μs | 145.0 μs | — | **10.98x** | **2.41x** |
| tensor | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 86.2 μs | 1.21 ms | 245.3 μs | — | **14.02x** | **2.85x** |
| tensor | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 54.6 μs | 637.2 μs | 144.0 μs | — | **11.67x** | **2.64x** |
| tensor | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 93.0 μs | 1.00 ms | 193.3 μs | — | **10.79x** | **2.08x** |
| tensor | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 70.8 μs | 929.0 μs | 146.3 μs | — | **13.12x** | **2.07x** |
| tensor | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 102.1 μs | 1.19 ms | 227.8 μs | — | **11.65x** | **2.23x** |
| tensor | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 72.3 μs | 750.2 μs | 145.7 μs | — | **10.37x** | **2.01x** |
| tensor | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 46.3 μs | 630.7 μs | 109.2 μs | — | **13.61x** | **2.36x** |
| tensor | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 57.8 μs | 678.0 μs | 116.4 μs | — | **11.74x** | **2.02x** |
| tensor | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 49.6 μs | 616.3 μs | 113.6 μs | — | **12.43x** | **2.29x** |
| tensor | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 98.4 μs | 1.36 ms | 258.5 μs | — | **13.79x** | **2.63x** |
| tensor | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 69.5 μs | 877.0 μs | 167.8 μs | — | **12.62x** | **2.41x** |
| tensor | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 71.9 μs | 890.9 μs | 169.5 μs | — | **12.39x** | **2.36x** |
| tensor | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 60.1 μs | 864.4 μs | 152.3 μs | — | **14.39x** | **2.54x** |
| tensor | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 77.5 μs | 1.02 ms | 189.0 μs | — | **13.14x** | **2.44x** |
| tensor | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 61.1 μs | 576.9 μs | 125.3 μs | — | **9.44x** | **2.05x** |
| tensor | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 34.4 μs | 495.9 μs | 92.2 μs | — | **14.43x** | **2.68x** |
| tensor | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 42.3 μs | 556.0 μs | 110.7 μs | — | **13.13x** | **2.61x** |
| tensor | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 57.2 μs | 799.0 μs | 140.4 μs | — | **13.98x** | **2.46x** |
| tensor | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 50.9 μs | 581.5 μs | 108.7 μs | — | **11.43x** | **2.14x** |
| tensor | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 63.3 μs | 711.1 μs | 143.0 μs | — | **11.23x** | **2.26x** |
| tensor | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 53.4 μs | 693.5 μs | 122.9 μs | — | **12.99x** | **2.30x** |
| tensor | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 41.3 μs | 508.6 μs | 107.5 μs | — | **12.30x** | **2.60x** |
| tensor | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 50.4 μs | 631.2 μs | 134.3 μs | — | **12.52x** | **2.66x** |
| tensor | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 60.0 μs | 742.1 μs | 137.4 μs | — | **12.36x** | **2.29x** |
| tensor | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 68.0 μs | 828.5 μs | 138.0 μs | — | **12.18x** | **2.03x** |
| tensor | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 66.3 μs | 926.2 μs | 177.0 μs | — | **13.96x** | **2.67x** |
| tensor | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 57.7 μs | 694.2 μs | 138.2 μs | — | **12.03x** | **2.39x** |
| tensor | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 50.2 μs | 632.0 μs | 116.1 μs | — | **12.59x** | **2.31x** |
| tensor | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 64.5 μs | 630.4 μs | 124.5 μs | — | **9.77x** | **1.93x** |
| tensor | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 100.8 μs | 1.25 ms | 193.4 μs | — | **12.43x** | **1.92x** |
| tensor | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 88.8 μs | 1.28 ms | 165.5 μs | — | **14.45x** | **1.87x** |
| tensor | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | n/a | **29.4 μs** | 35.6 μs | 5.96 ms | 464.0 μs | — | **202.47x** | **15.77x** |
| tensor | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 98.3 μs | 1.50 ms | 220.7 μs | — | **15.22x** | **2.24x** |
| tensor | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **15.74 ms** | 18.44 ms | 39.38 ms | 35.32 ms | — | **2.50x** | **2.24x** |
| tensor | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **13.60 ms** | 15.64 ms | 269.20 ms | 18.16 ms | — | **19.79x** | **1.33x** |
| tensor | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 75.1 μs | 913.5 μs | 182.4 μs | — | **12.16x** | **2.43x** |
| tensor | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 78.1 μs | 950.9 μs | 199.2 μs | — | **12.18x** | **2.55x** |
| tensor | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 83.0 μs | 1.01 ms | 179.0 μs | — | **12.12x** | **2.16x** |
| tensor | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 49.9 μs | 634.9 μs | 127.6 μs | — | **12.73x** | **2.56x** |
| tensor | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 50.8 μs | 653.2 μs | 129.5 μs | — | **12.85x** | **2.55x** |
| tensor | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 73.0 μs | 800.7 μs | 147.2 μs | — | **10.97x** | **2.02x** |
| tensor | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 42.8 μs | 524.2 μs | 103.9 μs | — | **12.25x** | **2.43x** |
| tensor | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 50.5 μs | 654.9 μs | 128.0 μs | — | **12.97x** | **2.53x** |
| tensor | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 90.3 μs | 1.01 ms | 163.4 μs | — | **11.23x** | **1.81x** |
| tensor | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 94.4 μs | 924.0 μs | 166.4 μs | — | **9.79x** | **1.76x** |
| tensor | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 80.8 μs | 911.8 μs | 164.5 μs | — | **11.28x** | **2.03x** |
| tensor | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 76.2 μs | 1.11 ms | 187.4 μs | — | **14.59x** | **2.46x** |
| tensor | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 44.0 μs | 600.9 μs | 122.7 μs | — | **13.67x** | **2.79x** |
| tensor | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 58.5 μs | 1.00 ms | 151.9 μs | — | **17.13x** | **2.59x** |
| tensor | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 98.4 μs | 935.5 μs | 141.9 μs | — | **9.51x** | **1.44x** |
| tensor | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 71.2 μs | 827.3 μs | 194.1 μs | — | **11.62x** | **2.73x** |
| tensor | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 59.1 μs | 738.7 μs | 150.6 μs | — | **12.49x** | **2.55x** |
| tensor | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 84.8 μs | 917.0 μs | 167.5 μs | — | **10.82x** | **1.97x** |
| tensor | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 47.3 μs | 645.2 μs | 137.5 μs | — | **13.64x** | **2.91x** |
| tensor | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 75.2 μs | 1.01 ms | 188.0 μs | — | **13.37x** | **2.50x** |
| tensor | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 83.0 μs | 820.5 μs | 159.8 μs | — | **9.89x** | **1.93x** |
| tensor | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 70.7 μs | 887.7 μs | 168.1 μs | — | **12.56x** | **2.38x** |
| tensor | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 38.7 μs | 518.7 μs | 97.5 μs | — | **13.40x** | **2.52x** |
| tensor | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 62.4 μs | 686.5 μs | 135.6 μs | — | **11.00x** | **2.17x** |
| tensor | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 87.3 μs | 1.06 ms | 189.0 μs | — | **12.16x** | **2.16x** |
| tensor | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 94.4 μs | 1.22 ms | 289.5 μs | — | **12.90x** | **3.07x** |
| tensor | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 47.2 μs | 520.0 μs | 97.1 μs | — | **11.02x** | **2.06x** |
| tensor | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 45.0 μs | 546.8 μs | 103.0 μs | — | **12.16x** | **2.29x** |
| tensor | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 40.6 μs | 577.6 μs | 99.5 μs | — | **14.23x** | **2.45x** |
| tensor | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 40.2 μs | 559.9 μs | 104.8 μs | — | **13.92x** | **2.61x** |
| tensor | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 56.7 μs | 709.9 μs | 127.9 μs | — | **12.52x** | **2.26x** |
| tensor | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 52.6 μs | 567.3 μs | 119.9 μs | — | **10.79x** | **2.28x** |
| tensor | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 51.6 μs | 654.8 μs | 134.5 μs | — | **12.69x** | **2.61x** |
| tensor | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 43.3 μs | 586.7 μs | 112.5 μs | — | **13.55x** | **2.60x** |
| tensor | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 36.2 μs | 488.4 μs | 108.8 μs | — | **13.47x** | **3.00x** |
| tensor | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 42.4 μs | 539.5 μs | 110.5 μs | — | **12.73x** | **2.61x** |
| tensor | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 71.7 μs | 933.0 μs | 165.7 μs | — | **13.00x** | **2.31x** |
| tensor | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 61.6 μs | 926.9 μs | 256.3 μs | — | **15.04x** | **4.16x** |
| tensor | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 55.1 μs | 719.0 μs | 140.0 μs | — | **13.04x** | **2.54x** |
| tensor | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 54.9 μs | 716.9 μs | 133.9 μs | — | **13.05x** | **2.44x** |
| tensor | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 53.7 μs | 643.9 μs | 129.9 μs | — | **11.99x** | **2.42x** |
| tensor | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 67.1 μs | 726.4 μs | 155.8 μs | — | **10.82x** | **2.32x** |
| tensor | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 42.7 μs | 588.8 μs | 111.2 μs | — | **13.79x** | **2.60x** |
| tensor | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 57.0 μs | 738.3 μs | 133.9 μs | — | **12.94x** | **2.35x** |
| tensor | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 52.3 μs | 692.5 μs | 131.8 μs | — | **13.24x** | **2.52x** |
| tensor | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 55.0 μs | 650.7 μs | 123.9 μs | — | **11.84x** | **2.25x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **—** | — | 196.24 ms | — | — | **—** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **76.99 ms** | — | 188.10 ms | — | — | **2.44x** | **—** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **31.77 ms** | 32.37 ms | 74.14 ms | 38.35 ms | — | **2.33x** | **1.21x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **32.56 ms** | 34.00 ms | 90.40 ms | 30.52 ms | — | **2.78x** | **0.94x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **45.16 ms** | 44.20 ms | 66.43 ms | 51.83 ms | — | **1.50x** | **1.17x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **18.16 ms** | 16.39 ms | 75.33 ms | 18.03 ms | — | **4.60x** | **1.10x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | off | **1.57 ms** | 1.79 ms | 2.70 ms | 1.30 ms | — | **1.73x** | **0.83x** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | off | **3.85 ms** | 3.64 ms | 11.35 ms | 5.25 ms | — | **3.12x** | **1.44x** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | off | **2.47 ms** | 2.27 ms | 4.55 ms | 2.49 ms | — | **2.01x** | **1.10x** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | off | **13.32 ms** | 13.64 ms | 15.83 ms | 9.95 ms | — | **1.19x** | **0.75x** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | off | **472.4 μs** | 482.5 μs | 1.59 ms | 543.6 μs | — | **3.38x** | **1.15x** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | off | **1.98 ms** | 1.86 ms | 4.59 ms | 2.11 ms | — | **2.46x** | **1.14x** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | off | **935.7 μs** | 991.5 μs | 2.64 ms | 1.05 ms | — | **2.82x** | **1.12x** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | off | **3.98 ms** | 4.02 ms | 8.45 ms | 4.15 ms | — | **2.13x** | **1.04x** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | off | **2.62 ms** | 2.44 ms | 5.83 ms | 3.57 ms | — | **2.39x** | **1.46x** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | off | **10.94 ms** | 11.09 ms | 17.62 ms | 11.17 ms | — | **1.61x** | **1.02x** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | off | **280.0 μs** | 262.7 μs | 1.00 ms | 1.16 ms | — | **3.81x** | **4.43x** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | off | **1.53 ms** | 1.43 ms | 3.59 ms | 5.71 ms | — | **2.50x** | **3.98x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | off | **7.29 ms** | 6.68 ms | 14.56 ms | 11.02 ms | — | **2.18x** | **1.65x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | off | **10.32 ms** | 10.61 ms | 20.84 ms | 13.86 ms | — | **2.02x** | **1.34x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | off | **480.9 μs** | 339.0 μs | 1.68 ms | 357.5 μs | — | **4.95x** | **1.05x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | off | **1.79 ms** | 2.10 ms | 2.81 ms | 1.12 ms | — | **1.57x** | **0.63x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | off | **2.66 ms** | 3.32 ms | 7.17 ms | 3.10 ms | — | **2.70x** | **1.17x** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | off | **706.1 μs** | 771.0 μs | 1.01 ms | 330.7 μs | — | **1.44x** | **0.47x** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | off | **3.90 ms** | 3.78 ms | 7.88 ms | 5.84 ms | — | **2.08x** | **1.54x** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | off | **4.79 ms** | 4.65 ms | 5.29 ms | 3.54 ms | — | **1.14x** | **0.76x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | off | **102.1 μs** | 117.9 μs | 465.4 μs | 116.5 μs | — | **4.56x** | **1.14x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | off | **563.5 μs** | 470.9 μs | 2.28 ms | 876.9 μs | — | **4.85x** | **1.86x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | off | **960.5 μs** | 1.00 ms | 1.65 ms | 619.8 μs | — | **1.72x** | **0.65x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | off | **111.8 μs** | 115.7 μs | 491.5 μs | 127.0 μs | — | **4.40x** | **1.14x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | off | **1.04 ms** | 936.5 μs | 2.34 ms | 956.8 μs | — | **2.50x** | **1.02x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | off | **1.48 ms** | 1.44 ms | 3.56 ms | 1.50 ms | — | **2.47x** | **1.04x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | off | **206.5 μs** | 207.0 μs | 689.2 μs | 226.8 μs | — | **3.34x** | **1.10x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | off | **2.58 ms** | 2.52 ms | 5.31 ms | 2.84 ms | — | **2.11x** | **1.13x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | off | **4.10 ms** | 3.92 ms | 5.33 ms | 3.62 ms | — | **1.36x** | **0.92x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | off | **129.0 μs** | 230.8 μs | 702.6 μs | 214.0 μs | — | **5.45x** | **1.66x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | off | **244.4 μs** | 233.3 μs | 979.5 μs | 1.15 ms | — | **4.20x** | **4.91x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | off | **557.5 μs** | 538.3 μs | 2.51 ms | 2.27 ms | — | **4.66x** | **4.22x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **1.70 ms** | 1.54 ms | 3.39 ms | 1.82 ms | — | **2.20x** | **1.18x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **1.76 ms** | 1.65 ms | 3.87 ms | 1.75 ms | — | **2.34x** | **1.06x** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | off | **271.6 μs** | 255.8 μs | 1.09 ms | 1.03 ms | — | **4.25x** | **4.04x** |
| tensor | mef_small | read_full | 0.45 MB | CPU | off | **125.3 μs** | 104.0 μs | 1.68 ms | 752.1 μs | — | **16.13x** | **7.23x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **152.8 μs** | 168.5 μs | 1.86 ms | 642.9 μs | — | **12.16x** | **4.21x** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | off | **4.10 ms** | 4.11 ms | 12.57 ms | 5.38 ms | — | **3.07x** | **1.31x** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | off | **1.10 ms** | 1.10 ms | 3.30 ms | 1.11 ms | — | **3.00x** | **1.01x** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | off | **161.8 μs** | 170.2 μs | 757.6 μs | 154.2 μs | — | **4.68x** | **0.95x** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | off | **98.6 μs** | 107.5 μs | 502.6 μs | 274.0 μs | — | **5.10x** | **2.78x** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | off | **349.9 μs** | 184.5 μs | 1.27 ms | 205.0 μs | — | **6.87x** | **1.11x** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | off | **677.2 μs** | 366.9 μs | 682.3 μs | 199.7 μs | — | **1.86x** | **0.54x** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | off | **315.7 μs** | 152.4 μs | 630.6 μs | 209.0 μs | — | **4.14x** | **1.37x** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | off | **206.5 μs** | 185.7 μs | 1.02 ms | 234.0 μs | — | **5.47x** | **1.26x** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | off | **390.8 μs** | 397.2 μs | 1.17 ms | 412.2 μs | — | **2.98x** | **1.05x** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | off | **168.3 μs** | 138.9 μs | 586.6 μs | 163.7 μs | — | **4.22x** | **1.18x** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | off | **368.5 μs** | 152.2 μs | 626.4 μs | 202.0 μs | — | **4.12x** | **1.33x** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | off | **128.5 μs** | 136.3 μs | 970.5 μs | 242.2 μs | — | **7.55x** | **1.89x** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | off | **325.2 μs** | 201.9 μs | 744.6 μs | 165.5 μs | — | **3.69x** | **0.82x** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | off | **235.2 μs** | 239.1 μs | 915.6 μs | 295.7 μs | — | **3.89x** | **1.26x** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | off | **300.3 μs** | 551.1 μs | 502.7 μs | 160.0 μs | — | **1.67x** | **0.53x** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | off | **109.7 μs** | 101.8 μs | 607.1 μs | 157.3 μs | — | **5.97x** | **1.55x** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | off | **386.8 μs** | 350.2 μs | 1.34 ms | 421.9 μs | — | **3.83x** | **1.20x** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | off | **479.9 μs** | 463.7 μs | 784.0 μs | 329.0 μs | — | **1.69x** | **0.71x** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | off | **86.6 μs** | 94.0 μs | 589.5 μs | 104.3 μs | — | **6.81x** | **1.20x** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | off | **119.7 μs** | 107.5 μs | 742.7 μs | 184.9 μs | — | **6.91x** | **1.72x** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | off | **124.9 μs** | 169.8 μs | 612.2 μs | 251.0 μs | — | **4.90x** | **2.01x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | off | **186.2 μs** | 152.3 μs | 781.4 μs | 190.4 μs | — | **5.13x** | **1.25x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | off | **178.8 μs** | 179.5 μs | 1.94 ms | 413.2 μs | — | **10.86x** | **2.31x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **290.2 μs** | 356.4 μs | 1.27 ms | 377.9 μs | — | **4.37x** | **1.30x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **551.5 μs** | 243.5 μs | 1.10 ms | 246.9 μs | — | **4.53x** | **1.01x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **329.3 μs** | 356.5 μs | 1.08 ms | 196.7 μs | — | **3.27x** | **0.60x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **149.9 μs** | 160.3 μs | 865.5 μs | 186.9 μs | — | **5.77x** | **1.25x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **127.3 μs** | 115.3 μs | 727.5 μs | 174.8 μs | — | **6.31x** | **1.52x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **81.4 μs** | 90.0 μs | 577.8 μs | 116.6 μs | — | **7.10x** | **1.43x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **155.2 μs** | 138.3 μs | 800.9 μs | 166.9 μs | — | **5.79x** | **1.21x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **176.7 μs** | 187.5 μs | 914.0 μs | 174.0 μs | — | **5.17x** | **0.98x** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **92.1 μs** | 78.6 μs | 420.7 μs | 103.0 μs | — | **5.35x** | **1.31x** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **100.0 μs** | 115.2 μs | 476.7 μs | 103.7 μs | — | **4.77x** | **1.04x** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **130.3 μs** | 155.3 μs | 1.24 ms | 267.6 μs | — | **9.52x** | **2.05x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **123.4 μs** | 93.9 μs | 1.20 ms | 203.3 μs | — | **12.74x** | **2.16x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **178.1 μs** | 210.8 μs | 1.03 ms | 208.0 μs | — | **5.76x** | **1.17x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **230.5 μs** | 216.9 μs | 911.6 μs | 183.0 μs | — | **4.20x** | **0.84x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **172.3 μs** | 175.1 μs | 855.5 μs | 342.0 μs | — | **4.97x** | **1.99x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **129.1 μs** | 109.4 μs | 506.3 μs | 120.1 μs | — | **4.63x** | **1.10x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **110.5 μs** | 110.5 μs | 780.5 μs | 195.3 μs | — | **7.07x** | **1.77x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **288.5 μs** | 274.3 μs | 740.7 μs | 171.3 μs | — | **2.70x** | **0.62x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **132.1 μs** | 141.5 μs | 523.0 μs | 124.2 μs | — | **3.96x** | **0.94x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **285.2 μs** | 129.6 μs | 637.2 μs | 161.3 μs | — | **4.92x** | **1.24x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **114.4 μs** | 106.5 μs | 949.7 μs | 130.8 μs | — | **8.91x** | **1.23x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **100.7 μs** | 114.6 μs | 715.7 μs | 127.9 μs | — | **7.11x** | **1.27x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **168.0 μs** | 169.8 μs | 716.4 μs | 134.6 μs | — | **4.26x** | **0.80x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **30.51 ms** | 32.94 ms | 85.55 ms | 33.30 ms | — | **2.80x** | **1.09x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **31.99 ms** | 28.48 ms | 185.38 ms | 49.41 ms | — | **6.51x** | **1.73x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **56.13 ms** | 55.01 ms | 82.24 ms | 60.22 ms | — | **1.50x** | **1.09x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **19.17 ms** | 19.35 ms | 76.82 ms | 17.64 ms | — | **4.01x** | **0.92x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | on | **2.54 ms** | 2.43 ms | 5.66 ms | — | — | **2.33x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | on | **9.08 ms** | 7.06 ms | 15.68 ms | — | — | **2.22x** | **—** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | on | **3.73 ms** | 3.99 ms | 9.52 ms | — | — | **2.55x** | **—** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | on | **15.08 ms** | 15.52 ms | 32.86 ms | — | — | **2.18x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | on | **1.13 ms** | 1.12 ms | 5.34 ms | — | — | **4.78x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | on | **4.26 ms** | 3.79 ms | 8.51 ms | — | — | **2.25x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.29 ms** | 1.30 ms | 2.39 ms | — | — | **1.85x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | on | **6.06 ms** | 6.34 ms | 10.16 ms | — | — | **1.68x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | on | **2.53 ms** | 2.62 ms | 6.37 ms | — | — | **2.52x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | on | **15.39 ms** | 15.58 ms | 28.96 ms | — | — | **1.88x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | on | **364.7 μs** | 350.7 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | on | **2.17 ms** | 2.15 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | on | **8.02 ms** | 7.11 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | on | **12.71 ms** | 12.95 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | on | **459.4 μs** | 470.2 μs | 1.22 ms | — | — | **2.66x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | on | **3.01 ms** | 1.98 ms | 6.56 ms | — | — | **3.31x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | on | **3.26 ms** | 3.20 ms | 8.61 ms | — | — | **2.69x** | **—** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | on | **413.8 μs** | 505.1 μs | 1.54 ms | — | — | **3.73x** | **—** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | on | **4.51 ms** | 4.70 ms | 8.78 ms | — | — | **1.95x** | **—** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | on | **7.84 ms** | 6.55 ms | 13.99 ms | — | — | **2.14x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | on | **259.1 μs** | 438.7 μs | 1.02 ms | — | — | **3.92x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | on | **1.24 ms** | 1.27 ms | 3.26 ms | — | — | **2.62x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | on | **1.53 ms** | 1.51 ms | 4.99 ms | — | — | **3.31x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | on | **322.5 μs** | 229.3 μs | 1.49 ms | — | — | **6.49x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | on | **1.77 ms** | 2.38 ms | 5.78 ms | — | — | **3.26x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | on | **3.07 ms** | 3.62 ms | 9.29 ms | — | — | **3.02x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | on | **936.0 μs** | 456.3 μs | 1.72 ms | — | — | **3.76x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | on | **15.60 ms** | 6.21 ms | 12.66 ms | — | — | **2.04x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | on | **11.07 ms** | 9.60 ms | 9.90 ms | — | — | **1.03x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | on | **6.10 ms** | 358.4 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | on | **839.0 μs** | 959.3 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | on | **1.23 ms** | 979.0 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **2.69 ms** | 1.99 ms | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **3.43 ms** | 4.13 ms | — | — | — | **—** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | on | **950.8 μs** | 654.7 μs | — | — | — | **—** | **—** |
| tensor | mef_small | read_full | 0.45 MB | CPU | on | **270.5 μs** | 319.0 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **275.3 μs** | 366.8 μs | — | — | — | **—** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | on | **9.49 ms** | 10.80 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | on | **2.05 ms** | 1.99 ms | — | — | — | **—** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | on | **299.7 μs** | 518.9 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | on | **215.2 μs** | 365.7 μs | 1.45 ms | — | — | **6.72x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | on | **542.9 μs** | 559.4 μs | 1.37 ms | — | — | **2.53x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | on | **418.2 μs** | 591.8 μs | 2.16 ms | — | — | **5.17x** | **—** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | on | **543.6 μs** | 455.2 μs | 1.18 ms | — | — | **2.60x** | **—** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | on | **736.0 μs** | 395.4 μs | 1.07 ms | — | — | **2.70x** | **—** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | on | **1.25 ms** | 876.5 μs | 1.76 ms | — | — | **2.01x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | on | **144.7 μs** | 187.8 μs | 1.27 ms | — | — | **8.81x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | on | **250.9 μs** | 356.5 μs | 1.26 ms | — | — | **5.03x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | on | **349.5 μs** | 337.0 μs | 1.74 ms | — | — | **5.15x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | on | **488.6 μs** | 207.7 μs | 1.28 ms | — | — | **6.16x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | on | **488.7 μs** | 379.9 μs | 1.18 ms | — | — | **3.11x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | on | **425.6 μs** | 459.5 μs | 4.01 ms | — | — | **9.43x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | on | **473.2 μs** | 367.0 μs | 1.13 ms | — | — | **3.07x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | on | **428.2 μs** | 432.4 μs | 1.06 ms | — | — | **2.48x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | on | **476.0 μs** | 660.6 μs | 1.62 ms | — | — | **3.40x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | on | **229.8 μs** | 180.1 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | on | **382.7 μs** | 343.1 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | on | **229.7 μs** | 353.1 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | on | **300.4 μs** | 285.9 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | on | **381.9 μs** | 343.8 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **350.5 μs** | 337.7 μs | 1.19 ms | — | — | **3.51x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **308.5 μs** | 236.5 μs | 772.7 μs | — | — | **3.27x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **283.3 μs** | 216.0 μs | 5.63 ms | — | — | **26.05x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **234.3 μs** | 308.5 μs | 808.0 μs | — | — | **3.45x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **291.4 μs** | 192.6 μs | 965.3 μs | — | — | **5.01x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **125.1 μs** | 111.8 μs | 623.7 μs | — | — | **5.58x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **211.6 μs** | 360.4 μs | 994.1 μs | — | — | **4.70x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **266.1 μs** | 117.8 μs | 1.02 ms | — | — | **8.67x** | **—** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **112.8 μs** | 116.8 μs | 709.2 μs | — | — | **6.29x** | **—** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **200.6 μs** | 143.3 μs | 1.24 ms | — | — | **8.67x** | **—** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **332.4 μs** | 442.1 μs | 1.32 ms | — | — | **3.97x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **233.4 μs** | 475.0 μs | 809.0 μs | — | — | **3.47x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **321.1 μs** | 264.4 μs | 5.31 ms | — | — | **20.08x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **287.5 μs** | 244.6 μs | 1.21 ms | — | — | **4.94x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **151.3 μs** | 193.9 μs | 1.17 ms | — | — | **7.74x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **266.0 μs** | 259.2 μs | 1.03 ms | — | — | **3.97x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **166.9 μs** | 251.4 μs | 792.2 μs | — | — | **4.75x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **116.1 μs** | 124.5 μs | 1.06 ms | — | — | **9.16x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **317.4 μs** | 208.3 μs | 1.11 ms | — | — | **5.35x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **334.8 μs** | 274.5 μs | 1.35 ms | — | — | **4.93x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **154.1 μs** | 236.7 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **285.1 μs** | 392.0 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **141.2 μs** | 160.1 μs | — | — | — | **—** | **—** |
| tensor | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | n/a | **1.76 ms** | 1.65 ms | 15.25 ms | 2.32 ms | — | **9.25x** | **1.41x** |
| tensor | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | n/a | **263.8 μs** | 322.0 μs | 6.27 ms | 634.6 μs | — | **23.75x** | **2.41x** |
| tensor | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | n/a | **30.75 ms** | 28.19 ms | 188.22 ms | 32.04 ms | — | **6.68x** | **1.14x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **34.32 ms** | 27.37 ms | 60.78 ms | 34.95 ms | — | **2.22x** | **1.28x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **24.29 ms** | 22.09 ms | 81.28 ms | 30.17 ms | — | **3.68x** | **1.37x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **45.81 ms** | 38.72 ms | 89.42 ms | 45.55 ms | — | **2.31x** | **1.18x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **14.53 ms** | 16.88 ms | 61.30 ms | 12.83 ms | — | **4.22x** | **0.88x** |
| tensor | large_float32_1d | read_full | 3.82 MB | MPS | off | **1.12 ms** | 1.33 ms | 2.61 ms | 1.25 ms | — | **2.34x** | **1.12x** |
| tensor | large_float32_2d | read_full | 16.00 MB | MPS | off | **5.97 ms** | 9.50 ms | 13.60 ms | 6.76 ms | — | **2.28x** | **1.13x** |
| tensor | large_int16_1d | read_full | 1.91 MB | MPS | off | **1.87 ms** | 1.21 ms | 4.88 ms | 2.51 ms | — | **4.02x** | **2.07x** |
| tensor | large_int16_2d | read_full | 8.00 MB | MPS | off | **2.27 ms** | 2.84 ms | 5.38 ms | 2.35 ms | — | **2.37x** | **1.03x** |
| tensor | large_int32_1d | read_full | 3.82 MB | MPS | off | **1.20 ms** | 1.17 ms | 3.03 ms | 1.71 ms | — | **2.60x** | **1.47x** |
| tensor | large_int32_2d | read_full | 16.00 MB | MPS | off | **5.48 ms** | 6.66 ms | 12.33 ms | 6.44 ms | — | **2.25x** | **1.18x** |
| tensor | large_int64_1d | read_full | 7.63 MB | MPS | off | **2.83 ms** | 2.92 ms | 7.05 ms | 3.02 ms | — | **2.49x** | **1.07x** |
| tensor | large_int64_2d | read_full | 32.00 MB | MPS | off | **12.53 ms** | 11.56 ms | 39.38 ms | 12.94 ms | — | **3.41x** | **1.12x** |
| tensor | large_int8_1d | read_full | 0.96 MB | MPS | off | **467.4 μs** | 558.9 μs | 1.51 ms | 1.46 ms | — | **3.23x** | **3.12x** |
| tensor | large_int8_2d | read_full | 4.00 MB | MPS | off | **1.84 ms** | 1.61 ms | 3.66 ms | 5.62 ms | — | **2.27x** | **3.48x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | MPS | off | **6.44 ms** | 8.16 ms | 11.52 ms | 6.61 ms | — | **1.79x** | **1.03x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | MPS | off | **10.81 ms** | 8.25 ms | 20.72 ms | 11.20 ms | — | **2.51x** | **1.36x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | MPS | off | **300.0 μs** | 346.0 μs | 1.01 ms | 416.1 μs | — | **3.37x** | **1.39x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | MPS | off | **1.42 ms** | 1.22 ms | 2.88 ms | 1.71 ms | — | **2.37x** | **1.41x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | MPS | off | **1.78 ms** | 1.88 ms | 4.53 ms | 1.85 ms | — | **2.54x** | **1.04x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | MPS | off | **260.8 μs** | 304.5 μs | 857.1 μs | 389.0 μs | — | **3.29x** | **1.49x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | MPS | off | **751.8 μs** | 773.3 μs | 2.34 ms | 866.3 μs | — | **3.11x** | **1.15x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | MPS | off | **1.07 ms** | 1.13 ms | 3.02 ms | 1.31 ms | — | **2.81x** | **1.22x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | MPS | off | **397.3 μs** | 318.0 μs | 999.5 μs | 629.1 μs | — | **3.14x** | **1.98x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | MPS | off | **3.24 ms** | 2.44 ms | 6.24 ms | 4.22 ms | — | **2.56x** | **1.73x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | MPS | off | **3.43 ms** | 4.10 ms | 8.43 ms | 5.78 ms | — | **2.45x** | **1.68x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | MPS | off | **794.4 μs** | 1.30 ms | 3.26 ms | 993.3 μs | — | **4.10x** | **1.25x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | MPS | off | **5.02 ms** | 6.84 ms | 8.76 ms | 8.63 ms | — | **1.74x** | **1.72x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | MPS | off | **7.42 ms** | 7.06 ms | 12.20 ms | 12.43 ms | — | **1.73x** | **1.76x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | MPS | off | **536.0 μs** | 432.7 μs | 1.98 ms | 909.5 μs | — | **4.57x** | **2.10x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | MPS | off | **615.5 μs** | 526.5 μs | 2.07 ms | 2.09 ms | — | **3.93x** | **3.96x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | MPS | off | **933.2 μs** | 1.43 ms | 2.50 ms | 2.79 ms | — | **2.68x** | **2.99x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **2.52 ms** | 2.75 ms | 3.96 ms | 4.30 ms | — | **1.57x** | **1.70x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **2.33 ms** | 2.15 ms | 5.43 ms | 2.32 ms | — | **2.53x** | **1.08x** |
| tensor | mef_medium | read_full | 7.02 MB | MPS | off | **506.7 μs** | 628.7 μs | 1.96 ms | 1.70 ms | — | **3.86x** | **3.36x** |
| tensor | mef_small | read_full | 0.45 MB | MPS | off | **284.5 μs** | 275.4 μs | 1.55 ms | 593.5 μs | — | **5.63x** | **2.16x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **242.6 μs** | 240.2 μs | 1.66 ms | 668.6 μs | — | **6.92x** | **2.78x** |
| tensor | scaled_large | read_full | 8.00 MB | MPS | off | **8.89 ms** | 6.04 ms | 17.23 ms | 8.41 ms | — | **2.85x** | **1.39x** |
| tensor | scaled_medium | read_full | 2.01 MB | MPS | off | **1.75 ms** | 1.62 ms | 5.78 ms | 1.77 ms | — | **3.56x** | **1.09x** |
| tensor | scaled_small | read_full | 0.13 MB | MPS | off | **367.5 μs** | 337.3 μs | 1.32 ms | 542.5 μs | — | **3.91x** | **1.61x** |
| tensor | small_float32_1d | read_full | 42.2 KB | MPS | off | **522.6 μs** | 307.6 μs | 1.12 ms | 870.3 μs | — | **3.64x** | **2.83x** |
| tensor | small_float32_2d | read_full | 0.26 MB | MPS | off | **374.9 μs** | 518.5 μs | 2.20 ms | 620.9 μs | — | **5.87x** | **1.66x** |
| tensor | small_float32_3d | read_full | 0.63 MB | MPS | off | **463.0 μs** | 571.0 μs | 1.67 ms | 665.5 μs | — | **3.60x** | **1.44x** |
| tensor | small_int16_1d | read_full | 22.5 KB | MPS | off | **454.0 μs** | 325.9 μs | 1.78 ms | 655.0 μs | — | **5.48x** | **2.01x** |
| tensor | small_int16_2d | read_full | 0.13 MB | MPS | off | **1.01 ms** | 373.5 μs | 1.71 ms | 2.10 ms | — | **4.58x** | **5.63x** |
| tensor | small_int16_3d | read_full | 0.32 MB | MPS | off | **721.0 μs** | 615.2 μs | 3.36 ms | 1.85 ms | — | **5.47x** | **3.01x** |
| tensor | small_int32_1d | read_full | 42.2 KB | MPS | off | **472.8 μs** | 422.8 μs | 1.65 ms | 663.4 μs | — | **3.90x** | **1.57x** |
| tensor | small_int32_2d | read_full | 0.26 MB | MPS | off | **589.3 μs** | 389.1 μs | 1.80 ms | 772.2 μs | — | **4.62x** | **1.98x** |
| tensor | small_int32_3d | read_full | 0.63 MB | MPS | off | **650.5 μs** | 801.2 μs | 2.80 ms | 982.2 μs | — | **4.31x** | **1.51x** |
| tensor | small_int64_1d | read_full | 0.08 MB | MPS | off | **446.8 μs** | 402.0 μs | 1.34 ms | 543.5 μs | — | **3.33x** | **1.35x** |
| tensor | small_int64_2d | read_full | 0.51 MB | MPS | off | **429.0 μs** | 350.0 μs | 1.07 ms | 622.2 μs | — | **3.07x** | **1.78x** |
| tensor | small_int64_3d | read_full | 1.26 MB | MPS | off | **598.8 μs** | 518.8 μs | 1.48 ms | 777.1 μs | — | **2.86x** | **1.50x** |
| tensor | small_int8_1d | read_full | 14.1 KB | MPS | off | **228.2 μs** | 432.3 μs | 2.27 ms | 375.9 μs | — | **9.97x** | **1.65x** |
| tensor | small_int8_2d | read_full | 0.07 MB | MPS | off | **794.0 μs** | 367.6 μs | 1.76 ms | 800.7 μs | — | **4.79x** | **2.18x** |
| tensor | small_int8_3d | read_full | 0.16 MB | MPS | off | **369.0 μs** | 220.5 μs | 981.0 μs | 565.8 μs | — | **4.45x** | **2.57x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | MPS | off | **337.5 μs** | 352.9 μs | 940.6 μs | 413.8 μs | — | **2.79x** | **1.23x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | MPS | off | **635.8 μs** | 647.6 μs | 2.58 ms | 1.46 ms | — | **4.05x** | **2.29x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **425.1 μs** | 337.1 μs | 2.03 ms | 642.9 μs | — | **6.01x** | **1.91x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **251.5 μs** | 228.1 μs | 837.7 μs | 383.0 μs | — | **3.67x** | **1.68x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **267.3 μs** | 268.1 μs | 933.7 μs | 453.8 μs | — | **3.49x** | **1.70x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **268.8 μs** | 251.7 μs | 1.17 ms | 396.9 μs | — | **4.64x** | **1.58x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **285.2 μs** | 327.7 μs | 1.48 ms | 443.0 μs | — | **5.19x** | **1.55x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **219.4 μs** | 209.1 μs | 787.0 μs | 349.0 μs | — | **3.76x** | **1.67x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **264.4 μs** | 226.5 μs | 1.01 ms | 470.2 μs | — | **4.44x** | **2.08x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **329.0 μs** | 266.8 μs | 1.03 ms | 385.1 μs | — | **3.85x** | **1.44x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **328.9 μs** | 205.0 μs | 911.3 μs | 578.1 μs | — | **4.45x** | **2.82x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **217.7 μs** | 206.5 μs | 846.0 μs | 361.5 μs | — | **4.10x** | **1.75x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **257.3 μs** | 2.79 ms | 853.5 μs | 408.3 μs | — | **3.32x** | **1.59x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **501.1 μs** | 457.7 μs | 1.60 ms | 623.5 μs | — | **3.49x** | **1.36x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **548.2 μs** | 419.0 μs | 1.31 ms | 993.5 μs | — | **3.12x** | **2.37x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **372.6 μs** | 289.7 μs | 1.11 ms | 585.0 μs | — | **3.82x** | **2.02x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **200.7 μs** | 196.9 μs | 839.3 μs | 381.5 μs | — | **4.26x** | **1.94x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **261.7 μs** | 252.5 μs | 817.3 μs | 369.5 μs | — | **3.24x** | **1.46x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **300.0 μs** | 280.5 μs | 931.3 μs | 467.9 μs | — | **3.32x** | **1.67x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **252.8 μs** | 212.1 μs | 1.33 ms | 411.8 μs | — | **6.26x** | **1.94x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **327.2 μs** | 267.2 μs | 1.13 ms | 610.0 μs | — | **4.21x** | **2.28x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **287.9 μs** | 196.7 μs | 1.06 ms | 396.0 μs | — | **5.39x** | **2.01x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **29.10 ms** | 37.26 ms | 115.23 ms | 34.80 ms | — | **3.96x** | **1.20x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **32.44 ms** | 27.88 ms | 88.07 ms | 38.33 ms | — | **3.16x** | **1.37x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **65.01 ms** | 56.63 ms | 71.84 ms | 58.49 ms | — | **1.27x** | **1.03x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **13.66 ms** | 22.58 ms | 40.81 ms | 13.88 ms | — | **2.99x** | **1.02x** |
| tensor | large_float32_1d | read_full | 3.82 MB | MPS | on | **1.06 ms** | 1.42 ms | 2.15 ms | — | — | **2.03x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | MPS | on | **4.86 ms** | 5.02 ms | 30.96 ms | — | — | **6.36x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | MPS | on | **1.85 ms** | 747.0 μs | 1.94 ms | — | — | **2.60x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | MPS | on | **2.98 ms** | 2.97 ms | 5.35 ms | — | — | **1.80x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | MPS | on | **1.19 ms** | 2.04 ms | 4.08 ms | — | — | **3.44x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | MPS | on | **4.82 ms** | 5.24 ms | 9.23 ms | — | — | **1.92x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | MPS | on | **2.59 ms** | 2.77 ms | 4.70 ms | — | — | **1.81x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | MPS | on | **13.81 ms** | 12.18 ms | 20.35 ms | — | — | **1.67x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | MPS | on | **693.3 μs** | 438.7 μs | 2.48 ms | — | — | **5.66x** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | MPS | on | **1.51 ms** | 1.61 ms | 4.14 ms | — | — | **2.75x** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | MPS | on | **5.86 ms** | 5.85 ms | 9.80 ms | — | — | **1.67x** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | MPS | on | **8.50 ms** | 13.75 ms | 25.12 ms | — | — | **2.95x** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | MPS | on | **1.16 ms** | 655.0 μs | 3.93 ms | — | — | **6.01x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | MPS | on | **4.08 ms** | 2.30 ms | 5.07 ms | — | — | **2.21x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | MPS | on | **3.43 ms** | 3.29 ms | 8.81 ms | — | — | **2.68x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | MPS | on | **274.6 μs** | 274.7 μs | 852.3 μs | — | — | **3.10x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | MPS | on | **868.6 μs** | 936.5 μs | 1.85 ms | — | — | **2.13x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | MPS | on | **1.46 ms** | 1.19 ms | 2.85 ms | — | — | **2.40x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | MPS | on | **343.5 μs** | 278.8 μs | 904.0 μs | — | — | **3.24x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | MPS | on | **1.59 ms** | 1.28 ms | 2.44 ms | — | — | **1.91x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | MPS | on | **2.31 ms** | 2.13 ms | 4.36 ms | — | — | **2.05x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | MPS | on | **365.3 μs** | 394.5 μs | 1.10 ms | — | — | **3.00x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | MPS | on | **2.75 ms** | 5.26 ms | 5.30 ms | — | — | **1.93x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | MPS | on | **4.15 ms** | 4.22 ms | 5.35 ms | — | — | **1.29x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | MPS | on | **299.1 μs** | 263.7 μs | 1.51 ms | — | — | **5.71x** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | MPS | on | **528.2 μs** | 451.9 μs | 1.89 ms | — | — | **4.18x** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | MPS | on | **790.8 μs** | 836.0 μs | 2.75 ms | — | — | **3.48x** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **1.58 ms** | 1.54 ms | 4.65 ms | — | — | **3.02x** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **2.25 ms** | 2.27 ms | 6.51 ms | — | — | **2.90x** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | MPS | on | **531.2 μs** | 515.5 μs | 2.54 ms | — | — | **4.93x** | **—** |
| tensor | mef_small | read_full | 0.45 MB | MPS | on | **231.5 μs** | 220.3 μs | 1.99 ms | — | — | **9.03x** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **281.8 μs** | 262.6 μs | 1.90 ms | — | — | **7.24x** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | MPS | on | **4.99 ms** | 5.64 ms | 13.18 ms | — | — | **2.64x** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | MPS | on | **1.55 ms** | 1.50 ms | 4.60 ms | — | — | **3.06x** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | MPS | on | **297.7 μs** | 287.5 μs | 4.28 ms | — | — | **14.89x** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | MPS | on | **248.3 μs** | 210.7 μs | 652.8 μs | — | — | **3.10x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | MPS | on | **276.2 μs** | 293.0 μs | 1.06 ms | — | — | **3.84x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | MPS | on | **730.2 μs** | 540.1 μs | 2.17 ms | — | — | **4.02x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | MPS | on | **230.7 μs** | 212.4 μs | 746.8 μs | — | — | **3.52x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | MPS | on | **392.4 μs** | 240.0 μs | 944.9 μs | — | — | **3.94x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | MPS | on | **755.2 μs** | 298.9 μs | 1.06 ms | — | — | **3.56x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | MPS | on | **230.4 μs** | 230.4 μs | 854.2 μs | — | — | **3.71x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | MPS | on | **251.2 μs** | 324.3 μs | 935.0 μs | — | — | **3.72x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | MPS | on | **365.2 μs** | 361.0 μs | 986.4 μs | — | — | **2.73x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | MPS | on | **239.5 μs** | 346.2 μs | 1.26 ms | — | — | **5.26x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | MPS | on | **401.1 μs** | 386.6 μs | 1.27 ms | — | — | **3.28x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | MPS | on | **540.4 μs** | 700.9 μs | 1.59 ms | — | — | **2.94x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | MPS | on | **262.6 μs** | 256.5 μs | 1.54 ms | — | — | **6.02x** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | MPS | on | **393.4 μs** | 261.7 μs | 1.54 ms | — | — | **5.90x** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | MPS | on | **367.0 μs** | 1.09 ms | 3.00 ms | — | — | **8.19x** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | MPS | on | **699.3 μs** | 537.6 μs | 2.99 ms | — | — | **5.56x** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | MPS | on | **385.1 μs** | 453.9 μs | 2.79 ms | — | — | **7.25x** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **281.0 μs** | 234.3 μs | 817.7 μs | — | — | **3.49x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **679.2 μs** | 658.8 μs | 2.62 ms | — | — | **3.97x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **502.7 μs** | 357.7 μs | 997.2 μs | — | — | **2.79x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **254.2 μs** | 330.1 μs | 1.26 ms | — | — | **4.97x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **325.5 μs** | 314.2 μs | 1.68 ms | — | — | **5.34x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **217.7 μs** | 468.7 μs | 2.18 ms | — | — | **10.02x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **536.9 μs** | 412.2 μs | 1.92 ms | — | — | **4.67x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **514.4 μs** | 348.6 μs | 1.72 ms | — | — | **4.94x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **245.5 μs** | 228.8 μs | 811.9 μs | — | — | **3.55x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **244.6 μs** | 255.4 μs | 901.4 μs | — | — | **3.69x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **375.3 μs** | 201.1 μs | 1.07 ms | — | — | **5.33x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **243.2 μs** | 280.3 μs | 897.2 μs | — | — | **3.69x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **308.0 μs** | 241.8 μs | 902.2 μs | — | — | **3.73x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **287.8 μs** | 271.0 μs | 783.7 μs | — | — | **2.89x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **266.3 μs** | 242.7 μs | 842.0 μs | — | — | **3.47x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **289.3 μs** | 297.1 μs | 991.0 μs | — | — | **3.43x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **326.0 μs** | 270.2 μs | 1.03 ms | — | — | **3.80x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **337.5 μs** | 240.8 μs | 1.76 ms | — | — | **7.31x** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **294.1 μs** | 243.7 μs | 1.64 ms | — | — | **6.72x** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **254.5 μs** | 257.0 μs | 1.98 ms | — | — | **7.79x** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **1.03 ms** | 805.3 μs | 6.64 ms | 1.24 ms | — | **8.25x** | **1.54x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **3.07 ms** | 3.35 ms | 29.32 ms | 8.12 ms | — | **9.56x** | **2.65x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **2.63 ms** | 3.58 ms | 28.57 ms | 7.83 ms | — | **10.88x** | **2.98x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **254.8 μs** | 253.8 μs | 5.55 ms | 1.38 ms | — | **21.86x** | **5.45x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **93.6 μs** | 64.2 μs | 630.1 μs | 140.8 μs | — | **9.81x** | **2.19x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **169.4 μs** | 254.8 μs | 3.36 ms | 417.5 μs | — | **19.85x** | **2.47x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **338.0 μs** | 319.3 μs | 5.92 ms | 1.27 ms | — | **18.54x** | **3.97x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **382.2 μs** | 355.9 μs | 5.31 ms | 1.20 ms | — | **14.93x** | **3.36x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **164.4 μs** | 144.5 μs | 4.84 ms | 524.6 μs | — | **33.51x** | **3.63x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **74.5 μs** | 113.5 μs | 1.21 ms | 284.7 μs | — | **16.19x** | **3.82x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **41.82 ms** | 50.47 ms | 34.94 ms | 95.16 ms | — | **0.84x** | **2.28x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **18.14 ms** | 18.72 ms | 33.91 ms | 157.67 ms | — | **1.87x** | **8.69x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **55.26 ms** | 44.94 ms | 959.99 ms | 313.65 ms | — | **21.36x** | **6.98x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **1.16 ms** | 954.5 μs | 31.72 ms | 5.26 ms | — | **33.23x** | **5.51x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **233.4 μs** | 256.3 μs | 2.45 ms | 676.2 μs | — | **10.48x** | **2.90x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **3.86 ms** | 5.10 ms | 9.78 ms | 8.95 ms | — | **2.53x** | **2.32x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **3.75 ms** | 4.64 ms | 10.98 ms | 10.54 ms | — | **2.93x** | **2.81x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **5.65 ms** | 7.12 ms | 98.48 ms | 31.45 ms | — | **17.43x** | **5.57x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **1.33 ms** | 1.28 ms | 20.72 ms | 6.38 ms | — | **16.20x** | **4.99x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **165.8 μs** | 129.1 μs | 1.08 ms | 296.7 μs | — | **8.33x** | **2.30x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **610.4 μs** | 815.4 μs | 4.90 ms | 1.46 ms | — | **8.03x** | **2.39x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **629.8 μs** | 945.7 μs | 7.94 ms | 1.41 ms | — | **12.60x** | **2.24x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **1.33 ms** | 1.29 ms | 14.70 ms | 4.07 ms | — | **11.36x** | **3.14x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **546.6 μs** | 213.5 μs | 5.78 ms | 831.0 μs | — | **27.08x** | **3.89x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **103.4 μs** | 155.4 μs | 1.32 ms | 392.2 μs | — | **12.80x** | **3.79x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **193.1 μs** | 398.2 μs | 5.36 ms | 761.2 μs | — | **27.76x** | **3.94x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **220.7 μs** | 299.5 μs | 5.01 ms | 753.6 μs | — | **22.70x** | **3.42x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **390.0 μs** | 340.7 μs | 7.00 ms | 1.30 ms | — | **20.53x** | **3.83x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **362.0 μs** | 264.7 μs | 8.38 ms | 816.9 μs | — | **31.65x** | **3.09x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **152.6 μs** | 160.1 μs | 2.24 ms | 381.6 μs | — | **14.66x** | **2.50x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **32.89 ms** | 32.71 ms | 23.09 ms | 50.76 ms | — | **0.71x** | **1.55x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **19.49 ms** | 15.79 ms | 23.30 ms | 102.17 ms | — | **1.48x** | **6.47x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **16.64 ms** | 19.64 ms | 26.70 ms | 18.38 ms | — | **1.60x** | **1.10x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **416.4 μs** | 1.77 ms | 22.10 ms | 7.35 ms | — | **53.08x** | **17.65x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **113.2 μs** | 147.4 μs | 1.48 ms | 353.7 μs | — | **13.03x** | **3.12x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **3.61 ms** | 3.57 ms | 11.70 ms | 5.74 ms | — | **3.27x** | **1.61x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **1.42 ms** | 1.98 ms | 7.57 ms | 12.02 ms | — | **5.34x** | **8.48x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **2.25 ms** | 2.68 ms | 6.70 ms | 5.21 ms | — | **2.97x** | **2.31x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **786.1 μs** | 949.2 μs | 7.33 ms | 2.41 ms | — | **9.32x** | **3.07x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **144.7 μs** | 113.6 μs | 988.1 μs | 222.6 μs | — | **8.70x** | **1.96x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **499.9 μs** | 823.0 μs | 4.29 ms | 908.2 μs | — | **8.58x** | **1.82x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **545.8 μs** | 522.6 μs | 4.52 ms | 1.52 ms | — | **8.65x** | **2.91x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **401.6 μs** | 449.3 μs | 8.54 ms | 848.9 μs | — | **21.26x** | **2.11x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **201.9 μs** | 272.1 μs | 5.18 ms | 727.0 μs | — | **25.66x** | **3.60x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **146.6 μs** | 149.5 μs | 1.63 ms | 345.3 μs | — | **11.14x** | **2.36x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **214.7 μs** | 299.1 μs | 3.86 ms | 601.3 μs | — | **17.98x** | **2.80x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **248.8 μs** | 554.2 μs | 9.70 ms | 1.01 ms | — | **38.99x** | **4.05x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **236.2 μs** | 290.0 μs | 4.02 ms | 539.4 μs | — | **17.01x** | **2.28x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **203.5 μs** | 280.5 μs | 5.58 ms | 708.7 μs | — | **27.40x** | **3.48x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **119.0 μs** | 139.3 μs | 1.89 ms | 335.8 μs | — | **15.87x** | **2.82x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **3.88 ms** | 3.41 ms | 7.23 ms | 5.66 ms | — | **2.12x** | **1.66x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **11.79 ms** | 13.82 ms | 101.58 ms | 49.59 ms | — | **8.61x** | **4.21x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **15.34 ms** | 13.39 ms | 73.80 ms | 39.79 ms | — | **5.51x** | **2.97x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **1.72 ms** | 2.50 ms | 18.41 ms | 8.99 ms | — | **10.71x** | **5.23x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **173.8 μs** | 198.7 μs | 2.34 ms | 395.7 μs | — | **13.45x** | **2.28x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **1.27 ms** | 760.4 μs | 4.39 ms | 1.17 ms | — | **5.78x** | **1.53x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **2.02 ms** | 1.38 ms | 10.00 ms | 4.05 ms | — | **7.24x** | **2.93x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **1.91 ms** | 1.99 ms | 11.73 ms | 4.98 ms | — | **6.14x** | **2.61x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **221.3 μs** | 230.8 μs | 4.71 ms | 1.17 ms | — | **21.27x** | **5.31x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **257.1 μs** | 124.0 μs | 1.33 ms | 342.6 μs | — | **10.70x** | **2.76x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **2.92 ms** | 1.81 ms | 3.83 ms | 3.02 ms | — | **2.12x** | **1.67x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **219.60 ms** | 248.79 ms | 1.709 s | 337.23 ms | — | **7.78x** | **1.54x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **236.16 ms** | 251.56 ms | 1.530 s | 324.38 ms | — | **6.48x** | **1.37x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **19.12 ms** | 19.87 ms | 148.86 ms | 36.32 ms | — | **7.79x** | **1.90x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **90.8 μs** | 94.2 μs | 969.9 μs | 205.1 μs | — | **10.68x** | **2.26x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **336.0 μs** | 548.1 μs | 2.85 ms | 871.5 μs | — | **8.47x** | **2.59x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **19.83 ms** | 19.20 ms | 131.03 ms | 29.26 ms | — | **6.82x** | **1.52x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **25.83 ms** | 25.46 ms | 167.11 ms | 36.79 ms | — | **6.56x** | **1.45x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **2.01 ms** | 2.03 ms | 18.56 ms | 4.11 ms | — | **9.21x** | **2.04x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **74.5 μs** | 98.0 μs | 1.03 ms | 229.3 μs | — | **13.88x** | **3.08x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **191.6 μs** | 178.9 μs | 2.39 ms | 400.7 μs | — | **13.34x** | **2.24x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **2.35 ms** | 2.14 ms | 23.02 ms | 4.86 ms | — | **10.76x** | **2.27x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **2.08 ms** | 3.99 ms | 30.22 ms | 5.35 ms | — | **14.55x** | **2.58x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **373.0 μs** | 279.8 μs | 4.35 ms | 696.6 μs | — | **15.54x** | **2.49x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **89.4 μs** | 133.5 μs | 1.63 ms | 393.1 μs | — | **18.21x** | **4.40x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **14.49 ms** | 14.01 ms | 32.29 ms | 16.45 ms | — | **2.31x** | **1.17x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **8.87 ms** | 11.21 ms | 28.65 ms | 21.08 ms | — | **3.23x** | **2.38x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **46.93 ms** | 61.24 ms | 465.99 ms | 174.17 ms | — | **9.93x** | **3.71x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **6.15 ms** | 5.57 ms | 85.74 ms | 24.14 ms | — | **15.39x** | **4.33x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **431.7 μs** | 443.2 μs | 1.77 ms | 948.7 μs | — | **4.11x** | **2.20x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **2.10 ms** | 2.36 ms | 18.86 ms | 2.79 ms | — | **8.97x** | **1.33x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **2.81 ms** | 2.84 ms | 26.85 ms | 3.48 ms | — | **9.55x** | **1.24x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **7.67 ms** | 6.74 ms | 81.79 ms | 27.92 ms | — | **12.13x** | **4.14x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **1.42 ms** | 1.41 ms | 46.68 ms | 4.64 ms | — | **33.17x** | **3.30x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **411.6 μs** | 380.9 μs | 1.82 ms | 849.7 μs | — | **4.77x** | **2.23x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **462.6 μs** | 592.9 μs | 18.68 ms | 1.82 ms | — | **40.38x** | **3.93x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **504.4 μs** | 512.0 μs | 19.91 ms | 1.83 ms | — | **39.48x** | **3.63x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **1.32 ms** | 2.00 ms | 28.80 ms | 3.58 ms | — | **21.76x** | **2.70x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **1.14 ms** | 1.07 ms | 27.73 ms | 1.70 ms | — | **26.01x** | **1.59x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **428.9 μs** | 478.4 μs | 5.59 ms | 4.63 ms | — | **13.05x** | **10.79x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **329.0 μs** | 172.6 μs | 5.85 ms | — | — | **33.90x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **298.5 μs** | 296.7 μs | 18.84 ms | — | — | **63.49x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **353.8 μs** | 393.8 μs | 26.21 ms | — | — | **74.07x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **219.2 μs** | 164.7 μs | 4.96 ms | — | — | **30.12x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **65.9 μs** | 74.2 μs | 753.5 μs | — | — | **11.44x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **154.7 μs** | 205.7 μs | 6.21 ms | — | — | **40.14x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **151.3 μs** | 157.5 μs | 3.70 ms | — | — | **24.46x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **164.5 μs** | 122.8 μs | 3.49 ms | — | — | **28.42x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **110.6 μs** | 146.8 μs | 3.29 ms | — | — | **29.76x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **84.2 μs** | 67.8 μs | 634.2 μs | — | — | **9.36x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **22.24 ms** | 26.42 ms | 48.70 ms | — | — | **2.19x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **34.13 ms** | 32.55 ms | 62.32 ms | — | — | **1.91x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **40.57 ms** | 72.67 ms | 1.020 s | — | — | **25.14x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **1.05 ms** | 772.0 μs | 45.91 ms | — | — | **59.47x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **111.3 μs** | 185.7 μs | 1.33 ms | — | — | **11.96x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **3.10 ms** | 2.99 ms | 12.37 ms | — | — | **4.14x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **1.98 ms** | 1.83 ms | 11.07 ms | — | — | **6.05x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **3.84 ms** | 3.91 ms | 89.44 ms | — | — | **23.29x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **655.7 μs** | 699.3 μs | 14.51 ms | — | — | **22.13x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **190.3 μs** | 107.8 μs | 866.4 μs | — | — | **8.03x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **565.4 μs** | 439.8 μs | 6.48 ms | — | — | **14.74x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **571.3 μs** | 499.5 μs | 6.24 ms | — | — | **12.50x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **821.4 μs** | 856.6 μs | 17.16 ms | — | — | **20.89x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **422.3 μs** | 469.3 μs | 10.99 ms | — | — | **26.02x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **164.7 μs** | 173.1 μs | 1.50 ms | — | — | **9.09x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **325.3 μs** | 278.5 μs | 6.41 ms | — | — | **23.03x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **335.3 μs** | 344.6 μs | 10.87 ms | — | — | **32.41x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **453.8 μs** | 389.5 μs | 15.37 ms | — | — | **39.46x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **400.5 μs** | 359.1 μs | 12.22 ms | — | — | **34.04x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **157.5 μs** | 180.2 μs | 1.50 ms | — | — | **9.54x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **7.15 ms** | 17.51 ms | 27.15 ms | — | — | **3.80x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **13.38 ms** | 19.42 ms | 35.47 ms | — | — | **2.65x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **12.87 ms** | 11.34 ms | 31.89 ms | — | — | **2.81x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **417.9 μs** | 444.6 μs | 19.02 ms | — | — | **45.51x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **136.7 μs** | 96.7 μs | 1.00 ms | — | — | **10.34x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **3.93 ms** | 1.56 ms | 4.39 ms | — | — | **2.81x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **821.5 μs** | 777.8 μs | 5.71 ms | — | — | **7.35x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **782.2 μs** | 1.13 ms | 5.72 ms | — | — | **7.31x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **224.5 μs** | 798.3 μs | 9.11 ms | — | — | **40.56x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **248.5 μs** | 132.7 μs | 1.46 ms | — | — | **10.97x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **527.9 μs** | 379.4 μs | 4.95 ms | — | — | **13.05x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **425.3 μs** | 440.8 μs | 5.49 ms | — | — | **12.90x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **370.1 μs** | 534.2 μs | 13.36 ms | — | — | **36.10x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **367.0 μs** | 344.9 μs | 5.73 ms | — | — | **16.60x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **136.8 μs** | 136.1 μs | 1.43 ms | — | — | **10.53x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **175.0 μs** | 290.7 μs | 13.35 ms | — | — | **76.28x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **314.0 μs** | 279.5 μs | 3.64 ms | — | — | **13.04x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **286.7 μs** | 280.6 μs | 4.02 ms | — | — | **14.32x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **228.0 μs** | 302.8 μs | 8.11 ms | — | — | **35.57x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **157.9 μs** | 106.0 μs | 1.04 ms | — | — | **9.85x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **853.4 μs** | 618.2 μs | 3.38 ms | — | — | **5.47x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **4.35 ms** | 3.68 ms | 97.82 ms | — | — | **26.55x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **3.46 ms** | 4.55 ms | 98.58 ms | — | — | **28.49x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **609.5 μs** | 1.30 ms | 14.23 ms | — | — | **23.34x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **86.1 μs** | 85.0 μs | 974.5 μs | — | — | **11.47x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **636.2 μs** | 235.2 μs | 4.08 ms | — | — | **17.35x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **455.6 μs** | 580.7 μs | 10.41 ms | — | — | **22.85x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **748.1 μs** | 381.4 μs | 10.43 ms | — | — | **27.34x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **216.8 μs** | 147.8 μs | 3.72 ms | — | — | **25.19x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **148.9 μs** | 166.2 μs | 1.70 ms | — | — | **11.44x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **1.53 ms** | 927.8 μs | 3.55 ms | — | — | **3.83x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **185.27 ms** | 165.10 ms | 1.068 s | — | — | **6.47x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **203.38 ms** | 238.99 ms | 1.666 s | — | — | **8.19x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **30.32 ms** | 30.37 ms | 191.65 ms | — | — | **6.32x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **113.0 μs** | 146.3 μs | 1.45 ms | — | — | **12.79x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **235.1 μs** | 118.7 μs | 2.03 ms | — | — | **17.11x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **19.93 ms** | 21.65 ms | 173.29 ms | — | — | **8.70x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **26.05 ms** | 25.51 ms | 179.82 ms | — | — | **7.05x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **1.28 ms** | 1.49 ms | 13.31 ms | — | — | **10.42x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **74.2 μs** | 92.8 μs | 859.7 μs | — | — | **11.58x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **190.8 μs** | 196.3 μs | 4.46 ms | — | — | **23.36x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **3.12 ms** | 2.62 ms | 25.34 ms | — | — | **9.69x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **2.16 ms** | 2.48 ms | 27.83 ms | — | — | **12.87x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **589.5 μs** | 518.4 μs | 6.58 ms | — | — | **12.69x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **122.2 μs** | 113.1 μs | 1.31 ms | — | — | **11.56x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **5.83 ms** | 7.56 ms | 41.53 ms | — | — | **7.13x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **9.29 ms** | 10.46 ms | 35.27 ms | — | — | **3.80x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **70.95 ms** | 32.49 ms | 385.76 ms | — | — | **11.87x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **3.74 ms** | 2.84 ms | 61.41 ms | — | — | **21.64x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **384.7 μs** | 453.5 μs | 1.85 ms | — | — | **4.80x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **740.3 μs** | 475.4 μs | 11.76 ms | — | — | **24.74x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **1.11 ms** | 1.62 ms | 26.09 ms | — | — | **23.54x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **6.43 ms** | 4.95 ms | 65.37 ms | — | — | **13.20x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **1.33 ms** | 977.2 μs | 21.69 ms | — | — | **22.20x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **325.5 μs** | 353.2 μs | 1.47 ms | — | — | **4.51x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **335.8 μs** | 348.8 μs | 16.76 ms | — | — | **49.91x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **372.4 μs** | 450.0 μs | 26.01 ms | — | — | **69.85x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **1.38 ms** | 1.88 ms | 26.26 ms | — | — | **19.01x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **1.06 ms** | 1.27 ms | 32.18 ms | — | — | **30.35x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **355.6 μs** | 372.0 μs | 2.03 ms | — | — | **5.71x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| macOS arm64 / MPS | tensor | compressed_rice_1 [read_full @ mps] | off | 14.53 ms | 477.2 | fitsio/fitsio_torch_device | 1.13× |
| macOS arm64 / MPS | tensor | compressed_hcompress_1 [read_full @ mps] | on | 65.01 ms | 434.6 | fitsio/fitsio_torch_device | 1.11× |
| macOS arm64 / MPS | tensor | scaled_large [read_full @ mps] | off | 8.89 ms | 612.0 | fitsio/fitsio_torch_device | 1.06× |
| macOS arm64 / MPS | tensor | tiny_int16_3d [read_full @ mps] | off | 2.79 ms | 500.7 | fitsio/fitsio_torch_device_specialized | 3.55× |
| macOS arm64 / CPU | tensor | repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | n/a | 15.64 ms | 818.2 | fitsio/fitsio_torch | 1.14× |
| macOS arm64 / MPS | tensor | medium_int64_2d [read_full @ mps] | off | 6.84 ms | 608.2 | fitsio/fitsio_torch_device_specialized | 1.05× |
| macOS arm64 / MPS | tensor | compressed_rice_1 [read_full @ mps] | on | 22.58 ms | 434.8 | fitsio/fitsio_torch_device_specialized | 1.02× |
| macOS arm64 / CPU | table | mixed_1000000 [predicate_filter] | off | 50.47 ms | 1109.6 | astropy/astropy | 1.44× |
| macOS arm64 / CPU | table | narrow_1000000 [predicate_filter] | off | 32.71 ms | 557.0 | astropy/astropy | 1.42× |
| macOS arm64 / CPU | table | narrow_1000000 [read_full] | off | 19.64 ms | 633.5 | fitsio/fitsio | 1.07× |
| Linux x86_64 / CPU | table | narrow_1000000 [predicate_filter] | off | 9.57 ms | 393.8 | astropy/astropy | 1.15× |
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| macOS arm64 / MPS | `exhaustive_mps_20260718_180230` | 3931 | 10 | 586.8 | lab + mmap-matrix + GPU |
| Linux x86_64 / CPU | `exhaustive_cpu_20260717_040146` | 2825 | 1 | 288.8 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260717_042840` | 4079 | 0 | 730.7 | lab + mmap-matrix + GPU |
<!-- BENCH_HOSTS_END -->

RC2 CANFAR re-soak sessions (`exhaustive_*_20260718_1808*`) were still **Pending** at rc3 ship time; Linux rows above remain the Jul-17 soak IDs. MPS is `exhaustive_mps_20260718_180230`. ML loader: `ml_20260718_191908` (local macOS arm64). MegaCam cutouts: `20260718_124403` (local).


Latest local quick benchmark evidence:

<!-- BENCH_QUICK_BEGIN -->
| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| — | FITS image I/O | _(no run yet)_ | — | — |
| — | FITS table I/O | _(no run yet)_ | — | — |
<!-- BENCH_QUICK_END -->

### ML DataLoader throughput

<!-- BENCH_ML_BEGIN -->
Source: `docs/assets/bench/ml_20260718_191908/ml_results.csv` (device=cpu).

| Case | Method | Median throughput |
|---|---|---:|
| ml_compressed_rice | `fitsio (comp)` | 406,509,087 pixels/s |
| ml_compressed_rice | `torchfits (comp)` | 388,652,395 pixels/s |
| ml_uncompressed | `fitsio + numpy` | 1,350,170,413 pixels/s |
| ml_uncompressed | `torchfits` | 1,387,206,443 pixels/s |
<!-- BENCH_ML_END -->

### CFHT MegaCam MEF cutouts (local)

<!-- BENCH_MEGACAM_BEGIN -->
Source: `docs/assets/bench/20260718_124403/megacam_results.csv` (160 OK rows).

| Method | Median throughput |
|---|---:|
| `fitsio_cached` | 31.3 MB/s |
| `torchfits_cached` | 34.8 MB/s |
| `torchfits_materialize` | 72.5 MB/s |
| `torchfits_naive` | 31.1 MB/s |
<!-- BENCH_MEGACAM_END -->


Keep this page current with the latest tensor and table benchmark run before
making performance claims.
