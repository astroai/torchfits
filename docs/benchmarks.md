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

- `docs/assets/bench/exhaustive_mps_20260719_065105/results.csv`
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

For **uncompressed** survey mosaics (e.g. CFHTLS MegaPipe float32 stacks),
`open_subset_reader` maps the data segment once and slices cutouts with
endian swap into torch tensors — see
[ML with FITS](examples-ml.md#survey-mosaic-cutouts-cfht-megapipe). Rice `.fz`
MegaCam cutouts remain a separate comparison (tile decompress inside CFITSIO).

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
| `bench_all.py` / `bench-fits` | fits | Scorecard path |
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
Source: `benchmarks_results/exhaustive_mps_20260719_065105/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.16 ms` (n=174) | `0.56 ms` (n=253) | `0.23 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.26 ms` (n=174) | `0.90 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.36 ms` (n=152) | `0.76 ms` (n=152) | `0.37 ms` (n=152) | — |
| `disk→RAM→GPU` | `0.53 ms` (n=152) | `1.43 ms` (n=152) | `19.54 ms` (n=8) | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.33 ms` (n=182) | `2.66 ms` (n=164) | `0.83 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.59 ms` (n=180) | `3.81 ms` (n=164) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large tensor read (Float32 2D, 16.0 MB) | CPU | **3.39 ms** | 3.48 ms | 7.05 ms | 3.64 ms | **2.08x** | **1.07x** |
| Large tensor read (Float32 2D @ CUDA) | CUDA | **3.76 ms** | 3.40 ms | 6.46 ms | 3.50 ms | **1.90x** | **1.03x** |
| Compressed tensor read (Rice, 1.1 MB) | CPU | **10.70 ms** | 10.50 ms | 30.49 ms | 11.55 ms | **2.90x** | **1.10x** |
| Compressed tensor read (Rice @ CUDA) | CUDA | **8.31 ms** | 8.07 ms | 23.10 ms | 7.86 ms | **2.86x** | **0.97x** |
| Repeated cutouts (50x 100x100) | CPU | **7.25 ms** | 7.21 ms | 110.50 ms | 8.00 ms | **15.32x** | **1.11x** |
| Table read (100k rows, 8 cols, mixed) | CPU | **2.66 ms** | 2.71 ms | 42.22 ms | 14.07 ms | **15.88x** | **5.29x** |
| Varlen table read (100k rows, 3 cols) | CPU | **104.88 ms** | 105.41 ms | 633.26 ms | 143.20 ms | **6.04x** | **1.37x** |
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
| tensor | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 165.9 μs | 1.51 ms | 237.3 μs | — | **9.10x** | **1.43x** |
| tensor | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 155.2 μs | 1.47 ms | 202.2 μs | — | **9.48x** | **1.30x** |
| tensor | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 161.6 μs | 1.53 ms | 208.4 μs | — | **9.46x** | **1.29x** |
| tensor | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | n/a | **945.8 μs** | 895.2 μs | 7.76 ms | 976.2 μs | — | **8.67x** | **1.09x** |
| tensor | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 186.3 μs | 1.60 ms | 238.6 μs | — | **8.59x** | **1.28x** |
| tensor | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 82.4 μs | 324.7 μs | 89.4 μs | — | **3.94x** | **1.08x** |
| tensor | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 66.5 μs | 280.7 μs | 88.3 μs | — | **4.22x** | **1.33x** |
| tensor | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 89.7 μs | 304.7 μs | 109.0 μs | — | **3.40x** | **1.22x** |
| tensor | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 88.3 μs | 345.8 μs | 173.3 μs | — | **3.92x** | **1.96x** |
| tensor | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 92.7 μs | 343.2 μs | 83.0 μs | — | **3.70x** | **0.89x** |
| tensor | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 70.6 μs | 326.2 μs | 74.3 μs | — | **4.62x** | **1.05x** |
| tensor | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 75.7 μs | 312.7 μs | 89.6 μs | — | **4.13x** | **1.18x** |
| tensor | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 86.6 μs | 385.7 μs | 84.3 μs | — | **4.46x** | **0.97x** |
| tensor | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 70.8 μs | 301.7 μs | 67.2 μs | — | **4.27x** | **0.95x** |
| tensor | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 66.5 μs | 302.8 μs | 71.6 μs | — | **4.55x** | **1.08x** |
| tensor | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 73.6 μs | 313.6 μs | 82.5 μs | — | **4.26x** | **1.12x** |
| tensor | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 75.6 μs | 332.5 μs | 69.2 μs | — | **4.40x** | **0.92x** |
| tensor | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 78.1 μs | 324.1 μs | 101.0 μs | — | **4.15x** | **1.29x** |
| tensor | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 70.3 μs | 335.9 μs | 152.5 μs | — | **4.78x** | **2.17x** |
| tensor | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 108.5 μs | 441.8 μs | 130.2 μs | — | **4.07x** | **1.20x** |
| tensor | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 91.0 μs | 391.7 μs | 97.9 μs | — | **4.31x** | **1.08x** |
| tensor | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 74.6 μs | 339.4 μs | 126.5 μs | — | **4.55x** | **1.69x** |
| tensor | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 84.8 μs | 329.1 μs | 76.7 μs | — | **3.88x** | **0.90x** |
| tensor | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 71.0 μs | 289.0 μs | 193.9 μs | — | **4.07x** | **2.73x** |
| tensor | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 63.9 μs | 312.1 μs | 66.6 μs | — | **4.89x** | **1.04x** |
| tensor | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 61.6 μs | 265.9 μs | 75.8 μs | — | **4.32x** | **1.23x** |
| tensor | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 67.0 μs | 297.6 μs | 67.2 μs | — | **4.44x** | **1.00x** |
| tensor | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 72.7 μs | 319.3 μs | 100.5 μs | — | **4.39x** | **1.38x** |
| tensor | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 69.7 μs | 286.0 μs | 70.5 μs | — | **4.10x** | **1.01x** |
| tensor | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 65.8 μs | 285.7 μs | 131.5 μs | — | **4.34x** | **2.00x** |
| tensor | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 65.5 μs | 295.6 μs | 69.3 μs | — | **4.52x** | **1.06x** |
| tensor | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 57.8 μs | 250.5 μs | 146.8 μs | — | **4.34x** | **2.54x** |
| tensor | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 86.9 μs | 341.2 μs | 95.7 μs | — | **3.93x** | **1.10x** |
| tensor | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 69.9 μs | 329.2 μs | 76.0 μs | — | **4.71x** | **1.09x** |
| tensor | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 70.5 μs | 344.3 μs | 77.4 μs | — | **4.89x** | **1.10x** |
| tensor | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 62.9 μs | 325.1 μs | 66.3 μs | — | **5.17x** | **1.05x** |
| tensor | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 71.5 μs | 336.0 μs | 94.8 μs | — | **4.70x** | **1.32x** |
| tensor | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 71.5 μs | 331.0 μs | 67.9 μs | — | **4.63x** | **0.95x** |
| tensor | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 70.2 μs | 326.6 μs | 64.8 μs | — | **4.65x** | **0.92x** |
| tensor | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 82.3 μs | 471.4 μs | 73.8 μs | — | **5.73x** | **0.90x** |
| tensor | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 78.8 μs | 493.1 μs | 73.0 μs | — | **6.26x** | **0.93x** |
| tensor | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | n/a | **277.0 μs** | 110.7 μs | 3.32 ms | 421.3 μs | — | **30.00x** | **3.80x** |
| tensor | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 85.2 μs | 504.4 μs | 98.5 μs | — | **5.92x** | **1.16x** |
| tensor | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **7.28 ms** | 7.48 ms | 10.04 ms | 8.15 ms | — | **1.38x** | **1.12x** |
| tensor | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **7.25 ms** | 7.21 ms | 110.50 ms | 8.00 ms | — | **15.32x** | **1.11x** |
| tensor | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 80.2 μs | 333.5 μs | 80.1 μs | — | **4.16x** | **1.00x** |
| tensor | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 70.7 μs | 332.4 μs | 94.8 μs | — | **4.70x** | **1.34x** |
| tensor | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 70.7 μs | 328.0 μs | 65.4 μs | — | **4.64x** | **0.92x** |
| tensor | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 71.3 μs | 313.0 μs | 80.2 μs | — | **4.39x** | **1.13x** |
| tensor | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 63.0 μs | 283.6 μs | 125.8 μs | — | **4.50x** | **2.00x** |
| tensor | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 71.6 μs | 342.5 μs | 67.7 μs | — | **4.78x** | **0.95x** |
| tensor | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 66.9 μs | 312.0 μs | 59.6 μs | — | **4.66x** | **0.89x** |
| tensor | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 72.5 μs | 281.4 μs | 63.1 μs | — | **3.88x** | **0.87x** |
| tensor | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 67.8 μs | 308.0 μs | 76.1 μs | — | **4.55x** | **1.12x** |
| tensor | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 64.0 μs | 257.1 μs | 55.5 μs | — | **4.02x** | **0.87x** |
| tensor | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 77.2 μs | 319.4 μs | 188.6 μs | — | **4.14x** | **2.44x** |
| tensor | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 68.2 μs | 322.5 μs | 127.1 μs | — | **4.73x** | **1.86x** |
| tensor | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 71.0 μs | 245.9 μs | 58.1 μs | — | **3.46x** | **0.82x** |
| tensor | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 80.4 μs | 297.9 μs | 70.4 μs | — | **3.71x** | **0.88x** |
| tensor | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 79.6 μs | 350.4 μs | 175.1 μs | — | **4.40x** | **2.20x** |
| tensor | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 62.1 μs | 254.1 μs | 61.6 μs | — | **4.09x** | **0.99x** |
| tensor | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 71.1 μs | 284.1 μs | 65.3 μs | — | **4.00x** | **0.92x** |
| tensor | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 62.0 μs | 303.2 μs | 58.9 μs | — | **4.89x** | **0.95x** |
| tensor | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 65.5 μs | 305.3 μs | 68.5 μs | — | **4.66x** | **1.05x** |
| tensor | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 65.3 μs | 300.5 μs | 61.6 μs | — | **4.60x** | **0.94x** |
| tensor | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 76.0 μs | 336.4 μs | 75.6 μs | — | **4.43x** | **1.00x** |
| tensor | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 70.9 μs | 305.5 μs | 67.2 μs | — | **4.31x** | **0.95x** |
| tensor | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 62.4 μs | 312.7 μs | 65.9 μs | — | **5.01x** | **1.06x** |
| tensor | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 75.3 μs | 280.3 μs | 71.8 μs | — | **3.72x** | **0.95x** |
| tensor | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 71.6 μs | 325.8 μs | 138.4 μs | — | **4.55x** | **1.93x** |
| tensor | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 67.8 μs | 291.9 μs | 93.8 μs | — | **4.30x** | **1.38x** |
| tensor | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 71.4 μs | 296.3 μs | 60.9 μs | — | **4.15x** | **0.85x** |
| tensor | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 63.5 μs | 276.4 μs | 62.5 μs | — | **4.35x** | **0.98x** |
| tensor | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 72.4 μs | 274.5 μs | 60.5 μs | — | **3.79x** | **0.84x** |
| tensor | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 67.1 μs | 263.4 μs | 69.2 μs | — | **3.93x** | **1.03x** |
| tensor | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 65.4 μs | 296.5 μs | 60.9 μs | — | **4.54x** | **0.93x** |
| tensor | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 73.1 μs | 289.2 μs | 117.5 μs | — | **3.96x** | **1.61x** |
| tensor | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 68.5 μs | 283.5 μs | 140.0 μs | — | **4.14x** | **2.04x** |
| tensor | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 69.7 μs | 311.4 μs | 85.5 μs | — | **4.47x** | **1.23x** |
| tensor | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 68.7 μs | 299.3 μs | 68.0 μs | — | **4.36x** | **0.99x** |
| tensor | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 70.9 μs | 266.5 μs | 93.2 μs | — | **3.76x** | **1.31x** |
| tensor | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 63.3 μs | 289.5 μs | 57.8 μs | — | **4.58x** | **0.91x** |
| tensor | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 65.0 μs | 257.2 μs | 170.7 μs | — | **3.95x** | **2.62x** |
| tensor | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 65.2 μs | 279.7 μs | 57.0 μs | — | **4.29x** | **0.87x** |
| tensor | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 73.6 μs | 313.9 μs | 88.1 μs | — | **4.27x** | **1.20x** |
| tensor | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 67.7 μs | 270.6 μs | 63.5 μs | — | **3.99x** | **0.94x** |
| tensor | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 67.0 μs | 278.7 μs | 58.2 μs | — | **4.16x** | **0.87x** |
| tensor | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 67.6 μs | 305.2 μs | 124.7 μs | — | **4.52x** | **1.85x** |
| tensor | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 64.4 μs | 283.4 μs | 59.2 μs | — | **4.40x** | **0.92x** |
| tensor | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 68.2 μs | 302.2 μs | 68.9 μs | — | **4.43x** | **1.01x** |
| tensor | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 67.7 μs | 320.7 μs | 72.3 μs | — | **4.74x** | **1.07x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **—** | — | 78.44 ms | — | — | **—** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **44.41 ms** | — | 82.25 ms | — | — | **1.85x** | **—** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **19.14 ms** | 19.17 ms | 41.45 ms | 23.55 ms | — | **2.17x** | **1.23x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **20.21 ms** | 20.25 ms | 68.83 ms | 24.87 ms | — | **3.41x** | **1.23x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **35.36 ms** | 35.39 ms | 44.77 ms | 35.13 ms | — | **1.27x** | **0.99x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **10.70 ms** | 10.50 ms | 30.49 ms | 11.55 ms | — | **2.90x** | **1.10x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | off | **767.1 μs** | 901.9 μs | 2.59 ms | 1.11 ms | — | **3.38x** | **1.44x** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | off | **3.39 ms** | 3.48 ms | 7.05 ms | 3.64 ms | — | **2.08x** | **1.07x** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | off | **1.90 ms** | 1.83 ms | 4.04 ms | 2.12 ms | — | **2.21x** | **1.16x** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | off | **8.65 ms** | 8.96 ms | 15.11 ms | 9.33 ms | — | **1.75x** | **1.08x** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | off | **390.7 μs** | 391.0 μs | 1.30 ms | 424.3 μs | — | **3.34x** | **1.09x** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | off | **1.62 ms** | 1.69 ms | 4.03 ms | 1.88 ms | — | **2.49x** | **1.16x** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | off | **775.1 μs** | 875.3 μs | 1.95 ms | 774.0 μs | — | **2.52x** | **1.00x** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | off | **3.41 ms** | 3.29 ms | 7.67 ms | 3.80 ms | — | **2.33x** | **1.15x** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | off | **2.04 ms** | 1.98 ms | 4.13 ms | 2.27 ms | — | **2.09x** | **1.15x** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | off | **9.75 ms** | 9.62 ms | 15.76 ms | 10.03 ms | — | **1.64x** | **1.04x** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | off | **246.9 μs** | 283.6 μs | 764.2 μs | 941.8 μs | — | **3.10x** | **3.81x** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | off | **1.12 ms** | 1.08 ms | 2.47 ms | 4.19 ms | — | **2.28x** | **3.87x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | off | **5.08 ms** | 5.12 ms | 7.93 ms | 5.13 ms | — | **1.56x** | **1.01x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | off | **6.73 ms** | 6.95 ms | 12.04 ms | 7.19 ms | — | **1.79x** | **1.07x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | off | **117.0 μs** | 134.6 μs | 437.9 μs | 127.3 μs | — | **3.74x** | **1.09x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | off | **965.1 μs** | 973.5 μs | 2.24 ms | 888.4 μs | — | **2.32x** | **0.92x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | off | **1.38 ms** | 1.36 ms | 3.42 ms | 1.37 ms | — | **2.51x** | **1.01x** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | off | **229.7 μs** | 235.4 μs | 890.0 μs | 262.3 μs | — | **3.87x** | **1.14x** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | off | **2.18 ms** | 2.27 ms | 4.04 ms | 2.20 ms | — | **1.85x** | **1.01x** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | off | **4.50 ms** | 4.27 ms | 5.51 ms | 3.76 ms | — | **1.29x** | **0.88x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | off | **110.1 μs** | 150.3 μs | 513.8 μs | 138.7 μs | — | **4.67x** | **1.26x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | off | **539.5 μs** | 594.5 μs | 1.06 ms | 445.7 μs | — | **1.97x** | **0.83x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | off | **622.5 μs** | 631.4 μs | 1.99 ms | 762.4 μs | — | **3.20x** | **1.22x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | off | **120.1 μs** | 128.1 μs | 429.7 μs | 131.5 μs | — | **3.58x** | **1.09x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | off | **859.7 μs** | 882.4 μs | 2.08 ms | 833.2 μs | — | **2.42x** | **0.97x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | off | **1.49 ms** | 1.43 ms | 3.48 ms | 1.44 ms | — | **2.43x** | **1.01x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | off | **193.5 μs** | 200.1 μs | 508.4 μs | 197.6 μs | — | **2.63x** | **1.02x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | off | **2.09 ms** | 2.28 ms | 4.30 ms | 2.52 ms | — | **2.05x** | **1.20x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | off | **3.16 ms** | 3.30 ms | 4.66 ms | 3.49 ms | — | **1.48x** | **1.11x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | off | **145.3 μs** | 369.4 μs | 714.7 μs | 230.0 μs | — | **4.92x** | **1.58x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | off | **225.1 μs** | 217.6 μs | 697.7 μs | 919.2 μs | — | **3.21x** | **4.22x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | off | **442.9 μs** | 449.0 μs | 1.24 ms | 1.55 ms | — | **2.81x** | **3.51x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **1.19 ms** | 1.16 ms | 2.24 ms | 1.08 ms | — | **1.93x** | **0.93x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **1.40 ms** | 1.51 ms | 3.16 ms | 1.52 ms | — | **2.25x** | **1.08x** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | off | **212.9 μs** | 259.5 μs | 1.23 ms | 1.02 ms | — | **5.79x** | **4.80x** |
| tensor | mef_small | read_full | 0.45 MB | CPU | off | **311.0 μs** | 155.5 μs | 692.9 μs | 190.5 μs | — | **4.46x** | **1.23x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **148.2 μs** | 327.8 μs | 792.5 μs | 295.2 μs | — | **5.35x** | **1.99x** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | off | **3.87 ms** | 3.80 ms | 8.92 ms | 4.06 ms | — | **2.35x** | **1.07x** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | off | **992.1 μs** | 949.7 μs | 2.21 ms | 834.2 μs | — | **2.33x** | **0.88x** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | off | **170.1 μs** | 122.1 μs | 547.6 μs | 146.0 μs | — | **4.48x** | **1.20x** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | off | **97.7 μs** | 163.0 μs | 362.0 μs | 94.6 μs | — | **3.70x** | **0.97x** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | off | **100.1 μs** | 93.5 μs | 515.6 μs | 91.9 μs | — | **5.51x** | **0.98x** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | off | **125.6 μs** | 122.5 μs | 454.6 μs | 125.4 μs | — | **3.71x** | **1.02x** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | off | **75.9 μs** | 201.5 μs | 313.8 μs | 175.1 μs | — | **4.13x** | **2.31x** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | off | **118.2 μs** | 111.3 μs | 423.5 μs | 144.5 μs | — | **3.81x** | **1.30x** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | off | **226.0 μs** | 201.0 μs | 533.9 μs | 212.2 μs | — | **2.66x** | **1.06x** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | off | **55.5 μs** | 52.1 μs | 288.4 μs | 75.3 μs | — | **5.53x** | **1.44x** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | off | **106.4 μs** | 199.4 μs | 368.0 μs | 132.3 μs | — | **3.46x** | **1.24x** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | off | **86.8 μs** | 84.0 μs | 331.6 μs | 89.5 μs | — | **3.95x** | **1.06x** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | off | **177.3 μs** | 70.2 μs | 274.2 μs | 73.3 μs | — | **3.91x** | **1.05x** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | off | **135.7 μs** | 104.2 μs | 363.1 μs | 89.3 μs | — | **3.48x** | **0.86x** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | off | **95.4 μs** | 114.8 μs | 398.4 μs | 111.8 μs | — | **4.18x** | **1.17x** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | off | **91.2 μs** | 80.7 μs | 280.5 μs | 157.5 μs | — | **3.48x** | **1.95x** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | off | **125.1 μs** | 109.2 μs | 394.5 μs | 115.0 μs | — | **3.61x** | **1.05x** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | off | **198.2 μs** | 187.3 μs | 503.3 μs | 205.9 μs | — | **2.69x** | **1.10x** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | off | **53.7 μs** | 47.5 μs | 335.0 μs | 74.0 μs | — | **7.05x** | **1.56x** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | off | **94.7 μs** | 108.0 μs | 357.6 μs | 101.3 μs | — | **3.78x** | **1.07x** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | off | **89.3 μs** | 143.6 μs | 493.6 μs | 161.2 μs | — | **5.53x** | **1.80x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | off | **125.2 μs** | 217.2 μs | 650.7 μs | 109.6 μs | — | **5.20x** | **0.88x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | off | **106.1 μs** | 136.7 μs | 435.8 μs | 122.3 μs | — | **4.11x** | **1.15x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **97.6 μs** | 74.4 μs | 357.6 μs | 100.0 μs | — | **4.81x** | **1.34x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **80.7 μs** | 88.3 μs | 319.2 μs | 83.7 μs | — | **3.96x** | **1.04x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **110.8 μs** | 95.8 μs | 338.2 μs | 78.7 μs | — | **3.53x** | **0.82x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **71.1 μs** | 91.2 μs | 306.0 μs | 84.0 μs | — | **4.31x** | **1.18x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **74.6 μs** | 101.9 μs | 378.9 μs | 94.2 μs | — | **5.08x** | **1.26x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **54.4 μs** | 61.7 μs | 261.9 μs | 67.0 μs | — | **4.82x** | **1.23x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **47.4 μs** | 57.9 μs | 270.5 μs | 65.2 μs | — | **5.70x** | **1.38x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **63.9 μs** | 51.2 μs | 277.4 μs | 64.2 μs | — | **5.42x** | **1.25x** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **46.2 μs** | 51.2 μs | 275.9 μs | 70.4 μs | — | **5.98x** | **1.52x** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **59.1 μs** | 60.9 μs | 276.5 μs | 70.2 μs | — | **4.68x** | **1.19x** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **91.3 μs** | 99.8 μs | 322.8 μs | 188.9 μs | — | **3.53x** | **2.07x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **51.6 μs** | 58.5 μs | 255.5 μs | 65.6 μs | — | **4.95x** | **1.27x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **55.5 μs** | 51.0 μs | 304.3 μs | 69.2 μs | — | **5.97x** | **1.36x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **50.1 μs** | 51.7 μs | 282.4 μs | 63.6 μs | — | **5.64x** | **1.27x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **52.5 μs** | 57.5 μs | 246.7 μs | 62.7 μs | — | **4.70x** | **1.19x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **58.2 μs** | 52.2 μs | 270.3 μs | 65.3 μs | — | **5.18x** | **1.25x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **68.4 μs** | 53.9 μs | 273.4 μs | 65.8 μs | — | **5.07x** | **1.22x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **52.6 μs** | 58.0 μs | 251.3 μs | 63.7 μs | — | **4.78x** | **1.21x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **52.5 μs** | 50.0 μs | 267.5 μs | 66.8 μs | — | **5.36x** | **1.34x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **102.2 μs** | 67.8 μs | 308.8 μs | 76.8 μs | — | **4.55x** | **1.13x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **60.7 μs** | 52.4 μs | 346.8 μs | 68.6 μs | — | **6.62x** | **1.31x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **58.7 μs** | 51.5 μs | 348.3 μs | 63.1 μs | — | **6.76x** | **1.22x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **59.0 μs** | 55.7 μs | 358.1 μs | 66.7 μs | — | **6.43x** | **1.20x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **28.95 ms** | 28.06 ms | 63.94 ms | 34.24 ms | — | **2.28x** | **1.22x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **27.57 ms** | 27.55 ms | 109.55 ms | 33.52 ms | — | **3.98x** | **1.22x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **51.36 ms** | 50.35 ms | 106.49 ms | 91.56 ms | — | **2.12x** | **1.82x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **23.68 ms** | 25.30 ms | 122.58 ms | 21.12 ms | — | **5.18x** | **0.89x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | on | **1.35 ms** | 1.10 ms | 3.04 ms | — | — | **2.77x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | on | **4.54 ms** | 5.54 ms | 14.03 ms | — | — | **3.09x** | **—** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | on | **3.05 ms** | 3.08 ms | 5.45 ms | — | — | **1.79x** | **—** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | on | **12.24 ms** | 12.49 ms | 21.84 ms | — | — | **1.78x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | on | **925.5 μs** | 1.03 ms | 1.96 ms | — | — | **2.12x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | on | **2.94 ms** | 3.17 ms | 5.89 ms | — | — | **2.00x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.47 ms** | 1.58 ms | 3.37 ms | — | — | **2.29x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | on | **5.65 ms** | 6.66 ms | 14.15 ms | — | — | **2.50x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | on | **2.59 ms** | 3.03 ms | 5.15 ms | — | — | **1.99x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | on | **12.63 ms** | 14.13 ms | 28.55 ms | — | — | **2.26x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | on | **604.5 μs** | 778.2 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | on | **1.67 ms** | 1.71 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | on | **7.82 ms** | 6.14 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | on | **9.90 ms** | 10.39 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | on | **376.3 μs** | 417.5 μs | 973.5 μs | — | — | **2.59x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | on | **1.42 ms** | 1.40 ms | 3.18 ms | — | — | **2.27x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | on | **2.22 ms** | 2.40 ms | 4.61 ms | — | — | **2.07x** | **—** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | on | **530.3 μs** | 355.0 μs | 1.15 ms | — | — | **3.25x** | **—** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | on | **3.56 ms** | 3.51 ms | 5.89 ms | — | — | **1.68x** | **—** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | on | **5.19 ms** | 4.96 ms | 6.24 ms | — | — | **1.26x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | on | **315.5 μs** | 207.4 μs | 1.21 ms | — | — | **5.82x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | on | **765.0 μs** | 754.3 μs | 1.91 ms | — | — | **2.54x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | on | **1.03 ms** | 1.15 ms | 3.13 ms | — | — | **3.05x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | on | **259.4 μs** | 250.4 μs | 692.5 μs | — | — | **2.77x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | on | **1.35 ms** | 1.42 ms | 2.86 ms | — | — | **2.12x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | on | **2.36 ms** | 2.47 ms | 4.66 ms | — | — | **1.98x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | on | **546.2 μs** | 311.3 μs | 1.02 ms | — | — | **3.29x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | on | **2.47 ms** | 3.02 ms | 5.18 ms | — | — | **2.10x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | on | **4.04 ms** | 4.43 ms | 10.67 ms | — | — | **2.64x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | on | **174.8 μs** | 152.3 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | on | **420.1 μs** | 416.0 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | on | **722.3 μs** | 656.1 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **1.59 ms** | 1.74 ms | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **2.44 ms** | 2.10 ms | — | — | — | **—** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | on | **327.4 μs** | 305.0 μs | — | — | — | **—** | **—** |
| tensor | mef_small | read_full | 0.45 MB | CPU | on | **182.0 μs** | 354.5 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **185.3 μs** | 306.7 μs | — | — | — | **—** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | on | **4.39 ms** | 4.36 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | on | **1.03 ms** | 1.03 ms | — | — | — | **—** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | on | **198.8 μs** | 150.7 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | on | **74.5 μs** | 144.5 μs | 297.5 μs | — | — | **4.00x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | on | **99.3 μs** | 106.2 μs | 380.1 μs | — | — | **3.83x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | on | **108.2 μs** | 115.5 μs | 398.6 μs | — | — | **3.68x** | **—** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | on | **116.0 μs** | 131.6 μs | 286.0 μs | — | — | **2.46x** | **—** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | on | **163.1 μs** | 185.6 μs | 446.5 μs | — | — | **2.74x** | **—** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | on | **206.0 μs** | 257.5 μs | 598.7 μs | — | — | **2.91x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | on | **57.5 μs** | 55.7 μs | 285.1 μs | — | — | **5.12x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | on | **169.3 μs** | 133.5 μs | 385.3 μs | — | — | **2.89x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | on | **115.8 μs** | 138.0 μs | 359.9 μs | — | — | **3.11x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | on | **127.0 μs** | 124.5 μs | 276.6 μs | — | — | **2.22x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | on | **106.3 μs** | 84.8 μs | 388.8 μs | — | — | **4.58x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | on | **135.0 μs** | 132.2 μs | 440.5 μs | — | — | **3.33x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | on | **132.6 μs** | 124.0 μs | 338.7 μs | — | — | **2.73x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | on | **125.3 μs** | 122.1 μs | 352.9 μs | — | — | **2.89x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | on | **187.5 μs** | 232.0 μs | 496.4 μs | — | — | **2.65x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | on | **50.3 μs** | 62.8 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | on | **134.5 μs** | 100.9 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | on | **89.8 μs** | 141.2 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | on | **173.4 μs** | 151.5 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | on | **257.7 μs** | 236.5 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **156.5 μs** | 198.3 μs | 564.2 μs | — | — | **3.61x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **139.3 μs** | 269.1 μs | 819.6 μs | — | — | **5.88x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **252.1 μs** | 175.2 μs | 927.3 μs | — | — | **5.29x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **259.3 μs** | 234.7 μs | 1.43 ms | — | — | **6.12x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **309.9 μs** | 263.4 μs | 866.3 μs | — | — | **3.29x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **147.7 μs** | 132.3 μs | 841.0 μs | — | — | **6.36x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **154.0 μs** | 110.5 μs | 704.6 μs | — | — | **6.38x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **129.4 μs** | 126.0 μs | 676.0 μs | — | — | **5.37x** | **—** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **128.2 μs** | 144.4 μs | 633.0 μs | — | — | **4.94x** | **—** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **159.0 μs** | 174.4 μs | 677.0 μs | — | — | **4.26x** | **—** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **293.2 μs** | 298.8 μs | 818.3 μs | — | — | **2.79x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **164.5 μs** | 309.6 μs | 745.2 μs | — | — | **4.53x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **113.3 μs** | 104.4 μs | 850.6 μs | — | — | **8.15x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **118.1 μs** | 118.8 μs | 466.3 μs | — | — | **3.95x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **262.0 μs** | 192.7 μs | 508.5 μs | — | — | **2.64x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **183.3 μs** | 99.8 μs | 565.2 μs | — | — | **5.67x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **100.4 μs** | 113.1 μs | 669.6 μs | — | — | **6.67x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **122.5 μs** | 288.4 μs | 499.4 μs | — | — | **4.08x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **118.9 μs** | 125.2 μs | 544.3 μs | — | — | **4.58x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **167.0 μs** | 158.4 μs | 522.5 μs | — | — | **3.30x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **132.8 μs** | 101.0 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **90.6 μs** | 97.7 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **128.0 μs** | 98.2 μs | — | — | — | **—** | **—** |
| tensor | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | n/a | **1.03 ms** | 1.10 ms | 7.04 ms | 1.06 ms | — | **6.83x** | **1.03x** |
| tensor | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | n/a | **386.3 μs** | 361.5 μs | 3.31 ms | 353.9 μs | — | **9.16x** | **0.98x** |
| tensor | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | n/a | **13.48 ms** | 14.37 ms | 118.10 ms | 14.13 ms | — | **8.76x** | **1.05x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **13.50 ms** | 13.92 ms | 29.49 ms | 16.52 ms | — | **2.18x** | **1.22x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **14.36 ms** | 14.86 ms | 48.89 ms | 17.38 ms | — | **3.40x** | **1.21x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **25.33 ms** | 26.05 ms | 31.23 ms | 25.10 ms | — | **1.23x** | **0.99x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **8.32 ms** | 8.07 ms | 20.67 ms | 8.27 ms | — | **2.56x** | **1.03x** |
| tensor | large_float32_1d | read_full | 3.82 MB | MPS | off | **1.05 ms** | 844.5 μs | 2.05 ms | 997.2 μs | — | **2.43x** | **1.18x** |
| tensor | large_float32_2d | read_full | 16.00 MB | MPS | off | **3.24 ms** | 3.40 ms | 6.38 ms | 3.44 ms | — | **1.97x** | **1.06x** |
| tensor | large_int16_1d | read_full | 1.91 MB | MPS | off | **826.0 μs** | 519.1 μs | 1.21 ms | 597.3 μs | — | **2.33x** | **1.15x** |
| tensor | large_int16_2d | read_full | 8.00 MB | MPS | off | **1.79 ms** | 2.07 ms | 3.63 ms | 1.95 ms | — | **2.03x** | **1.09x** |
| tensor | large_int32_1d | read_full | 3.82 MB | MPS | off | **866.0 μs** | 867.3 μs | 2.23 ms | 974.2 μs | — | **2.57x** | **1.12x** |
| tensor | large_int32_2d | read_full | 16.00 MB | MPS | off | **3.48 ms** | 3.29 ms | 6.08 ms | 3.84 ms | — | **1.85x** | **1.17x** |
| tensor | large_int64_1d | read_full | 7.63 MB | MPS | off | **2.20 ms** | 2.21 ms | 4.43 ms | 1.92 ms | — | **2.02x** | **0.87x** |
| tensor | large_int64_2d | read_full | 32.00 MB | MPS | off | **8.78 ms** | 8.93 ms | 12.72 ms | 8.90 ms | — | **1.45x** | **1.01x** |
| tensor | large_int8_1d | read_full | 0.96 MB | MPS | off | **366.3 μs** | 505.7 μs | 962.5 μs | 839.2 μs | — | **2.63x** | **2.29x** |
| tensor | large_int8_2d | read_full | 4.00 MB | MPS | off | **1.31 ms** | 1.48 ms | 2.50 ms | 3.37 ms | — | **1.91x** | **2.57x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | MPS | off | **4.00 ms** | 3.85 ms | 5.92 ms | 3.83 ms | — | **1.54x** | **1.00x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | MPS | off | **5.58 ms** | 5.34 ms | 8.92 ms | 5.82 ms | — | **1.67x** | **1.09x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | MPS | off | **358.6 μs** | 295.5 μs | 736.3 μs | 320.5 μs | — | **2.49x** | **1.08x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | MPS | off | **998.9 μs** | 945.7 μs | 2.46 ms | 979.0 μs | — | **2.60x** | **1.04x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | MPS | off | **1.30 ms** | 1.65 ms | 3.14 ms | 1.31 ms | — | **2.42x** | **1.01x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | MPS | off | **247.7 μs** | 261.8 μs | 577.2 μs | 260.4 μs | — | **2.33x** | **1.05x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | MPS | off | **551.6 μs** | 560.5 μs | 1.55 ms | 577.7 μs | — | **2.81x** | **1.05x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | MPS | off | **929.1 μs** | 751.9 μs | 1.65 ms | 778.5 μs | — | **2.20x** | **1.04x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | MPS | off | **272.3 μs** | 299.4 μs | 919.0 μs | 288.7 μs | — | **3.38x** | **1.06x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | MPS | off | **884.0 μs** | 982.6 μs | 2.29 ms | 954.4 μs | — | **2.59x** | **1.08x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | MPS | off | **1.74 ms** | 1.90 ms | 3.00 ms | 1.59 ms | — | **1.73x** | **0.92x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | MPS | off | **426.9 μs** | 336.2 μs | 818.7 μs | 405.5 μs | — | **2.44x** | **1.21x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | MPS | off | **2.04 ms** | 2.44 ms | 3.46 ms | 2.25 ms | — | **1.70x** | **1.10x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | MPS | off | **3.30 ms** | 3.19 ms | 4.36 ms | 3.53 ms | — | **1.37x** | **1.11x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | MPS | off | **346.0 μs** | 348.8 μs | 592.3 μs | 333.0 μs | — | **1.71x** | **0.96x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | MPS | off | **404.0 μs** | 389.5 μs | 1.01 ms | 904.4 μs | — | **2.58x** | **2.32x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | MPS | off | **556.0 μs** | 1.22 ms | 1.38 ms | 1.32 ms | — | **2.49x** | **2.37x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **1.17 ms** | 1.02 ms | 2.01 ms | 1.17 ms | — | **1.96x** | **1.14x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **1.38 ms** | 1.56 ms | 2.66 ms | 1.58 ms | — | **1.93x** | **1.15x** |
| tensor | mef_medium | read_full | 7.02 MB | MPS | off | **415.4 μs** | 448.2 μs | 1.15 ms | 937.9 μs | — | **2.77x** | **2.26x** |
| tensor | mef_small | read_full | 0.45 MB | MPS | off | **344.3 μs** | 336.5 μs | 715.9 μs | 408.1 μs | — | **2.13x** | **1.21x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **357.7 μs** | 341.7 μs | 830.4 μs | 380.8 μs | — | **2.43x** | **1.11x** |
| tensor | scaled_large | read_full | 8.00 MB | MPS | off | **3.32 ms** | 3.35 ms | 7.82 ms | 3.45 ms | — | **2.36x** | **1.04x** |
| tensor | scaled_medium | read_full | 2.01 MB | MPS | off | **1.12 ms** | 1.16 ms | 2.44 ms | 1.04 ms | — | **2.19x** | **0.93x** |
| tensor | scaled_small | read_full | 0.13 MB | MPS | off | **279.5 μs** | 261.0 μs | 767.8 μs | 264.1 μs | — | **2.94x** | **1.01x** |
| tensor | small_float32_1d | read_full | 42.2 KB | MPS | off | **222.1 μs** | 483.9 μs | 565.1 μs | 220.9 μs | — | **2.54x** | **0.99x** |
| tensor | small_float32_2d | read_full | 0.26 MB | MPS | off | **271.0 μs** | 334.3 μs | 910.6 μs | 277.6 μs | — | **3.36x** | **1.02x** |
| tensor | small_float32_3d | read_full | 0.63 MB | MPS | off | **314.4 μs** | 300.4 μs | 813.0 μs | 318.4 μs | — | **2.71x** | **1.06x** |
| tensor | small_int16_1d | read_full | 22.5 KB | MPS | off | **227.8 μs** | 221.8 μs | 449.2 μs | 282.2 μs | — | **2.02x** | **1.27x** |
| tensor | small_int16_2d | read_full | 0.13 MB | MPS | off | **381.0 μs** | 426.1 μs | 538.1 μs | 258.6 μs | — | **1.41x** | **0.68x** |
| tensor | small_int16_3d | read_full | 0.32 MB | MPS | off | **273.0 μs** | 260.9 μs | 604.0 μs | 272.2 μs | — | **2.31x** | **1.04x** |
| tensor | small_int32_1d | read_full | 42.2 KB | MPS | off | **318.2 μs** | 366.7 μs | 664.5 μs | 231.4 μs | — | **2.09x** | **0.73x** |
| tensor | small_int32_2d | read_full | 0.26 MB | MPS | off | **291.7 μs** | 326.2 μs | 710.0 μs | 281.6 μs | — | **2.43x** | **0.97x** |
| tensor | small_int32_3d | read_full | 0.63 MB | MPS | off | **308.7 μs** | 302.3 μs | 742.3 μs | 323.5 μs | — | **2.46x** | **1.07x** |
| tensor | small_int64_1d | read_full | 0.08 MB | MPS | off | **354.7 μs** | 295.8 μs | 579.7 μs | 414.2 μs | — | **1.96x** | **1.40x** |
| tensor | small_int64_2d | read_full | 0.51 MB | MPS | off | **385.9 μs** | 368.6 μs | 721.0 μs | 365.3 μs | — | **1.96x** | **0.99x** |
| tensor | small_int64_3d | read_full | 1.26 MB | MPS | off | **503.5 μs** | 405.6 μs | 1.16 ms | 481.5 μs | — | **2.87x** | **1.19x** |
| tensor | small_int8_1d | read_full | 14.1 KB | MPS | off | **290.1 μs** | 250.0 μs | 560.4 μs | 272.8 μs | — | **2.24x** | **1.09x** |
| tensor | small_int8_2d | read_full | 0.07 MB | MPS | off | **342.0 μs** | 269.5 μs | 587.0 μs | 361.2 μs | — | **2.18x** | **1.34x** |
| tensor | small_int8_3d | read_full | 0.16 MB | MPS | off | **264.6 μs** | 265.3 μs | 676.1 μs | 371.0 μs | — | **2.56x** | **1.40x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | MPS | off | **567.4 μs** | 410.7 μs | 777.5 μs | 356.1 μs | — | **1.89x** | **0.87x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | MPS | off | **363.0 μs** | 289.7 μs | 1.79 ms | 324.0 μs | — | **6.19x** | **1.12x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **326.0 μs** | 275.7 μs | 649.6 μs | 383.5 μs | — | **2.36x** | **1.39x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **323.7 μs** | 278.5 μs | 691.2 μs | 284.2 μs | — | **2.48x** | **1.02x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **273.3 μs** | 295.5 μs | 742.2 μs | 323.2 μs | — | **2.72x** | **1.18x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **291.5 μs** | 324.7 μs | 668.0 μs | 306.6 μs | — | **2.29x** | **1.05x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **479.6 μs** | 310.5 μs | 617.3 μs | 508.0 μs | — | **1.99x** | **1.64x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **357.6 μs** | 296.5 μs | 705.2 μs | 330.0 μs | — | **2.38x** | **1.11x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **282.7 μs** | 224.4 μs | 575.8 μs | 340.3 μs | — | **2.57x** | **1.52x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **249.0 μs** | 223.0 μs | 531.5 μs | 294.5 μs | — | **2.38x** | **1.32x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **226.0 μs** | 253.5 μs | 522.0 μs | 269.5 μs | — | **2.31x** | **1.19x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **229.7 μs** | 225.9 μs | 472.0 μs | 273.7 μs | — | **2.09x** | **1.21x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **227.1 μs** | 207.3 μs | 613.2 μs | 373.0 μs | — | **2.96x** | **1.80x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **221.3 μs** | 244.9 μs | 547.6 μs | 284.4 μs | — | **2.47x** | **1.29x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **241.4 μs** | 215.1 μs | 466.0 μs | 371.6 μs | — | **2.17x** | **1.73x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **221.8 μs** | 209.3 μs | 488.6 μs | 233.8 μs | — | **2.33x** | **1.12x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **213.5 μs** | 209.7 μs | 537.7 μs | 268.5 μs | — | **2.56x** | **1.28x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **226.5 μs** | 236.4 μs | 519.5 μs | 397.2 μs | — | **2.29x** | **1.75x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **375.2 μs** | 302.7 μs | 560.5 μs | 245.4 μs | — | **1.85x** | **0.81x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **211.8 μs** | 251.2 μs | 532.8 μs | 286.2 μs | — | **2.52x** | **1.35x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **213.8 μs** | 259.0 μs | 519.4 μs | 286.5 μs | — | **2.43x** | **1.34x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **202.0 μs** | 206.5 μs | 591.2 μs | 250.9 μs | — | **2.93x** | **1.24x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **16.25 ms** | 15.38 ms | 33.24 ms | 19.61 ms | — | **2.16x** | **1.27x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **15.74 ms** | 15.95 ms | 55.48 ms | 19.47 ms | — | **3.52x** | **1.24x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **29.90 ms** | 29.77 ms | 38.59 ms | 30.06 ms | — | **1.30x** | **1.01x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **11.36 ms** | 10.77 ms | 30.89 ms | 12.04 ms | — | **2.87x** | **1.12x** |
| tensor | large_float32_1d | read_full | 3.82 MB | MPS | on | **1.13 ms** | 1.06 ms | 2.35 ms | — | — | **2.22x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | MPS | on | **3.93 ms** | 4.74 ms | 6.16 ms | — | — | **1.57x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | MPS | on | **711.3 μs** | 831.3 μs | 1.42 ms | — | — | **2.00x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | MPS | on | **1.97 ms** | 2.27 ms | 3.76 ms | — | — | **1.90x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | MPS | on | **1.14 ms** | 1.19 ms | 1.85 ms | — | — | **1.63x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | MPS | on | **3.99 ms** | 4.18 ms | 6.11 ms | — | — | **1.53x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | MPS | on | **1.82 ms** | 2.29 ms | 3.67 ms | — | — | **2.02x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | MPS | on | **8.06 ms** | 9.85 ms | 13.40 ms | — | — | **1.66x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | MPS | on | **390.2 μs** | 537.5 μs | 1.47 ms | — | — | **3.77x** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | MPS | on | **1.46 ms** | 1.34 ms | 2.93 ms | — | — | **2.19x** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | MPS | on | **4.93 ms** | 6.31 ms | 6.98 ms | — | — | **1.42x** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | MPS | on | **7.03 ms** | 8.13 ms | 12.94 ms | — | — | **1.84x** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | MPS | on | **341.5 μs** | 368.5 μs | 1.02 ms | — | — | **2.98x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | MPS | on | **1.72 ms** | 1.34 ms | 2.58 ms | — | — | **1.92x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | MPS | on | **1.65 ms** | 2.53 ms | 3.78 ms | — | — | **2.30x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | MPS | on | **326.8 μs** | 628.1 μs | 846.5 μs | — | — | **2.59x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | MPS | on | **1.05 ms** | 830.9 μs | 1.75 ms | — | — | **2.11x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | MPS | on | **985.0 μs** | 1.63 ms | 2.15 ms | — | — | **2.18x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | MPS | on | **373.2 μs** | 364.0 μs | 826.5 μs | — | — | **2.27x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | MPS | on | **1.78 ms** | 1.40 ms | 2.51 ms | — | — | **1.79x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | MPS | on | **2.14 ms** | 2.32 ms | 3.00 ms | — | — | **1.41x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | MPS | on | **434.5 μs** | 455.0 μs | 922.5 μs | — | — | **2.12x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | MPS | on | **4.30 ms** | 2.73 ms | 4.86 ms | — | — | **1.78x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | MPS | on | **4.22 ms** | 4.28 ms | 4.46 ms | — | — | **1.06x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | MPS | on | **548.8 μs** | 403.9 μs | 1.05 ms | — | — | **2.60x** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | MPS | on | **536.4 μs** | 503.7 μs | 1.88 ms | — | — | **3.73x** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | MPS | on | **634.4 μs** | 658.0 μs | 2.33 ms | — | — | **3.67x** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **1.28 ms** | 1.38 ms | 3.09 ms | — | — | **2.42x** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **1.96 ms** | 2.18 ms | 4.36 ms | — | — | **2.23x** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | MPS | on | **643.2 μs** | 580.5 μs | 1.86 ms | — | — | **3.20x** | **—** |
| tensor | mef_small | read_full | 0.45 MB | MPS | on | **337.6 μs** | 499.3 μs | 1.90 ms | — | — | **5.64x** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **303.2 μs** | 600.1 μs | 1.61 ms | — | — | **5.30x** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | MPS | on | **4.70 ms** | 4.87 ms | 10.81 ms | — | — | **2.30x** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | MPS | on | **1.46 ms** | 1.43 ms | 3.72 ms | — | — | **2.59x** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | MPS | on | **297.8 μs** | 688.6 μs | 2.02 ms | — | — | **6.78x** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | MPS | on | **395.0 μs** | 653.4 μs | 970.7 μs | — | — | **2.46x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | MPS | on | **413.6 μs** | 379.1 μs | 854.5 μs | — | — | **2.25x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | MPS | on | **489.4 μs** | 597.8 μs | 1.22 ms | — | — | **2.49x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | MPS | on | **328.2 μs** | 276.5 μs | 779.1 μs | — | — | **2.82x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | MPS | on | **668.8 μs** | 347.0 μs | 797.7 μs | — | — | **2.30x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | MPS | on | **398.6 μs** | 530.8 μs | 944.6 μs | — | — | **2.37x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | MPS | on | **498.6 μs** | 585.0 μs | 828.3 μs | — | — | **1.66x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | MPS | on | **459.5 μs** | 374.1 μs | 783.6 μs | — | — | **2.09x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | MPS | on | **461.0 μs** | 633.2 μs | 1.16 ms | — | — | **2.51x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | MPS | on | **625.7 μs** | 483.4 μs | 1.05 ms | — | — | **2.18x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | MPS | on | **409.6 μs** | 608.9 μs | 927.6 μs | — | — | **2.26x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | MPS | on | **909.8 μs** | 831.6 μs | 1.67 ms | — | — | **2.01x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | MPS | on | **288.7 μs** | 300.2 μs | 1.34 ms | — | — | **4.64x** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | MPS | on | **504.8 μs** | 394.6 μs | 1.32 ms | — | — | **3.35x** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | MPS | on | **328.0 μs** | 430.3 μs | 1.71 ms | — | — | **5.21x** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | MPS | on | **468.5 μs** | 465.6 μs | 2.36 ms | — | — | **5.08x** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | MPS | on | **493.7 μs** | 450.4 μs | 1.87 ms | — | — | **4.16x** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **372.0 μs** | 392.5 μs | 1.25 ms | — | — | **3.37x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **348.3 μs** | 632.8 μs | 793.3 μs | — | — | **2.28x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **378.5 μs** | 359.2 μs | 898.7 μs | — | — | **2.50x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **405.6 μs** | 429.6 μs | 946.3 μs | — | — | **2.33x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **399.6 μs** | 350.4 μs | 858.7 μs | — | — | **2.45x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **297.3 μs** | 238.0 μs | 696.4 μs | — | — | **2.93x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **234.9 μs** | 258.7 μs | 847.1 μs | — | — | **3.61x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **299.6 μs** | 314.0 μs | 715.4 μs | — | — | **2.39x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **497.8 μs** | 310.8 μs | 777.2 μs | — | — | **2.50x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **247.2 μs** | 255.5 μs | 647.8 μs | — | — | **2.62x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **342.9 μs** | 269.7 μs | 685.3 μs | — | — | **2.54x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **314.5 μs** | 617.5 μs | 611.2 μs | — | — | **1.94x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **244.3 μs** | 253.1 μs | 743.2 μs | — | — | **3.04x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **293.0 μs** | 604.3 μs | 699.0 μs | — | — | **2.39x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **323.3 μs** | 265.4 μs | 669.2 μs | — | — | **2.52x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **249.7 μs** | 285.4 μs | 693.1 μs | — | — | **2.78x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **526.8 μs** | 449.8 μs | 682.3 μs | — | — | **1.52x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **341.5 μs** | 247.3 μs | 1.25 ms | — | — | **5.06x** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **350.5 μs** | 267.5 μs | 1.13 ms | — | — | **4.23x** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **230.7 μs** | 253.8 μs | 1.12 ms | — | — | **4.85x** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **623.5 μs** | 331.4 μs | 3.14 ms | 405.9 μs | — | **9.49x** | **1.22x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.35 ms** | 1.30 ms | 10.44 ms | 2.72 ms | — | **8.04x** | **2.10x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.32 ms** | 1.27 ms | 10.44 ms | 2.61 ms | — | **8.23x** | **2.05x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **183.9 μs** | 196.1 μs | 2.64 ms | 821.6 μs | — | **14.34x** | **4.47x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **85.3 μs** | 87.3 μs | 357.3 μs | 87.0 μs | — | **4.19x** | **1.02x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **122.6 μs** | 106.7 μs | 1.49 ms | 167.1 μs | — | **14.00x** | **1.57x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **174.6 μs** | 188.5 μs | 2.36 ms | 410.2 μs | — | **13.53x** | **2.35x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **178.2 μs** | 182.5 μs | 2.03 ms | 368.1 μs | — | **11.39x** | **2.07x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **105.4 μs** | 117.2 μs | 1.67 ms | 203.7 μs | — | **15.81x** | **1.93x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **83.2 μs** | 111.4 μs | 353.0 μs | 87.9 μs | — | **4.24x** | **1.06x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **18.39 ms** | 22.75 ms | 18.49 ms | 41.29 ms | — | **1.01x** | **2.25x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.80 ms** | 10.79 ms | 18.70 ms | 75.03 ms | — | **1.73x** | **6.95x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **23.02 ms** | 21.64 ms | 396.98 ms | 140.54 ms | — | **18.34x** | **6.49x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **456.0 μs** | 865.6 μs | 17.21 ms | 2.32 ms | — | **37.74x** | **5.09x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **469.7 μs** | 147.5 μs | 453.3 μs | 104.0 μs | — | **3.07x** | **0.71x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.83 ms** | 2.20 ms | 3.23 ms | 4.64 ms | — | **1.76x** | **2.54x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.26 ms** | 1.69 ms | 3.64 ms | 8.42 ms | — | **2.89x** | **6.69x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **2.66 ms** | 2.71 ms | 42.22 ms | 14.07 ms | — | **15.88x** | **5.29x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **341.0 μs** | 453.7 μs | 7.06 ms | 2.12 ms | — | **20.70x** | **6.22x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **111.6 μs** | 109.6 μs | 493.6 μs | 211.9 μs | — | **4.50x** | **1.93x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **367.6 μs** | 328.5 μs | 1.64 ms | 587.1 μs | — | **5.00x** | **1.79x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **251.3 μs** | 259.6 μs | 1.76 ms | 901.2 μs | — | **7.01x** | **3.59x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **365.0 μs** | 394.9 μs | 5.47 ms | 1.51 ms | — | **14.98x** | **4.14x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **158.6 μs** | 196.7 μs | 3.25 ms | 499.2 μs | — | **20.46x** | **3.15x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **103.0 μs** | 139.7 μs | 431.8 μs | 107.8 μs | — | **4.19x** | **1.05x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **104.5 μs** | 94.0 μs | 1.48 ms | 183.8 μs | — | **15.78x** | **1.96x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **122.7 μs** | 123.9 μs | 1.56 ms | 219.3 μs | — | **12.73x** | **1.79x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **151.7 μs** | 173.3 μs | 1.88 ms | 376.2 μs | — | **12.40x** | **2.48x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **132.7 μs** | 139.4 μs | 2.37 ms | 196.9 μs | — | **17.87x** | **1.48x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **127.5 μs** | 105.5 μs | 384.6 μs | 102.1 μs | — | **3.64x** | **0.97x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **11.56 ms** | 13.83 ms | 9.93 ms | 20.38 ms | — | **0.86x** | **1.76x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **5.06 ms** | 5.27 ms | 8.10 ms | 38.78 ms | — | **1.60x** | **7.67x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **6.34 ms** | 6.46 ms | 9.20 ms | 7.74 ms | — | **1.45x** | **1.22x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **232.7 μs** | 161.5 μs | 5.42 ms | 725.6 μs | — | **33.59x** | **4.49x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **88.5 μs** | 108.1 μs | 419.8 μs | 373.2 μs | — | **4.75x** | **4.22x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.38 ms** | 1.56 ms | 2.21 ms | 2.21 ms | — | **1.60x** | **1.60x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **663.4 μs** | 708.2 μs | 2.49 ms | 4.32 ms | — | **3.75x** | **6.51x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **661.0 μs** | 678.5 μs | 2.13 ms | 801.2 μs | — | **3.22x** | **1.21x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **459.5 μs** | 236.4 μs | 2.13 ms | 825.2 μs | — | **9.03x** | **3.49x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **99.9 μs** | 254.3 μs | 403.1 μs | 96.1 μs | — | **4.03x** | **0.96x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **414.1 μs** | 207.0 μs | 1.11 ms | 323.0 μs | — | **5.38x** | **1.56x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **164.6 μs** | 214.1 μs | 1.09 ms | 518.0 μs | — | **6.64x** | **3.15x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **232.0 μs** | 222.6 μs | 1.12 ms | 260.4 μs | — | **5.03x** | **1.17x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **113.2 μs** | 120.5 μs | 1.54 ms | 207.6 μs | — | **13.59x** | **1.83x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **88.1 μs** | 88.0 μs | 369.7 μs | 81.0 μs | — | **4.20x** | **0.92x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **95.4 μs** | 84.0 μs | 1.01 ms | 174.9 μs | — | **12.03x** | **2.08x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **108.3 μs** | 116.4 μs | 1.16 ms | 188.8 μs | — | **10.68x** | **1.74x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **130.2 μs** | 136.6 μs | 1.25 ms | 203.7 μs | — | **9.61x** | **1.56x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **115.7 μs** | 117.6 μs | 1.44 ms | 176.0 μs | — | **12.47x** | **1.52x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **102.2 μs** | 107.2 μs | 432.7 μs | 96.1 μs | — | **4.23x** | **0.94x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **1.54 ms** | 1.19 ms | 2.33 ms | 2.02 ms | — | **1.95x** | **1.69x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **6.15 ms** | 6.12 ms | 40.28 ms | 19.83 ms | — | **6.58x** | **3.24x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **7.38 ms** | 7.24 ms | 39.01 ms | 20.76 ms | — | **5.39x** | **2.87x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **966.9 μs** | 1.25 ms | 5.56 ms | 2.66 ms | — | **5.75x** | **2.75x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **95.1 μs** | 115.3 μs | 432.9 μs | 215.4 μs | — | **4.55x** | **2.26x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **239.3 μs** | 195.3 μs | 1.06 ms | 293.5 μs | — | **5.45x** | **1.50x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **1.11 ms** | 814.9 μs | 4.63 ms | 1.71 ms | — | **5.68x** | **2.10x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **963.8 μs** | 813.6 μs | 4.84 ms | 2.73 ms | — | **5.95x** | **3.35x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **177.4 μs** | 146.0 μs | 1.88 ms | 490.3 μs | — | **12.86x** | **3.36x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **88.7 μs** | 90.7 μs | 371.6 μs | 87.6 μs | — | **4.19x** | **0.99x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **1.27 ms** | 1.00 ms | 2.83 ms | 1.95 ms | — | **2.82x** | **1.95x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **102.10 ms** | 103.23 ms | 626.62 ms | 144.96 ms | — | **6.14x** | **1.42x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **104.88 ms** | 105.41 ms | 633.26 ms | 143.20 ms | — | **6.04x** | **1.37x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **10.07 ms** | 8.62 ms | 63.10 ms | 14.63 ms | — | **7.32x** | **1.70x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **140.0 μs** | 88.5 μs | 412.6 μs | 87.5 μs | — | **4.66x** | **0.99x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **474.2 μs** | 169.7 μs | 1.24 ms | 286.8 μs | — | **7.31x** | **1.69x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **10.12 ms** | 9.59 ms | 63.53 ms | 14.84 ms | — | **6.62x** | **1.55x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **9.45 ms** | 9.68 ms | 65.74 ms | 14.78 ms | — | **6.96x** | **1.56x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **907.9 μs** | 852.8 μs | 7.23 ms | 1.94 ms | — | **8.48x** | **2.27x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **89.6 μs** | 92.9 μs | 414.0 μs | 98.6 μs | — | **4.62x** | **1.10x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **118.7 μs** | 99.0 μs | 1.47 ms | 187.6 μs | — | **14.83x** | **1.89x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **972.9 μs** | 973.6 μs | 7.84 ms | 1.72 ms | — | **8.06x** | **1.77x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **837.3 μs** | 859.2 μs | 6.91 ms | 1.47 ms | — | **8.25x** | **1.75x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **216.0 μs** | 256.7 μs | 2.85 ms | 413.5 μs | — | **13.18x** | **1.91x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **102.5 μs** | 108.0 μs | 462.4 μs | 112.5 μs | — | **4.51x** | **1.10x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **5.59 ms** | 6.30 ms | 10.17 ms | 7.01 ms | — | **1.82x** | **1.25x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **4.13 ms** | 4.90 ms | 10.85 ms | 9.07 ms | — | **2.62x** | **2.19x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **19.57 ms** | 17.91 ms | 167.30 ms | 68.78 ms | — | **9.34x** | **3.84x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **2.25 ms** | 1.87 ms | 27.50 ms | 7.81 ms | — | **14.69x** | **4.17x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **235.3 μs** | 316.8 μs | 611.3 μs | 333.2 μs | — | **2.60x** | **1.42x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **776.4 μs** | 870.2 μs | 5.97 ms | 1.06 ms | — | **7.69x** | **1.37x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **682.0 μs** | 593.8 μs | 6.08 ms | 1.24 ms | — | **10.23x** | **2.09x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.98 ms** | 2.04 ms | 21.26 ms | 6.72 ms | — | **10.71x** | **3.38x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **734.5 μs** | 642.5 μs | 10.94 ms | 1.26 ms | — | **17.03x** | **1.96x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **200.3 μs** | 191.8 μs | 534.6 μs | 304.5 μs | — | **2.79x** | **1.59x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **169.0 μs** | 174.4 μs | 5.34 ms | 480.5 μs | — | **31.61x** | **2.84x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **195.0 μs** | 211.0 μs | 5.06 ms | 513.1 μs | — | **25.97x** | **2.63x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **661.7 μs** | 629.2 μs | 6.86 ms | 1.14 ms | — | **10.90x** | **1.82x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **549.2 μs** | 541.0 μs | 9.04 ms | 689.1 μs | — | **16.72x** | **1.27x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **193.6 μs** | 210.5 μs | 561.6 μs | 330.0 μs | — | **2.90x** | **1.70x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **584.0 μs** | 262.3 μs | 4.93 ms | — | — | **18.81x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **375.4 μs** | 403.8 μs | 17.10 ms | — | — | **45.54x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **436.4 μs** | 352.4 μs | 12.97 ms | — | — | **36.80x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **558.7 μs** | 285.5 μs | 2.99 ms | — | — | **10.49x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **124.7 μs** | 94.5 μs | 405.8 μs | — | — | **4.29x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **249.0 μs** | 113.6 μs | 1.39 ms | — | — | **12.23x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **462.5 μs** | 576.9 μs | 2.58 ms | — | — | **5.59x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **242.7 μs** | 449.7 μs | 2.30 ms | — | — | **9.49x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **457.9 μs** | 492.0 μs | 1.75 ms | — | — | **3.83x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **150.6 μs** | 106.0 μs | 415.2 μs | — | — | **3.92x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **11.85 ms** | 16.74 ms | 32.15 ms | — | — | **2.71x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **15.96 ms** | 16.11 ms | 31.37 ms | — | — | **1.97x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **36.57 ms** | 32.30 ms | 600.80 ms | — | — | **18.60x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **689.7 μs** | 675.0 μs | 24.34 ms | — | — | **36.07x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **176.9 μs** | 186.8 μs | 666.4 μs | — | — | **3.77x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.80 ms** | 2.48 ms | 5.31 ms | — | — | **2.95x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **2.29 ms** | 1.58 ms | 7.12 ms | — | — | **4.52x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **3.14 ms** | 3.85 ms | 75.78 ms | — | — | **24.14x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **771.4 μs** | 1.11 ms | 13.70 ms | — | — | **17.76x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **463.5 μs** | 213.2 μs | 1.09 ms | — | — | **5.11x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **789.6 μs** | 359.1 μs | 3.47 ms | — | — | **9.65x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **602.2 μs** | 464.1 μs | 3.44 ms | — | — | **7.42x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **719.3 μs** | 735.6 μs | 10.64 ms | — | — | **14.80x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **635.8 μs** | 912.5 μs | 5.50 ms | — | — | **8.65x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **181.0 μs** | 167.1 μs | 699.9 μs | — | — | **4.19x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **381.6 μs** | 180.8 μs | 2.64 ms | — | — | **14.62x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **554.3 μs** | 624.9 μs | 2.66 ms | — | — | **4.79x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **550.2 μs** | 411.7 μs | 3.37 ms | — | — | **8.18x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **552.7 μs** | 653.0 μs | 4.15 ms | — | — | **7.51x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **186.6 μs** | 172.5 μs | 686.7 μs | — | — | **3.98x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **6.36 ms** | 12.97 ms | 21.28 ms | — | — | **3.35x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **5.83 ms** | 6.30 ms | 12.39 ms | — | — | **2.13x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **8.03 ms** | 6.75 ms | 14.49 ms | — | — | **2.15x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **423.5 μs** | 444.0 μs | 6.68 ms | — | — | **15.78x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **204.5 μs** | 177.1 μs | 697.0 μs | — | — | **3.93x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.21 ms** | 1.18 ms | 3.10 ms | — | — | **2.62x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **922.1 μs** | 717.3 μs | 2.56 ms | — | — | **3.57x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **797.7 μs** | 790.4 μs | 2.78 ms | — | — | **3.52x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **404.6 μs** | 337.4 μs | 3.07 ms | — | — | **9.09x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **131.0 μs** | 193.5 μs | 509.0 μs | — | — | **3.89x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **589.9 μs** | 390.8 μs | 3.19 ms | — | — | **8.16x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **464.3 μs** | 462.5 μs | 2.64 ms | — | — | **5.70x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **624.8 μs** | 428.3 μs | 2.65 ms | — | — | **6.18x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **534.4 μs** | 670.0 μs | 3.20 ms | — | — | **5.98x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **198.1 μs** | 186.4 μs | 774.6 μs | — | — | **4.16x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **427.1 μs** | 180.4 μs | 2.02 ms | — | — | **11.17x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **441.9 μs** | 498.2 μs | 1.91 ms | — | — | **4.33x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **494.9 μs** | 439.0 μs | 2.00 ms | — | — | **4.56x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **646.4 μs** | 598.7 μs | 2.51 ms | — | — | **4.20x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **201.0 μs** | 150.5 μs | 657.2 μs | — | — | **4.37x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **1.36 ms** | 485.4 μs | 2.64 ms | — | — | **5.44x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.99 ms** | 2.35 ms | 49.22 ms | — | — | **24.70x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.99 ms** | 2.68 ms | 45.90 ms | — | — | **23.04x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **499.3 μs** | 630.4 μs | 6.46 ms | — | — | **12.93x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **114.3 μs** | 122.9 μs | 461.5 μs | — | — | **4.04x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **630.3 μs** | 135.2 μs | 1.28 ms | — | — | **9.48x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **390.6 μs** | 340.2 μs | 5.27 ms | — | — | **15.48x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **364.9 μs** | 613.6 μs | 5.71 ms | — | — | **15.64x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **646.1 μs** | 389.0 μs | 2.62 ms | — | — | **6.73x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **112.5 μs** | 119.1 μs | 523.9 μs | — | — | **4.66x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **1.71 ms** | 397.8 μs | 1.62 ms | — | — | **4.07x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **140.14 ms** | 126.35 ms | 766.45 ms | — | — | **6.07x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **138.87 ms** | 154.68 ms | 960.26 ms | — | — | **6.91x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **11.62 ms** | 10.82 ms | 78.43 ms | — | — | **7.25x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **125.2 μs** | 132.2 μs | 610.0 μs | — | — | **4.87x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **705.3 μs** | 208.5 μs | 1.48 ms | — | — | **7.08x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **14.92 ms** | 14.34 ms | 99.27 ms | — | — | **6.92x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **15.02 ms** | 14.32 ms | 98.91 ms | — | — | **6.91x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **2.16 ms** | 1.52 ms | 11.63 ms | — | — | **7.63x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **160.9 μs** | 127.8 μs | 553.7 μs | — | — | **4.33x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **528.7 μs** | 147.7 μs | 1.49 ms | — | — | **10.12x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **2.24 ms** | 1.51 ms | 10.37 ms | — | — | **6.85x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **1.68 ms** | 2.07 ms | 13.83 ms | — | — | **8.21x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **952.4 μs** | 800.6 μs | 7.62 ms | — | — | **9.51x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **131.7 μs** | 155.8 μs | 659.1 μs | — | — | **5.00x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **3.62 ms** | 3.57 ms | 20.77 ms | — | — | **5.81x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **4.55 ms** | 3.88 ms | 15.50 ms | — | — | **4.00x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **30.34 ms** | 26.59 ms | 264.56 ms | — | — | **9.95x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **2.49 ms** | 2.44 ms | 47.98 ms | — | — | **19.70x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **359.8 μs** | 364.5 μs | 964.3 μs | — | — | **2.68x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **725.0 μs** | 521.5 μs | 8.89 ms | — | — | **17.05x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **666.8 μs** | 672.8 μs | 10.09 ms | — | — | **15.13x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **2.45 ms** | 3.20 ms | 34.85 ms | — | — | **14.24x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **1.32 ms** | 1.33 ms | 15.60 ms | — | — | **11.85x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **521.5 μs** | 273.0 μs | 756.5 μs | — | — | **2.77x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **488.9 μs** | 345.1 μs | 10.67 ms | — | — | **30.91x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **951.4 μs** | 1.01 ms | 10.98 ms | — | — | **11.54x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **1.45 ms** | 1.23 ms | 13.72 ms | — | — | **11.15x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **1.62 ms** | 1.63 ms | 23.43 ms | — | — | **14.44x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **407.0 μs** | 346.1 μs | 918.8 μs | — | — | **2.65x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| macOS arm64 / CPU | tensor | small_int32_1d [read_full] | off | 177.3 μs | 814.8 | fitsio/fitsio_torch | 1.92× |
| macOS arm64 / MPS | tensor | small_uint16_2d [read_full @ mps] | off | 567.4 μs | 819.0 | fitsio/fitsio_torch_device | 1.59× |
| macOS arm64 / MPS | tensor | tiny_int64_3d [read_full @ mps] | off | 375.2 μs | 819.0 | fitsio/fitsio_torch_device | 1.53× |
| macOS arm64 / MPS | tensor | small_int16_2d [read_full @ mps] | off | 381.0 μs | 819.0 | fitsio/fitsio_torch_device | 1.47× |
| macOS arm64 / MPS | tensor | large_int16_1d [read_full @ mps] | off | 826.0 μs | 817.0 | fitsio/fitsio_torch_device | 1.38× |
| macOS arm64 / MPS | tensor | small_int32_1d [read_full @ mps] | off | 318.2 μs | 819.0 | fitsio/fitsio_torch_device | 1.38× |
| macOS arm64 / CPU | tensor | multi_mef_10ext [cutout_100x100] | n/a | 277.0 μs | 814.8 | fitsio/fitsio_torch | 1.36× |
| macOS arm64 / MPS | tensor | medium_int16_3d [read_full @ mps] | off | 929.1 μs | 817.0 | fitsio/fitsio_torch_device | 1.19× |
| macOS arm64 / MPS | tensor | large_int64_1d [read_full @ mps] | off | 2.20 ms | 817.0 | fitsio/fitsio_torch_device | 1.15× |
| macOS arm64 / MPS | tensor | timeseries_frame_001 [read_full @ mps] | off | 323.7 μs | 819.0 | fitsio/fitsio_torch_device | 1.14× |
| macOS arm64 / MPS | tensor | small_uint32_2d [read_full @ mps] | off | 363.0 μs | 819.0 | fitsio/fitsio_torch_device | 1.12× |
| macOS arm64 / MPS | tensor | medium_float32_1d [read_full @ mps] | off | 358.6 μs | 817.0 | fitsio/fitsio_torch_device | 1.12× |
| macOS arm64 / CPU | tensor | mef_small [read_full] | off | 311.0 μs | 814.8 | fitsio/fitsio_torch | 1.11× |
| macOS arm64 / CPU | tensor | small_float32_1d [read_full] | off | 97.7 μs | 814.8 | fitsio/fitsio_torch | 1.09× |
| macOS arm64 / MPS | tensor | medium_int32_3d [read_full @ mps] | off | 1.74 ms | 817.0 | fitsio/fitsio_torch_device | 1.09× |
| macOS arm64 / MPS | tensor | tiny_float32_1d [read_full @ mps] | off | 357.6 μs | 819.0 | fitsio/fitsio_torch_device | 1.08× |
| macOS arm64 / CPU | tensor | compressed_rice_1 [read_full] | on | 23.68 ms | 487.4 | fitsio/fitsio_torch | 1.08× |
| macOS arm64 / MPS | tensor | scaled_medium [read_full @ mps] | off | 1.12 ms | 819.0 | fitsio/fitsio_torch_device | 1.07× |
| macOS arm64 / MPS | tensor | small_int8_1d [read_full @ mps] | off | 290.1 μs | 819.0 | fitsio/fitsio_torch_device | 1.06× |
| macOS arm64 / CPU | tensor | tiny_float64_3d [read_full] | off | 91.3 μs | 814.8 | fitsio/fitsio_torch | 1.06× |
| macOS arm64 / MPS | tensor | scaled_small [read_full @ mps] | off | 279.5 μs | 819.0 | fitsio/fitsio_torch_device | 1.06× |
| macOS arm64 / MPS | tensor | large_float32_1d [read_full @ mps] | off | 1.05 ms | 817.0 | fitsio/fitsio_torch_device | 1.06× |
| macOS arm64 / MPS | tensor | small_int64_2d [read_full @ mps] | off | 385.9 μs | 819.0 | fitsio/fitsio_torch_device | 1.06× |
| macOS arm64 / MPS | tensor | medium_int64_1d [read_full @ mps] | off | 426.9 μs | 817.5 | fitsio/fitsio_torch_device | 1.05× |
| macOS arm64 / MPS | tensor | small_int64_3d [read_full @ mps] | off | 503.5 μs | 819.0 | fitsio/fitsio_torch_device | 1.05× |
| macOS arm64 / MPS | tensor | large_uint16_2d [read_full @ mps] | off | 4.00 ms | 817.0 | fitsio/fitsio_torch_device | 1.04× |
| macOS arm64 / MPS | tensor | medium_int8_1d [read_full @ mps] | off | 346.0 μs | 817.5 | fitsio/fitsio_torch_device | 1.04× |
| macOS arm64 / MPS | tensor | small_int32_2d [read_full @ mps] | off | 291.7 μs | 819.0 | fitsio/fitsio_torch_device | 1.04× |
| macOS arm64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 51.36 ms | 475.2 | fitsio/fitsio_torch | 1.02× |
| macOS arm64 / MPS | tensor | medium_float32_2d [read_full @ mps] | off | 998.9 μs | 817.0 | fitsio/fitsio_torch_device | 1.02× |
| macOS arm64 / CPU | tensor | repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | n/a | 7.25 ms | 814.8 | fitsio/fitsio_torch | 1.01× |
| macOS arm64 / MPS | tensor | compressed_hcompress_1 [read_full @ mps] | off | 25.33 ms | 817.0 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | small_float32_1d [read_full @ mps] | off | 222.1 μs | 819.0 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | compressed_rice_1 [read_full @ mps] | off | 8.32 ms | 817.0 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | medium_uint16_2d [read_full @ mps] | off | 1.17 ms | 819.0 | fitsio/fitsio_torch_device | 1.00× |
| macOS arm64 / MPS | tensor | small_int16_3d [read_full @ mps] | off | 273.0 μs | 819.0 | fitsio/fitsio_torch_device | 1.00× |
| macOS arm64 / CPU | tensor | small_float64_1d [read_full] | off | 201.5 μs | 814.8 | fitsio/fitsio_torch | 2.08× |
| macOS arm64 / CPU | tensor | small_float32_1d [read_full] | off | 163.0 μs | 814.8 | fitsio/fitsio_torch | 1.82× |
| macOS arm64 / MPS | tensor | small_float32_1d [read_full @ mps] | off | 483.9 μs | 819.0 | fitsio/fitsio_torch_device_specialized | 1.71× |
| macOS arm64 / CPU | tensor | small_int16_2d [read_full] | off | 199.4 μs | 814.8 | fitsio/fitsio_torch | 1.65× |

_…and 84 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| macOS arm64 / MPS | `exhaustive_mps_20260719_065105` | 3931 | 123 | 802.4 | lab + mmap-matrix + GPU |
| Linux x86_64 / CPU | `exhaustive_cpu_20260717_040146` | 2825 | 1 | 288.8 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260717_042840` | 4079 | 0 | 730.7 | lab + mmap-matrix + GPU |
<!-- BENCH_HOSTS_END -->

Round-2 local re-soak: MPS `exhaustive_mps_20260719_065105`; Linux CPU/CUDA rows remain Jul-17 soak IDs (`exhaustive_cpu_20260717_040146`, `exhaustive_cuda_20260717_042840`). ML loader: `ml_20260719_070024`. MegaCam: `20260719_000538`.



Latest local quick benchmark evidence:

<!-- BENCH_QUICK_BEGIN -->
| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| — | FITS image I/O | _(no run yet)_ | — | — |
| — | FITS table I/O | _(no run yet)_ | — | — |
<!-- BENCH_QUICK_END -->

### ML DataLoader throughput

<!-- BENCH_ML_BEGIN -->
Source: `docs/assets/bench/ml_20260719_070024/ml_results.csv` (device=cpu).

| Case | Method | Median throughput |
|---|---|---:|
| ml_compressed_rice | `fitsio (comp)` | 406,679,282 pixels/s |
| ml_compressed_rice | `torchfits (comp)` | 406,784,747 pixels/s |
| ml_uncompressed | `fitsio + numpy` | 1,484,808,470 pixels/s |
| ml_uncompressed | `torchfits` | 1,390,761,465 pixels/s |
<!-- BENCH_ML_END -->

### CFHT MegaCam MEF cutouts (local)

<!-- BENCH_MEGACAM_BEGIN -->
Source: `docs/assets/bench/20260719_000538/megacam_results.csv` (160 OK rows).

| Method | Median throughput |
|---|---:|
| `fitsio_cached` | 54.2 MB/s |
| `torchfits_cached` | 61.7 MB/s |
| `torchfits_materialize` | 121.7 MB/s |
| `torchfits_naive` | 53.5 MB/s |
<!-- BENCH_MEGACAM_END -->


Keep this page current with the latest tensor and table benchmark run before
making performance claims.
