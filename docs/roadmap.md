# Roadmap

`torchfits` is FITS I/O for PyTorch tensors and dataframes. This page is
forward-looking only.

## Current focus (toward 1.0.0)

**`1.0.0rc4`** is on PyPI with manylinux + macOS arm64 wheels
(`pip install "torchfits==1.0.0rc4"`). Collaborator soak is in progress; SemVer
**`1.0.0`** waits for feedback. Larger scope cuts (cache merge, dataset zoo,
CLI trim) stay deferred.

## Parity tiers

| Tier | Meaning |
|------|---------|
| Public contract | Docs and examples match the implemented surface |
| fitsio / Astropy workflows | Common read/write/header/checksum/compression paths interoperate |
| Selected CFITSIO | Documented where we expose CFITSIO-backed semantics |
| Non-goals | Full CFITSIO API parity; WCS / sphere / HEALPix / sky simulation |

## Near-term (through 1.0.0)

- Soak `1.0.0rc4` with collaborators; fold feedback before SemVer `1.0.0`
- Benchmark honesty: tensor vs table domains, CPU↔GPU deficits visible, CFITSIO-direct in the exhaustive table, MegaCam multi-cutout suite
- CLI depth: HIERARCH keys, batch header edit, honest `verify` scope, real RGB demos
- Publish exhaustive CSVs with release artifacts for user analysis

## Deferred (cosmetic / low priority)

From the rc1 static audits (not blocking 1.0):

- F1–F5 stylistic / macOS sysctl / CTYPE5 / END HISTORY notes
- C3 fold `insert_rows`+`update_rows` into one CFITSIO open (perf only)
- E5–E6 bench CSV cosmetics

### Deferred from prep review (post-rc4 / 1.1)

- M1 dtype-validation dedup across `write_api` / `mutation`
- M4 `HDUList.fromfile` fallback loop cleanup; M7 cache test isolation
- L2 Header `_mutate()` helper; L4 WHERE LRU sizing for dynamic literals
- T1–T4/T7 test-harness polish; E2/E5/E6 example assertion depth
- B6–B8 bench suite consolidation / `no_gpu` docs / `bench_ml_loader` review

## 1.1 candidates

- **Disk→GPU path** — today CFITSIO delivers host memory; a direct GPU path would need GPUDirect Storage, a CFITSIO change, or a different backend. Investigation only until there is a clear PyTorch-native API and tests.
- Richer compressed write coverage and public-file replay snapshots
- Broader Astropy table / ASCII / VLA cases where they stay small and tested

### CFITSIO (from 2026-07 CFITSIO audit leftovers)

- `fits_get_bcolparms` in `TableReader::analyze_table` (cold wide-table metadata)
- Expose quantize / HCOMPRESS / noise-bits / dither setters on write + `convert` CLI
- `fits_flush_file` between large multi-HDU writes (bound CFITSIO write buffers)
- `fits_open_data` for empty-primary MEF cold opens (marginal vs Python auto-HDU)
- `fits_set_bufsize` for CompImage decompression fallbacks
- Virtual image section / tile getters — only if compressed-mosaic cutout deficit measured
- `fits_img_stats_*` for CLI CPU stats; `fits_read_pixnull*` simplification
- Optional `thread_local` `fitsfile*` pool if per-read open cost shows up in scorecards

## Permanent design choices (not gaps)

- VLA and scaled-column **mmap** updates stay on the buffered path (format / safety).
- **GPU writes** stay host-copy through CFITSIO unless a 1.1 backend lands.
- PyArrow is the table runtime; Polars/Pandas/DuckDB remain optional.
- Concurrent reads use **private** `fitsfile*` handles (CFITSIO R2); do not
  reintroduce a shared-handle LRU across threads (option C — serialize on
  `meta->mutex` — is also rejected).
- No CFITSIO `fits_iterate_data`, `fits_calculator` / `fits_select_rows`,
  histogram binning, hierarchical grouping, table compress, WCS rebin, pixel
  filter, IRAF delete, template exec, or CFITSIO HTTPS/stream drivers (own
  HTTP cache after the `sh://` security fix).
- `fits_set_huge_hdu` / `fits_get_eqcoltype` / `fits_decode_chksum` /
  `fits_translate_keywords` stay unused unless a concrete user need appears.
