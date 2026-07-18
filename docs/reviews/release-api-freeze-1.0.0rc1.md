# Release API freeze — 1.0.0rc1

**Date:** 2026-07-17  
**Tree version:** `1.0.0rc1`  
**Verdict:** **Ready to tag `v1.0.0rc1`** — b1 deferrals closed below; SemVer
`1.0.0` still waits for the final freeze stamp after rc soak.

## Public-boundary freeze (rc1)

Public FITS → tensor / dataframe surface for 1.0:

- Core I/O: `read` / `read_tensor` / `write` / `open` / subset / batch / checksums
- Tables: `torchfits.table.*` (Arrow default; `read_torch` / `scan_torch`)
- Data: `Dataset` + `make_loader`
- Transforms: catalog in `docs/api-transforms.md`
- CLI: one-word MEF surface in `docs/cli.md`

Root `read_table` / `stream_table` / `read_table_rows` emit
`DeprecationWarning` and stay callable for compatibility.

## Gate status (b1 → rc1 deferrals)

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | Multi-host exhaustive scorecard | **Good** | Published IDs unchanged from b1 same-day refresh: `exhaustive_mps_20260717_040150`, `exhaustive_cpu_20260717_040146`, `exhaustive_cuda_20260717_042840` (cited in `docs/benchmarks.md`). rc1 local release-suite confirm: `20260717_212321` (2825 rows, 3 deficits, exit 0). |
| 2 | Clean-install wheel matrix | **Good** | Local: `bash scripts/clean_install_smoke.sh` — `torchfits-1.0.0rc1` cp313 macOS arm64 wheel installed in a fresh venv; image+table smoke OK. Full CPython 3.10–3.13 Linux/macOS matrix via `build_wheels.yml` on tag. |
| 3 | Network fixtures + replay | **Good** | `tests/test_http_probe_fixture.py` — local Range-capable HTTP server + `probe` JSON path. |
| 4 | Compatibility matrix | **Good** | `docs/compatibility.md` (Python / PyTorch / Arrow / platforms). |
| 5 | Formal public-boundary freeze stamp | **Good** | This document. Final `1.0.0` stamp after rc soak. |
| 6 | FITSH peer CLI | **Good (SKIP)** | Unavailable via Homebrew; same SKIP as b1 R4 (`docs/reviews/release-1.0b1-realdata-cli.md`). |
| 7 | Deprecate root table helpers | **Good** | `DeprecationWarning` on `read_table` / `stream_table` / `read_table_rows`. |

## Should-fix from b1 (closed)

- `verify` no-checksum messaging → `OK (no checksum keywords)` / exit 0 / `status=no_checksums`
- Example-runner REQUIRED examples cannot skip
- CLI process tax documented in `docs/cli.md` (cold-start note)

## Freeze rule

Bugfixes and docs allowed on the rc line. Breaking public API changes require a
new rc. Do **not** claim SemVer `1.0.0` until the final freeze stamp after soak.
