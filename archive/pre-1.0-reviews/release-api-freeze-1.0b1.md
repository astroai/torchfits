# Release API freeze review — 1.0b1

**Date:** 2026-07-17  
**Version:** 1.0b1  
**Branch:** `release/1.0b1`  
**Verdict:** **Ship with notes**

## Summary

1.0b1 is a **beta** of the public story (images → tensors; FITS tables →
dataframes under `torchfits.table`), not the final SemVer 1.0.0 freeze.
Four specialist reviews plus real-data CLI peer checks are **Ship with notes**.
The only release-identity Block (version still 0.9.3) is resolved by this cut.

## Specialist reviews

| Track | Doc | Verdict |
|-------|-----|---------|
| R1 Rendered docs | [`release-1.0b1-docs.md`](release-1.0b1-docs.md) | Ship with notes (0 blocking) |
| R2 API adoption | [`release-1.0b1-api-adoption.md`](release-1.0b1-api-adoption.md) | Block→cleared by version bump; else Ship with notes |
| R3 Deep code | [`release-1.0b1-code.md`](release-1.0b1-code.md) | Ship with notes (0 blocking) |
| R4 Real-data CLI | [`release-1.0b1-realdata-cli.md`](release-1.0b1-realdata-cli.md) | Ship with notes |
| Security | [`_security_1.0b1.md`](_security_1.0b1.md) | No Blocking/Important |

## Blocking (must be clear before tag)

None remaining.

- Version triplet synced to **1.0b1** (`pyproject.toml`, `pixi.toml`, `__init__.py`).
- CLI `transform` on integer HDUs fixed (float promote + header policy).

## Should-fix (noted, non-blocking for b1)

- CLI cold-start ~0.8–1 s vs gnuastro/CFITSIO ms-scale (document as process tax).
- `verify` messaging when checksums absent.
- Alias demotion in docs (partially done in `api.md` which-reader).
- Example-runner “skipping” substring PASS; probe double-print (R3).
- Benchmark claim footnotes / single scorecard provenance across migration pages (R1/R3).

## Defer to 1.0.0rc1

**Closed in `release-api-freeze-1.0.0rc1.md`.**

1. Fresh multi-host exhaustive scorecard refresh (cite new IDs in README/benchmarks).
2. Clean-install wheel matrix across supported Python versions.
3. Network-mounted storage fixtures + replay bundle.
4. Published compatibility matrix (Python / PyTorch / Arrow / downstream).
5. Formal public-boundary freeze stamp for SemVer 1.0.0.
6. FITSH peer CLI (unavailable via brew here; SKIP in R4).
7. Deprecation warnings for root `read_table` / `stream_table` if desired.

## Evidence

- Baseline: `pixi run preflight-push`, `release-gate`, `docs-build` green on branch.
- Targeted re-verify after transform fix: `tests/test_cli.py`,
  `test_public_boundary.py`, `test_docs_integrity.py`; HorseHead transform CLI.
- Scorecard: continue citing published IDs from 0.9.3 docs (no new exhaustive
  for b1 by plan).

## Freeze rule for 1.0b1

Bugfixes and doc corrections allowed on the beta line. Breaking public API
changes require a new beta/rc. Do **not** claim SemVer 1.0.0 stability until
rc1 gates above are green.
