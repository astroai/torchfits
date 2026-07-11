# Release API freeze review — 0.7.0

**Date:** 2026-07-10  
**Verdict:** **Ship with notes** (feature-freeze + API-freeze for 0.7.0 pre-1.0)

## Summary

0.7.0 completes the ML data layer, removes legacy `FITSDataset` aliases, adds
Zensical docs + expanded `release-gate`, and renames the table C++ backend to
`"cpp"` with a torch-first read path. Public root I/O, `table.*`, `cache.*`,
`data.*`, and `transforms.*` are stable for the 0.7.0 tag. GPU I/O (0.8.0) and
blocking mypy remain deferred before `1.0.0rc1`.

## Blocking

None for **0.7.0** tag.

## Should-fix (non-blocking)

| Item | Notes |
|---|---|
| `transforms.*` symbols in root `__all__` | 17 transform classes exported at package root but not listed in `docs/api.md` Quick Paths — document or defer re-export trim to 1.0 |
| `torchfits.table.TABLE_BACKENDS` | Now exported from `torchfits.table` (was documented but missing) |
| `bench_ml_loader` throughput row | Roadmap item still pending lab snapshot |

## Defer to 0.8.0 / 1.0

- GPU I/O E1–E3 + CANFAR bench sign-off
- Blocking mypy (T31)
- Remove `cpp_numpy` deprecation alias
- `1.0.0rc1` → `1.0.0` after GPU + mypy

## Evidence gaps

| Parity row | Gap |
|---|---|
| Arrow/Polars/DuckDB interop | Partial — optional deps; covered when installed |
| Scaled table columns mmap | Partial — documented limitation |
| VLA mmap | Partial — documented limitation |

All **Supported** tier-1/2 rows in `docs/parity.md` have named test files.

## Evidence

| Gate | Result |
|---|---|
| `pixi run release-gate` | **414 passed**, 3 skipped (2026-07-10) |
| Public API inventory | 50 root `__all__` symbols; `table`/`cache`/`cpp`/`data` namespaces |
| `tests/test_docs_integrity.py` | Included in release-gate |
| `tests/test_package_isolation.py` | No Python `astropy`/`fitsio` imports in `src/torchfits` |
| Benchmark snapshot | `20260709_163739` — 2754 rows, **3 deficits** (README + `docs/benchmarks.md` aligned) |

## Version triplet

- `pyproject.toml`, `pixi.toml`, `__init__.py`: **0.7.0** ✓

## Public API changes in 0.7.0

| Change | Breaking? | Migration |
|---|---|---|
| `FITSDataset` / `IterableFITSDataset` removed | Yes | `docs/migration_datasets.md` → `torchfits.data` |
| Table backend `"cpp_numpy"` → `"cpp"` | Yes (alias warns) | Pass `backend="cpp"`; old name works with `DeprecationWarning` |
| `torchfits.table` no longer re-exports `_` helpers | Yes | Import from `torchfits._table.*` only if you were using private API |
| `FitsTableIterableDataset`, `FitsCutoutDataset` | Additive | `torchfits.data` |

## High-impact defaults (unchanged)

- `read(..., scale_on_device=True)`, `mmap="auto"`, `device="cpu"`
- `table.read(..., backend="auto")` — C++ pushdown for `where=` on safe tables
- `TORCHFITS_TABLE_SCANNER_THRESHOLD`, handle/reader cache env vars

## API-freeze declaration

From **v0.7.0** until **1.0.0rc1**: bugfixes and doc corrections only on the
frozen surface (`read`/`write`/`open`, `table.*`, `cache.*`, `data.*`,
`transforms.*`). New public symbols or signature changes require changelog entry
and minor bump; breaking changes require migration guide (pre-1.0 still allowed
with notice).

## Freeze checklist

| Criterion | Pass? |
|---|---|
| No undocumented *required* public exports | ✓ (`TABLE_BACKENDS` fixed) |
| Parity Supported rows backed by tests | ✓ |
| `release-gate` green (CPU) | ✓ |
| Examples runnable | ✓ (`examples/test_examples.py` in gate) |
| Version/changelog aligned | ✓ |
| No README/docs scope creep (WCS/sphere/HEALPix) | ✓ |
| GPU bench-gate | Deferred 0.8.0 |
