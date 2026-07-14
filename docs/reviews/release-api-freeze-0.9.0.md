# Release API freeze review — 0.9.0

**Date:** 2026-07-14  
**Verdict:** **Release gates pass**

## Summary

Torchfits 0.9.0 consolidates the unreleased 0.8 work into a tested wheel,
finishes the Arrow/Polars table boundary, hardens native-handle invalidation and
atomic table mutation, and makes strict type checking a release gate. The
candidate has 53 unique root exports and no undocumented required public API.

The first exact-commit exhaustive run did what a release benchmark should do:
it found an accidental byte-by-byte tensor-storage conversion that made table
predicates 2.6–434.9× slower than the comparison floor. Commit `8232ef7`
replaced that conversion with a shared NumPy buffer and selects the faster
native-full-read plus Arrow-filter plan by default. The replacement CANFAR
report confirms the repair on the release GPU with no deficit above 1.439×.

## Blocking

None. Tagging and publication may proceed from the reviewed commit after this
report is committed and its documentation checks pass.

## Should-fix

None known for 0.9.0 after the benchmark repair.

## Defer

- Python 3.14, Linux aarch64, and macOS x86_64 until CI and wheel builds cover
  those targets.
- VLA mmap and scaled-table mmap; both remain documented partial-parity rows.
- Device-native FITS writes; GPU inputs currently cross the documented host
  boundary before CFITSIO writes.
- Optional dataframe/database adapters beyond the supported Arrow and Polars
  surfaces.

## Public API and defaults

- 53 root `__all__` entries, all unique and covered by the documentation
  contract.
- `read(..., hdu=None, device="cpu", mmap="auto", mode="auto")`.
- `read_tensor(..., device="cpu", mmap=True, raw_scale=False)`.
- `table.read(..., mmap=True, decode_bytes=True,
  include_fits_metadata=False, apply_fits_nulls=True, backend="auto")`.
- `table.scan(..., batch_size=65536, mmap=True, backend="auto")`.
- The removed `"cpp_numpy"` table backend now raises `ValueError`; callers use
  `backend="cpp"`. The environment-dependent `TensorFrame` alias was also
  removed in favor of the stable Arrow/Polars boundary.

## Evidence

| Gate | Result |
|---|---|
| Version triplet | `pyproject.toml`, `pixi.toml`, and `torchfits.__version__` report 0.9.0. |
| Focused predicate/interop/API tests | 73 passed. |
| Local table smoke | 60 benchmark rows; a 100k-row predicate measured 1.5 ms in default mode versus 6.4 ms with explicit native row-wise pushdown. |
| `pixi run preflight-push` | Ruff, formatting, strict mypy, and compileall pass at `8232ef7`. |
| Local release gate | 536 passed, 3 skipped in 102.94 s outside the macOS shared-memory sandbox. |
| Full local test suite | 785 passed, 6 skipped in 134.67 s outside the macOS shared-memory sandbox. |
| GitHub exact-commit matrix | Commit `8232ef7`: lint, docs contract, strict mypy/release gate, benchmark smoke, and Linux/macOS CPython 3.10–3.13 all pass ([run 29313075401](https://github.com/astroai/torchfits/actions/runs/29313075401)). |
| Clean candidate wheel | Commit `8232ef7`, macOS arm64 CPython 3.13; 864,920 bytes, 58 entries, no C/C++ headers or sources; SHA256 `4e5c2f78d332d05c619334d3a5e47a1d03dda2714674089e63b6433edb8eeeb3`; isolated image, table, and predicate smoke pass. |
| Initial exhaustive benchmark | `exhaustive_cuda_0.9.0_20260714_061550`, commit `b136bff`: 3,648 rows including 1,140 fitstable rows; rejected because 18 predicate deficits exceeded the 2× floor. |
| Replacement exhaustive benchmark | `exhaustive_cuda_0.9.0_20260714_065950`, commit `8232ef7`: 3,648 normalized rows, 7 deficits, maximum 1.439×, no large-N deficit, 894 CUDA transport rows; archived at `vos:sfabbro/torchfits-gpu-bench/exhaustive_cuda_0.9.0_20260714_065950`. Results SHA256 `350451b56268866235306a604c29383757d9a4b5b6af98189b7ed10a2a457587`. |

## Evidence gaps

No technical gap remains. The tag, GitHub release, and PyPI artifact do not yet
exist because this is the pre-tag audit; they must be verified after the tag
workflow publishes them.
