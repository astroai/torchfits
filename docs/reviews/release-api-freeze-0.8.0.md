# Release API freeze review — 0.8.0

**Date:** 2026-07-12  
**Verdict:** **Usable locally; hold publication for remote promotion gates**

## Summary

The 0.8.0 source and a clean macOS arm64 wheel pass image and table
round-trips, including the `TableHDU.from_fits` path implicated in the
published 0.7 downstream crash. The review removed duplicate native read
plumbing, made type checking blocking, corrected the wheel platform contract,
stopped shipping CFITSIO development artifacts inside the wheel, and made
existing-file rewrites transactional.

No known local correctness blocker remains for ordinary CPU FITS I/O. Do not
publish 0.8.0 until the GitHub wheel matrix and CANFAR Linux/CUDA gates pass.

## Blocking

| Promotion gate | Required evidence |
|---|---|
| GitHub wheel matrix | Installed-wheel smoke passes for CPython 3.10–3.13 on Linux x86_64 and macOS arm64. |
| CANFAR staging | Linux clean-wheel table/image matrix, bounded-memory streaming, deterministic cache cleanup, and GPU E1–E3 on `/scratch`. |
| Independent review | Sol-high or human review of the native table changes, wheel workflow, and public compatibility boundary. |

These block publication, not continued local/downstream development against
the reviewed 0.8 source.

## Should-fix

- Continue decomposing `transforms.py` and `_table/read.py` only along existing
  feature boundaries; their size is a maintenance risk but not a reason for a
  speculative rewrite.

## Defer

- Python 3.14 support and classifiers until CI and wheel builds cover it.
- Linux aarch64 and macOS x86_64 wheel claims until those artifacts are built.
- Exhaustive performance, large-file, CUDA, and network-storage runs to CANFAR
  staging; do not run them on the workstation.
- Learned I/O or domain-specific sky functionality; those remain downstream.

## Evidence gaps

| Area | Gap |
|---|---|
| Published artifact | PyPI still serves 0.7.0; the reviewed MIT 0.8.0 artifact is not published. |
| Linux native lifecycle | Local evidence is macOS arm64; CANFAR must cover Linux process, cache, and table-reader lifecycle. |
| CUDA | No local CUDA device; CANFAR E1–E3 remains required. |
| Performance | Existing 0.7 CANFAR snapshot remains the latest accepted benchmark evidence; this review makes no new speed claim. |

## Repairs made

- `TableHDU.from_fits` now delegates to the public `read_table` pipeline and
  reuses its header result instead of issuing separate native table and header
  reads.
- Root `__all__` is unique and the public API inventory is fully documented.
- Lazy root functions retain their real signatures, identities, and tracebacks;
  bare `import torchfits` remains light.
- `torchfits.cpp` exposes an explicit function-level compatibility inventory
  instead of wildcard-exporting the complete extension module.
- `TableHDU` no longer changes type when the optional `torch_frame` package is
  installed. Tensor/list mappings are internal, Arrow is the interchange
  boundary, and Polars is the supported dataframe surface.
- Mypy checks untyped bodies, runs in preflight, and exposed/fixed the
  `TableHDURef` tuple/list boundary mismatch.
- Existing-file overwrite and HDU mutation complete in a sibling temporary
  file and use atomic replacement; failed writes preserve the original bytes.
- Native image materialization and header-write errors now propagate instead
  of silently producing empty HDUs, and table metadata buffers use standard
  RAII containers.
- Opened `HDUList` summaries correctly parse numeric string header values and
  no longer fail in `repr`.
- Multi-worker tests use the required macOS spawn guard and retain subprocess
  stderr on timeout.
- Release-wheel tests now execute actual image and table round-trips, including
  `TableHDU.from_fits`, instead of checking import alone.
- macOS arm64 deployment target is 11.0 and platform documentation matches the
  actual wheel matrix.
- Vendored CFITSIO is still statically linked, but headers, the static archive,
  and build metadata are excluded from wheels. The local compressed wheel fell
  from about 2.4 MB to 0.8 MB and retained native round-trip parity.

## Evidence

| Gate | Result |
|---|---|
| Public API inventory | 50 unique root exports; all documented or recognized namespaces. |
| Focused API/HDU/release smoke | 46 passed. |
| Multi-worker DataLoader matrix | 5 passed in 56.24 s; default macOS spawn path covered. |
| Clean installed wheel smoke | 3 passed; imports from the wheel outside the source tree. |
| `pixi run preflight-push` | Ruff, formatting, blocking mypy, and compileall pass. |
| Version triplet | `pyproject.toml`, `pixi.toml`, and `torchfits.__version__` are 0.8.0. |
| Full local release gate | 463 passed, 3 skipped in 78.44 s with PyTorch shared-memory process permission. |
| Native/table/write regression set | 112 passed, 1 skipped after native error-propagation and RAII changes. |

## High-impact defaults

- `read(..., device="cpu", mmap="auto", scale_on_device=True)`.
- `read_tensor(..., device="cpu", mmap=True, raw_scale=False)`.
- `table.read(..., mmap=True, backend="auto", decode_bytes=True,
  apply_fits_nulls=True)`.
- `table.scan(..., batch_size=65536, mmap=True, backend="auto")`.

Historical metrics and benchmark rows are unchanged by this repair pass.
