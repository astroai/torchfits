# Release API freeze review — 0.9.2

**Date:** 2026-07-16  
**Package under review:** 0.9.1 → **0.9.2** quality freeze  
**Verdict:** **Ship with notes** after the lean cuts below land

## Summary

0.9.2 is a quality freeze on the Python library surface before the 0.9.3 CLI
and the 1.0 transforms review. Performance work is largely in place; this
release trims confusing dual surfaces, aligns docs with code, refreshes
multi-host benchmarks, and swaps the site logo. It does **not** tag 1.0.

Inventory (pre-cut): root `__all__` reports 50 symbols from the inventory
script (unique names; namespaces counted separately in docs as 53 with
namespaces). Root re-exports 16 of 33 transforms. `read_fast` remains public.
`read_image` is documented as a deprecated alias but is not in root `__all__`.

## Lean cuts (settled for 0.9.2)

| Issue | Action |
|---|---|
| `read_image` docs vs code | Document as **removed**; do not re-export |
| `read_fast` | Remove from root `__all__` and `io.__all__`; keep module-private impl if needed for tests |
| Root transform re-exports (16) | Remove from root; use `torchfits.transforms` only |
| `table.can_use_*`, `column_name_index_map` | Remove from `table.__all__` (remain importable from private modules if tests need them) |
| `torchfits.hdu` | Add `hdu` to root `_NAMESPACES` |
| Dual table APIs | Keep; Quick Paths lead image→`read_tensor`, Arrow→`table.read`, tensor dict→`read_table` |

## Persona notes

| Persona | Missing | Unnecessary | Non-intuitive |
|---|---|---|---|
| Astro student | Clear “open file → look at HDUs” path | Root transform names; `cpp` | Dual `read_table` vs `table.read` |
| CS student | Minimal `read_tensor(path, device=)` | Cache env knobs at root | `read_fast` vs `read` |
| Astro professional | Checksums, MEF mutation, compression parity | Root spectral transforms | Expects Astropy-shaped names |
| CFITSIO developer | Explicit mmap/compress limits | Full C API claims | `cpp` shim vs `_C` |
| PyTorch ML engineer | `data.make_loader`, device kwargs | Legacy dataset names (already gone) | GPU write = host stage |
| ML-on-FITS | `read_subset` + `FitsCutoutDataset` | Half of transforms at root | Which transform import path |
| Frontier model engineer | Honest deficit tables, batch/subset | Marketing floor claims | MPS deficits vs Linux 0 |

Bias: cut root noise; keep namespaces.

## Blocking (must fix before 0.9.2 tag)

1. Apply lean cuts above and update `tests/test_public_boundary.py` /
   `tests/test_docs_integrity.py`.
2. Remove `read_image` from deprecated-but-supported docs.
3. Sync README performance run IDs with `docs/benchmarks.md` after benches.
4. Point Zensical logo/favicon at `docs/torchfits-logo.png`.

## Should-fix

- Roadmap: mark CUDA snapshot refresh done relative to current scorecard policy.
- Clarify Linux strict-gate 0 deficits vs MPS deficits in README.

## Defer

- CLI and archive probe → **0.9.3**
- Profound transforms catalog review → **1.0**
- Device-direct GPU write (GPUDirect) — host staging remains
- Scratch/network replay expansion if strict bench gate already met

## Defaults (unchanged)

- `read(..., device="cpu", mmap="auto", mode="auto")`
- `read_tensor(..., device="cpu", mmap=True, raw_scale=False)`
- Non-CPU write inputs copy to host before CFITSIO

## Evidence checklist

| Criterion | Target |
|---|---|
| No undocumented required root exports | After lean cuts |
| Parity Supported rows test-backed | Keep |
| `pixi run release-gate` | Green |
| Examples use `torchfits.transforms` (already) | Green |
| Version triplet → 0.9.2 | At tag time |
| No WCS/sphere/HEALPix ownership claims | Integrity tests |

## Freeze rule for 0.9.2

After cuts land: bugfixes and doc corrections only until tag. New public
symbols wait for 0.9.3 (CLI entry point) or 1.0 (transforms settlement).
