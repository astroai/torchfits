# Thermo-nuclear code quality — 0.9.2 (public façade + hot paths)

**Date:** 2026-07-16  
**Scope:** Lean API cuts (`__init__` / `io` / `table` / docs) plus the I/O and table hot paths they expose  
**Rubric:** thermo-nuclear maintainability (1k-line rule, spaghetti dispatch, thin abstractions, code judo)  
**Verdict:** **Ship with notes** — lean cuts are a net delete and do not regress structure; hot-path debt is pre-existing and mostly deferred. One incomplete cut (`read_fast` / `_fastio`) should be finished before the tag.

---

## 1. Verdict

The 0.9.2 façade cuts are the right kind of change: **delete concepts, do not rearrange them**. Root `__init__` lost 16 transform re-exports and `read_fast`; `table.__all__` lost probe helpers; `hdu` joined `_NAMESPACES`. That improves the mental model without touching dispatch.

What remains is not introduced by this diff, but it **fails the thermo-nuclear bar** if “quality freeze” means the hot path is tidy:

| Area | Lines | Status |
|---|---:|---|
| `transforms.py` | 3184 | Over 1k — already deferred to **1.0** |
| `_table/read.py` | 1593 | Over 1k — **0.9.3+** |
| `_table/mutation.py` | 1223 | Over 1k — **0.9.3+** |
| `_io_engine/write_api.py` | 1005 | At the 1k cliff — **0.9.3** |
| `_io_engine/_read_pipeline.py` | 765 + 365 fallback | Under 1k but strategy-ladder spaghetti |
| `io.py` | 535 | Façade of near-identity `_impl` wrappers + orphaned `read_fast` |

**Approval bar vs this PR:** no structural regression from the lean cuts.  
**Approval bar vs “0.9.2 quality freeze”:** finish the `read_fast` demotion (delete the dead dual path). Do not open transforms / table-read rewrites before tag.

---

## 2. High-signal fixes worth doing NOW for 0.9.2

Concrete, small, public-API-preserving (or aligning with already-documented removals). Prefer documenting over large edits; these are the ones that earn a pre-tag pass.

### 2.1 Delete orphaned `read_fast` + `_fastio` (code judo)

**Problem:** Docs and freeze already say `read_fast` is removed. It was dropped from root/`io.__all__`, but:

- `io.read_fast` still exists as a callable (`src/torchfits/io.py` ~L77–104)
- `src/torchfits/_fastio.py` (111 lines) is **only** imported by that wrapper
- No test imports `read_fast` / `_fastio` (boundary test only asserts absence from `__all__`)

This is an incomplete cut: a second image fast path parallel to `_read_pipeline` / `read_tensor`, still importable as `from torchfits.io import read_fast`.

**Change (small):**

1. Delete `read_fast` from `io.py`.
2. Delete `src/torchfits/_fastio.py` (or fold any unique C++ entry into `_read_pipeline` / `image.py` only if something still needs it — today nothing does).
3. Update `tests/test_docs_integrity.py` wording that still treats `read_fast` as a live owner of `scale_on_device`.
4. Optionally harden boundary test: `assert not hasattr(io, "read_fast")` (or `pytest.raises` on import).

**Why NOW:** matches documented removal; deletes a whole dual path; zero public-API breakage relative to the freeze doc.

### 2.2 Stop identity wrappers that only exist to rename `_impl`

**Problem:** `io.py` is mostly:

```text
def foo(...): return _foo_impl(...)
```

Worse: private one-liners like `_invalidate_path_caches`, `_cpp_module`, `_write_header_cards_if_supported`, `_normalize_cpp_table_data` add a hop without binding env state.

**Change (small, optional pre-tag):**

- Inline `_cpp_module()` → pass `cpp` / `torchfits._C` directly where `read()` builds kwargs.
- Delete `_invalidate_path_caches` and pass `_invalidate_path_caches_impl` (or the cache helper) straight into `_read_check_cache`.
- Keep **public** thin wrappers (`write_tensor`, `read_table_rows`) — those are API surface, not indirection tax.

Do **not** collapse the whole `io` → `_io_engine` façade in 0.9.2; that is a larger boundary move.

### 2.3 One predicate helper in `read_unified` (anti-spaghetti, tiny)

**Problem:** The full-image fast-path guard is copy-pasted three times in `_read_pipeline.read_unified`:

- batch-HDU branch
- CPU fast path
- generic fast path  

Pattern: `columns is None and start_row == 1 and num_rows == -1` (+ variants with `return_header` / `isinstance(hdu, int)`).

**Change (small):**

```python
def _is_full_single_image_read(hdu, columns, start_row, num_rows, return_header) -> bool:
    ...
```

Use it in the three sites. No behavior change. Makes the next strategy edit safer.

### 2.4 Remove production `unittest.mock` sniff

**Problem:** `read_unified` imports `unittest.mock` and skips the CPU fast path when `cpp_module.read_full` is a `Mock` (`_read_pipeline.py` ~L338–340). Test concern leaked into the hot dispatcher.

**Change (small):** Prefer a test double that lacks `read_full` / raises, or a private `_TORCH FITS_FORCE_FALLBACK` env already in the cold-path spirit — not `isinstance(..., Mock)`.

### 2.5 Boundary test: assert `hdu` namespace + demotions (mostly done)

Already landed in `tests/test_public_boundary.py`. **NOW polish:** also assert `"read_fast" not in io.__all__` and that transform names are absent from `dir(torchfits)` after import (lazy `__getattr__` must not resurrect them). Skip if already covered by docs integrity.

---

## 3. Defer to 0.9.3 / 1.0

Do not start these before the 0.9.2 tag. Freeze rule: bugfix + doc sync only after lean cuts.

### 0.9.3 (CLI era — structure, not product flash)

| Item | File(s) | Code judo |
|---|---|---|
| Split `write_api.py` at the 1k cliff | `_io_engine/write_api.py` (1005) | Extract compress / MEF / table-dict branches from `write()` isinstance ladder into `write_image.py` / `write_table_dict.py` |
| Split table mutation | `_table/mutation.py` (1223) | Coercion helpers (`_coerce_*`) → `_table/mutation_coerce.py`; keep public mutators thin |
| Split Arrow table read | `_table/read.py` (1593) | where-pushdown / scan / schema / chunk C++ into focused modules; keep `table.read` signature |
| Collapse DI into `read_unified` | `_read_pipeline.py` + `io.py` | Today `read()` injects ~10 callables for cache/env binding. Prefer module-level policy object or closed-over config once; delete the callback parade |
| Break circular import | `_read_pipeline.py` ↔ `_read_pipeline_fallback.py` | Fallback imports helpers from pipeline; pipeline lazy-imports fallback. Move shared scale/unsigned helpers to `_scale.py` / `_unsigned_image.py` |
| Dual image entry names | `image.read_image` vs public `read_tensor` | Rename private `read_image` → `read_image_tensor` (or similar) so “removed `read_image`” docs stay true |
| `read_table_rows` | `io.py` | Keep for compatibility; later decide fold into `read_table(..., num_rows=)` only docs, or keep as explicit range sugar |

### 1.0 (transforms settlement — already on freeze roadmap)

| Item | File(s) | Notes |
|---|---|---|
| Decompose `transforms.py` | 3184 lines, 131 `if`/`elif` | Catalog review + split by domain (stretch / continuum / spectral / clip). **Do not** re-export at root |
| Dual table APIs | `read_table` (tensor dict) vs `table.read` (Arrow) | Intentional for 0.9.x; settle naming/docs story at 1.0, not by deleting one |
| Profound continuum / wavelet / ALS | same file | Product review, not a drive-by split |

---

## 4. Findings by rubric priority

### Structural

1. **Incomplete `read_fast` demotion** — dual fast path still on disk (`io.read_fast` + `_fastio`). Highest-conviction pre-tag fix.
2. **Four modules ≥ ~1k lines** on the maintenance surface users hit via root/table/transforms. Cuts did not enlarge them; freeze must not pretend they are healthy.

### Missed code judo (behavior-preserving deletes)

3. **Delete `_fastio` entirely** once `read_fast` is gone — whole module, not a rename.
4. **`read_unified` strategy ladder** — three overlapping “is this a full image read?” predicates; one helper deletes repetition without a new abstraction layer.
5. **Callback-injection façade** — `io.read` → `read_unified(..., autodetect_hdu=..., batch_to_device=..., ...)` is testability theater that obscures the real graph. Collapse later; do not add more injectables in 0.9.2.

### Spaghetti / branching

6. **`write()` isinstance / compress ladder** in `write_api.py` — classic ad-hoc growth; file is already past 1k. Next write feature must split first.
7. **Fallback HDU-name resolution** — nested try/except loops in `_read_pipeline_fallback.py`; acceptable for rare path, but keep new logic out of it.
8. **Production Mock sniff** — special-case in the busiest dispatcher.

### Boundary / abstraction

9. **`io.py` as rename layer** — public functions earning their keep (docstrings, kwargs shaping); private `_impl` aliases often do not.
10. **`table.can_use_*` demotion** — correct; helpers remain private in `_table/read.py` where they belong. Do not re-export.
11. **Root still re-exports HDU types and interop** — fine; `hdu` namespace addition removes the “where do types live?” ambiguity.

### File size

12. `transforms.py` 3184 — blocker for any transforms change until split (1.0).
13. `_table/read.py` 1593 / `mutation.py` 1223 / `write_api.py` 1005 — blocker for feature PRs that would grow them further.

### Legibility

14. Version triplet still `0.9.1` in `__init__` / `pyproject.toml` — tag-time only, not structure.
15. `test_docs_integrity` still narrates `read_fast` as the home of `scale_on_device` — sync when deleting.

---

## 5. What this PR got right (do not undo)

- Root stops loading transform classes via `__getattr__` maps — import stays light.
- `read_fast` out of `__all__` is the correct direction (finish by deleting the body).
- `table.__all__` no longer advertises path probes — public surface matches “Arrow I/O helpers.”
- `hdu` in `_NAMESPACES` matches how types already live.
- Docs “Removed names” table replaces “Deprecated but still supported” — honest.

---

## 6. Explicit non-goals for this review

- No implementation of §2–§3 except trivial one-liners if an agent is already mid-edit.
- No CLI / archive work (0.9.3).
- No transforms catalog rewrite (1.0).
- No performance claims; this is structure only.

---

## 7. Suggested pre-tag checklist

- [ ] `read_fast` / `_fastio` deleted or proven still required by an in-tree caller
- [ ] Boundary + docs integrity green under pixi
- [ ] No new public root symbols
- [ ] No net line growth in files already ≥ 1k
- [ ] Version bump to 0.9.2 at tag time only
