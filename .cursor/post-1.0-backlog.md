# Post-1.0 backlog (from deep_review rounds)

Deferred after the 1.0 triage passes. Do not block the 1.0 tag on these.

## Scope cuts (later)

- Merge dual cache subsystems (`cache.py` vs `_io_engine/caches.py`) — document relationship first
- Collapse Dataset class zoo (`FitsImage*` / `FitsCube*` / …) into fewer constructors
- CLI trim (`compress` / `decompress` / `arith` vs fpack/numpy)

## Round 7 deferrals (safe post-1.0)

- R7-HDU1 — `TensorHDU.to_tensor()` closed-handle guard
- R7-HDU2 — `TableDataAccessor` squeeze on `(N,1)` (intentional FITS scalar shape)
- R7-CPP1 — floating-point equality for unsigned TZERO (malformed files only)
- R7-CPP2 — thread-local HDU cache stale after shared-meta invalidation
- R7-CPP3 — `cache.cpp:clear()` retain borrowed handles (ponytail)
- R7-MUT4 — `_normalize_mutation_rows` preprocesses all columns
- B1 — duplicate mmap/torch capability-check helpers in `_table/read.py`
- B3 — Python byte-swap in `http_subset` (network-bound)

## Hygiene / structure

- Header HISTORY/`remove` O(N²) for huge HISTORY lists
- Split `_table/read.py` mega-function strategies
- Vestigial `UnifiedCache` shared-handle path cleanup
- Broader `except Exception: pass` audit (soft fallthroughs in strategy probes;
  Round-2 glm notes: batch `read_images_batch` silent fallthrough, NAXIS2→0,
  tnull fill swallow, `update_rows` mmap=auto swallow)

## Spectroscopy / continuum (not in torchfits)

Continuum and spectral `FITSTransform`s were **deleted** from torchfits (no
deprecation). Absorb-vs-new design belongs in the sibling astronomy stack repo.
