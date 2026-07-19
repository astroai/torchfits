# Post-1.0 backlog (from deep_review rounds)

Deferred after the 1.0 triage passes. Do not block the 1.0 tag on these.

## Shipped in thin-I/O wave (Pass-3 + skinny meta)

- Skinny `read_*` metadata set + caller wiring (Datasets, image_meta, examples)
- `open_table_reader`, `table.read_torch(where=)`, thin table dispatch
- Pass-3 P1/P2 HTTP cutout byteswap/clone; P6 SigmaClip scalar fill; 8.1 end_row hoist
- Review archive: `archive/pre-1.0-reviews/deep_review_1.0-r3.md`

## Scope cuts (later)

- Merge dual cache subsystems (`cache.py` vs `_io_engine/caches.py`) — document relationship first
- Collapse Dataset class zoo (`FitsImage*` / `FitsCube*` / …) into fewer constructors
- CLI trim (`compress` / `decompress` / `arith` vs fpack/numpy)
- Scorecard / CANFAR re-soak after thin-I/O — **done** (Round-3: MPS/CPU/CUDA
  `exhaustive_*_20260719_14*`; see `docs/benchmarks.md`)
- MegaCam `torchfits_cached` median lags `fitsio_cached` on Round-3 local soak
  (materialize path still leads) — investigate handle reuse vs fitsio cache
- Narrow-table `read_full` ~1.06–1.15× behind fitsio (small; polish later)
- **predicate_filter Round-3 investigation (done):** fused `where=` engages.
  Significant deficits were `specialized` only (tensor vs astropy **numpy**);
  `smart` already won. Dense `col > 0` (~50% keep) stresses gather; selective
  tails favor it. Bench now has both ops: `predicate_filter` (dense) and
  `predicate_filter_selective`. Optional later: C++ high-keep-rate fast path.

## Round 7 deferrals (safe post-1.0)

- R7-HDU1 — `TensorHDU.to_tensor()` closed-handle guard
- R7-HDU2 — `TableDataAccessor` squeeze on `(N,1)` (intentional FITS scalar shape)
- R7-CPP1 — floating-point equality for unsigned TZERO (malformed files only)
- R7-CPP2 — thread-local HDU cache stale after shared-meta invalidation
- R7-CPP3 — `cache.cpp:clear()` retain borrowed handles (ponytail)
- R7-MUT4 — `_normalize_mutation_rows` preprocesses all columns
- B1 — duplicate mmap/torch capability-check helpers in `_table/read.py`

## Pass-3 deferrals (low ROI)

- P3 batch `stack().to().unbind()` VRAM
- P5/P8 SharedReadMeta mutex coalescing / `shared_mutex`
- P10 OrderedDict cache locks
- P11/P12 GPU fallback cache / table device move
- NIT-CPP-2 tl_cache LRU (overlaps R7-CPP2)

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
