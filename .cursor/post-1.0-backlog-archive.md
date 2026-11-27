# Post-1.0 backlog: archive

Entries moved out of [`post-1.0-backlog.md`](post-1.0-backlog.md) because they
are fixed, moot, or record state that no longer holds. Kept for history; do not
work from this file.

## Archived 2026-11-27 (1.4 correctness wave)

### 1.2 audit deferrals, verified fixed or moot

- r1b-08 — `docs/api-core-io.md` shows `read_extname(path, hdu=1)`; `io.read_extname` defaults `hdu=0`.
- r1c-11 — `docs/api-tables.md` predicate helpers omit `where_identifier_re` (it is in `where.__all__`).
- r3a-16 — `FitsSpectrumDataset` accepts `mmap=` and does not pass it to `read_torch`.
- r4a-14 — `image_meta._cache_get` / `_cache_set` and `subset.read_subset` have no callers.
- r5c-12 — `table.write_csv` / `table.write_ipc` are in `table.__all__` and have no docs section.
- r7c-15 — `arith -o <same path>` is allowed. The read finishes before the write, so it is in-place rather than corrupt.
- r7c-17 — `compress` then `decompress` of a single-image file yields an empty primary plus an image extension.
- r7c-21 — a header dict whose commentary value is a list (`{"HISTORY": ["h1", "h2"]}`) hits `std::bad_cast` in card replay.
- r11a-07 — `tests/test_bench_suites.py` pins source text with `inspect.getsource`.
- r11a-08 — `TestCaching.test_cache_clearing` calls `get_cache_performance()` and asserts nothing.
- r11a-09 — `test_clear_cache_disk_true_parameter` calls `clear_cache(disk=False)`.
- r11a-10 — `tests/test_api.py` repeats the `mode="invalid"` block.
- Image BSCALE/BZERO accumulation is float32 for 1.2 (`read_full_scaled_cpu`). The float64 image path is 2.0.
  — Stale: scaled BITPIX=32/64 images are read as float64 since 1.3 (`docs/compatibility.md`).

### Stale notes

- Install lane note (was under "Hygiene / structure"). Stale since 1.3.3: there
  are no per-PyTorch-minor lanes any more (no libtorch, DLPack, stable ABI;
  `scripts/torch_support.json` holds a floor with no upper bound). Original text:

  > - ~~Install: consider a **2.11+ / 2.13** wheel ABI lane only after scorecard
  >   re-soak~~ — **landed.** The lane is **PyTorch 2.13.x**: `constraints-wheel.txt`,
  >   `pyproject.toml` (runtime *and* `build-system.requires`), `pixi.toml`, the conda
  >   recipe, and `src/torchfits/__init__.py` all pin `>=2.13,<2.14`, and
  >   `pixi run check-lane` enforces it. This line previously read "today: wheels +
  >   pixi stay on PyTorch 2.10; source builds allow ≥2.10" — both halves were stale,
  >   and the source-build half is now *stricter* than ≥2.10: an isolated build
  >   resolves the lane pin, so a plain `pip install .` can no longer compile against
  >   a different torch minor than the one the extension's ABI stamp expects.

### Shipped in thin-I/O wave (Pass-3 + skinny meta)

- Skinny `read_*` metadata set + caller wiring (Datasets, image_meta, examples)
- `open_table_reader`, `table.read_torch(where=)`, thin table dispatch
- Pass-3 P1/P2 HTTP cutout byteswap/clone; P6 SigmaClip scalar fill; 8.1 end_row hoist
