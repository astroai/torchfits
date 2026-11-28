# Post-1.0 backlog: archive

Entries moved out of [`post-1.0-backlog.md`](post-1.0-backlog.md) because they
are fixed, moot, or record state that no longer holds. Kept for history; do not
work from this file.

## Archived 2026-11-28 (1.4 correctness wave + hygiene pass)

Everything below was open in `post-1.0-backlog.md` before this reconciliation.
Hashes are on the integrated branch (`fix/assessment-wave`, then `main`).

### Scope cuts

- Merge dual cache subsystems (`cache.py` vs `_io_engine/caches.py`) —
  decided: they are layers, not duplicates (`torchfits.cache` is the
  torch-free public façade owning the disk roots; the engine holds all
  in-process state). Relationship documented in `docs/architecture.md`
  ("Caching") and the one real duplication (a double native
  `SharedReadMeta` clear in `clear_cache`) removed: `a67bc76`.
- Collapse Dataset class zoo — decided: keep every public class. Image/cube
  datasets were already thin presets of `FitsTensor*`; the copied iterable
  `__iter__` is one `_EpochStream` mixin and `docs/api-data.md` names the
  recommended entry points: `5e32f8c`. Worker split already one helper
  (`_shard_work_plan`, `4820002`).
- CLI trim (`compress` / `decompress` / `arith` vs fpack/numpy) — decided:
  keep (rationale under "Decisions" in the backlog). Shipped meanwhile:
  `-j`/`-J`, imarith-class `arith`, batch copy/transform/cutout, stats
  std/median, compress `--algorithm`, header wildcards, setkey
  `--delete`/`@list`; fpack tile/quantize/dither/`-i2f` (`93310b0`,
  `b8037df`, `5d42ddb`, `43e4160`, `0de21c9`).
- Scorecard / CANFAR re-soak after thin-I/O — done (Round-3 MPS/CPU/CUDA
  `exhaustive_*_20260719_14*`, `docs/benchmarks.md`).
- MegaCam `torchfits_cached` vs `fitsio_cached` — moot: real OK-row medians
  were 49.3 vs 52.7 MB/s, a fresh lab soak (2026-07-30) led ~8%, no
  handle-reuse bug; revisit only with a new same-host repro.
- predicate_filter Round-3 (fused `where=` lagged Astropy on dense `col > 0`)
  — fixed by project + torch-mask, then the native windowed filter
  `b4c2ee3`; dataset reads `81dd472`, `1510d1f`.

### Performance history (folded into the one open re-benchmark item)

- Narrow-table buffered full-read residual (1.06-1.25x vs fitsio, several
  2026-08-22 notes) and buffered projection (`read_columns_buffered` read
  whole rows to de-interleave one column; predicate_filter 26-32% behind
  astropy-numpy) — `a645818` tiled parallel pread: narrow 1-col 13.4 -> 2.7 ms
  (mmap 3.8), 2M x 8 full 216 -> 9.7 ms, where= mmap-off 228 -> 18 ms.
  fitsio not re-run (not in env): tracked as the open re-benchmark.
- Single-pass arena decode (API-visible, "1.2 design") — not doing: after
  `a645818` the native full read is memory-bandwidth bound and Arrow
  conversion dominates.
- Per-column CFITSIO fan-out (3-8x worse) — reverted, history only.
- hcompress residual (~1.02-1.03x) — build-variance floor, not algorithmic.

### 1.1 audit deferrals

- C++ robustness (GIL in `read_full_numpy`/`get_header`/`get_num_hdus`, mmap
  truncation bounds, strided DLPack in `update_rows_mmap`, repeat int32 /
  duplicate-TTYPE UB, unsigned `read_full_numpy`) — fixed (R8/R9).
- `schema()` complex columns, unnamed-column capability checks, empty-result
  unknown columns — already fixed (verified in wave 3).
- `TableHDURef.head` / `TableHDU.head(-n)` — fixed (R6).
- Width-1 `FixedSizeList` off the main decode path — `b0fc10f`.
- `to_astropy` TNULL object dtype / TUNIT unit; `TableHDURef.to_arrow`
  kwarg collision — already fixed (verified, astropy 8.0.3).
- `table.write(quantize=)` silent no-op — raises `QuantizeError` (R5).
- Mutation error types (KeyError vs ValueError) — `UnknownColumnError`
  `13a07ce`; "already exists" unified as `ColumnExistsError` `40a3fd0`.
- Broad `except Exception` fallbacks in `table_api.py` / `_read_scan.py` —
  gone (`1825f3f`, `4fec0cc`); library-wide audit `ce29126`.
- Remote: multiprocess `.partial` race, If-Range/ETag resume,
  Content-Length-less truncation, staged-cutout cleanup races, make_loader
  double download — `6c6251b`, `df334e4`, `8d63214` (double download not
  reproduced; pinned by tests).
- CLI/http: exit codes, JSON `null`, same-path copy, Ctrl-C, per-file `-J`
  transforms, DNS-rebinding docstring — fixed (R7, R10/R15).
- Hygiene/tests: suites writing into the process cwd — verified clean in
  the hygiene pass: the full suite run from an empty temporary cwd created
  no file there, and every relative write sits under
  `monkeypatch.chdir(tmp_path)`. Examples write to the gitignored
  `examples/output/` by design.
- Dead/duplicated code sweep — `read_scaled_cpu_fast` gone (`b1e837f`),
  `clear_file_cache(handles=)` removed in 1.3.0 (`b1e837f`, documented in
  `docs/api-core-io.md`), `_normalize_cpp_chunk` gone (`857d209`), unused
  header_parser regexes (`ff60be3`; the one left is used), clip.py inf
  sentinel removed (`eb1691d`; the remaining `isfinite` guard catches the
  NaN of all-masked groups), worker split one helper (`4820002`),
  `_normalize_row_slice` one copy (r6b-02), fallback-table double-open
  (r5a-11), negative meta lookups cached (`fdfbfd6`), HTTP cutout header
  walk once (`2ae2698`), table device move one helper with batched VLA rows
  (`e33540b`).

### Round 7 / Pass-3 deferrals

- R7-HDU1, R7-HDU2 — fixed. R7-CPP1 — #235. R7-CPP2 and NIT-CPP-2 —
  `729c4ee`. R7-CPP3 — obsolete (`cache.cpp` removed, `fcd8d54`). R7-MUT4,
  B1 — already fixed (verified).
- P3 batch VRAM — `81a4e4c`. P5/P8 — already `shared_mutex` (~100 ns; not
  worth more). P10 — verified locked. P11 — `5b7b7bb`. P12 table device move
  — no correctness issue; duplication and per-row VLA transfers fixed in
  `e33540b`.

### Hygiene / structure

- Split `_table/read.py` mega-function strategies — files split in
  `ba0022d`; the native read strategies are named functions in
  `_table/_read_strategies.py`: `1edae80`.
- Split `_io_engine/write_api.py` / `_table/mutation.py` coerce vs ops —
  files split in `ba0022d`; `torchfits.write` and `update_rows` coercion
  separated from the writes (`_io_engine/_write_coerce.py`,
  `_mutation_coerce._normalize_update_rows`): `b45ed19`.
- Broader `except Exception: pass` audit (incl. Round-2 glm notes: batch
  `read_images_batch` fallthrough, NAXIS2 -> 0, tnull fill, `update_rows`
  mmap=auto) — `d4178b1`, r4c-14 (verified), `ce29126` (all 82 sites
  narrowed, logged, or justified).
- `torchfits verify` said "OK (no checksum keywords)" for a data-truncated
  file — now exit 3 naming the truncated HDU: `42df7d9`.

### 1.2 audit deferrals (all closed)

- r1b-09 `8bdc70a` (verified). r1c-13 `ecf5314` (breaking: strict decode,
  `decode_errors=` opt-in), interop `b90f394`. r2a-12 `cd7ad28`. r2a-16
  `f183865`. r2a-17 / r2b-12 `9f24964`. r3a-17 `d4178b1`. r3b-09 `1825f3f`.
  r4a-12 `fdfbfd6`. r4c-14 already fixed (verified). r5a-13 backend wording
  matches `read_policy.py` since `b4c2ee3`. r5a-14 `d63907f`. r5a-16
  `58de60f`. r5b-06 `d2b535c`. r5b-08 `4fec0cc`. r5c-13 `b0fc10f`. r6a-06
  `1362b00` (verified). r7a-08 `ddd3fe1` (+ verify `42df7d9`). r7a-09
  `fcacb41`. r7b-12 `de36823`. r7b-13 and r7c-16 `290eb0f`. r7c-18
  `9749377`. r7c-19 `d73dbdb`. r7c-23 `2063b3d`. r7c-25 `004a146`. r8b-08
  `516a500`, `6e24ddc`, `8da20d1`. r8b-09 `cbdb7a1`. r9b-06 `a034adf`.
- r12a-05 (MB/s of compressed images, GPU `read_full` throughput) — not in
  this repo: throughput reporting lives in torchfits-harness.
- "Arena decode and the `table_reader.h` / `write_api.py` splits stay out of
  1.2" — arena: not doing (see above); `write_api.py`: `b45ed19`;
  `table_reader.h`: still open in the backlog.
- `tests/test_remote_resume.py` Barrier + sleep note — not a race (moot).
- Struck before this pass: r8b-10 (#259), r1c-12, r1c-14, r1b-10, r4a-15,
  r4b-13, r5c-09, r5a-12 / r5b-07 / r5c-15, r5b-05, r6a-07, r7c-24.

### Extra defects found and fixed in the 1.4 wave

- CFITSIO first-use global-lock deadlock `fd67be4`; reads after file
  replacement `106d48c` / `8da20d1`; NaN pixels in lossy float compression
  `5d42ddb`; float literals not pushed down and rL logical vector gather
  `b4c2ee3`; table preview on complex / non-ASCII `de36823`.

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
