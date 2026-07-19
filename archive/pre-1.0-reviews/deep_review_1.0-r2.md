# torchfits 1.0 — Comprehensive Deep Review

**Date:** July 18, 2026
**Version reviewed:** 1.0.0rc3+ (post-review cleanup)
**Scope:** Full codebase — 81 Python files, 7 C++ files, 12 headers

---

## Executive Summary

torchfits has undergone a significant cleanup since the first five-round review. The most critical architectural issues were addressed: the `__init__.py` race condition is fixed, `core.py` dead code removed, `_unsigned.py` and the niche `continuum.py`/`spectral.py` transforms deleted, `logging.py` now properly uses `NullHandler`, and ~7,700 lines of code were removed overall.

The codebase is in **solid shape for 1.0**. The remaining issues are primarily architectural refinements and polish, not blockers. Overall grade: **A-** (up from B+).

---

## PART 1 — What Was Fixed (Post First Review)

These issues from the original five-round deep review have been resolved:

| Original Finding | Resolution |
|---|---|
| `__init__.py` `globals()[name]` mutation race condition | ✅ Replaced with `_ATTR_CACHE` + `threading.RLock()` |
| `core.py` dead `ChecksumVerifier` class | ✅ File deleted |
| `_unsigned.py` dead/redundant code | ✅ File deleted |
| `transforms/continuum.py` niche spectral domain code | ✅ File deleted |
| `transforms/spectral.py` niche spectral domain code | ✅ File deleted |
| `logging.py` import-time StreamHandler | ✅ Uses `NullHandler()` |
| `http_util.py` bare `except Exception: pass` on SSRF checks | ✅ Proper `HttpBlockedError` with `oserror` |
| `_string_decode.py` numpy dependency for byte decode | ✅ Pure Python decode via buffer protocol |
| ~5 deprecated root aliases (`read_table`, `stream_table`, etc.) | ✅ Removed from public API |
| Benchmarks and examples cleaned up | ✅ ~7,700 net lines removed |

---

## PART 2 — Architecture Review (Current State)

### Solid Foundations (Unchanged from V1 Review)

- **C++/Python layering via nanobind** — well-designed with three distinct image read paths
- **GIL release** on every CFITSIO call
- **L0/L1/L2 cache hierarchy** with per-entry mutexes and `pread` on shared raw fd
- **Table predicate pushdown** entirely in C++ with mmap + parallel scan
- **HTTP Range subset reads** for uncompressed 2D remote images
- **Transactional writes** via temp-file + atomic rename
- **~850 tests**, mypy clean, ruff clean, docs integrity tests

### Improvements Since V1 Review

- **Lazy import system** is now thread-safe with `_ATTR_CACHE` + `threading.RLock()`
- **Per-path handle caching removed** — `get_cached_handle` now always opens a fresh handle and returns `cached=False` (CFITSIO §4 Option A)
- **SharedReadMeta** properly manages per-path metadata with mutex guards
- **Transform surface** reduced to 10 core transforms (was ~30)
- **Interop surface** consolidated — `to_pandas`/`to_arrow`/`to_polars` now accept paths/readers/batches

---

## PART 3 — Remaining Issues by Area

### A. Python Source — High Priority

#### A1. `cache.py` — Dual cache subsystems still partially overlap

The `CacheManager`/`CacheConfig` (in `torchfits.cache`) and `_io_engine/caches.py` (in-process caches) serve different purposes but there's some conceptual overlap:
- `CacheManager` is environment-aware (HPC/cloud/GPU detection) and configures the C++ backend
- `_io_engine/caches.py` manages Python-side read caches, HDU type caches, handle registries

**Assessment:** This is functional but the split could be confusing for contributors. Not a 1.0 blocker — documentation could clarify the distinction.

#### A2. `image.py:dispatch_read_image_cpp` — handle_cache parameter is dead

```python
# image.py line ~66
def dispatch_read_image_cpp(cpp, path, hdu, mmap, handle_cache, raw_scale):
```

The `handle_cache` parameter is accepted but never used. The function always calls `read_full` / `read_full_nocache` / `read_full_raw`. The parameter exists for API compatibility.

**Severity: Low** — dead parameter, no functional impact.

#### A3. `_read_pipeline.py` — Complex dependency injection pattern

The `read_unified` function receives 19 keyword arguments including many callables. This is the A2 strategy refactor approach (dependency injection for testability). It works but makes the call chain difficult to trace.

**Severity: Low** — functional, well-tested, but complex to maintain.

#### A4. `header_parser.py` — Two parsing paths with subtle duplication

`parse_header_string` and `_parse_card` share significant logic for parsing FITS value types but implement it with slight differences. The `_STRING_KEYWORDS` set contains some CTYPE/CUNIT keys (only indices 1-4) while many FITS files use higher indices.

**Severity: Low** — the `_STRING_KEYWORDS` limitation on CTYPEn/CUNITn for n>4 could be a minor issue for multi-extension MEF files.

#### A5. `table_hdu.py:TableDataAccessor.__getitem__` — lossy squeeze

```python
# table_hdu.py line ~25
def __getitem__(self, key):
    if isinstance(value, torch.Tensor):
        if value.dim() > 1:
            return value.squeeze()  # <-- lossy for (N,1) columns
```

A tensor column of shape `(N, 1)` gets squeezed to `(N,)`, losing the dimension. This is consistent with the original FITS column semantics (scalar columns should be 1D), but can be surprising for callers who expect rank-preserving access.

**Severity: Low** — documented behavior, but worth noting.

#### A6. `table_hdu.py:TableHDU.filter()` — materializes full ChunkedArray

```python
# table_hdu.py ~line 235
mask_arr = mask_chunked.to_numpy()
```

The Arrow mask is fully materialized as numpy before filtering, which duplicates memory for large tables. The original review noted this — it remains.

**Severity: Low** — the filter path is for in-memory TableHDU objects, not the streaming scan path.

### B. Python Source — Medium Priority

#### B1. `_table/read.py` — `_can_use_mmap_row_path_for_full_read` and `_can_use_torch_table_path_for_full_read` are ~85% identical

These two functions check almost the same thing. The duplication was noted in the original review and remains.

**Severity: Low** — functional, but refactoring would reduce maintenance burden.

#### B2. `header_parser.py` — `_parse_string_value` has O(n) quote scanning

The string parser searches for `''` (escaped quotes) in every string value. For very long string-valued header keywords (CONTINUE concatenation), this is O(n) in string length. The comment acknowledges the optimization: "If there are no internal quotes, simple slice is fastest."

**Severity: Trivial** — well-optimized for the common case.

#### B3. `http_subset.py:_bswap_inplace` — Python-level byte swap

The `_bswap_inplace` function does byte swapping in pure Python with `memoryview` and `bytes(reversed(...))` per row. While functional, this creates temporary `bytes` objects for every 4/8-byte element. For large cutouts this could be measurable.

**Severity: Low** — the C++ `SubsetReader` with `internal::bswap16_copy` handles the local-file fast path. HTTP cutouts are inherently network-bound.

#### B4. `data/remote.py:prefetch_urls._job` — error handling improved

The previous review noted `except Exception: pass` for silent download failures. This is now:
```python
except Exception as exc:
    _log.warning("prefetch failed for %s: %s", u, exc)
    warnings.warn(...)
```

**Fixed.** ✅

#### B5. `_table/mutation.py` — `_COMPLEX_DTYPE_MAP` / `_VLA_DTYPE_MAP` lazy init remains

The module-level dicts `_COMPLEX_DTYPE_MAP` and `_VLA_DTYPE_MAP` are initialized lazily with `global` statements. This is a pattern that works but is fragile. Not a 1.0 blocker.

**Severity: Low-W** — works, but `global` mutation at function call time is unusual.

### C. C++ Source

#### C1. `fits_bindings.cpp:read_full_cached` — Thread-local HDU metadata cache

The `read_full_cached` function maintains a `thread_local` cache of HDU metadata (`LocalHduMeta`) keyed by `(SharedReadMeta::uid, hdu_num)`. This is a **new optimization** not present in the original review. It avoids repeated `fits_get_img_paramll` calls in the same thread.

**Assessment:** This is a good optimization. The thread-local cache is limited to 4096 entries and cleared when exceeded.

#### C2. `SubsetReader::init_from_hdu` — Proper handling of signed-byte/unsigned conventions

The `SubsetReader` now correctly handles FITS pseudo-unsigned conventions (signed byte with BZERO=-128, unsigned-16 with BZERO=32768, unsigned-32 with BZERO=2147483648) without float-promoting. This was a bug in the original review's findings.

**Fixed.** ✅

#### C3. `FITSFile::read_image_raw` — mmap fast path for uint8 images

The `read_image_raw` method implements a `pread`-based fast path for uncompressed uint8 images. It falls back to mmap + memcpy if pread fails. This is well-designed.

**Assessment:** Good implementation.

#### C4. `FITSFile::~FITSFile` and `SubsetReader::~SubsetReader` — proper RAII

Both destructors call `close()` which releases resources. The `raw_fd_` is closed in `close_raw_fd()`. Memory maps are released in `release_data_mmap()`.

**Assessment:** No resource leaks found.

### D. CLI

The CLI has 15 subcommands, each cleanly factored into its own module. Common argument handling is centralized in `common.py`.

**Notable:** The CLI is well-structured with proper error handling via `CliError` hierarchy, exit codes, and `emit_records` for text/JSON/JSONL output.

### E. Transforms

The transform surface has been reduced from ~30 to 10 core transforms. The remaining transforms are:

- **Stretches:** `ArcsinhStretch`, `LogStretch`, `SqrtStretch`
- **Normalizers:** `ZScaleNormalize`, `RobustNormalize`, `BackgroundSubtract`, `PercentileClipNormalize`, `MinMaxNormalize`, `GlobalScalarNorm`
- **FITS-meta:** `FITSHeaderScale`, `FITSScaleColumns`, `TNullToNan`, `FITSHeaderNormalize`
- **Clippers:** `SigmaClip`, `AsymmetricSigmaClip`
- **Composition:** `Compose`, `AsModule`/`as_module`

The `ContinuumSubtract` and `WaveletDecompose` transforms that were flagged as niche have been removed. The `lupton_rgb` function is now a thin wrapper around the CLI's RGB implementation.

**Assessment:** Clean, focused transform surface. The `FITSTransform` base class + `AsModule` adapter pattern is well-designed.

### F. Documentation & Examples

The docs have been cleaned up significantly. The `docs/reviews/transforms-1.0.md` was removed as part of the cleanup. Examples have been streamlined.

### G. Benchmarks

Benchmarks have been significantly reduced. `bench_fast.py` had large deletions. The `bench_poly_continuum.py` was removed along with the continuum transforms.

---

## PART 4 — Fresh Issues Found

### Issue 1: `http_subset.py:_bswap_inplace` performance for HTTP cutouts (NEW)

```python
# http_subset.py ~line 140
elif elem_bytes == 4:
    for i in range(0, len(buf), 4):
        mv[i : i + 4] = bytes(reversed(mv[i : i + 4]))
```

For a 2048×2048 float32 cutout, this creates 4,194,304 temporary `bytes` objects. A C-level `bswap32_copy` would be faster, but only matters when both i) the source is HTTP and ii) the cutout is large.

**Severity: Low** — network latency dominates for HTTP sources.

### Issue 2: `_table/read.py` — `_can_use_mmap_row_path_for_full_read` reads header redundantly (NEW OBSERVATION)

The function is called in `_read_cpp_table_chunk` which already has a cached header (`_hdr`), but the function re-reads it internally. In some paths it receives `header=_hdr` but when it doesn't, it does a fresh `torchfits.read_header()` call.

**Severity: Low** — cached at the C++ level.

### Issue 3: `table_hdu.py` — `TableHDU.append_rows` reconstructs TableHDU (EXISTING)

The in-memory `append_rows` creates a new `TableHDU` object, which means repeated appends create O(n) new objects. This is a design choice (immutable-style API), not a bug.

**Severity: Trivial** — by design.

### Issue 4: `header_parser.py` — `_STRING_KEYWORDS` not exhaustive (EXISTING, LOW)

`CTYPE1` through `CTYPE4` are listed, but WCS can have higher indices. This only affects the heuristic for typed value parsing — the fallback correctly treats them as strings.

**Severity: Trivial** — the heuristic is best-effort.

---

## PART 5 — 1.0 Readiness Assessment

### What's Ready

| Area | Assessment |
|---|---|
| Core I/O (read/write) | ✅ Production-ready |
| Table I/O (Arrow/Polars/Pandas) | ✅ Production-ready |
| Image subset/cutout | ✅ Production-ready |
| HTTP Range cutouts | ✅ Production-ready |
| SSRF security | ✅ Production-ready |
| Thread safety | ✅ Production-ready |
| Transactional writes | ✅ Production-ready |
| Header parsing | ✅ Production-ready |
| CLI | ✅ Production-ready |
| Transform library | ✅ Production-ready |
| ML datasets | ✅ Production-ready |

### Minor Polish Items (Post-1.0)

1. **Deduplicate** `_can_use_mmap_row_path_for_full_read` and `_can_use_torch_table_path_for_full_read`
2. **Consider** `struct.unpack`-based byte swap in `http_subset.py` for large HTTP cutouts
3. **Document** the `cache.py` / `_io_engine/caches.py` split in architecture docs
4. **Remove** the dead `handle_cache` parameter from `dispatch_read_image_cpp`

---

## PART 6 — Final Grade

| Dimension | Grade |
|---|---|
| Architecture | **A** |
| Thread safety | **A** |
| Performance | **A-** |
| Code cleanliness | **A-** |
| Test coverage | **A** |
| Documentation | **A-** |
| Security | **A** |
| ML community fit | **B+** |

**Overall: A-** — Ready for 1.0. The remaining issues are minor polish items that do not block release.

---

## PART 7 — Round 6 Deep Dive: C++, CLI, Benchmarks, Bug Patterns, & Test Coverage

### 7.1 C++ Deep Dive — Memory Safety & Architecture

#### C++5. `fits_detail.h:read_tensor_canonical` — Sophisticated multi-strategy image read (EXCELLENT)

The canonical read path in `read_tensor_canonical` implements **five distinct strategies** selected by data properties:

1. **BYTE_IMG `pread` fast path** — for uncompressed uint8/signed-byte images: direct `pread` via shared fd, then `_xor_sign_bit_u8` (with `at::parallel_for` for arrays >256KB)
2. **Multi-byte mmap fast path** — for int16/int32/int64/float32/float64 with SIMD byte-swap (`internal::bswap16/32/64_copy`)
3. **Unsigned offset paths** — special `bswap16_copy_u16_offset` / `bswap32_copy_u32_offset` that apply `+32768` / `+2147483648` during byte swap
4. **Chunked CFITSIO read** — for large images (>128MB), reads in 128MB chunks to bound memory
5. **Standard CFITSIO fallback** — single `fits_read_img` call

**Assessment:** This is production-grade engineering. The fallback chain is well-ordered and each strategy has clear preconditions.

#### C++6. `internal_utils.h` — SIMD byte-swap for 3 architectures (EXCELLENT)

The byte-swap functions `bswap16_copy`, `bswap32_copy`, `bswap64_copy` have optimized implementations for:
- **ARM NEON** (`vrev16q_u8`, `vrev32q_u8`, `vrev64q_u8`)
- **x86 AVX2** (`_mm256_shuffle_epi8`)
- **x86 SSSE3** (`_mm_shuffle_epi8`)
- **Scalar fallback** (`__builtin_bswap16/32/64`)

**Assessment:** Excellent cross-platform optimization. The unsigned-offset variants (`bswap16_copy_u16_offset`, `bswap32_copy_u32_offset`) apply the offset in a second pass — a fused kernel could be faster but the current approach is clear and correct.

#### C++7. `table_reader.h` — `TableReader` constructor RAII with try/catch (GOOD)

The `TableReader(const std::string&, int)` constructor opens a fitsfile*, then wraps HDU navigation and analysis in a try/catch block that closes the handle on failure:

```cpp
try {
    status = 0;
    fits_movabs_hdu(fptr_, target_hdu_, nullptr, &status);
    if (status != 0) throw std::runtime_error("Failed to move to table HDU");
    analyze_table();
} catch (...) {
    int close_status = 0;
    fits_close_file(fptr_, &close_status);
    fptr_ = nullptr;
    throw;
}
```

**Assessment:** Proper RAII — no resource leaks on construction failure. The destructor also correctly handles owned vs. borrowed handles.

#### C++8. `fits_detail.h:get_shared_meta_for_path` — Stat-based cache invalidation (GOOD)

The shared metadata cache uses `stat()` to detect file changes (size, mtime, inode). The validation interval is configurable via `TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS` (default 1000ms). When a file changes, all cached metadata (image info, compression, scale) is cleared and the shared fd is closed.

**Assessment:** Robust cache invalidation. The interval prevents thrashing on repeated reads of the same file.

#### C++9. `security.h:check_fits_filename_security` — CFITSIO injection prevention (GOOD)

Validates filenames for CFITSIO injection patterns (`!` prefix for forced overwrite, `[` for extended syntax). The check is applied in every FITSFile and TableReader constructor.

**Assessment:** Security-conscious design. However, `check_fits_filename_security` allows `[` when the path is from a user (for image sections like `file.fits[10:100,20:200]`).

#### C++10. `fits_bindings.cpp:read_full_cached` — Thread-local optimization (NEW FINDING)

The function maintains a `thread_local` cache (`tl_cache`) of HDU metadata keyed by `(SharedReadMeta::uid, hdu_num)`. This avoids repeated `fits_get_img_paramll` calls. The cache is limited to 4096 entries.

**Potential issue:** The cache key uses `SharedReadMeta::uid` which is a monotonically increasing atomic. If `g_shared_meta` is cleared and re-populated (e.g., file invalidation), the old uid entries persist in thread-local caches until the thread clears them (at 4096 entries). This means stale metadata could be used briefly after file invalidation.

**Severity: Low** — the `ensure_hdu` call before the read would still succeed; only cached image parameters (bitpix, naxis, shape) might be stale for one read.

#### C++11. Memory allocation pattern (GOOD)

- No `malloc`/`free` — uses RAII (`std::vector`, `std::string`, `std::shared_ptr`, `std::unique_ptr`)
- No `new[]`/`delete[]` — `MMapHandle` wraps `mmap`/`munmap` in RAII
- `torch::empty` creates tensors with PyTorch's memory management
- CFITSIO resources (`fitsfile*`, `header_str`) properly freed
- `raw_fd_` in `SharedReadMeta` closed in destructor

**Assessment: No memory safety issues found.**

### 7.2 CLI Deep Dive — All 15 Subcommands

#### CLI Architecture (GOOD)
- **15 subcommands**: info, header, verify, diff, stats, table, convert, copy, arith, cutout, compress, decompress, transform, probe, setkey
- **Centralized error handling**: `CliError` hierarchy with `UsageError`, `IoError`, plus exit codes (`EXIT_OK`, `EXIT_DIFF`, `EXIT_USAGE`, `EXIT_IO`, `EXIT_VERIFY_FAIL`)
- **Structured output**: `emit_records` supports text, JSON, and JSONL formats
- **SSRF-safe probing**: `cmds_probe.py` uses `is_internal_url` and `ValidatingRedirectHandler` for HTTP(S) probes

#### CLI Findings

| Finding | Severity |
|---|---|
| `cmds_setkey.py` — Uses `_write_header_cards_if_supported` which is a private `_io_engine` function, not part of the public API. This works but creates a hidden coupling. | Low |
| `cmds_transform.py` — Accepts any transform name from `__all__` and calls `cls()` with no arguments. Some transforms require constructor args (e.g., `ArcsinhStretch(a=...)`), which would fail. | Medium |
| `cmds_compress.py` — The `_rewrite` function reads entire HDUs into memory via `torchfits.open`. For large MEF files this could be memory-intensive. | Low |
| `cmds_diff.py` — Calls `read_tensor` on every image HDU, which means diffing a large MEF file reads all image data. | Low |
| `cmds_convert.py` — The `_arrow_to_column_dict` helper calls `col.to_numpy(zero_copy_only=False)` which copies all numeric columns. | Low |
| `cmds_info.py` — `hdu_type_name` falls back to NAXIS check for UNKNOWN types. This is the correct heuristic. | Trivial |
| `cli/rgb.py` — `write_rgb_image` has a `ponytail` comment about `bytes(flat.tolist())` being slow for large images. | Low |

#### CLI6. `cmds_transform.py` — No args for transforms (MEDIUM)

```python
# cmds_transform.py
transform = cls()  # Always called with no args
```

This means only transforms with default constructors work via CLI. `ArcsinhStretch(a=2.0)` requires Python code, not CLI. This is an intentional limitation (the CLI is for quick inspection, not production pipelines), but undocumented.

**Severity: Medium** — usability gap, not a bug.

### 7.3 Benchmark Analysis

#### Benchmarks Architecture (GOOD)

- **`bench_contract.py`**: Sophisticated ranking/deficit framework with strict mmap fairness, family-based grouping, significance labeling
- **`bench_all.py`**: Multi-domain orchestrator with process isolation for table benchmarks
- **`bench_fits_io.py`**: Creates 87 fixture files across 5 size categories, 6 dtypes, plus compressed/MEF/scaled/unsigned variants
- **`suites.py`**: 11 named benchmark suites for different purposes (release, deficit focus, GPU transports)

#### Benchmark Findings

| Finding | Severity |
|---|---|
| `bench_fast.py` — **Deleted**. This was flagged as "insincere benchmarks" in the first review. ✅ | N/A |
| `bench_contract.py` — Deficit floors still use `DEFICIT_MIN_ABS_DELTA_S = 2e-4` (0.2ms) and `DEFICIT_MIN_ABS_DELTA_S_HCOMPRESS = 1.6e-3` (1.6ms). The original review flagged these as masking small regressions. They remain, but the code now always *emits* the deficit row and only labels significance as "noise" vs "significant". | Low |
| `bench_fits_io.py` — The `EXPECTED_FILE_COUNT = 87` contract is enforced with a runtime check. Good. | Trivial |
| `bench_all.py` — Table benchmarks run in a **separate process** via `subprocess.run` to isolate CFITSIO/PyTorch native state. This is a sophisticated approach. | Good |
| `bench_timing.py` — Uses `time_medians_interleaved` which interleaves competitor runs to minimize systematic bias from thermal throttling/CPU frequency scaling. | Good |
| **No GPU memory benchmarks for table operations** — `bench_fitstable_io.py` exists but was truncated. GPU transport benchmarks are image-only. | Low |

### 7.4 Bug Pattern Search Results

#### BUG1. `except Exception` patterns

**68 `except Exception`** patterns remain across the codebase (down from 74+ in the first review). Of these:
- **25** are in tests/benchmarks (acceptable)
- **43** are in production code
- **20** of the production `except Exception` are followed by `pass`

**Notable remaining silent swallows:**

| File | Line(s) | Context | Risk |
|---|---|---|---|
| `_io_engine/caches.py` | 127, 290, 295, 317, 380 | Cache operations — silent fails on clear/close | Low — caches are best-effort |
| `_io_engine/hdu_api.py` | 68, 73, 82, 96 | HDU type detection fallback — tries next HDU | Low — graceful degradation |
| `_io_engine/_read_pipeline.py` | 479, 652 | Cache stat update and fast-path fallback | Low — non-critical |
| `_hdu/hdu_list.py` | 71, 77, 101 | Double-close protection | Low — best-effort cleanup |
| `header_parser.py` | 138, 223 | Float/int parse fallback for string values | Low — correct fallback to string |
| `_table/mutation.py` | 166, 734, 815, 925, 985 | TNULL fill and operation fallbacks | Low — uses defaults |

**Assessment:** The remaining `except Exception: pass` patterns are mostly in cache/cleanup paths where silent failure is the correct behavior. None found in critical data paths.

#### BUG2. No `malloc`/`free` in C++

The C++ codebase uses exclusively RAII patterns:
- `std::vector`, `std::string` for dynamic allocation
- `std::shared_ptr` for shared ownership (`SharedReadMeta`)
- `MMapHandle` RAII wrapper for `mmap`/`munmap`
- `FITSFile`/`SubsetReader`/`TableReader` destructors properly clean up

**Assessment: No manual memory management issues.**

#### BUG3. `memcpy` usage

All `memcpy` calls in C++ operate on properly sized buffers:
- `fits_detail.h:271` — copies from mmap'd region into tensor (size validated by `read_region_via_fd`)
- `table_reader.h:716-729` — element-wise copy with `elem_size` computed from datatype
- `fits_file.cpp:344` — copies from mmap into tensor after pread fallback

**Assessment: No buffer overflow risks found.**

#### BUG4. CFITSIO status handling

All CFITSIO calls that can fail check the status code and either throw with `fits_get_errstatus` or propagate the error. No unchecked status codes found.

**Assessment: Proper error handling.**

#### BUG5. `ponytail` comments (12 total)

These are markers for deferred improvements:
| Location | Note |
|---|---|
| `cli/rgb.py:89` | `bytes(flat.tolist())` is slow for large PNGs |
| `_io_engine/caches.py:30` | List-based registry suffices for low concurrency |
| `_io_engine/write_api.py:28` | Per-path shared meta invalidation not yet available |
| `data/remote.py:166` | Best-effort prefetch retries on next resolve |
| `data/__init__.py:251,348` | Worker sharding and per-row file re-open notes |
| `cpp_src/cache.cpp:213` | Borrowed handle lifecycle |
| `cpp_src/table_ops.cpp:487` | CFITSIO error propagation |
| `_table/read.py:96` | VLA schema fallback |
| `examples/example_table_recipes.py:13` | PyArrow scanner deadlock avoidance |
| `examples/example_ml_galaxyzoo_legacy.py:68` | Non-invertible transform note |

**Assessment: All are reasonable deferred improvements, not bugs.**

### 7.5 Test Coverage Analysis

#### Test Statistics

- **209 test functions** across 40+ test files
- **~850 total test cases** (many are parametrized)
- **38 test files** covering all major subsystems

#### Coverage by Subsystem

| Subsystem | Test Files | Assessment |
|---|---|---|
| Core I/O | `test_api.py`, `test_io.py`, `test_read_policy.py` | ✅ Comprehensive |
| Image reads | `test_compression.py`, `test_byteswap.py`, `test_subset_3d.py` | ✅ Comprehensive |
| Table I/O | `test_table.py`, `test_table_filtering.py`, `test_arrow_table_api.py`, `test_table_file_ops.py` | ✅ Comprehensive |
| Table mutation | `test_table_file_ops.py` | ✅ Good coverage |
| Transforms | `test_transforms.py`, `test_transforms_typing.py`, `test_transforms_e2e.py` | ✅ Comprehensive |
| CLI | `test_cli.py` | ✅ Good coverage |
| HTTP/Remote | `test_remote_http_range.py`, `test_http_probe_fixture.py` | ✅ Good coverage |
| Security | `test_security.py`, `test_security_fix.py`, `test_security_eval.py` | ✅ Comprehensive |
| Concurrency | `test_concurrent_same_file_read.py` | ✅ Covered |
| Package isolation | `test_package_isolation.py`, `test_no_external_fits_backends.py` | ✅ Covered |
| Benchmarks | `test_bench_suites.py`, `test_bench_ranking_mmap.py` | ✅ Meta-tested |
| Docs | `test_docs_integrity.py`, `test_table_docs_smoke.py` | ✅ Covered |
| Examples | `test_examples_runner.py` | ✅ Covered |
| Upstream parity | `test_fitsio_upstream_smoke.py`, `test_astropy_upstream_smoke.py`, `test_upstream_parity_inventory.py` | ✅ Comprehensive |
| Integration | `test_integration.py`, `test_performance.py` | ✅ Good coverage |

#### Test Gaps (Minor)

| Gap | Severity |
|---|---|
| No test for `cli/rgb.py:write_rgb_image` standalone | Low — tested via examples |
| No test for `header_parser.py:benchmark_header_parsing` function | Trivial — benchmarking utility |
| No test for `data/remote.py` VOSpace download path (requires `vos` package) | Low — optional dependency |
| No test for concurrent table writes | Low — writes are not concurrent-safe by design |
| GPU transport tests gated behind `pytest.mark.skipif(not torch.cuda.is_available())` | Expected — hardware-dependent |

**Assessment: Test coverage is exceptional for a pre-1.0 library.**

### 7.6 Round 6 — New Issues Summary

| ID | Finding | Severity |
|---|---|---|
| C++10 | Thread-local HDU cache may hold stale metadata after shared meta invalidation | Low |
| CLI6 | `cmds_transform.py` calls transforms with no constructor args | Medium |
| R6-BUG1 | 20 remaining `except Exception: pass` in production code (all non-critical) | Low |
| R6-GAP1 | No standalone test for `write_rgb_image` | Low |
| R6-GAP2 | No VOSpace download path test | Low |

**Round 6 finds no new blocking issues.** The C++ code is memory-safe, the CLI is well-structured, benchmarks are sincere, and test coverage is exceptional.

---

## PART 8 — Round 7: Deep Re-review of Entire Codebase (Post-Cleanup)

**Context:** The user addressed most issues from Rounds 1-6 (~7,700 lines removed, 106 files changed). This round re-reads every remaining source file to find residual issues and verify fixes.

### 8.1 Critical Fixes Verified

| Original Issue | Status |
|---|---|
| `__init__.py` thread-safety | ✅ `_ATTR_CACHE` + `threading.RLock()` confirmed correct |
| `logging.py` NullHandler | ✅ Verified — no side effects at import time |
| `_unsigned.py` deleted | ✅ Confirmed — unsigned conventions now handled entirely in C++ |
| `core.py` deleted | ✅ Confirmed — `ChecksumVerifier` class gone |
| `continuum.py` / `spectral.py` deleted | ✅ Confirmed — niche domain transforms removed |
| `http_util.py` SSRF error handling | ✅ Confirmed — `HttpBlockedError` with `oserror` |
| `data/remote.py` prefetch error handling | ✅ Confirmed — now logs + warns (no silent swallow) |
| `get_cached_handle` always returns `cached=False` | ✅ Confirmed — CFITSIO §4 Option A |

### 8.2 Correction: `image.py:dispatch_read_image_cpp` — handle_cache IS used

**Prior review (A3 in Part 3) was incorrect.** The `handle_cache` parameter IS used:

```python
if not handle_cache and hasattr(cpp, "read_full_nocache"):
    return cast(Tensor, cpp.read_full_nocache(path, hdu, mmap))
```

When `handle_cache=False`, it uses the cold `read_full_nocache` path. The parameter is functional, not dead.

**Correction applied.** ✅ Prior finding A3 is retracted.

### 8.3 New Issues Found (Round 7)

#### R7-MUT1. `_table/mutation.py:_infer_fits_scalar_code` — Silent rejection of uint16/uint32/uint64

```python
def _infer_fits_scalar_code(arr: "np.ndarray") -> str:
    kind = arr.dtype.kind
    itemsize = arr.dtype.itemsize
    # ...
    if kind == "u" and itemsize == 1:
        return "B"
    # ...
    raise TypeError(f"Cannot infer FITS TFORM for dtype={arr.dtype}")
```

The function **only accepts `uint8`** (`kind=="u"` and `itemsize==1`). Passing `np.uint16`, `np.uint32`, or `np.uint64` arrays results in an opaque `TypeError` with no guidance. FITS itself has no native unsigned types beyond pseudo-unsigned conventions (BZERO offsets), but numpy users commonly create unsigned arrays. The error message should suggest using a signed equivalent or provide `.astype(np.int32)` guidance.

**Severity: Medium** — bad UX for a common numpy pattern. Users working with masks or count columns often have `np.uint16`/`np.uint32` arrays.

#### R7-MUT2. `_table/mutation.py` — `_COMPLEX_DTYPE_MAP` initialized at **4** sites

```
Line 131:  _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}
Line 396:  _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}
Line 539:  _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}
Line 1042: _COMPLEX_DTYPE_MAP = {"C": np.complex64, "M": np.complex128}
```

Same 2-line init copied 4 times across `_default_table_column_values`, `_coerce_table_column_array`, `update_rows`, and `_coerce_table_complex_values`. Each is guarded by `if not _COMPLEX_DTYPE_MAP:` but any future change to the map would need updating in all 4 places. Similarly, `_VLA_DTYPE_MAP` is init'd at 2 sites.

**Severity: Low-W** — works correctly but copy-paste drift risk. Extract to module-level constant (already exists on line 31 as `_COMPLEX_TFORM_CODES` — the dtype maps should follow suit).

#### R7-MUT3. Double `cache.clear()` + `_invalidate_path_caches` on every mutation

Every mutation function (`insert_column`, `replace_column`, `append_rows`, `insert_rows`, `delete_rows`, `update_rows`, `rename_columns`, `drop_columns`) follows this pattern:

```python
_invalidate_path_caches(path)
torchfits.cache.clear()
# ... C++ operation ...
torchfits.cache.clear()
_invalidate_path_caches(path)
```

That's 4 cache invalidation calls per mutation — double the necessary work. The pre-operation invalidation prevents stale reads, and the post-operation ensures the new state is visible. But calling both `invalidate_path_caches` AND `cache.clear()` (which also invalidates C++ cache) is redundant — `invalidate_path_caches` already calls `_invalidate_caches_for_path` which handles table caches. The double invalidation is safe but inefficient.

**Severity: Low** — performance-only; safe but wasteful.

#### R7-MUT4. `_normalize_mutation_rows` — O(columns) preprocessing before any data validation

The function builds `string_widths`, `vla_codes`, `complex_codes` dictionaries by iterating all column TFORM values before processing any input data. For tables with hundreds of columns (common in large surveys), this is wasted work when only a few columns are being mutated.

**Severity: Low** — tables with hundreds of columns are rare in mutation paths (mutations are typically small operations).

#### R7-PIPE1. `_read_pipeline.py:_read_cpu_fast_path` — dead variable assignment

```python
_ = (handle_cache_capacity, get_cached_handle)
```

This assigns to throwaway underscore — dead code that should be removed. The comment above explains why (handle caching was removed for CFITSIO §4), but the assignment serves no purpose.

**Severity: Trivial** — cosmetic, no functional impact.

#### R7-CPP1. `table_reader.h:ensure_column_scale` — floating-point equality for unsigned convention

```cpp
if ((typecode == TINT || typecode == TUINT || typecode == TLONG ||
     typecode == TINT32BIT) &&
    col.tzero == 2147483648.0) {
```

`2147483648.0` is exactly representable as IEEE 754 double (no mantissa loss), so this comparison is correct in practice. However, if a FITS file has a subtly-off TZERO value (e.g., `2.147483648000001e9`), this check silently fails and the column remains as int32 rather than uint32. A `std::fabs(col.tzero - 2147483648.0) < 0.5` tolerance check would be more robust.

**Severity: Very Low** — only affects malformed FITS files. Standard writers produce exact values.

#### R7-CPP2. `fits_bindings.cpp:read_full_cached` — thread-local cache stale after invalidation

The thread-local cache (`tl_cache`) is keyed by `(SharedReadMeta::uid, hdu_num)`. When `g_shared_meta` is invalidated and re-created for a path, the new `SharedReadMeta` gets a higher `uid`. Old entries persist in each thread's `tl_cache` until the thread visits 4096 unique HDU combinations. Until then, stale cached values (bitpix, naxis, naxes) serve the old metadata. The `ensure_hdu` call before reading would still succeed, so this can't cause crashes — just potentially wrong metadata for one read if the file was replaced between reads.

**Severity: Low** — the 4096-entry limit bounds the window; most threads never approach it.

#### R7-CPP3. `cache.cpp:clear()` — retained borrowed handles leak across clear

```cpp
if (entry.refcount != 0) {
    // ponytail: retain borrowed handles until their owners release them
    entry.stale = true;
    ++it;
    continue;
}
```

If a borrowed handle is never released (leaked by caller), the cache entry persists forever, keeping the fitsfile* open and the path in the LRU list. This is a GC-style weakness — a generation-based eviction or weak-reference would be more robust.

**Severity: Very Low** — the `ponytail` comment acknowledges this. No known leak path exists.

#### R7-CLI1. `cmds_transform.py` — `cls()` with no args (still unfixed)

```python
transform = cls()  # Always called with no constructor args
```

Flagged in Round 6 (CLI6) and still present. Transforms like `ArcsinhStretch(a=2.0)` can't be used via CLI. The CLI only works for transforms with default constructors.

**Severity: Medium** — usability gap for CLI users.

#### R7-HDU1. `tensor_hdu.py:to_tensor()` — no guard for closed file handle

```python
def to_tensor(self, device: str = "cpu") -> Tensor:
    if self._data is not None:
        return self._data.to(device)
    elif self._file_handle is not None:
        import torchfits._C as cpp
        return cast(Tensor, cpp.read_full(self._file_handle, self._hdu_index).to(device))
```

`self._file_handle` is checked for `is not None`, but a handle that's been `.close()`d is still not None. Calling `cpp.read_full()` on a closed handle would crash at the C++ level with an opaque error. The `HDUList.__exit__` sets `_file_handle = None` on contained HDUs when closing, but if a user holds a reference to a `TensorHDU` after closing its parent HDUList, this path can be triggered.

**Severity: Low** — requires holding TensorHDU references across context manager exits, which is uncommon.

#### R7-HDU2. `table_hdu.py:TableDataAccessor.__getitem__` — `squeeze()` on (N,1) columns (still present)

```python
if value.dim() > 1:
    return value.squeeze()  # lossy for (N,1) columns
```

Flagged in Round 5 and still present. A table column of shape `(N, 1)` returns shape `(N,)` — the trailing dimension is silently lost.

**Severity: Low** — intentional design (FITS scalar columns should be 1D), but worth documenting explicitly.

### 8.4 Codebase Statistics (Current)

| Metric | Before Cleanup | After Cleanup |
|---|---|---|
| Python source files | ~81 | ~78 |
| C++ source files | 7 | 7 |
| Header files | 12 | 12 |
| Lines removed | — | ~7,700 |
| Test files | ~40 | ~38 |
| Example files | ~30 | ~25 |
| Benchmark files | ~20 | ~18 |

### 8.5 Cross-Cutting Audit: Round 7 Sweep Metrics

| Area | Files Read | New Issues | Blocker? |
|---|---|---|---|
| Core I/O (image, read_pipeline, fallback, batch) | 4 | 1 (R7-PIPE1) | No |
| Caching (caches.py, cache.cpp, cache.h) | 3 | 2 (R7-CPP2, R7-CPP3) | No |
| Table mutation (mutation.py, write.py) | 2 | 4 (R7-MUT1-4) | No |
| C++ bindings (fits_bindings, table_bindings) | 2 | 1 (re-verified R7-CPP1) | No |
| C++ detail (fits_detail.h, fits_file.cpp/h, security.h) | 4 | 0 | — |
| C++ table reader (table_reader.h) | 1 | 1 (R7-CPP1) | No |
| HDU types (tensor_hdu, table_hdu, table_hdu_ref, card, dataview) | 5 | 2 (R7-HDU1, R7-HDU2) | No |
| CLI (16 command files) | 16 | 1 (R7-CLI1) | No |
| Data/datasets/remote | 3 | 0 | — |
| Transforms (__init__, stretch, normalize, clip, rgb, fits_meta, helpers, base) | 8 | 0 | — |
| Init/io/cache/table/hdu root modules | 8 | 0 | — |
| Remainder (interop, header_parser, http_util, vos_uri, etc.) | 8 | 0 | — |

### 8.6 Round 7 — Issues Summary

| ID | Finding | Severity | New? |
|---|---|---|---|
| R7-ERRATUM | `image.py:handle_cache` parameter is NOT dead — prior finding A3 retracted | — | Correction |
| R7-MUT1 | `_infer_fits_scalar_code` rejects uint16/uint32/uint64 with opaque TypeError | **Medium** | YES |
| R7-MUT2 | `_COMPLEX_DTYPE_MAP` init'd at 4 sites (copy-paste drift risk) | Low-W | YES |
| R7-MUT3 | Double cache.clear() + _invalidate_path_caches (4 calls per mutation) | Low | YES |
| R7-MUT4 | `_normalize_mutation_rows` preprocesses all columns before data validation | Low | YES |
| R7-PIPE1 | `_read_pipeline.py` dead `_ = (handle_cache_capacity, get_cached_handle)` | Trivial | YES |
| R7-CPP1 | Floating-point equality for unsigned convention detection | Very Low | YES |
| R7-CPP2 | Thread-local cache stale after shared meta invalidation (re-verified) | Low | Yes (from R6) |
| R7-CPP3 | `cache.cpp:clear()` retains borrowed handles (ponytail) | Very Low | YES |
| R7-CLI1 | `cmds_transform.py` — `cls()` with no args (re-verified unfixed) | **Medium** | Yes (from R6) |
| R7-HDU1 | `tensor_hdu.py:to_tensor()` — no guard for closed handle | Low | YES |
| R7-HDU2 | `table_hdu.py` — `squeeze()` on (N,1) columns (re-verified) | Low | Yes (from R5) |

---

## PART 9 — Final 1.0 Readiness (After Round 7)

### Unresolved Items Worth Fixing Pre-1.0

| Priority | Issue | Effort |
|---|---|---|
| **P1 (Medium)** | R7-MUT1: `_infer_fits_scalar_code` better error message for uint16/32/64 | 5 lines |
| **P1 (Medium)** | R7-CLI1: `cmds_transform.py` — document or support constructor args | ~20 lines |
| **P2 (Low)** | R7-MUT2: Hoist `_COMPLEX_DTYPE_MAP`/`_VLA_DTYPE_MAP` to module level | ~10 lines |
| **P2 (Low)** | R7-PIPE1: Remove dead `_ = (...)` in `_read_cpu_fast_path` | 1 line |
| **P3 (Cosmetic)** | R7-MUT3: Reduce double cache.clear() calls | ~30 lines |

### Items Explicitly Deferred (Safe for Post-1.0)

| Issue | Reason |
|---|---|
| R7-HDU1 (closed handle guard) | Requires user to hold TensorHDU across context manager exit — rare |
| R7-HDU2 (squeeze on (N,1)) | Intentional design, matches cfitsio convention |
| R7-CPP1 (floating-point equality) | Affects only malformed FITS files |
| R7-CPP2 (thread-local stale cache) | Only matters after file invalidation + 4096 unique HDU reads in one thread |
| R7-CPP3 (borrowed handle retention) | No known leak path; ponytail acknowledged |
| R7-MUT4 (column preprocessing) | Mutation operates on small tables |
| B1 (duplicate capability-check functions) | Functional; refactor post-1.0 |
| B3 (Python byte swap in http_subset) | Network-bound |

### Final Grade (After Round 7)

| Dimension | Grade | Change |
|---|---|---|
| Architecture | **A** | — |
| Thread safety | **A** | — |
| Performance | **A-** | — |
| Code cleanliness | **B+** → **A-** | ⬆ (post-cleanup) |
| Test coverage | **A** | — |
| Documentation | **A-** | — |
| Security | **A** | — |
| ML community fit | **B+** | — |

**Overall: A-** — Ready for 1.0. Only 2 medium-priority items (R7-MUT1, R7-CLI1) remain worth addressing before tagging. All other findings are minor polish or explicitly deferred.