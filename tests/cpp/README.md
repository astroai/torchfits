# `tests/cpp` — standalone C++ self-checks

Seven small programs that exercise C++ behaviour the Python suite cannot reach
directly: SIMD byte-swaps, the `parallel_for` pool (nesting and fork safety),
CFITSIO's open-mode conflict contract, `FitsReader`'s thread safety, and the
shared read cache's descriptor handling.

Each is a plain `main()` that **counts** its failures and returns non-zero,
deliberately framework-free so it can be compiled and run against a header or a
library in isolation. They do not use `assert`: under `-DNDEBUG` every `assert`
compiles to nothing, so an assert-based self-check prints "all checks passed"
and exits 0 having verified nothing at all (measured — see the audit ledger).

## Run them

```bash
pixi run dev            # once: builds the core library and CFITSIO under build/
tests/cpp/run_all.sh
```

The runner needs nothing but a C++ compiler and a `pixi run dev` build: it
locates the core library and CFITSIO's static archive under `build/`, writes the
FITS fixture the three file-based checks need with torchfits itself, builds all
of them, runs each **under a timeout**, and exits non-zero if any fails. It
prints `PASS` / `FAIL` / `BUILD FAILED` per check, and `TIMEOUT` /
`FAIL (exit 124)` for a check that never returns.

It also runs an extra check: the **x86_64 build of `test_bswap_helpers`,
executed under Rosetta**, because the byte-swap helpers have three `#if`
branches and a native build compiles only one. On Apple Silicon that is NEON,
so the SSSE3 branch — which `CMakeLists.txt` compiles into every x86_64 build
via `-mssse3` — would otherwise never be executed here. Set
`TORCHFITS_CPP_X86=0` to skip it; without an x86 runtime it is skipped with a
printed note. The AVX2 branch is not attempted (no shipped build enables it,
and Rosetta raises `SIGILL` on it here): `tests/test_simd_shuffle_masks.py`
checks both x86 mask tables statically, on any platform.

`CXX` selects the compiler (default `clang++`) and may include flags, the way
the pytest drivers read it (`CXX="c++ -pipe"` works).
`TORCHFITS_CPP_CHECK_TIMEOUT` sets the per-check budget in seconds (default
180). The timeout is load-bearing, not defensive: the regression
`test_parallel_for_nesting` guards *is* a deadlock, so a regression makes that
check hang rather than fail.

## Five of them also run under pytest

- `test_bracket_detection.cpp` — `tests/test_native_tree.py::test_native_cfitsio_bracket_detector_probe_runs`
  compiles and runs it, and asserts on its "all checks passed" line.
- `test_bswap_helpers.cpp` — `tests/test_native_tree.py::test_native_bswap_helpers_match_a_byte_wise_reference`
  compiles and runs it, and asserts on its exit status and summary line.
- `test_parallel_for_nesting.cpp` — `tests/test_cpp_self_checks.py::test_parallel_for_nesting_self_check`
  compiles it (with `core/parallel.cpp`) and runs it at pool sizes 1, 2, 4 and 8,
  so the deadlock regression in `parallel_for` is covered by `pytest tests/ -q`
  and not only by this runner (`tests/test_core_library.py::test_thread_pool_survives_worker_side_nesting`
  runs it too).
- `test_fitsreader_threads.cpp` — `tests/test_core_library.py::test_fitsreader_is_safe_to_share_across_threads`
  links it against the built core library and runs it on a synthetic MEF.
- `test_open_for_write_conflict.cpp` — `tests/test_core_library.py::test_cfitsio_readonly_handle_blocks_readwrite_open`
  links it against the built core library and runs it.

The two `test_core_library.py` drivers need the `build/` tree that `pixi run dev`
writes, and skip with that reason when it is absent (a plain `pip install`
links CFITSIO statically into the extension and keeps no build tree).

All drivers read `CXX` with `shlex.split`, so a value carrying flags
(`ccache c++`, `c++ -pipe`) works, and a `CXX` that cannot be run at all
**skips with a reason** instead of erroring. `tests/test_cpp_self_checks.py`
also guards the runner itself: a check cannot be dropped from the case table
without failing a test, and every `tests/<file>.py::<name>` this README cites
must exist.

`test_fork_after_pool.cpp` and `test_raw_fd_retry.cpp` stay behind the runner:
no pytest fixture builds them. `test_raw_fd_retry`'s Python half — the *cap* on
retained descriptors, which needs the whole extension — is
`tests/test_raw_fd_cap.py::test_shared_read_cache_does_not_retain_one_descriptor_per_path`.
Throughput and upstream-parity checks live in
[torchfits-harness](https://github.com/astroai/torchfits-harness), not here.

## What each one covers

| File | Covers | Needs a file argument |
|---|---|---|
| `test_bracket_detection.cpp` | `has_cfitsio_extended_filename_syntax`, including the `'['`-in-a-directory false positives | no |
| `test_bswap_helpers.cpp` | the big-endian→host byte-swaps in `internal_utils.h`: 19 sizes × 5 helpers × 4 source/destination alignments against a reference that moves bytes one at a time, plus a host-order pin tying the permutation to the FITS convention. Prints which SIMD branch compiled in, because a silent no-op branch would still "pass". | no |
| `test_parallel_for_nesting.cpp` | `core::parallel_for` nesting (a deadlock regression), that the outer chunks tile the range exactly once, and error propagation from both top-level and nested bodies | no |
| `test_fork_after_pool.cpp` | that a child forked after the pool has started can still run `parallel_for` | no |
| `test_open_for_write_conflict.cpp` | the CFITSIO contract `open_fits_for_write` relies on: a held READONLY handle makes a READWRITE open return `FILE_NOT_OPENED` (104) | yes |
| `test_fitsreader_threads.cpp` | one `FitsReader` shared by 4 threads — CFITSIO keeps one mutable current-HDU cursor per handle, so unlocked queries could interleave into another HDU's answer. The full shape vector is compared, not just its first axis. | yes (≥ 2 usable HDUs) |
| `test_raw_fd_retry.cpp` | that `get_shared_raw_fd` does not memoise a *failed* open: `chmod` the file unreadable (which leaves inode, size and mtime — the identity the invalidation watches — untouched), then make it readable again and require a live descriptor on the next call. Skips as root, where `chmod` cannot make `open` fail. | yes |

`test_parallel_for_nesting`, `test_fork_after_pool`, `test_fitsreader_threads`
and `test_raw_fd_retry` are regressions for real defects that were measured,
not hypothesised; each file's header records the measurement.
