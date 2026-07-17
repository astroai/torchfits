# Release 1.0b1 — Deep code review (src, tests, benches, examples)

**Status:** complete
**Model:** claude-opus-4-8-thinking-high (+ bugbot + security-review run separately)
**Branch:** `release/1.0b1` @ `ee154c6` (identical to `main` HEAD; the diff-vs-main is
empty, so this is a full-state review of the 1.0b1 tree, incl. the table API
coherence commit `ee154c6`).

## Verdict

**Ship with notes.**

No blocking product defects found. A clean rebuild passes the entire suite
(**830 passed, 6 skipped, 0 failed**; skips are CUDA-gated on this Mac host).
The command-injection security guard (`|`, `sh://`, `!`/whitespace-bypass) is
present, correct, and wired into every C++ entry point. The recent table-API
reframe (`table.read_arrow` / `read_torch` / `read_polars`, root
`read_table` alias) is coherent and covered by `test_public_boundary.py`.

The notes below are real but non-blocking: one CLI cosmetic-correctness bug, a
test-runner silent-skip hazard, and a build-hygiene trap that produces a
*false* security-test failure when the test env is not rebuilt.

## Blocking

**Count: 0.**

Nothing must block the 1.0b1 tag. In particular, the security test failure that
shows up when running `pixi run -e test pytest tests/test_security.py` directly
is **not** a product bug — see Should-fix #3 (stale binary), verified below.

## Should-fix

### SF-1 — `torchfits probe <url>` prints every remote record twice (text mode)
`cli/cmds_probe.py::run()` calls `emit_records(...)` (which already prints
`key=val` lines when neither `--json` nor `--jsonl` is set), then immediately
re-prints the identical lines in a manual loop.

```137:147:src/torchfits/cli/cmds_probe.py
    if remote_records:
        emit_records(remote_records, json_mode=args.json, jsonl=args.jsonl)
        if not (args.json or args.jsonl):
            for record in remote_records:
                parts = [f"{key}={record[key]!r}" for key in sorted(record)]
                print(" ".join(parts))
        return EXIT_OK
```

`emit_records` in `cli/common.py` (lines 131-133) already emits exactly this
format, so plain `probe http://…` / `probe vos:…` duplicates output. Untested:
the only remote probe tests use `--json`
(`tests/test_cli.py::test_vos_probe_bad_uri_is_io_error_when_vos_present`), so
the text path is never exercised. Fix: drop the manual loop (emit_records
already handles all three modes).

### SF-2 — Example runner marks REQUIRED failures as PASS on a substring match
`examples/test_examples.py::_run_example` treats any non-zero exit whose
combined stdout+stderr contains `"not installed"` or `"skipping"` **anywhere**
as a pass — even for scripts in the `REQUIRED` list.

```74:78:examples/test_examples.py
    output = (result.stderr or "") + (result.stdout or "")
    skip_markers = ("not installed", "skipping")
    if any(marker in output.lower() for marker in skip_markers):
        return True, "skipped (optional dependency missing)"
    return False, output[:1500]
```

A genuinely-broken required example that happens to print "skipping" (e.g. the
gallery scripts that log "skipping plot" when matplotlib is absent — commit
`28e5892`) would be silently green. This is the exact "silent skip masking a
failure" pattern the release gate is supposed to prevent. Fix: only honor skip
markers for the `OPTIONAL` list, or gate on a structured sentinel line
(`TORCHFITS_EXAMPLE_SKIP:`) rather than free-text.

### SF-3 — Stale `_C.so` in the `test` env yields a *false* security-test failure
Running `pixi run -e test pytest tests/test_security.py` fails
`test_security_cve_cfitsio_command_injection` with
`Actual message: "Could not open FITS file: sh://echo 'pwned'"`. This is a
build-staleness artifact, not a source defect:

- Source guard is correct (`sh://` after stripping leading `!`/whitespace):
  `src/torchfits/cpp_src/security.h` lines 24-33, committed in `16a7899`
  ("Fix command injection vulnerability via CFITSIO sh:// protocol",
  2026-07-16 17:02).
- `test` env binary: `.pixi/envs/test/.../torchfits/_C.cpython-313-darwin.so`
  built **Jul 16 07:46** — ~9 h *before* the fix commit.
- `default` env binary rebuilt **Jul 17 14:27** (during `pixi run dev`), and the
  security test passes there.

Verified: `pixi run -e default pytest tests/test_security.py` → **4 passed**;
full `pixi run -e default pytest tests/` → **830 passed, 6 skipped**.

Risk: any developer/CI invocation that runs pytest without first running
`pixi run dev` (rebuild) against a given env can get stale, misleading results
(here, a *passing-in-source* CVE test that appears to fail). Fix: make the
`test`/`release-gate` tasks the only supported entry points (they already
prepend `pixi run dev`), and/or install torchfits as a true editable build so
the extension can't silently lag the source. Document "never run bare pytest".

## Defer

### D-1 — Benchmark "Win vs X" columns cherry-pick the faster torchfits variant
In `docs/benchmarks.md` "Performance highlights", the win ratios are computed
against whichever of the two torchfits columns (default vs persistent) is
faster, per row (e.g. Large Image CPU uses 6.51 ms → 2.14× vs astropy;
Repeated Cutouts uses the persistent 18.34 ms → 18.28×). Defensible as
"best-of-torchfits", but the header doesn't say so. Add a one-line footnote.

### D-2 — Table I/O speedups (34–628× vs fitsio) warrant a fairness footnote
The FITS-table `read_full` category claims **34–628× vs fitsio**. Numbers come
from a recorded scorecard (`exhaustive_cuda_20260717_042840`, 4,079 rows, 0
deficits) and the methodology discloses mmap/non-comparable gating, but a
two-to-three-order-of-magnitude gap against a mature C library reads as a
comparator-shape effect (per-column torch conversion / handle reuse asymmetry).
Not re-runnable in this review; recommend a short "why tables are this fast"
note (direct CFITSIO→tensor columns vs structured-array→per-column copy) so the
claim doesn't look like a measurement error to reviewers.

### D-3 — Silent broad `except` in batch image fallback
`_io_engine/_read_pipeline.py::_read_batch_paths` swallows all exceptions from
`cpp_module.read_images_batch` with a bare `except Exception: pass` (lines
477-484) before dropping to the per-file loop. Correct behavior, but a genuine
batch-path regression would be invisible. Consider a debug-level log like the
other fast-path fallbacks already do.

## Evidence

- Branch state: `git merge-base main HEAD == main == HEAD == ee154c6`;
  `git diff main..HEAD` empty → full-tree review.
- Clean-build full suite (`pixi run -e default pytest tests/`):
  `830 passed, 6 skipped, 1 warning in 124.57s`.
- Security guard source: `src/torchfits/cpp_src/security.h` (checks `|` prefix,
  `sh://` prefix after `!`/whitespace strip, trailing `|`); wired via
  `check_fits_filename_security(...)` in `fits_file.cpp`, `table_reader.h`,
  `table_bindings.cpp`, `fits_bindings.cpp`, `cache.cpp`.
- Stale-binary proof: `test` env `_C.so` mtime `Jul 16 07:46` vs fix commit
  `16a7899` `2026-07-16 17:02`; `default` env `_C.so` `Jul 17 14:27`;
  `pixi run -e test pytest tests/test_security.py` → 1 failed,
  `pixi run -e default …` → 4 passed.
- No `eval`/`exec`/`os.system`/`subprocess`/`shell=True`/`pickle.load` in
  `src/torchfits` (only `re.compile`); `table.filter` uses numexpr and rejects
  `__import__(...)`/`print(...)` (`tests/test_security_eval.py`).
- Where-clause parsing (`_table/read.py::_compile_where_to_simple_predicates`)
  returns an immutable cached tuple, so `functools.lru_cache` values can't be
  mutated by callers.
- Skips across the suite are legitimate: CUDA-gated
  (`test_scale_on_device`, `test_integration`, `test_writing`,
  `test_performance`) and optional-dep `importorskip` (pyarrow/pandas/fitsio/
  astropy are hard deps in the prod+test env, so they don't skip in CI).
- Table-API coherence (`ee154c6`): `table.read_arrow is table.read`,
  `table.read_torch`/`read_polars` public, root `read_table` aliases
  `table.read_torch`; guarded by
  `tests/test_public_boundary.py::test_table_destination_readers_are_public`.

## External subagent notes

- **Security Review:** no Blocking/Important on the 1.0b1 surface; notes in
  [`_security_1.0b1.md`](_security_1.0b1.md).
- **Bugbot:** no separate findings file written; R3 full-suite + security
  coverage above stands in for the branch audit.
- **R4 real-data CLI:** transform-on-int16 write crash fixed in
  `cli/cmds_transform.py` before freeze; see
  [`release-1.0b1-realdata-cli.md`](release-1.0b1-realdata-cli.md).
