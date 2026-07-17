# Release 1.0b1 — Real-data CLI multi-tool correctness + speed

**Status:** complete  
**Model:** composer-2.5 (parent)  
**Samples:** HorseHead, Chandra events (cached under `~/.cache/torchfits/samples/`)  
**Date:** 2026-07-17

## Verdict

**Ship with notes**

Correctness vs astropy/fitsio on HorseHead pixels and Chandra row counts passes. CLI recipes and core commands work after a transform write fix. CLI process overhead dominates wall time vs gnuastro/CFITSIO (expected cold-start); FITSH unavailable (SKIP).

## Blocking

- ~~`torchfits transform` crashed on integer HDUs~~ — **fixed on branch**: promote to float before transform; drop integer BITPIX header on float write ([`cmds_transform.py`](../../src/torchfits/cli/cmds_transform.py)).

## Should-fix

1. **CLI cold-start ~0.8–1.1 s** for header/stats/table vs gnuastro/CFITSIO ~7–15 ms — document as Torch/Python import tax; optional lazy CLI entry for rc1.
2. **`torchfits verify` reports FAIL** when checksum keywords absent (HorseHead) while `fitsverify` exits 0 with warnings — align messaging with “no checksum keywords” vs corrupt.
3. **Docs:** `table --preview` (not `--limit`); cutout `--box` is exclusive end (100,100,256,256 → 156×156) — recipes already correct; examples that say `--limit` should be fixed if any remain.

## Defer

- FITSH (`fiinfo`/`fiarith`) — **SKIP**: no Homebrew formula; not installed.
- Fresh multi-host scorecard microbench is not this report (CLI peer smoke only).

## Install notes

| Family | Status |
|--------|--------|
| torchfits CLI | pixi env `.pixi/envs/default/bin/torchfits` |
| astropy / fitsio | pixi |
| CFITSIO utils | `fitscopy`, `fitsverify`, `fitsheader`, `fitsinfo` (Homebrew + pixi) |
| gnuastro 0.24 | installed via `brew install gnuastro` (`astfits`, `aststatistics`, `asttable`, …) |
| FITSH | **SKIP** — not available via brew in this environment |
| hyperfine | Homebrew |

## Correctness

| Check | Result | Notes |
|-------|--------|-------|
| HDU list HorseHead | **PASS** | torchfits/astropy/fitsio: 2 HDUs (IMAGE + TABLE mask) |
| Header BITPIX/NAXIS* | **PASS** | BITPIX=16, NAXIS2=893, NAXIS1=891 |
| Image pixels vs astropy/fitsio | **PASS** | `allclose` after float64 compare; min=3759 max=22918 |
| Image stats vs `aststatistics --hdu=0` | **PASS** | mean≈9831.48, min=3759, max=22918 |
| Cutout 100,100,256,256 | **PASS** | exclusive end → 156×156; matches slice semantics |
| Arith +100 | **PASS** | exit 0 |
| Transform Arcsinh/ZScale on int16 | **PASS** after fix | float32 output written |
| `imstat_imarith.sh` recipe | **PASS** | exit 0 |
| Chandra `table.read` rows | **PASS** | 483964 rows, 19 cols vs astropy |
| `fitsverify` HorseHead | **PASS** (exit 0) | no checksum keywords |
| `astfits` HorseHead | **PASS** | exit 0 |
| FITSH | **SKIP** | not installed |

Intentional deltas: `verify` FAIL without checksum cards; transform outputs float32 without copying int BITPIX header.

## Speed

CLI peer smoke (hyperfine, 5 runs, warmup 1; **not** release scorecard). Medians:

| Workload | torchfits | Peer | Ratio (tf/peer) |
|----------|-----------|------|-----------------|
| header | 806 ms | `astfits` 7.5 ms | ~107× |
| header | 806 ms | `fitsheader` 215 ms | ~3.7× |
| stats | 822 ms | `aststatistics --hdu=0` 10.7 ms | ~77× |
| verify | 827 ms | `fitsverify` 6.6 ms | ~125× |
| table preview | 1090 ms | `asttable -h1 --head=5` 99 ms | ~11× |

Interpretation: torchfits CLI pays PyTorch import every process; in-process library reads remain competitive (see published exhaustive scorecards). Label as **CLI process overhead**, not I/O engine regression.

## Evidence

Artifacts under `/tmp/tf-1.0b1-cli/` (`speed2.json`, `speed2.txt`, recipe log). Commands re-runnable from this doc.
