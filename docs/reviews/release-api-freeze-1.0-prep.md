# Release API freeze review — 1.0 prep status

**Date:** 2026-07-16  
**Version on tree:** 0.9.2 (CLI surface included; formal 0.9.3 tag optional)  
**Verdict:** **Not ready to tag 1.0.0** — proceed through 0.9.2/0.9.3 evidence first

## Gate status

| Prerequisite | Status |
|---|---|
| Lean Python API (0.9.2 cuts) | Done |
| Thermonuclear review | Done (`docs/reviews/thermo-nuclear-0.9.2.md`) |
| Transforms profound review | Done (`docs/reviews/transforms-1.0.md`); catalog keep-all with doc demotions |
| CLI one-word MEF surface | Done (`docs/cli.md`, `torchfits` entry point) |
| Archive probe | Local + HTTP(S) range header probe; `vos:` deferred |
| Fresh multi-host exhaustive scorecard | Published IDs kept (`…190646` / `…191252` / `…191255`); scorecard re-patched 2026-07-16. Local `exhaustive_mps_20260717_000853` started; CANFAR re-create returned HTTP 400 from this workspace |
| Compatibility matrix (Python/PyTorch/Arrow) | Pending publish |
| Clean-install release-gate | Pending at rc |
| Split `transforms.py` (~3k LOC) | Blocking structural item from transforms review |

## Freeze rule

Do not tag `1.0.0` until:

1. Transforms file split (or explicit waive in freeze notes)
2. Fresh Darwin MPS + Linux CPU + CUDA scorecard cited in README and `docs/benchmarks.md`
3. Compatibility matrix published
4. `1.0.0rc1` clean-install matrix green

Until then, SemVer remains pre-1.0; CLI may still tighten.
