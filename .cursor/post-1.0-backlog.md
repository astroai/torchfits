# Post-1.0 backlog (from deep_review rounds)

Only genuinely open items live here, each with a one-line current status.
Entries verified fixed, decided, or moot are moved to
[`post-1.0-backlog-archive.md`](post-1.0-backlog-archive.md) with the commit
that closed them. Reconciled 2026-11-28 after the 1.4 correctness wave and the
hygiene pass.

## Open

- **`imexpr` / STILTS-style expression CLI.** Status: not started; a feature
  idea, not a defect. If done, it stays on the torchfits HTTP stack (with its
  SSRF guard), never CFITSIO's HTTPS drivers.

## Decisions (in force)

- **CLI trim: keep `compress` / `decompress` / `arith`.** Decided 2026-11-28.
  They are not fpack/funpack/imarith clones to drop in favour of NumPy: waves
  1-3 hardened and extended them (atomic outputs, `-j`/`-J`, exact and
  saturating integer `arith`, `BUNIT` checks, fpack tile/quantize/dither/
  `-i2f` controls) and the docs and tests pin their contracts. Removing them
  would break scripts for no maintenance win, since they are thin layers over
  `torchfits.write(compress=...)` and the tensor readers.
- **`torchfits verify` is checksum verification plus a structural size check,
  and stays self-contained.** It walks a local uncompressed file's headers so
  a truncated file is an I/O error (`42df7d9`), and it must never shell out to
  `fitsverify`. Cross-checking its verdict against `fitsverify` belongs in
  torchfits-harness (`tests/test_verify_vs_fitsverify.py`).
- **World coordinates stay out.** WCS/catalog cutouts, units and source
  finding belong to other packages (`docs/roadmap.md`).
- **Spectroscopy / continuum transforms stay out.** They were deleted from
  torchfits (no deprecation); absorb-vs-new design belongs in the sibling
  astronomy stack repo.
