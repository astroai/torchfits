# Post-1.0 backlog (from deep_review rounds)

Only genuinely open items live here, each with a one-line current status.
Entries verified fixed, decided, or moot are moved to
[`post-1.0-backlog-archive.md`](post-1.0-backlog-archive.md) with the commit
that closed them. Reconciled 2026-11-28 after the 1.4 correctness wave and the
hygiene pass.

## Open

- **Narrow-table buffered reads vs fitsio (`mmap=False`).** Status: the tiled
  parallel pread (`a645818`) took a 1-column projection of a narrow table from
  13.4 to 2.7 ms and a 2M x 8 full read from 216 to 9.7 ms (where= with mmap
  off: 228 to 18 ms); the fitsio comparison was not re-run because fitsio is
  not in the dev environment. Next step: re-benchmark in torchfits-harness
  (same host, same `case_id`) before deciding on a strided per-column read.
- **`table_reader.h` split / persistent `TableReader` direct I/O by path.**
  Status: still one header; reads through a borrowed handle are serialized
  and checked against file replacement (`106d48c`, `8da20d1`), so this is
  structure work, not a correctness gap.
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
