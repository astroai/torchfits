# torchfits CLI

After `pip install torchfits`, the `torchfits` command inspects and transforms
FITS files from the shell. It wraps the same C++ engine as the Python API.

Inventory commands (`info`, `header`, `verify`, `stats`, `table`, `probe`) take
paths on the command line or from stdin / `--stdin`. Mutation commands
(`copy`, `cutout`, `arith`, …) take explicit input/output paths.

### Process tax (cold start)

Each CLI invocation pays a **PyTorch / extension import** (~0.8–1 s on typical
laptops) before any FITS work. That dominates wall time versus gnuastro or
raw CFITSIO tools for tiny files. Prefer the in-process Python API for tight
loops; use the CLI for shell pipelines and one-shot inspection.

## Install and help

```bash
pip install torchfits
torchfits --help
torchfits info --help
```

## Quick examples

```bash
torchfits info science.fits
torchfits header science.fits --keyword OBJECT --json
torchfits verify science.fits
torchfits stats science.fits --hdu 0 --format jsonl
torchfits table catalog.fits --hdu 1 --preview 5
torchfits cutout science.fits cutout.fits --hdu 0 --box 100,100,256,256
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
torchfits convert catalog.fits out.parquet --hdu 1
torchfits convert catalog.fits out.csv --hdu 1
torchfits convert catalog.fits out.tsv --hdu 1
torchfits convert catalog.fits out.arrow --hdu 1
torchfits convert catalog.fits out.parquet --to parquet --hdu 1
torchfits convert r.fits g.fits b.fits rgb.png
```

Pipe paths into inventory commands:

```bash
find . -name '*.fits' | torchfits info --stdin --format jsonl
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | files differ (`diff`) |
| 2 | usage error |
| 3 | I/O error |
| 4 | checksum verification failed (`verify`) |

## Subcommands

| Command | What it does |
|---------|----------------|
| `info` | list HDUs (type, shape, rows) |
| `header` | dump cards; `--keyword` filter; `--fitsort` multi-file table |
| `verify` | check `DATASUM` / `CHECKSUM` |
| `stats` | image min / max / mean |
| `table` | Arrow schema + preview rows |
| `cutout` | write a pixel box to a new FITS file |
| `convert` | table → Parquet/CSV/TSV/Arrow IPC; Lupton RGB → PNG |
| `probe` | local inventory; HTTP(S) range probe; optional `vos:` |
| `diff` | compare two files (exit 1 if they differ) |
| `copy` | MEF-preserving FITS → FITS copy |
| `arith` | image ±×÷ by a constant |
| `compress` / `decompress` | tile-compress or expand image HDUs |
| `transform` | apply a named `torchfits.transforms` class |
| `setkey` | set one header keyword |

### Multi-extension FITS (MEF)

Most commands walk **all HDUs** by default. Narrow with `--hdu 0,1,2`.
JSONL mode emits one record per `(file, hdu)`.

### Output formats

Inventory commands (`info`, `header`, `verify`, `stats`, `table`, `probe`)
accept:

| Flag | Meaning |
|------|---------|
| (default) | human text |
| `--format json` / `--json` | JSON array |
| `--format jsonl` / `--jsonl` | one JSON object per line |

### `cutout`

Two equivalent ways to extract a pixel box:

- **`--box x1,y1,x2,y2`** — torchfits 0-based, half-open (Python slice end).
- **CFITSIO image section** on the path — 1-based inclusive, e.g.
  `torchfits cutout 'img.fits[10:100,20:200]' out.fits`.

- Supported/smoke-tested: image pixel sections via path (cutout CLI / `read_tensor`).
- Prefer torchfits APIs for the same jobs when available: `--box` / `read_subset`,
  `hdu=` / EXTNAME indexing, `table.read(..., where=)`.
- Not certified: path HDU selectors (`file.fits[1]` / `[EVENTS]` via `open`),
  binspec/histogram filenames, complex column filters as a substitute for
  `where=`, remote URL extended forms beyond existing `probe`, stacking
  section+`--box`.

### `verify`

Checks `DATASUM` / `CHECKSUM` keywords for each HDU (CFITSIO `ffvcks`).

Text output uses three labels:

| Label | Meaning | Exit code |
|-------|---------|-----------|
| `OK (no checksum keywords)` | HDU has no `DATASUM`/`CHECKSUM` — nothing to verify | 0 |
| `OK` | Checksums present and valid | 0 |
| `FAIL` | Checksums present but incorrect (corrupt) | 4 |

A file without checksum keywords is **not** a failure — it simply has nothing
to verify. This matches `fitsverify` semantics (warnings, not errors, for
missing keywords). Use `torchfits write_checksums(path, hdu=...)` to add
`DATASUM`/`CHECKSUM` keywords before verification.

```bash
torchfits verify science.fits
torchfits verify *.fits --format jsonl
```

JSON/JSONL output adds a `"status"` field (`"ok"`, `"no_checksums"`,
`"fail"`) alongside `"ok"`, `"datastatus"`, and `"hdustatus"`.

### `header --fitsort`

Print a keyword table across many files (same idea as qfits `dfits | fitsort`):

```bash
torchfits header *.fits --fitsort --keyword OBJECT --keyword DATE-OBS
torchfits header *.fits --fitsort --keyword BITPIX --json
```

### `convert`

- **parquet** / **csv** / **tsv** / **arrow** — export a table HDU (`--hdu`,
  default 1). Streaming writers keep large catalogs out-of-core.
  - `csv` / `tsv` are for flat columns (nested / list columns need parquet or
    arrow).
  - `arrow` is Arrow IPC / Feather V2 (``.arrow``) — opens in Polars
    (`pl.read_ipc`) and PyArrow.
- **png** — Lupton asinh RGB preview from one cube (`--bands 0,1,2`) or three
  band files. Writes PNG with torch + stdlib only (no Pillow dependency).

`--to` is optional when the output extension is unambiguous
(`.parquet`, `.csv`, `.tsv`/`.tab`, `.arrow`/`.feather`/`.ipc`, `.png`).

Defaults are for previews, not journal figures — retune stretch / Q per survey.

### `probe`

- **Local paths** — same inventory shape as `info`.
- **HTTP(S)** — range-fetch enough bytes for the primary header (no extra deps).
- **`vos:` / `vos://`** — optional; install the `vos` package for CANFAR VOSpace.
  Auth uses the client’s normal config.

Archive *search* (CAOM / `astquery`-style queries) is out of scope.

## Familiar-tool mapping

| torchfits | Closest classic tools |
|-----------|------------------------|
| `info` | `fitsinfo`, CFITSIO structure dump |
| `header` | `fitsheader`, `dfits` / `fitsort` |
| `verify` | `fitscheck`, HEASARC `fitsverify` (checksum subset) |
| `stats` | `imstat`, `aststatistics` |
| `table` | `tablist`, `asttable` |
| `cutout` | `astcrop`, CFITSIO sections |
| `convert` | `astconvertt` (tables); Lupton RGB preview |
| `copy` | `fitscopy` / `imcopy` |
| `arith` | `imarith` (constant operand) |
| `compress` / `decompress` | `fpack` / `funpack` |
| `transform` | `imfunction`-style stretches / named transforms |
| `setkey` | `modhead`, `replacekey` |
| `probe` | local `info` + remote header peek |

Step-by-step shell recipes (HorseHead, Chandra events): [CLI recipes](cli-recipes.md).

torchfits is FITS I/O oriented — it does not clone photometry, WCS, or
source-detection pipelines.

## Scripting notes

- No prompts; stable exit codes.
- Prefer `--format json` / `jsonl` (or `--json` / `--jsonl`) for automation.
- GPU tensors are staged through host memory before any FITS write (same as the
  Python API — not GPUDirect).
