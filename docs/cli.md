# torchfits CLI

After `pip install torchfits`, the `torchfits` command inspects and transforms
FITS files from the shell. It wraps the same C++ engine as the Python API.

Inventory commands (`info`, `header`, `verify`, `stats`, `table`, `probe`) take
paths on the command line or from stdin / `--stdin`. Mutation commands
(`copy`, `cutout`, `arith`, …) take explicit input/output paths.

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
torchfits stats science.fits --hdu 0 --jsonl
torchfits table catalog.fits --hdu 1 --preview 5
torchfits cutout science.fits cutout.fits --hdu 0 --box 100,100,256,256
torchfits convert catalog.fits out.parquet --to parquet --hdu 1
torchfits header *.fits --fitsort --keyword OBJECT --keyword DATE-OBS
torchfits probe https://example.edu/data.fits --json
```

Pipe paths into inventory commands:

```bash
find . -name '*.fits' | torchfits info --stdin --jsonl
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
| `convert` | table → Parquet; Lupton RGB → PNG |
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

### `header --fitsort`

Print a keyword table across many files (same idea as qfits `dfits | fitsort`):

```bash
torchfits header *.fits --fitsort --keyword OBJECT --keyword DATE-OBS
torchfits header *.fits --fitsort --keyword BITPIX --json
```

### `convert`

- **parquet** — export a table HDU (`--hdu`, default 1). Uses streaming
  batch writes so large tables stay out-of-core (bounded memory).
- **png** — Lupton asinh RGB preview from one cube (`--bands 0,1,2`) or three
  band files. Writes PNG with torch + stdlib only (no Pillow dependency).

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
| `setkey` | `modhead`, `replacekey` |
| `probe` | local `info` + remote header peek |

torchfits is FITS I/O oriented — it does not clone photometry, WCS, or
source-detection pipelines.

## Scripting notes

- No prompts; stable exit codes.
- Prefer `--json` / `--jsonl` for automation.
- GPU tensors are staged through host memory before any FITS write (same as the
  Python API — not GPUDirect).
