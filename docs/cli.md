# torchfits CLI

After `pip install torchfits`, the `torchfits` command inspects and transforms
FITS files from the shell. It wraps the same C++ engine as the Python API.

**Every flag is on `torchfits <cmd> --help`.** This page is a tour, not an
exhaustive flag list.

Inventory commands (`info`, `header`, `verify`, `stats`, `table`, `probe`) take
paths on the command line or from stdin / `--stdin`. Mutation commands
(`copy`, `cutout`, `arith`, …) take explicit input/output paths.

Common shorts (where applicable):

| Short | Long | Notes |
|-------|------|-------|
| `-e` | `--hdu` | HDU index / list (`-e` avoids clashing with `-h` help) |
| `-f` | `--format` | inventory output: `text` / `json` / `jsonl` |
| `-o` | `--out` | output path (also positional on copy/cutout/convert/compress) |
| `-w` / `-c` | `--where` / `--columns` | `convert` table filter (STILTS-like) |
| `-n` | `--rows` | `table` preview row count |
| `-k` | `--keyword` / `--key` | **same short, different commands:** `header -k` filters cards; `setkey -k` sets a keyword |

`probe --header-bytes` controls the remote header peek size; `--timeout` is the
HTTP timeout.

### Process tax (cold start)

Each CLI invocation pays a **PyTorch / extension import** (~0.8–1 s on typical
laptops) before any FITS work. That dominates wall time versus gnuastro or
raw CFITSIO tools for tiny files. Prefer the in-process Python API for tight
loops ([Python workflows](python-workflows.md)); use the CLI for shell
pipelines and one-shot inspection.

## Install and help

```bash
pip install torchfits
torchfits --help
torchfits info --help
torchfits convert --help
```

## Quick examples

```bash
torchfits info science.fits
torchfits header science.fits -k OBJECT -f json
torchfits verify science.fits
torchfits stats science.fits -e 0 -f jsonl
torchfits table catalog.fits -e 1 -n 5
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
torchfits cutout science.fits -o cutout.fits -e 0 --box 100,100,256,256
torchfits convert catalog.fits -o out.parquet -e 1
torchfits convert catalog.fits out.csv -e 1
torchfits convert catalog.fits -o filtered.parquet -e 1 -w "flux > 2" -c ra,dec,flux
torchfits convert r.fits g.fits b.fits -o rgb.png
torchfits copy science.fits -o science_copy.fits
torchfits table catalog.fits -e 1 -n 10
torchfits probe https://example.edu/file.fits --header-bytes 5760 --timeout 30
```

Multi-file / stdin inventory:

```bash
torchfits info a.fits b.fits c.fits -f jsonl
find . -name '*.fits' | torchfits info --stdin -f jsonl
printf '%s\n' *.fits | torchfits header --stdin --keyword-table -k OBJECT -k NAXIS1
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
| `header` | dump cards; `-k` filter; `--keyword-table` multi-file table |
| `verify` | check `DATASUM` / `CHECKSUM` |
| `stats` | image min / max / mean |
| `table` | Arrow schema + preview rows |
| `cutout` | write a pixel box to a new FITS file |
| `convert` | table → Parquet/CSV/TSV/Arrow/FITS; filter with `--where`; Lupton RGB → PNG |
| `probe` | local = `info`; HTTP(S)/vos = header peek (`--header-bytes` / `--timeout`) |
| `diff` | compare two files (exit 1 if they differ) |
| `copy` | MEF-preserving FITS → FITS copy |
| `arith` | image ±×÷ by a constant |
| `compress` / `decompress` | tile-compress or expand image HDUs |
| `transform` | apply a named `torchfits.transforms` class |
| `setkey` | set one header keyword |

### Multi-extension FITS (MEF)

Most commands walk **all HDUs** by default. Narrow with `-e 0,1,2`.
JSONL mode emits one record per `(file, hdu)`.

### Output formats

Inventory commands (`info`, `header`, `verify`, `stats`, `table`, `probe`)
accept:

| Flag | Meaning |
|------|---------|
| (default) | human text |
| `-f json` / `--json` | JSON array |
| `-f jsonl` / `--jsonl` | one JSON object per line |

### `cutout`

Same job, two syntaxes (do not combine them):

- **CFITSIO image section** on the path (1-based inclusive) — what most
  `imcopy` / CFITSIO users already type:
  `torchfits cutout 'img.fits[10:100,20:200]' out.fits`
- **`--box x1,y1,x2,y2`** — torchfits 0-based half-open (same coords as
  `read_subset`):
  `torchfits cutout img.fits -o out.fits --box 9,19,100,200`

Supported/smoke-tested: image pixel sections via path (cutout CLI /
`read_tensor`). Not certified: path HDU selectors (`file.fits[1]` /
`[EVENTS]` via `open`), binspec/histogram filenames, stacking
section+`--box`, or using path filters instead of `table.read(..., where=)`.

### `verify`

Checks **`DATASUM` / `CHECKSUM` only** (CFITSIO `ffvcks`). Covers checksum
keywords only; HEASARC `fitsverify` structural checks (mandatory keywords,
XTENSION rules, etc.) are out of scope.

Text output uses three labels:

| Label | Meaning | Exit code |
|-------|---------|-----------|
| `OK (no checksum keywords)` | HDU has no `DATASUM`/`CHECKSUM` — nothing to verify | 0 |
| `OK` | Checksums present and valid | 0 |
| `FAIL` | Checksums present but incorrect (corrupt) | 4 |

Files without checksum keywords exit **0** with label
`OK (no checksum keywords)` — there is nothing to verify (fitsverify-style
warning, not corruption). Use `torchfits write_checksums(path, hdu=...)` to
add checksums before verification.

```bash
torchfits verify science.fits
torchfits verify *.fits -f jsonl
```

JSON/JSONL output adds a `"status"` field (`"ok"`, `"no_checksums"`,
`"fail"`) alongside `"ok"`, `"datastatus"`, and `"hdustatus"`.

### `setkey`

Set or rename header keywords. Supports short cards, **HIERARCH** / long names,
`-e all` (or a comma list), and multiple files (`--out-dir` for copies).

```bash
torchfits setkey science.fits -k OBJECT --value NGC1234
torchfits setkey science.fits -k "ESO DET CHIP1 ID" --value "42" -e all
torchfits setkey *.fits --rename OBJECT=TARGET -e 0 --out-dir /tmp/edited
```

### `header --keyword-table`

Print a keyword table across many files:

```bash
torchfits header *.fits --keyword-table -k OBJECT -k DATE-OBS
torchfits header *.fits --keyword-table -k BITPIX -f json
```

### `convert`

- **parquet** / **csv** / **tsv** / **arrow** / **fits** — export a table HDU
  (`-e`, default 1). Streaming writers keep large catalogs out-of-core when
  no filter is applied.
  - `--where` + optional `--columns` — filter+export (STILTS-like subset, not
    full STILTS). Same predicate syntax as `table.read(..., where=)`.
  - `csv` / `tsv` are for flat columns (nested / list columns need parquet or
    arrow).
  - `arrow` is Arrow IPC / Feather V2 (``.arrow``).
- **png** — Lupton asinh RGB preview from one cube (`--bands 0,1,2`) or three
  band files. Writes PNG with torch + stdlib only (no Pillow dependency).

`--to` is optional when the output extension is unambiguous
(`.parquet`, `.csv`, `.tsv`/`.tab`, `.arrow`/`.feather`/`.ipc`, `.fits`, `.png`).

Defaults are for previews, not journal figures — retune stretch / Q per survey.

### `transform`

`--name` is a class from `torchfits.transforms.__all__`. Append
`:key=val,key2=val2` to pass constructor kwargs (values are parsed as
bool/int/float, else left as a string); unknown kwargs are rejected before
construction.

```bash
torchfits transform image.fits --name ArcsinhStretch -o out.fits
torchfits transform image.fits --name ArcsinhStretch:a=2.0 -o out.fits
torchfits transform image.fits --name PercentileClipNormalize:lower_pct=1.0,upper_pct=99.0 -o out.fits
```

### `probe`

- **Local paths** — same inventory as `info` (`-e` selects HDUs).
- **HTTP(S)** — range-fetch primary header (`--header-bytes`, `--timeout`); `-e` is
  ignored for remote peeks (primary only). Follows redirects with SSRF checks;
  optional `TORCHFITS_HTTP_AUTHORIZATION` / `TORCHFITS_HTTP_TOKEN`.
- **`vos:` / `vault:` / `vos://`** — optional; install the `vos` package.
  Short `vos:<user>/...` and `vault:<user>/...` map to
  `vos://cadc.nrc.ca~vault/<user>/...`. Auth uses the client’s normal config.

```bash
torchfits probe science.fits
torchfits probe https://example.edu/data.fits --header-bytes 5760 --timeout 15 -f json
torchfits probe vos:alice/data/sample.fits
```

Archive *search* (CAOM / `astquery`-style queries) is out of scope.

### `cutout`

Pixel box extraction. HTTP(S) **uncompressed 2D** inputs use Range GETs;
compressed remotes download into the cache first (same as `read_subset`).

## Familiar-tool mapping

| torchfits | Closest classic tools |
|-----------|------------------------|
| `info` | `fitsinfo`, CFITSIO structure dump |
| `header` | `fitsheader`, `dfits` / `fitsort` |
| `verify` | `fitscheck` / HEASARC `fitsverify` **checksum subset only** |
| `stats` | `imstat`, `aststatistics` |
| `table` | `tablist`, `asttable` |
| `cutout` | `astcrop`, CFITSIO sections |
| `convert` | `astconvertt` / STILTS-like filter+export; Lupton RGB preview |
| `copy` | `fitscopy` / `imcopy` |
| `arith` | `imarith` (constant operand) |
| `compress` / `decompress` | `fpack` / `funpack` |
| `transform` | `imfunction`-style stretches / named transforms |
| `setkey` | `modhead` / `hedit`-style set + rename (MEF / multi-file) |
| `probe` | local `info` + remote header peek |

Step-by-step shell recipes (HorseHead, Chandra events): [CLI recipes](cli-recipes.md).

## Scripting notes

- No prompts; stable exit codes.
- Prefer `-f json` / `jsonl` (or `--json` / `--jsonl`) for automation.
- GPU tensors are staged through host memory before any FITS write (same as the
  Python API — not GPUDirect).
