# torchfits CLI

Command-line FITS utilities built on the torchfits Python API. Use `--json` or
`--jsonl` for machine-readable output. Inventory commands (`info`, `header`,
`verify`, `stats`, `table`, `probe`) accept paths on the argv or via stdin /
`--stdin`. Mutation commands take explicit path arguments.

## Quick start

```bash
torchfits info image.fits
torchfits header image.fits --keyword BITPIX --json
torchfits verify image.fits
torchfits stats image.fits --hdu 0 --jsonl
torchfits table catalog.fits --hdu 1 --preview 3
torchfits cutout image.fits cutout.fits --hdu 0 --box 10,10,50,50
torchfits convert catalog.fits out.parquet --to parquet --hdu 1
torchfits convert r.fits g.fits b.fits rgb.ppm --to ppm
torchfits copy in.fits out.fits
torchfits diff a.fits b.fits
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

| Subcommand | Description |
|------------|-------------|
| `info` | HDU inventory (type, shape, rows) |
| `header` | dump header cards; `--keyword` filter |
| `verify` | `DATASUM` / `CHECKSUM` verification |
| `stats` | image min/max/mean via `read_tensor` |
| `table` | Arrow schema + preview rows |
| `cutout` | pixel subset via `read_subset` |
| `convert` | table→parquet; Lupton RGB→PPM |
| `probe` | local inventory; HTTP(S) range header probe |
| `diff` | compare headers and image shape/stats |
| `copy` | MEF-preserving FITS→FITS copy |
| `arith` | image ±×÷ by a constant |
| `compress` | tile-compress via `write(..., compress=True)` |
| `decompress` | expand compressed image HDUs |
| `transform` | apply a named `torchfits.transforms` class |
| `setkey` | set one header keyword |

### Multi-extension FITS (MEF)

Most commands default to **all HDUs**. Pass `--hdu 0,1,2` to select specific
extensions. JSONL mode emits one record per `(file, hdu)` pair.

### `convert`

- **parquet** — `torchfits.table.write_parquet` on a table HDU (`--hdu`, default 1).
- **ppm** — Lupton+ (2004) asinh RGB from one FITS (`--bands 0,1,2`) or three
  band files; writes binary PPM (no Pillow dependency).

## Tool mapping

| torchfits CLI | CFITSIO | Astropy | Gnuastro | qfits |
|---------------|---------|---------|----------|-------|
| `info` | `fits_get_num_hdus`, `fits_read_key` | `fits.open`, `len(hdul)` | `fitsfile` metadata | `qfits_header_get` |
| `header` | `fits_read_record` | `hdul[i].header` | `fitsheader` | `qfits_header` dump |
| `verify` | `fits_verify_chksum` | manual / third-party | — | — |
| `stats` | `fits_read_pix` + stats | `data.min/max/mean` | `arith` stats | — |
| `table` | table column metadata | `Table` schema | `table` | `qfits_table` |
| `cutout` | tiled `fits_read_subset` | `data[y1:y2, x1:x2]` | `crop` | — |
| `convert` | export | `Table.write` / PNG | `convert` | — |
| `probe` | remote header fetch | `open` URL | — | HTTP header |
| `diff` | — | `fitsdiff` | `fitsdiff` | — |
| `copy` | `fits_copy_hdu` | `HDUList` copy | `fits copy` | — |
| `arith` | pixel ops | `numpy` ops | `arith` | — |
| `compress` | `fits_compress_img` | `compimg` | `compress` | — |
| `decompress` | implicit on read | implicit | — | — |
| `transform` | — | `CCDData` ops | `mkcatalog` pipeline | — |
| `setkey` | `fits_update_key` | `header[key]=` | `fitskeyword` | `qfits_header_set` |

## Agent notes

- No interactive prompts; stable exit codes.
- Paths from argv or stdin (`--stdin` or piped input when argv is empty).
- Prefer `--jsonl` for multi-file / multi-HDU automation.
