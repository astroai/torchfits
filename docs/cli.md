# torchfits CLI

After installing `torchfits`, the `torchfits` command line interface lets you
inspect, manipulate, and transform FITS files directly from your shell, backed by the same high-performance C++ engine and vendored CFITSIO as the Python library.

Run `torchfits --help` or `torchfits <subcommand> --help` anytime for the complete list of options.
Run `torchfits --version` to print the installed torchfits version.

---

## Quick Tour

```bash
# Inspection & metadata
torchfits info science.fits
torchfits header science.fits -k OBJECT -k 'NAXIS*'
torchfits header *.fits --keyword-table -k OBJECT -k FILTER
torchfits stats science.fits -e 0 -f json
torchfits table catalog.fits -e 1 -n 5
torchfits verify science.fits

# File comparison & copies
torchfits diff file_a.fits file_b.fits
torchfits copy science.fits science_copy.fits

# Transformations & arithmetic
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
torchfits arith science.fits --op mul --value 2.0 -o doubled.fits
torchfits arith science.fits flat.fits --op div -o calibrated.fits
torchfits transform science.fits --name ArcsinhStretch -o stretched.fits

# Format conversions & previews
torchfits convert catalog.fits catalog.parquet -e 1
torchfits convert catalog.fits bright.parquet -e 1 -w "MAG_G < 18.0" -c RA,DEC,MAG_G
torchfits convert g.fits r.fits i.fits -o rgb.png --to png

# Compression
torchfits compress science.fits science.fits.fz
torchfits decompress science.fits.fz science_decomp.fits

# Header modification
torchfits setkey science.fits -k OBJECT --value "NGC 1234"
torchfits setkey science.fits --rename OLDNAME=NEWNAME

# Remote peeking (HTTP/HTTPS/VOSpace)
torchfits probe https://example.edu/survey/image.fits --header-bytes 5760
```

---

## Global Options & Flags

Shared flags (availability varies by subcommand):

| Flag | Name | Purpose | Default | Available on |
|---|---|---|---|---|
| `-e` | `--hdu` | Target HDU index (e.g. `-e 0`, `-e 1,2`, or `-e all` for `setkey`); a repeated index (`-e 0,0`) is a usage error | All HDUs or 0/1 depending on command | every subcommand except `copy`, `diff`; table-only on `convert`, local-only on `probe`, and requires `--split hdu` on compression commands |
| `-f` | `--format` | Output format: `text`, `json`, or `jsonl` (aliases `--json`, `--jsonl`) | `text` | `info`, `header`, `verify`, `stats`, `table`, `probe` |
| `-o` | `--out` | Output path (alias of the positional `OUTPUT`) | positional `OUTPUT` | `convert`, `copy`, `arith`, `cutout`, `compress`, `decompress`, `transform`, `setkey` |
| `--out-dir` | `--out-dir` | Output directory for batch file processing | unset (required for multi-input batches) | `copy`, `arith`, `cutout`, `compress`, `decompress`, `transform`, `setkey` |
| `--stdin` | `--stdin` | Read input file paths from standard input (stdin is also read implicitly when no paths are given and stdin is not a terminal) | `False` | `info`, `header`, `verify`, `stats`, `table`, `probe`, `setkey` |
| `-j` | `--jobs` | PyTorch intra-op CPU threads (`torch.set_num_threads`) | `0` (all CPU cores) | `stats`, `arith`, `compress`, `decompress`, `transform` |
| `-J` | `--file-jobs` | Parallel worker thread pool across multiple files | `0` (all CPU cores if $\ge 2$ files, else 1) | `verify`, `stats`, `copy`, `arith`, `cutout`, `compress`, `decompress`, `transform`, `setkey` |

### Parallelism & Multi-Core Execution

The CLI provides two orthogonal parallelism knobs:

1. **`-j` / `--jobs` (Intra-file parallelism):** Controls the number of PyTorch internal threads used for array computations (e.g. image arithmetic reductions, statistics calculations). This is the setting that applies whenever a single file (or `-J 1`) is processed.
2. **`-J` / `--file-jobs` (Inter-file batch parallelism):** Spawns a thread pool to process multiple independent FITS files concurrently. When batching files with `-J`, each worker automatically sets intra-op threads to 1 to prevent CPU oversubscription — under fan-out the `-j` value yields to that cap, because `-J` alone already occupies the cores.

So `-j 8 one.fits out.fits` runs with 8 intra-op threads, while `-j 8 -J 4 a.fits b.fits c.fits d.fits --out-dir out/` runs 4 workers of 1 thread each.

### Performance: Shell CLI vs Python API

Metadata commands (`info`, `header`, `probe`, `verify`, `table`, `copy`, and
`setkey`) start without importing Python `torch`. Commands that read or write
pixel/tensor payloads initialize the tensor runtime when they need it, so
startup cost is intentionally paid only by the operation that uses tensors.
For shell automation, scripting, and file management, this distinction keeps
metadata checks lightweight; for tight computational loops (such as dataset
iterators and deep learning training pipelines), use the in-process Python API
([Python workflows](python-workflows.md)).

---

## Exit Codes

| Code | Meaning | Example |
|---|---|---|
| `0` | Success | Command completed successfully |
| `1` | Difference found | `torchfits diff` found header, pixel, or table-data discrepancies |
| `2` | Usage error | Missing arguments, invalid flags, or unknown syntax; an `-e`/`--hdu` (or `--bands`) index the file does not have, in every subcommand |
| `3` | I/O error | File not found, permission denied, or invalid FITS structure (a truncated file is reported as such, not as a bare CFITSIO status) |
| `4` | Checksum verification failure | `torchfits verify` detected invalid `DATASUM` or `CHECKSUM` |
| `5` | Internal error | Unexpected exception; traceback printed to stderr |
| `130` | Interrupted | `KeyboardInterrupt` / Ctrl-C |

A downstream consumer that closes the pipe early (`torchfits info many.fits | head -1`) is **not** an error: the output is truncated and the command still exits `0`. Without this, a block-buffered stdout would only fail during interpreter shutdown, which CPython reports as an undocumented exit `120` plus `Exception ignored on flushing sys.stdout` noise on stderr.

---

## Subcommand Reference

| Subcommand | Description |
|---|---|
| [`info`](#info) | Summary of HDUs (extension index, name, type, dimensions, data type) |
| [`header`](#header) | Display header cards, filter keywords, or build multi-file summary tables |
| [`verify`](#verify) | Validate `DATASUM` and `CHECKSUM` integrity across HDUs |
| [`diff`](#diff) | Compare header keywords, image pixels, and table data between two FITS files |
| [`stats`](#stats) | Compute image statistics (min, max, mean, standard deviation, median) |
| [`table`](#table) | Inspect binary/ASCII table schemas and preview rows |
| [`cutout`](#cutout) | Extract sub-regions using pixel coordinates or CFITSIO sections |
| [`convert`](#convert) | Export tables to Parquet, CSV, TSV, Arrow, or render 1–7 band RGB PNGs |
| [`copy`](#copy) | Byte-identical copy of a FITS file (CompImage tiles stay compressed) |
| [`arith`](#arith) | Perform scalar or image-to-image arithmetic (+, -, *, /) |
| [`compress`](#compress) | Tile-compress images using Rice, Gzip, or Hcompress algorithms |
| [`decompress`](#decompress) | Uncompress tile-compressed FITS files to standard FITS |
| [`transform`](#transform) | Apply astronomical stretch and normalization transforms |
| [`setkey`](#setkey) | Insert, update, rename, or delete header keywords |
| [`probe`](#probe) | Peek at local or remote HTTP(S) / VOSpace FITS headers without downloading |

---

### `info`

Lists all HDU extensions in one or more FITS files, showing HDU index, name, extension type, dimensions, and data type.

```bash
# Inspect a single file
torchfits info science.fits

# Inspect multiple files with JSON Lines output
torchfits info file1.fits file2.fits -f jsonl

# Read paths from standard input
find . -name "*.fits" | torchfits info --stdin -f jsonl
```

---

### `header`

Dumps header cards from one or more HDUs in standard FITS card format, with support for keyword filtering and multi-file summary catalogs.

```bash
# Print all headers in all HDUs
torchfits header science.fits

# Select specific HDUs
torchfits header science.fits -e 0,1

# Filter for specific keywords (supports wildcards)
torchfits header science.fits -k OBJECT -k 'NAXIS*'

# Export header cards to structured JSON
torchfits header science.fits -e 0 -f json

# Build a tabular summary across multiple files
torchfits header *.fits --keyword-table -k OBJECT -k FILTER -k EXPTIME
```

---

### `verify`

Validates FITS `DATASUM` and `CHECKSUM` keywords using CFITSIO's fast verification engine. It is deliberately self-contained: a structure checker such as `fitsverify` is never invoked. Cross-checking the two verdicts lives in [torchfits-harness](https://github.com/astroai/torchfits-harness).

```bash
# Verify checksums for all HDUs
torchfits verify science.fits

# Batch verify multiple files in parallel
torchfits verify *.fits -J 0 -f jsonl
```

Output statuses:

- `OK`: Checksums are present and valid.
- `OK (no checksum keywords)`: File has no checksum keywords (returns exit code 0).
- `FAIL`: Checksum does not match data (returns exit code 4).

Before checking checksums, `verify` makes sure a local, uncompressed file is as
long as its headers say: every header and every data unit must be present (a
missing final 2880-byte pad is tolerated, as CFITSIO tolerates it). A truncated
file is an I/O error (exit code 3) naming the HDU that is cut short, even when
it carries no checksum keywords or `--hdu` selects a different HDU. Compressed
(`.gz`, `.bz2`, ...) files are not decompressed for this size check.

---

### `diff`

Compares two FITS files HDU by HDU. Header card order, repeated cards, values,
and comments must match (transient `CHECKSUM`/`DATASUM` cards are ignored).
Image HDUs are compared pixel by pixel (shape, dtype, and every value exactly;
`NaN`, including `BLANK` pixels read as `NaN`, equals `NaN`), and table HDUs
row by row in every column. Returns exit code `0` only if headers and data are
identical, `1` if anything differs; each difference is listed on stderr.

```bash
# Compare two files
torchfits diff image_v1.fits image_v2.fits
```

---

### `stats`

Computes pixel statistics for image HDUs: minimum, maximum, mean, standard deviation, and median.

- Only finite pixels count: `NaN`, `±Infinity`, and integer `BLANK` pixels are excluded, and their number is reported as `nbad`.
- Statistics are computed in float64; integer images report exact integer `min` / `max`.
- `std` is the population standard deviation, and `median` is the conventional midpoint median (as in numpy and IRAF `imstat`): the mean of the two middle values for an even pixel count.

```bash
# Compute statistics for primary HDU
torchfits stats science.fits -e 0

# Output in JSON format
torchfits stats science.fits -e 0 -f json

# Process multiple files in parallel
torchfits stats *.fits -e 0 -J 0 -f jsonl
```

Non-finite values in JSON output: `stats` and `table` serialize `NaN`/`±Infinity` floats as `null` in `-f json` / `-f jsonl` (`--json` / `--jsonl`) — JSON has no NaN or Infinity literals. An empty image region, or one with no finite pixel, therefore reports `null` for `min`, `max`, `mean`, `std`, and `median`.

---

### `table`

Inspects binary or ASCII table extensions, displaying the PyArrow schema, column types, and a preview of rows.

```bash
# Preview table schema and first 5 rows
torchfits table catalog.fits -e 1 -n 5

# Output rows in JSON format
torchfits table catalog.fits -e 1 -n 5 -f json

# Schema only, no preview rows
torchfits table catalog.fits -e 1 -n 0
```

`-n` / `--rows` takes a non-negative count; a negative value is a usage error (exit `2`).

The preview (in every output format) is strict JSON:

- non-finite floats (`NaN`, `±inf`) are `null`;
- complex values (`TFORM` `C` / `M`) are objects `{"re": 1.0, "im": -2.0}`
  (a non-finite part is `null`); the schema lists them as `complex64` /
  `complex128` (Arrow has no complex type, so these columns are read with
  `torchfits.table.read_torch`);
- strings decode as ASCII like `torchfits.table.read`, but a byte that does
  not decode becomes U+FFFD (`"\ufffd"`) instead of failing the listing;
  any raw `bytes` value is emitted as its UTF-8 decoding with the same
  replacement.

---

### `cutout`

Extracts a sub-region from an image HDU and writes it to a new FITS file. The output header is a copy of the source header with the reference pixel moved to the cutout origin: `CRPIX1` / `CRPIX2` (and alternate-WCS `CRPIX1a` / `CRPIX2a`) are shifted by the cutout's offset in both coordinate formats, so the stamp stays on the sky. Other WCS cards are copied unchanged. `DATAMIN` / `DATAMAX` describe the full source image and are dropped.

Supports two coordinate formats:

1. **CFITSIO Section Syntax (1-based inclusive):**
   ```bash
   torchfits cutout 'science.fits[101:256,101:256]' cutout.fits
   # A section of extension 1: put the extension in the path
   torchfits cutout 'science.fits[1][101:256,101:256]' cutout.fits
   ```
   A bare section always applies to the primary HDU, so combining it with `-e N` (N ≠ 0) is a usage error (exit 2).
2. **Bounding Box Syntax (0-based half-open `x1,y1,x2,y2`):**
   ```bash
   torchfits cutout science.fits -o cutout.fits -e 0 --box 100,100,256,256
   ```

`--box` values must be integers with non-negative `x1,y1` and `x1 < x2`, `y1 < y2` (0-based, half-open); an empty, inverted, or negative-origin box is a usage error (exit 2). A box that extends past the image edge is clamped to the image, so the output region is the box intersected with the image. A box that misses the image entirely — `x1 >= NAXIS1` or `y1 >= NAXIS2` — selects no pixels and is also a usage error (exit 2) rather than a silently written 0x0 image. On 3D+ cubes the box addresses the trailing `(y, x)` axes, so it is checked against `NAXIS2 x NAXIS1`. For a tile-compressed image HDU the check uses the image size (`ZNAXIS2 x ZNAXIS1`), not the tile table's `NAXIS*`.

```bash
# Batch extract cutouts across multiple files
torchfits cutout *.fits --box 100,100,256,256 --out-dir /tmp/cutouts -J 0
```

---

### `convert`

Converts FITS binary/ASCII tables to modern data science formats (Parquet, CSV, TSV, Arrow IPC) with optional in-engine row filtering, or renders 1–7 band RGB PNG images.

#### Table Conversion & Filtering
```bash
# Convert to Apache Parquet (format inferred from .parquet suffix)
torchfits convert catalog.fits catalog.parquet -e 1

# Convert to CSV selecting specific columns
torchfits convert catalog.fits -o catalog.csv -e 1 -c RA,DEC,MAG_G

# Pushdown row filtering (STILTS-style SQL predicate)
torchfits convert catalog.fits -o filtered.parquet -e 1 -w "MAG_G < 19.5 AND DEC > 0" -c RA,DEC,MAG_G
```

`--to fits` (or a `.fits` OUTPUT) copies the table HDU in its stored form:
column formats (`TFORMn`), `TSCALn`/`TZEROn`, `TNULLn` sentinels, `TDIMn`,
`TUNITn` and other per-column cards (renumbered for `--columns`), `EXTNAME`,
and the HDU's other non-structural keywords are kept.

Output is streamed batch by batch, with or without `--columns`/`--where`, so
memory stays bounded by the batch size. FITS output copies the stored row bytes
of the selected columns chunk by chunk through `torchfits.table.writer()`
(every cell byte-identical); a selection that includes a variable-length
(`P`/`Q`) column is the exception and is rewritten from the table read into
memory.

An unknown `--columns` name or an invalid `--where` predicate (unknown column,
syntax error, type mismatch) is a usage error (exit `2`), as is an `-e` index
the file does not have; an unreadable file or a corrupt HDU stays an I/O error
(exit `3`).

CSV/TSV cells hold one value each: a selection that includes a vector
(`repeat > 1`, `TDIMn`) or variable-length (`P`/`Q`) column is refused as a
usage error (exit `2`) naming those columns, before the output is touched.
Select scalar columns with `--columns`, or convert to Parquet or Arrow, which
store array cells natively.

An OUTPUT that names the same file as an INPUT — the same path, a symlink, or
a hard link — is refused (exit `2`) before anything is written.

#### RGB PNG Rendering
```bash
# Auto RGB (default): files in blue → red order
torchfits convert g.fits r.fits i.fits -o preview.png --to png

# Four filters, photometric zeropoints (AB mag of 1 count)
torchfits convert u.fits g.fits i.fits z.fits -o preview.png --zeropoints 30,30,30,30

# Astropy-parity Lupton (reddest file first; same as 1.0)
torchfits convert r.fits g.fits b.fits -o preview.png --to png --recipe lupton --q 8.0 --stretch 0.5

# Cube or MEF: HDU list is shortest → longest
torchfits convert multi_band.fits -o preview.png --bands 0,1,2 --to png

# Keep array row 0 at the top of the PNG instead of the bottom
torchfits convert g.fits r.fits i.fits -o preview.png --origin upper
```

`--origin {lower,upper}` sets the PNG row order. `lower` (default) puts FITS
pixel (1,1) at the bottom-left, so standard WCS images display north up as in
ds9; `upper` puts array row 0 on top.

With several input files, `--bands` is one HDU index **per input file** and its length must match the number of inputs. A band read that fails names the file and HDU it was reading (for example `g.fits: HDU 1: Could not move to HDU`), so a bad index is attributable.

Table-only flags (`--hdu`, `--where`, `--columns`) are rejected for PNG
output. PNG-only flags (`--bands`, `--recipe`, `--brightness`, `--saturation`,
`--calibrated`, `--zeropoints`, `--q`, `--stretch`) are rejected for table
output instead of being silently ignored.

---

### `copy`

Performs an exact byte-for-byte copy of a FITS file (`shutil.copy2` locally).
CompImage tiles stay compressed. Same-path `INPUT OUTPUT` is refused
(exit `2`), including an OUTPUT that is a hard link to INPUT.
Private/loopback `http`/`https`/`ftp` URLs are blocked; public `http`/`https`
are fetched with the same redirect checks as `probe`.

```bash
# Copy single file
torchfits copy science.fits backup.fits

# Batch copy multiple files
torchfits copy *.fits --out-dir /backup/fits -J 0
```

---

### `arith`

Performs scalar and image-to-image arithmetic (`add`, `sub`, `mul`, `div`). Same-shape HDUs are stacked for accelerated PyTorch vector execution.
`--hdu2` is valid only with an image operand B. `--split hdu` requires
`--out-dir` and rejects `--out`. Every generated destination is checked before
processing so it cannot overwrite either input operand.
The output keeps operand A's header except `DATAMIN` / `DATAMAX`, which no
longer describe the result and are dropped (as `transform` does).
`BUNIT` is handled conservatively (torchfits does no unit algebra): `add` /
`sub` keep it, and adding or subtracting two images whose `BUNIT` values are
both set and differ is a usage error (exit `2`); `mul` / `div` by an image, or
by a scalar other than 1, drop it.

| Flag | Purpose |
|---|---|
| `--value NUMBER` | Scalar operand B; an integer literal (`--value 9223372036854775807`) is kept exact |
| `--operand2 PATH` | Image operand B; required for several A inputs with `--out-dir`, otherwise an alternative to the second positional path |
| `--hdu2 N` | HDU of image operand B (default: the same index as each A HDU, else 0) |
| `--dtype {auto,float32,float64}` | `auto` (default) keeps each A HDU's dtype (`div` on an integer A gives float64); integer results are saturated to the dtype's range and fractional parts truncated, with a warning. `float32` / `float64` cast both operands to that dtype, compute, and store in it |

```bash
# Add scalar value
torchfits arith science.fits --op add --value 50.0 -o offset.fits

# Multiply image by calibration factor
torchfits arith science.fits --op mul --value 1.25 -o scaled.fits

# Image-to-image subtraction (science - dark)
torchfits arith science.fits dark.fits --op sub -o dark_subtracted.fits

# Batch process multiple files against a shared scalar
torchfits arith *.fits --op mul --value 2.0 --out-dir /tmp/scaled -J 0
```

Integer `add` / `sub` / `mul` are exact for every integer width, including
64-bit (`int64`, `uint64`) images and operands: results outside the output
dtype saturate to its minimum or maximum with a `RuntimeWarning` naming how
many pixels were clamped, instead of wrapping or being rounded through
float64. A fractional scalar or a floating-point image B on an integer A (with
`--dtype auto`) is applied exactly too: each pixel's exact result is truncated
toward zero (warning how many pixels lost a fraction) and saturated to A's
dtype, so 64-bit pixels near `2**63` are never rounded through float64.
Non-finite results into an integer output: `+inf` / `-inf` (an infinite
operand) saturate to the dtype's maximum / minimum; a NaN result (a NaN
operand, or `0 * inf`) has no integer value and is refused as a usage error
(exit `2`, nothing written) — use `--dtype float32/float64` to keep it. (An
integer A with a `BLANK` keyword already reads as floating point, with NaN at
its blank pixels, so its results are floating point.)

---

### `compress`

Tile-compresses image HDUs using CFITSIO's compression architecture. Preserves non-image HDUs and table extensions.

The default `GZIP_1` codec is lossless for integer and floating-point images.
Also available: `RICE_1`, `GZIP_2`, `HCOMPRESS_1`, and `PLIO_1`; `BZIP2_1` is
listed when the build includes bzip2. The non-default codecs use CFITSIO float
quantization and are lossy for floating-point images (tune it with
`--quantize-level`, or pass `-q 0` with `GZIP_2` for lossless floats);
`PLIO_1` rejects floats. Integer images are always compressed losslessly:
quantization flags on an input file with no floating-point image HDU are a
usage error (exit `2`) unless `--int-to-float` is given, and in a mixed file
they apply to its float HDUs only.
No codec handles 64-bit integer images (`BITPIX = 64`): `compress` reports
`tile compression does not support 64-bit integer images` and exits `3`
without writing the output.
`--hdu` applies only with `--split hdu`.

| Flag | Default | Description |
|---|---|---|
| `--algorithm` | `GZIP_1` | Codec: `GZIP_1`, `GZIP_2`, `RICE_1`, `HCOMPRESS_1`, `PLIO_1` (and `BZIP2_1` when built with bzip2) |
| `-t`, `--tile N1,N2,...` | one image row per tile (16 rows for `HCOMPRESS_1`) | Tile size per axis in **FITS axis order** (`NAXIS1` first), like `fpack -t`. Axes left out get size 1; `-1` (or a size past the axis length) means the whole axis — write it as `--tile=-1,-1`. `HCOMPRESS_1` tiles must span exactly two axes of at least 4 pixels. A tile with more axes than an image is a usage error (exit `2`). |
| `-q`, `--quantize-level Q` | lossless for `GZIP_1`, `4` otherwise | Quantization of floating-point images, like `fpack -q`: `Q > 0` makes the step the tile's noise sigma divided by `Q` (larger = finer, bigger files); `Q < 0` is an absolute step of `abs(Q)`; `0` is lossless and needs `GZIP_1` or `GZIP_2`. |
| `--quantize-method` | `subtractive_dither_1` | `subtractive_dither_1`, `subtractive_dither_2` (also keeps exact zeros; not with `HCOMPRESS_1`) or `no_dither` (default level `16`). |
| `--dither-seed N` | picked from the clock | Dither seed `1..10000`; fixes the quantization so repeated runs write identical files. |
| `--int-to-float` | off | Compress integer images as quantized floats, like `fpack -i2f` (lossy; default level `4` even with `GZIP_1`). Physical values (`BSCALE`/`BZERO` applied, `BLANK` pixels as NaN) are stored, so `decompress` writes them as `BITPIX = -32`. Not with `PLIO_1` or `-q 0`. |

```bash
# Compress losslessly with default GZIP_1
torchfits compress science.fits science.fits.fz

# Compress using GZIP algorithm
torchfits compress science.fits science.fits.fz --algorithm GZIP_1

# 100x100-pixel tiles; one tile for the whole image
torchfits compress science.fits science.fits.fz --algorithm RICE_1 --tile 100,100
torchfits compress science.fits science.fits.fz --tile=-1,-1

# fpack-like lossy float compression: finer quantization, reproducible dither
torchfits compress science.fits science.fits.fz --algorithm RICE_1 -q 16 --dither-seed 42

# Absolute quantization step of 0.01, zeros kept exact
torchfits compress science.fits science.fits.fz --algorithm RICE_1 -q -0.01 --quantize-method subtractive_dither_2

# Lossless floats with the byte-shuffled GZIP_2
torchfits compress science.fits science.fits.fz --algorithm GZIP_2 -q 0

# fpack -i2f: quantize an integer (e.g. raw counts) image for higher compression
torchfits compress raw.fits raw.fits.fz --algorithm RICE_1 --int-to-float -q 4

# Split MEF file into individual compressed HDU files
torchfits compress mef.fits --split hdu --out-dir /tmp/compressed_hdus

# Batch compress all files in a directory in parallel
torchfits compress *.fits --out-dir /tmp/compressed -J 0
```

The same controls are keyword arguments of [`torchfits.write()`](api-core-io.md#write)
(`tile_shape=`, `quantize_level=`, `quantize_method=`, `dither_seed=`,
`int_to_float=`); note that `tile_shape` is in tensor axis order (rows first)
while `--tile` is in FITS order. Runnable side by side:
[`example_fpack_compression.py`](published-examples/example_fpack_compression.py).

---

### `decompress`

Expands tile-compressed `.fits.fz` files back into standard uncompressed FITS files.

A FITS primary HDU cannot hold a compressed image, so `compress` moves a
primary image into HDU 1 behind an empty primary and marks it with `ZSIMPLE = T`
(as `fpack` does). `decompress` restores such an image as the primary, like
`funpack`, so HDU indices round-trip; files whose primary was already empty
keep it. Compression bookkeeping cards (`ZSIMPLE`, `ZTENSION`, `ZPCOUNT`, …)
are not copied into the uncompressed output.

```bash
# Decompress single file
torchfits decompress science.fits.fz science.fits

# Batch decompress in parallel
torchfits decompress *.fits.fz --out-dir /tmp/uncompressed -J 0
```

---

### `transform`

Applies a registered astronomical normalization or stretch transform directly
from the shell. Supported names are `AffineTransform`, `ArcsinhStretch`,
`AsymmetricSigmaClip`, `BackgroundSubtract`, `GlobalScalarNorm`,
`InterquantileNormalize`, `InterquantileScale`, `LogStretch`,
`MeshBackgroundSubtract`, `MinMaxNormalize`, `PercentileClipNormalize`,
`RobustNormalize`, `SigmaClip`, `SigmaNormalize`, `SqrtStretch`, and
`ZScaleNormalize`. Constructor values after `:` are parsed as booleans,
integers, floats, or strings; duplicate and unknown names are usage errors.

```bash
# Apply Arcsinh stretch
torchfits transform science.fits --name ArcsinhStretch -o stretched.fits

# Apply ZScale normalization
torchfits transform science.fits --name ZScaleNormalize -o zscale.fits

# Pass custom parameters via Name:key=val syntax
torchfits transform science.fits --name PercentileClipNormalize:lower_pct=2.0,upper_pct=98.0 -o clipped.fits
torchfits transform science.fits --name ArcsinhStretch:a=0.05 -o custom_arcsinh.fits

# Batch transform multiple files
torchfits transform *.fits --name LogStretch --out-dir /tmp/log_images -J 0
```

The output keeps the source header except storage and range cards (`BSCALE`,
`BZERO`, `BLANK`, `DATAMIN`, `DATAMAX`, checksums). `BUNIT` is kept only by
transforms whose output stays in the input unit — `BackgroundSubtract`,
`MeshBackgroundSubtract`, `SigmaClip`, `AsymmetricSigmaClip`, and
`AffineTransform` with `scale=1`; every normalization or stretch drops it.

---

### `setkey`

Inserts, updates, renames, or deletes header keywords in place via CFITSIO card updates. Preserves tile compression on compressed files.

Structural keywords describe the data layout and cannot be set, renamed, or
deleted: `SIMPLE`, `XTENSION`, `BITPIX`, `NAXIS`, `NAXISn`, `PCOUNT`, `GCOUNT`,
`TFIELDS`, `TFORMn`, `TBCOLn`, `EXTEND`, `THEAP`, and, on tile-compressed HDUs,
the compression cards (`ZIMAGE`, `ZBITPIX`, `ZNAXISn`, `ZTILEn`, `ZCMPTYPE`,
`ZNAMEn`, `ZVALn`, `ZQUANTIZ`, …). Such an edit is a usage error (exit `2`)
and leaves the file untouched.

An edited HDU that already carries `CHECKSUM`/`DATASUM` has both recomputed
after the edit, so `torchfits verify` keeps passing; HDUs without them do not
gain them. An edit that names `CHECKSUM` or `DATASUM` itself is applied as
written.

With `-o` / `--out-dir` the edit is validated against the input before the
copy is made, so a rejected edit (missing keyword, structural keyword, …)
leaves no output file behind.

!!! note "Card comments"
    `setkey` writes the value only; it cannot set a card's comment text. Use a
    full header edit (`torchfits header`, or `HDUList` / `Header` in Python)
    when the card's comment matters.

```bash
# Set or update keyword
torchfits setkey science.fits -k OBJECT --value "NGC 1234"

# Values are coerced: true/false become booleans, plain numeric literals
# (12, -3, 300.0, 1.5e3) become int or float; anything else stays a string
torchfits setkey science.fits -k OKFLAG --value true
torchfits setkey science.fits -k EXPTIME --value 300.0
# A leading-zero ID such as 007 stays the string '007'
torchfits setkey science.fits -k OBSID --value 007

# Set HIERARCH / long keyword (--delete and --rename accept the same spelling)
torchfits setkey science.fits -k "ESO DET CHIP1 ID" --value "42"
torchfits setkey science.fits --delete "ESO DET CHIP1 ID"

# Apply across all HDUs
torchfits setkey science.fits -k OBSERVER --value "astronomer" -e all

# Rename a keyword card
torchfits setkey science.fits --rename OBJECT=TARGET

# Delete a keyword card
torchfits setkey science.fits --delete TEMPKEY

# Batch update files from a text file list (@list syntax)
torchfits setkey @file_list.txt -k PROCESSED --value "TRUE" --out-dir /tmp/updated -J 0

# Standard input may be the only path source
printf '%s\n' a.fits b.fits | torchfits setkey --stdin -k PROCESSED --value true
```

---

### `probe`

Inspects local files or peeks at remote FITS headers over HTTP(S) and VOSpace using HTTP Range requests without downloading the full file.
`--hdu` is local-only. `--header-bytes` and `--timeout` are remote-only;
`--header-bytes` must be between 2880 and 16777216. A remote response that does
not contain a complete FITS header ending in an `END` card within that limit is
an I/O error rather than a partial result.

```bash
# Probe local file
torchfits probe science.fits

# Peek at remote HTTP header (fetches only 5760 header bytes)
torchfits probe https://example.edu/survey/galaxy.fits --header-bytes 5760 --timeout 15 -f json

# Probe CADC VOSpace URI
torchfits probe vos:username/data/sample.fits
```

---

## Familiar-Tool Mapping

| torchfits command | Closest classic tools | Purpose & Description |
|---|---|---|
| `info` | [`fitsinfo`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsinfo) (Astropy) | Overview of HDU extensions, dimensions, and data types |
| `header` | [`fitsheader`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsheader) (Astropy), [`dfits` / `fitsort`](https://www.eso.org/sci/software/eclipse/eug/eug/node13.html) (ESO Eclipse) | Dump header cards; filter by keyword; multi-file summary tables |
| `verify` | [`fitscheck`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitscheck) (Astropy), [`fitsverify`](https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/) (NASA HEASARC) | Check `DATASUM` and `CHECKSUM` keyword integrity |
| `diff` | [`fitsdiff`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsdiff) (Astropy) | Compare headers, image pixels, and table data between two FITS files |
| `stats` | [`imstat`](https://iraf.net/irafdocs/imstat.php) (IRAF), [`aststatistics`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-aststatistics.html) (Gnuastro) | Compute min, max, mean, standard deviation, and median pixel values |
| `table` | [`asttable`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-asttable.html) (Gnuastro), `tablist` (NASA HEASARC FTOOLS) | Preview binary/ASCII table schema and row values |
| `cutout` | [`astcrop`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-astcrop.html) (Gnuastro), CFITSIO image sections | Extract sub-regions using pixel ranges (`[x1:x2, y1:y2]`) or bounding boxes |
| `convert` | [`astconvertt`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-astconvertt.html) (Gnuastro), [`STILTS`](https://www.star.bristol.ac.uk/~mbt/stilts/) (Starlink) | Export tables to Parquet/CSV/Arrow with SQL filters; render 1–7 band RGB PNGs |
| `copy` | [`fitscopy` / `imcopy`](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) (CFITSIO) | Byte-identical copy (not an HDU rewrite) |
| `arith` | [`imarith`](https://iraf.net/irafdocs/imarith.php) (IRAF) | Perform scalar or image-to-image addition, subtraction, multiplication, division |
| `compress` / `decompress` | [`fpack` / `funpack`](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) (NASA HEASARC) | Lossless / lossy tile compression (Rice, Gzip, Hcompress) and expansion |
| `transform` | [`imfunction`](https://iraf.net/irafdocs/imfunction.php) (IRAF) | Apply astronomical stretches (Arcsinh, Sqrt, Log) and ZScale / percentile scaling |
| `setkey` | [`hedit`](https://iraf.net/irafdocs/hedit.php) (IRAF), `modhead` (WCSTools) | Insert, update, rename, or delete header keywords across single or multiple files |
| `probe` | HTTP Range peek | Inspect remote FITS headers over HTTP(S) or VOSpace without downloading whole files |

For practical examples, see [CLI recipes](cli-recipes.md).

## Scripting notes

- No prompts; stable exit codes.
- Every output file (FITS, Parquet, CSV/TSV, Arrow, PNG, `copy`, `setkey -o`) is
  written to a hidden temp file beside it and renamed into place only when
  complete, so a failed command leaves an existing OUTPUT unchanged and no
  partial file behind.
- Prefer `-f json` / `jsonl` (or `--json` / `--jsonl`) for automation.
- GPU tensors are staged through host memory before any FITS write (same as the
  Python API — not GPUDirect).
