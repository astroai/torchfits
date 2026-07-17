---
template: home.html
---

# Welcome to torchfits

**FITS I/O for PyTorch** — read FITS images, tables, and headers directly as
tensors. A multi-threaded C++ engine with vendored CFITSIO delivers
significantly faster I/O than astropy or fitsio, with native GPU placement,
column and row filtering at C++ speed, and ML-ready datasets and transforms.
A `torchfits` CLI covers common inspect / convert jobs from the shell.

??? info "New to FITS?"
    **FITS** (Flexible Image Transport System) is the standard file format in
    astronomy. A FITS file contains one or more **HDUs** (Header Data Units) —
    think of them as numbered sections, each with a header (metadata key-value
    pairs) and data (an image array or a table). HDU 0 is usually the primary
    image; extension HDUs (1, 2, …) hold tables or additional images. Some files
    use **EXTNAME** labels (like `'SCI'` or `'EVENTS'`) to name extensions.
    torchfits reads these natively — no conversion needed.

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Get Started in 30 Seconds__

    ---

    Install with pip, read your first FITS file as a tensor.

    [:octicons-arrow-right-24: Installation](install.md)

-   :material-console:{ .lg .middle } __Command-Line Tools__

    ---

    `torchfits info`, `header`, `verify`, `stats`, `cutout`, and more.

    [:octicons-arrow-right-24: CLI guide](cli.md)

-   :material-image-multiple:{ .lg .middle } __Read FITS Images__

    ---

    `read_tensor` loads any N-D FITS image directly to CPU, CUDA, or MPS.

    [:octicons-arrow-right-24: Core I/O Reference](api-core-io.md)

-   :material-table-large:{ .lg .middle } __Query Tables at C++ Speed__

    ---

    Filter tables at C++ speed — often tens of times faster than astropy.

    [:octicons-arrow-right-24: Table Reference](api-tables.md)

-   :material-brain:{ .lg .middle } __Train with PyTorch__

    ---

    `FitsImageDataset`, `make_loader`, and transforms for ML pipelines.

    [:octicons-arrow-right-24: Data & Transforms](api-data.md)

</div>

---

## At a Glance

```python
import torchfits

# Read an image directly as a tensor
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cuda")

# Filter a table at C++ speed (column + row filtering)
table = torchfits.table.read("catalog.fits", hdu=1, where="MAG_G < 20")

# Train a model
from torchfits.data import FitsImageDataset, make_loader
loader = make_loader(FitsImageDataset("images/*.fits"), batch_size=32, num_workers=4)
```

```bash
torchfits info image.fits
torchfits header image.fits --keyword OBJECT --json
```

## Why torchfits?

| | astropy / fitsio | torchfits |
|---|---|---|
| **Image read (16 MB, CPU)** | 16.67 ms | **3.85 ms** (4.3x faster) |
| **Table read (100k rows)** | 6.74 ms | **95 us** (70x faster) |
| **Repeated cutouts (50x)** | 75.36 ms | **4.68 ms** (16x faster) |
| **GPU placement** | manual `.to(device)` | native `device="cuda"` |
| **Table filtering** | Python mask | C++ pushdown |
| **PyTorch Dataset** | hand-roll | built-in |
| **Shell tooling** | fitsinfo / fitsheader / … | `torchfits` CLI |

## I Want To...

| Goal | Start with |
|---|---|
| Read a FITS image as a tensor | [`read_tensor`](api-core-io.md#read_tensor) |
| Inspect a file from the shell | [CLI `info` / `header`](cli.md) |
| Read a table with SQL-like filters | [`table.read(..., where=...)`](api-tables.md#tableread) |
| Build a DataLoader for training | [`FitsImageDataset` + `make_loader`](api-data.md) |
| Stream a huge table in constant memory | [`FitsTableIterableDataset`](api-data.md#fitstableiterabledataset) |
| Apply FITS-aware preprocessing | [`torchfits.transforms`](api-transforms.md) |
| Migrate from astropy | [Side-by-side migration guide](migration_astropy.md) |
| Migrate from fitsio | [Side-by-side migration guide](migration_fitsio.md) |
| Check feature support | [Parity matrix](parity.md) |
| See performance numbers | [Benchmarks](benchmarks.md) |

## How the Docs Are Organized

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Getting Started__

    ---

    Installation, quick start, first examples, CLI.

    [:octicons-arrow-right-24: install.md](install.md) · [:octicons-arrow-right-24: quickstart.md](quickstart.md) · [:octicons-arrow-right-24: cli.md](cli.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Core I/O, tables, datasets, transforms — organized by task.

    [:octicons-arrow-right-24: api.md](api.md)

-   :material-book-open-variant:{ .lg .middle } __Guides__

    ---

    Runnable examples, benchmark data, feature parity matrix.

    [:octicons-arrow-right-24: examples.md](examples.md)

-   :material-swap-horizontal:{ .lg .middle } __Migration__

    ---

    Side-by-side replacements from astropy, fitsio, and legacy datasets.

    [:octicons-arrow-right-24: migration_astropy.md](migration_astropy.md)

</div>

## Scope

torchfits is **not** a full Astropy replacement. It owns FITS I/O:
images, tables, headers, compression, checksums, and ML data pipelines.
WCS, sky coordinates, HEALPix, and simulation frameworks belong in
downstream sky-domain packages. See [Parity](parity.md) for the full
compatibility contract.

**FITS standard coverage:** BSCALE/BZERO scaled data, unsigned integer
conventions (BZERO/TZERO), binary and ASCII tables, variable-length array
columns, complex columns, compressed images (RICE_1, GZIP_1, PLIO_1,
HCOMPRESS_1), and FITS checksums are all supported. See
[Parity](parity.md) for the full matrix.
