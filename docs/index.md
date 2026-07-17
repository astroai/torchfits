---
template: home.html
title: FITS I/O for PyTorch
---

<div class="tf-below" markdown>

# torchfits

## Start here

<ul class="tf-paths" markdown>
<li markdown>
[Install](install.md)

Wheels for Linux and macOS. PyTorch 2.10.
</li>
<li markdown>
[Quick start](quickstart.md)

First `read_tensor`, table filter, DataLoader.
</li>
<li markdown>
[CLI](cli.md)

`info`, `header`, `verify`, `cutout`, and more.
</li>
<li markdown>
[API reference](api.md)

I/O, tables, datasets, transforms.
</li>
</ul>

??? info "New to FITS?"
    **FITS** (Flexible Image Transport System) is the standard file format in
    astronomy. A FITS file contains one or more **HDUs** (Header Data Units) —
    each with a header and data (image or table). HDU 0 is usually the primary
    image; higher HDUs hold tables or more images. Some files use **EXTNAME**
    labels such as `'SCI'` or `'EVENTS'`. torchfits reads these natively.

## At a glance

```python
import torchfits

tensor = torchfits.read_tensor("image.fits", hdu=0, device="cuda")
table = torchfits.table.read("catalog.fits", hdu=1, where="MAG_G < 20")

from torchfits.data import FitsImageDataset, make_loader
loader = make_loader(FitsImageDataset("images/*.fits"), batch_size=32)
```

```bash
torchfits info image.fits
torchfits header image.fits --keyword OBJECT --json
```

## Why torchfits?

| | astropy / fitsio | torchfits |
|---|---|---|
| **Image read (16 MB, CPU)** | 16.67 ms | **3.85 ms** (4.3×) |
| **Table read (100k rows)** | 6.74 ms | **95 μs** (70×) |
| **Repeated cutouts (50×)** | 75.36 ms | **4.68 ms** (16×) |
| **GPU placement** | manual `.to(device)` | `device="cuda"` |
| **Table filtering** | Python mask | C++ pushdown |
| **Training loop** | hand-rolled Dataset | `FitsImageDataset` + `make_loader` |
| **Shell tooling** | fitsinfo / fitsheader / … | `torchfits` CLI |

Numbers from the lab scorecard — methodology in [Benchmarks](benchmarks.md).

## I want to…

| Goal | Go to |
|---|---|
| Read a FITS image as a tensor | [`read_tensor`](api-core-io.md#read_tensor) |
| Inspect a file from the shell | [CLI](cli.md) |
| Filter a table in C++ | [`table.read(..., where=...)`](api-tables.md#tableread) |
| Build a DataLoader | [`FitsImageDataset`](api-data.md) |
| Stream a huge table | [`FitsTableIterableDataset`](api-data.md#fitstableiterabledataset) |
| Preprocess for a model | [`torchfits.transforms`](api-transforms.md) |
| Migrate from Astropy | [Migration guide](migration_astropy.md) |
| Migrate from fitsio | [Migration guide](migration_fitsio.md) |
| Check what is supported | [Parity matrix](parity.md) |

## Scope

torchfits owns FITS I/O: images, tables, headers, compression, checksums, and
ML data pipelines. It is **not** a full Astropy replacement — WCS, sky
coordinates, HEALPix, and simulation belong in downstream packages. See
[Parity](parity.md).

Supported conventions include BSCALE/BZERO, unsigned integers, binary/ASCII
tables, variable-length arrays, complex columns, compressed images (RICE_1,
GZIP_1, PLIO_1, HCOMPRESS_1), and FITS checksums.

</div>
