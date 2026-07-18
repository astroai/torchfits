---
template: home.html
title: FITS I/O for PyTorch
---

<div class="tf-below" markdown>

## Browse

<ul class="tf-paths" markdown>
<li markdown>
[Install](install.md)

Wheels, source builds, GPU / accelerator notes.
</li>
<li markdown>
[Quick start](quickstart.md)

`read_tensor`, table filters, DataLoader.
</li>
<li markdown>
[CLI](cli.md)

`info`, `header`, `verify`, `cutout`, …
</li>
<li markdown>
[API](api.md)

Core I/O, tables, datasets, transforms.
</li>
<li markdown>
[Examples](examples.md)

Runnable scripts and transform galleries.
</li>
<li markdown>
[Benchmarks](benchmarks.md)

Methodology and scorecards.
</li>
<li markdown>
[Migration](migration_astropy.md)

From Astropy / fitsio.
</li>
<li markdown>
[Parity](parity.md)

What torchfits covers today.
</li>
</ul>

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
torchfits header image.fits -k OBJECT -f json
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

Docs channels: [stable](https://astroai.github.io/torchfits/) (latest `v*` tag) ·
[edge](https://astroai.github.io/torchfits/edge/) (`main` tip).

</div>
