# torchfits

[![PyPI stable](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)
[![GitHub release](https://img.shields.io/github/v/release/astroai/torchfits?include_prereleases&label=latest%20release)](https://github.com/astroai/torchfits/releases)
[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**torchfits** reads and writes FITS as PyTorch tensors and Arrow dataframes.
A multi-threaded C++ engine (vendored CFITSIO) handles images, tables, headers,
compression, and MEF files. Optional datasets, transforms, and a `torchfits`
CLI sit on top.

**PyPI stable:** 0.9.x · **Current pre-release:** 1.0.0rc1.
Docs: [stable](https://astroai.github.io/torchfits/) (latest `v*` tag, may be an
rc) · [edge](https://astroai.github.io/torchfits/edge/) (`main` tip). The docs
“stable” channel is **not** the same as the latest non-prerelease on PyPI.

## Install

```bash
pip install torchfits
```

Requires **Python 3.10+** and **PyTorch 2.10**. Pre-built wheels for Linux
x86_64 and macOS arm64 (CFITSIO is vendored).

## At a Glance

| Task | torchfits |
|---|---|
| Image → GPU tensor | `torchfits.read_tensor("img.fits", device="cuda")` |
| Write a tensor | `torchfits.write("out.fits", tensor)` |
| Filter a catalog in C++ | `table.read(..., where="MAG < 20")` |
| Open a MEF | `with torchfits.open("mef.fits") as hdul: …` |
| Train | `FitsImageDataset` + `make_loader(..., num_workers=4)` |
| Shell | `torchfits info` / `header` / `convert` / … |

## Quick Start

```python
import torchfits

tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")

# FITS tables → dataframes (Arrow by default)
table = torchfits.table.read(
    "catalog.fits",
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0",
)
```

```bash
torchfits info science.fits
torchfits convert catalog.fits out.parquet --hdu 1   # --to optional
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
```

## Learn more

| | |
|---|---|
| [Documentation](https://astroai.github.io/torchfits/) | Quickstart, API, CLI, galleries |
| [Examples](https://astroai.github.io/torchfits/examples/) | Runnable scripts + transform plots |
| [Benchmarks](https://astroai.github.io/torchfits/benchmarks/) | Methodology and scorecards |
| [Changelog](https://astroai.github.io/torchfits/changelog/) | Release notes |

## Contributing

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
pixi install
pixi run test
```

## License

[MIT](LICENSE)
