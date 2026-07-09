# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)

[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green)](LICENSE)

**torchfits** is a focused FITS I/O library for PyTorch. It reads and writes
FITS images, headers, HDUs, compressed images, and FITS tables through a
multi-threaded C++ engine with vendored CFITSIO, returning tensor-native data
without requiring users to build NumPy-to-torch glue code.

It is not a full replacement for Astropy, fitsio, or CFITSIO. The supported
surface is documented explicitly in [docs/parity.md](docs/parity.md), with
source-backed tests for each claimed parity area. WCS, sky-coordinate models,
HEALPix, sphere geometry, and sky-domain simulation workflows are out of scope
for torchfits.

## At a Glance

| Task | Traditional stack | torchfits equivalent |
|---|---|---|
| Read image to GPU | astropy/fitsio &rarr; numpy &rarr; torch &rarr; `.to(device)` | `torchfits.read("img.fits", device="cuda")` |
| Write tensor to FITS | tensor &rarr; numpy &rarr; astropy HDU &rarr; writeto | `torchfits.write("out.fits", tensor)` |
| Filter large table | load all rows &rarr; mask in Python | `where="MAG < 20"` pushdown in C++ |
| Read multi-extension files | manual HDU dispatch | `with torchfits.open("mef.fits") as hdul: ...` |
| Verify FITS checksums | comparator-specific helpers | `torchfits.verify_checksums(path)` |

## Features

**FITS I/O** &mdash; Multi-threaded C++ core with SIMD-optimized type conversion,
memory-mapped image reads, intelligent chunking, and adaptive buffering. Reads
and writes images, binary/ASCII tables, compressed images, and multi-extension
FITS files with header round-trip coverage.

**Table Engine** &mdash; Arrow-native table API with predicate pushdown (`where=`), column projection, row slicing, streaming `scan()`, and in-place mutations (append, insert, update, delete rows and columns). Interop with Pandas, Polars, DuckDB, and PyArrow.

**Compatibility Contract** &mdash; Parity is tracked by tier: truthful public docs,
fitsio core workflow parity, Astropy common workflow parity, selected CFITSIO
backend behavior, and explicit non-goals. See [docs/parity.md](docs/parity.md).

## What's New in 0.6.0

0.6.0 is a focused FITS I/O release with a maintainable Python/C++ core,
first-class `torch.utils.data` integration, and header-aware preprocessing.
Key improvements:

- **C++ engine hardening** — Rule-of-5 fixes, RAII guards, unified BITPIX mapping.
- **Predicate filter C++ pushdown** — all table sizes use C++ for `where=` filtering.
- **Lightweight is_compressed check** — O(1) header probe for compressed images.
- **Thread-safe caches** — `std::shared_mutex` for concurrent multi-worker access.
- **torchfits.data module** — `FitsImageDataset`, `make_loader` with automatic cache tuning.
- **25+ ML-friendly transforms** — all with `.inverse()` for model output decoding.
- **7 deficits remaining** (down from 22) in the lab exhaustive benchmark suite.

Full history: [docs/changelog.md](docs/changelog.md). Roadmap for 0.6.x and beyond: [docs/roadmap.md](docs/roadmap.md).

## Transforms

torchfits includes over 20 ML-friendly FITS data transforms — all with `.inverse()`
for decoding model outputs back to physical units.  See [docs/api.md](docs/api.md)
for full signatures and the example scripts in `examples/`.

```python
from torchfits import ArcsinhStretch, BackgroundSubtract, Compose, ZScaleNormalize

pipeline = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1), ZScaleNormalize()])
normalized = pipeline(image)     # forward → model input
restored  = pipeline.inverse(normalized)  # inverse → physical flux
```

### Image stretches & normalizers (invertible)

| Transform | Description |
|---|---|
| `ArcsinhStretch(a)` | LSST/SDSS standard for high-DR images |
| `LogStretch(a, eps)` | Logarithmic stretch, negatives clamped |
| `SqrtStretch()` | Poisson variance-stabilising |
| `ZScaleNormalize(contrast, dim)` | IRAF auto-contrast → [0, 1] |
| `RobustNormalize(dim)` | Median + MAD standardization (P1) |
| `BackgroundSubtract(dim)` | Subtract median background |
| `PercentileClipNormalize(lower, upper, dim)` | Percentile-clip → [0, 1] |
| `MinMaxNormalize(dim)` | Min-max → [0, 1] |
| `GlobalScalarNorm(stat, dim)` | Divide by median/max/mean/rms (P5) |

### Spectral & hyperspectral (astronomy-specific)

| Transform | Description |
|---|---|
| `ContinuumNormalize(order, n_sigma)` | Fit continuum + **divide** by it. Inverse multiplies back. |
| `ContinuumRemoval(method, order, n_knots)` | Fit continuum + **subtract** it. Inverse adds back. |
| `DopplerShift(z)` | Redshift/blueshift via resampling. Inverse is opposite shift. |
| `SpectralBinning(factor, mode, dim)` | Bin adjacent channels. Inverse nearest-neighbour upsamples. |
| `BandMath(func, band_dim)` | Band ratios (NDVI etc.) via `unbind`. Inverse not available. |

### Continuum / baseline estimators (additive decomposition)

All use `Original = Estimate + Residuals` for perfect recovery.
Based on post-2021 astro-ML research (SUPPNet, RASSINE, AstroCLIP).

| Transform | Description |
|---|---|
| `AsymmetricLeastSquares(lam, p, max_iter)` | Eilers 2003 penalised baseline (Raman/NIR) |
| `AlphaShapeContinuum(half_window, iterations)` | Morphological closing — guaranteed upper envelope |
| `SavitzkyGolayFilter(window, polyorder)` | Polynomial smoothing (P4) |
| `RunningPercentile(percentile, window)` | Sliding-window percentile continuum (P6) |
| `UpperEnvelopeContinuum(window, smooth)` | Local-max interpolation (RASSINE-like) (P3) |
| `WaveletDecompose(levels)` | Multi-level Haar DWT frequency split (P2) |

### Time-domain & meta

| Transform | Description |
|---|---|
| `PhaseFold(period, n_bins)` | Fold time series into phase bins |
| `FITSHeaderScale(bscale, bzero)` | Apply/remove BSCALE/BZERO |
| `FITSHeaderNormalize(header)` | Auto-normalize from BITPIX |
| `SigmaClip(n_sigma, max_iter, dim)` | Iterative outlier rejection |
| `AsymmetricSigmaClip(n_low, n_high, dim)` | One-pass asymmetric sigma-clip (median+MAD) |
| `Compose(transforms)` | Chain transforms; inverse unwinds in reverse |

## Performance

Median wall-clock from the lab exhaustive benchmark suite (`exhaustive_mmap_0.5.0b4_20260630_162835`, H100 CUDA). See [docs/benchmarks.md](docs/benchmarks.md) for methodology, deficit transparency, and reproducible commands.

| Case | torchfits | astropy | Speedup |
|---|---:|---:|---:|
| Large float32 image read (16 MB, CPU) | 4.89 ms | 15.66 ms | **3.3×** |
| Same read @ CUDA | 3.26 ms | 15.46 ms | **4.7×** |
| Compressed Rice image (CPU) | 9.22 ms | 28.70 ms | **3.1×** |
| 50× repeated 100×100 cutouts (CPU) | 6.34 ms | 80.25 ms | **13.3×** |
| Table read (100k rows, 8 cols) | 102 μs | 6.37 ms | **62×** |

**ML DataLoader (local diagnostic, 30×512² float32, CPU, 2 epochs):** torchfits **1.12×** vs fitsio on Rice-compressed files; uncompressed within ~4% (handle-cache tuning matters — call `torchfits.cache.optimize_for_dataset` before training loops).

**GPU integer reads:** Default `read(..., device="cuda")` applies BSCALE/BZERO on device and returns `float32` for generic scaled pixels — good for ML. For native integer dtypes (int8, uint16) matching fitsio, use `read_tensor(..., raw_scale=True)` or rely on the automatic signed-byte / unsigned-integer fast paths (see benchmarks doc). Tables remain CPU-resident in all backends; GPU rows measure host decode + H2D copy, not disk→GPU bypass.

## Install

```bash
pip install torchfits
```

Pre-built wheels are available for Linux and macOS (x86_64, arm64). No system CFITSIO needed&mdash;it's vendored and compiled automatically.

From source:

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
pip install -e .
```

Requires Python 3.10+, a C++17 compiler, CMake 3.21+, and PyTorch 2.0+.

## Quick Start

### Read an image to GPU

```python
import torchfits

data, header = torchfits.read("science.fits", device="cuda", return_header=True)
# data: torch.Tensor on CUDA, shape e.g. (4096, 4096), dtype torch.float32
```

### Filter and stream a catalog

```python
# Predicate pushdown — only matching rows leave C++
table = torchfits.table.read(
    "catalog.fits",
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.9",
)
# table: pyarrow.Table

# Stream 100M rows in constant memory
for batch in torchfits.table.scan("survey.fits", batch_size=50_000):
    process(batch)  # batch: pyarrow.RecordBatch
```

### Multi-HDU access

```python
with torchfits.open("multi_ext.fits") as hdul:
    print(hdul)            # pretty-printed summary
    img = hdul[0].data     # image tensor
    tbl = hdul[1].data     # dict-like table accessor
    tbl_filtered = hdul[1].filter("FLUX > 100 AND FLAG = 0")
```

### Write back

```python
torchfits.write("output.fits", data, header=header, overwrite=True)
# table_dict is a dict of column names to 1D arrays/tensors
torchfits.table.write("catalog_out.fits", table_dict, header=header, overwrite=True)
```

## Benchmarks

torchfits benchmark evidence is limited to FITS image I/O and FITS table I/O.
Comparators are `astropy.io.fits` and `fitsio`; selected CFITSIO behavior is
validated through the torchfits native backend and smoke tests.

Methodology, reproducible commands, results, and known deficits: [`docs/benchmarks.md`](docs/benchmarks.md)

## Documentation

| | |
|---|---|
| [API Reference](docs/api.md) | Full public API with signatures and examples |
| [Roadmap](docs/roadmap.md) | FITS I/O roadmap and parity tiers |
| [Parity Matrix](docs/parity.md) | Supported, partial, unsupported, and out-of-scope features |
| [Examples](docs/examples.md) | Runnable scripts for every major workflow |
| [Installation](docs/install.md) | Build from source, GPU setup, troubleshooting |
| [Benchmarks](docs/benchmarks.md) | Methodology, commands, and latest numbers |
| [Changelog](docs/changelog.md) | Version history and migration notes |
| [Release Checklist](docs/release.md) | Maintainer guide for cutting releases |

## Contributing

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
pixi install
pixi run test
```

The project uses [pixi](https://pixi.sh) for environment management, [ruff](https://github.com/astral-sh/ruff) for linting, and [pytest](https://docs.pytest.org) for testing.

## License

[GPL-2.0](LICENSE)
