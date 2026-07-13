# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)

[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**torchfits** is FITS I/O for PyTorch: a multi-threaded C++ engine with vendored
CFITSIO that reads and writes images, headers, HDUs, compressed images, and
tables as tensors — without NumPy-to-torch glue. Optional **`torchfits.data`**
datasets and **`torchfits.transforms`** provide header-aware preprocessing for
ML training loops.

It is not a full replacement for Astropy, fitsio, or CFITSIO. The supported
surface is documented explicitly in [docs/parity.md](docs/parity.md), with
source-backed tests for each claimed parity area. WCS, sky-coordinate models,
HEALPix, sphere geometry, sky-domain simulation, and photometric physics are out
of scope. Transforms cover FITS scale/null/dtype and tensor preprocessing only.

## At a Glance

| Task | Traditional stack | torchfits equivalent |
|---|---|---|
| Read image to GPU | astropy/fitsio &rarr; numpy &rarr; torch &rarr; `.to(device)` | `torchfits.read_tensor("img.fits", device="cuda")` |
| Write tensor to FITS | tensor &rarr; numpy &rarr; astropy HDU &rarr; writeto | `torchfits.write("out.fits", tensor)` |
| Filter large table | load all rows &rarr; mask in Python | `where="MAG < 20"` pushdown in C++ |
| Read multi-extension files | manual HDU dispatch | `with torchfits.open("mef.fits") as hdul: ...` |
| PyTorch training loop | hand-rolled `Dataset` + cache tuning | `FitsImageDataset` + `make_loader(..., num_workers=4)` |
| Normalize for model input | ad-hoc scaling in the training script | `Compose([BackgroundSubtract(), ZScaleNormalize()])` |
| Verify FITS checksums | comparator-specific helpers | `torchfits.verify_checksums(path)` |

## Features

**FITS I/O** &mdash; Multi-threaded C++ core with SIMD-optimized type conversion,
memory-mapped image reads, intelligent chunking, and adaptive buffering. Reads
and writes images, binary/ASCII tables, compressed images, and multi-extension
FITS files with header round-trip coverage.

**Table Engine** &mdash; Arrow-native table API with predicate pushdown (`where=`),
column projection, row slicing, streaming `scan()`, and in-place mutations
(append, insert, update, delete rows and columns). Interop with Pandas, Polars,
DuckDB, and PyArrow.

**ML Data Layer** &mdash; `torchfits.data` ships `FitsImageDataset`,
`FitsImageIterableDataset`, `FitsTableDataset`, `FitsTableIterableDataset`,
`FitsCutoutDataset`, and `make_loader` with automatic handle-cache warm-up.

**Transforms** &mdash; 25+ `FITSTransform` classes for image stretches,
header-aware scaling (`FITSHeaderScale`, `FITSScaleColumns`, `TNullToNan`),
spectral/hyperspectral preprocessing, and continuum estimators. Most ship
`.inverse()` for decoding model outputs back to physical units. See
[docs/api.md](docs/api.md#transforms) and `examples/example_transforms.py`.

**Compatibility Contract** &mdash; Parity is tracked by tier: truthful public docs,
fitsio core workflow parity, Astropy common workflow parity, selected CFITSIO
backend behavior, and explicit non-goals. See [docs/parity.md](docs/parity.md).

## What's New in 0.7.0

0.7.0 completes the ML data layer and removes legacy dataset aliases:

- **`FitsTableIterableDataset`** — stream large catalogs via `table.scan` with
  multi-worker batch sharding.
- **`FitsCutoutDataset`** — patch training from a cutout index table.
- **Legacy removal** — `FITSDataset` / `IterableFITSDataset` removed from the
  root namespace; see [migration_datasets.md](docs/migration_datasets.md).
- **Docs site** — Zensical + GitHub Pages at
  [astroai.github.io/torchfits](https://astroai.github.io/torchfits/).

0.6.0 shipped the core ML surface (`torchfits.data` image/table datasets,
`torchfits.transforms`, C++ predicate pushdown, thread-safe caches). See
[docs/changelog.md](docs/changelog.md#060---2026-07-09).

Full history: [docs/changelog.md](docs/changelog.md). Roadmap for 1.0:
[docs/roadmap.md](docs/roadmap.md).

## Transforms

```python
from torchfits.transforms import ArcsinhStretch, BackgroundSubtract, Compose, ZScaleNormalize

pipeline = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1), ZScaleNormalize()])
normalized = pipeline(image)              # forward → model input
restored = pipeline.inverse(normalized)   # inverse → physical flux
```

Representative classes (full catalog in [docs/api.md](docs/api.md#transforms)):

| Category | Examples | Inverse |
|---|---|---|
| Image stretches | `ArcsinhStretch`, `LogStretch`, `ZScaleNormalize`, `RobustNormalize` | ✓ |
| Header / table | `FITSHeaderScale`, `FITSHeaderNormalize`, `FITSScaleColumns`, `TNullToNan` | ✓ |
| Spectral | `ContinuumNormalize`, `ContinuumRemoval`, `DopplerShift`, `SpectralBinning` | ✓ (except `BandMath`) |
| Continuum estimators | `AsymmetricLeastSquares`, `AlphaShapeContinuum`, `WaveletDecompose`, `SavitzkyGolayFilter` | ✓ |
| Outlier / time | `SigmaClip`, `AsymmetricSigmaClip`, `PhaseFold` | ✗ (lossy or many-to-one) |

Runnable demos: `examples/example_transforms.py` (image pipeline),
`examples/example_hyperspectral.py` (spectral cube).

## Performance

Median wall-clock from the lab exhaustive benchmark suite (run
`exhaustive_cuda_0.7.0_20260711_055635`, CANFAR staging GPU + CPU rows); see
[docs/benchmarks.md](docs/benchmarks.md) for methodology, deficit transparency,
and reproducible commands.

| Case | torchfits | astropy | fitsio | Speedup vs astropy |
|---|---:|---:|---:|---:|
| Large float32 image read (16 MB, CPU) | 3.93 ms | 7.60 ms | 6.09 ms | **1.9×** |
| Compressed Rice image (CPU) | 8.99 ms | 28.14 ms | 9.36 ms | **3.1×** |
| 50× repeated 100×100 cutouts (CPU) | 4.63 ms | 76.04 ms | 4.76 ms | **16×** |
| Table read (100k rows, 8 cols) | 86.9 μs | 6.32 ms | 59.41 ms | **73×** |
| Varlen table read (100k rows, 3 cols) | 90.8 μs | 3.49 ms | 220 ms | **38×** |

**ML DataLoader (local diagnostic, not in lab CSV):** 30×512² float32, CPU, 2
epochs — torchfits **1.12×** vs fitsio on Rice-compressed files; uncompressed
within ~4%. `make_loader(..., optimize_cache=True)` warms handle caches
automatically when the dataset exposes a `files` attribute.

**GPU integer reads:** Default `read(..., device="cuda")` applies BSCALE/BZERO on
device and returns `float32` for generic scaled pixels — good for ML. For native
integer dtypes (int8, uint16) matching fitsio, use
`read_tensor(..., raw_scale=True)` or rely on the automatic signed-byte /
unsigned-integer fast paths (see benchmarks doc). Tables remain CPU-resident in
all backends; GPU rows measure host decode + H2D copy, not disk→GPU bypass.

## Install

```bash
pip install torchfits
```

Pre-built wheels are available for Linux x86_64 and macOS arm64. No system
CFITSIO is needed&mdash;it is vendored and compiled automatically. Other
architectures install from source when a compatible compiler and PyTorch are
available.

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

tensor = torchfits.read_tensor("science.fits", hdu=0, device="cuda")
```

### PyTorch DataLoader

```python
from torchfits.data import FitsImageDataset, make_loader

ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
loader = make_loader(ds, batch_size=32, num_workers=4)

for images, labels in loader:
    ...  # images: [B, 1, H, W] when add_channel_dim=True (default)
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

Published site: [astroai.github.io/torchfits](https://astroai.github.io/torchfits/)

| | |
|---|---|
| [Documentation site](https://astroai.github.io/torchfits/) | Browse all docs on GitHub Pages |
| [API Reference](docs/api.md) | Full public API with signatures and examples |
| [Migration from Astropy](docs/migration_astropy.md) | Side-by-side workflow translation |
| [Migration from fitsio](docs/migration_fitsio.md) | Side-by-side workflow translation |
| [Dataset migration](docs/migration_datasets.md) | Removed `FITSDataset` → `torchfits.data` |
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

[MIT](LICENSE)
