# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)

[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**torchfits** reads and writes FITS files as PyTorch tensors. A multi-threaded
C++ engine (vendored CFITSIO) handles images, tables, headers, compression, and
MEF files. Optional datasets, transforms, and a `torchfits` CLI sit on top.

```bash
pip install torchfits
```

Requires **Python 3.10+** and **PyTorch 2.10**. Docs:
[astroai.github.io/torchfits](https://astroai.github.io/torchfits/).

## At a Glance

| Task | torchfits |
|---|---|
| Image → GPU tensor | `torchfits.read_tensor("img.fits", device="cuda")` |
| Write a tensor | `torchfits.write("out.fits", tensor)` |
| Filter a catalog in C++ | `table.read(..., where="MAG < 20")` |
| Open a MEF | `with torchfits.open("mef.fits") as hdul: …` |
| Train | `FitsImageDataset` + `make_loader(..., num_workers=4)` |
| Shell | `torchfits info` / `header` / `convert` / … |

## Features

- **Fast FITS I/O** — mmap image reads, compressed images, MEF, checksums
- **Tables** — Arrow-native with `where=` pushdown, scan/stream, Parquet/CSV/TSV/Arrow IPC export
- **ML** — `torchfits.data` datasets + `make_loader`
- **Transforms** — stretches, FITS scale/null handling, spectral prep (`torchfits.transforms`)
- **CLI** — MEF-aware inspect/convert tools ([docs/cli.md](docs/cli.md))

Supported feature matrix: [docs/parity.md](docs/parity.md).

## What's New in 0.9.2 / 0.9.3

- **CLI** — `torchfits` for `info`, `header`, `verify`, `stats`, `table`, `cutout`,
  `convert`, … ([docs/cli.md](docs/cli.md))
- **Leaner imports** — transforms from `torchfits.transforms`; use `read` /
  `read_tensor` (not `read_fast` / `read_image`)
- **Convert** — tables → Parquet, CSV, TSV, or Arrow IPC; images → Lupton PNG
- **Scorecard** — Linux CPU/CUDA **0** strict deficits; Mac MPS **4**
  ([docs/benchmarks.md](docs/benchmarks.md))

Full notes: [docs/changelog.md](docs/changelog.md).

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
| Image stretches | `ArcsinhStretch`, `LogStretch`, `SqrtStretch`, `ZScaleNormalize`, `RobustNormalize`, `MinMaxNormalize`, `PercentileClipNormalize` | ✓ |
| Background / normalization | `BackgroundSubtract`, `GlobalScalarNorm`, `FITSHeaderNormalize` | ✓ |
| Header / table | `FITSHeaderScale`, `FITSScaleColumns`, `TNullToNan` | ✓ |
| Spectral | `ContinuumNormalize`, `ContinuumRemoval`, `DopplerShift`, `SpectralBinning` | ✓ (except `BandMath`) |
| Continuum estimators | `AsymmetricLeastSquares`, `AlphaShapeContinuum`, `WaveletDecompose`, `SavitzkyGolayFilter`, `RunningPercentile`, `UpperEnvelopeContinuum` | ✓ |
| Outlier / time | `SigmaClip`, `AsymmetricSigmaClip`, `PhaseFold` | ✗ (lossy or many-to-one) |

Runnable demos: `examples/example_transforms.py` (image pipeline),
`examples/example_hyperspectral.py` (spectral cube),
`examples/example_time_series.py` (light curves).

## Performance

Lab multi-host exhaustive scorecard
(`exhaustive_mps_20260717_000853` Mac MPS,
`exhaustive_cpu_20260716_191252` CANFAR CPU,
`exhaustive_cuda_20260716_191255` CANFAR CUDA); see
[docs/benchmarks.md](docs/benchmarks.md) for methodology, full exhaustive
table, category summaries, RSS columns, and deficit transparency.

Under the **strict** gate (images: any lag; Arrow tables: ≤1.05×), Linux CPU
and CUDA currently report **0** TorchFits deficits. Mac MPS reports **4**
deficits on this refresh (down from an earlier 101-row snapshot); MPS is not
the Linux CUDA release gate.

### Headline numbers

| Case | torchfits | astropy | fitsio | Speedup vs astropy |
|---|---:|---:|---:|---:|
| Large float32 image read (16 MB, CPU) | 3.85 ms | 16.67 ms | 5.89 ms | **4.3×** |
| Compressed Rice image (CPU) | 9.06 ms | 27.77 ms | 9.43 ms | **3.1×** |
| 50× repeated 100×100 cutouts (CPU) | 4.68 ms | 75.36 ms | 4.94 ms | **16.7×** |
| Table read (100k rows, 8 cols) | 95.3 μs | 6.74 ms | 59.84 ms | **70.6×** |
| Varlen table read (100k rows, 3 cols) | 93.9 μs | 3.52 ms | 288.81 ms | **37.5×** |

### By benchmark category

| Category | Best speedup vs astropy | Best speedup vs fitsio | Notes |
|---|---:|---:|---|
| **1D images** (float32/64, int8–int64) | **7.8×** | **2.3×** | All sizes, CPU |
| **2D images** (float32/64, int8–int64, uint16/32) | **7.7×** | **2.4×** | All sizes, CPU |
| **3D cubes** (float32/64, int8–int64) | **7.5×** | **2.1×** | Small–medium, CPU |
| **Compressed** (gzip, rice, hcompress) | **4.3×** | **1.1×** | rice and gzip dominate; hcompress slightly behind fitsio |
| **Scaled** (BSCALE/BZERO) | **6.1×** | **1.8×** | Automatic integer→float scaling |
| **MEF** (multi-extension) | **9.9×** | **2.4×** | Small–medium files |
| **Repeated cutouts** (50× 100×100) | **16.7×** | **1.1×** | Open-once, subset many times |
| **Time series frames** | **5.1×** | **1.9×** | 5 sequential frames |
| **Header reads** (all fixtures) | **9.5×** | **1.5×** | Sub-100 μs for all backends |
| **Table: read_full** | **115×** | **628×** | 100k rows, 8 cols |
| **Table: projection** | **147×** | **91×** | Column subset |
| **Table: row_slice** | **147×** | **162×** | Row range |
| **Table: predicate_filter** | **57×** | **25×** | WHERE clause |
| **GPU (CUDA) images** | **78×** | **3.0×** | tiny–large, all dtypes |

### Current deficits

Scorecard policy (same-mmap peers):

- **Images / cubes / spectra / cutouts:** any lag above float-timer ε is a deficit
  (rice/hcompress included — no percent floor).
- **Arrow tables:** allow up to **1.05×**.

Prior “0 deficit” claims used a 25% lag floor and are retracted. Re-score after
the SIMD endian + thin device + WHERE⇒mmap-scan fixes; see
`docs/benchmarks.md`.

**GPU integer reads:** Default `read(..., device="cuda")` applies BSCALE/BZERO on
device and returns `float32` for generic scaled pixels — good for ML. For native
integer dtypes (int8, uint16) matching fitsio, use
`read_tensor(..., raw_scale=True)` or rely on the automatic signed-byte /
unsigned-integer fast paths (see benchmarks doc). Tables remain CPU-resident in
all backends; GPU rows measure host decode + H2D copy, not disk→GPU bypass.

**ML DataLoader (local diagnostic, not in lab CSV):** 30×512² float32, CPU, 2
epochs — torchfits **1.12×** vs fitsio on Rice-compressed files; uncompressed
within ~4%. `make_loader(..., optimize_cache=True)` warms handle caches
automatically when the dataset exposes a `files` attribute.

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

Requires Python 3.10+, a C++17 compiler, CMake 3.21+, and PyTorch 2.10.

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

### Shell (CLI)

```bash
torchfits info science.fits
torchfits header science.fits --keyword OBJECT --json
torchfits verify science.fits
torchfits stats science.fits --hdu 0
torchfits convert catalog.fits out.csv --to csv --hdu 1
```

## Benchmarks

torchfits is benchmarked across FITS image I/O (1D/2D/3D, all integer and
float dtypes, compressed, scaled, MEF, cutouts, time series) and FITS table
I/O (read, projection, row slicing, predicate filtering, streaming). GPU
(CUDA) transport rows are included for image reads.

Comparators are `astropy.io.fits` and `fitsio`; selected CFITSIO behavior is
validated through the torchfits native backend and smoke tests.

Methodology, full exhaustive table, category summaries, and known deficits:
[`docs/benchmarks.md`](docs/benchmarks.md)

## Documentation

Published site: [astroai.github.io/torchfits](https://astroai.github.io/torchfits/)

| | |
|---|---|
| [Documentation site](https://astroai.github.io/torchfits/) | Browse all docs on GitHub Pages |
| [API Reference](docs/api.md) | Full public API with signatures and examples |
| [CLI](docs/cli.md) | `torchfits` command-line tools |
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
