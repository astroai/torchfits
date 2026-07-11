# torchfits

**torchfits** is FITS I/O for PyTorch: a multi-threaded C++ engine with vendored
CFITSIO that reads and writes images, headers, HDUs, compressed images, and
tables as tensors — without NumPy-to-torch glue.

## Quick start

```bash
pip install torchfits
```

```python
import torchfits

data = torchfits.read_tensor("image.fits", device="cuda")
```

See [Installation](install.md) for wheels, source builds, and GPU setup.

## Documentation map

| Topic | Page |
|---|---|
| Public API | [API Reference](api.md) |
| Runnable workflows | [Examples](examples.md) |
| Supported vs out-of-scope | [Parity Matrix](parity.md) |
| From Astropy / fitsio | [Migration guides](migration_astropy.md) |
| Dataset API changes | [Dataset migration](migration_datasets.md) |
| Table / cutout datasets | [Examples](examples.md) — `example_data_catalogs.py` |
| Performance numbers | [Benchmarks](benchmarks.md) |
| Release path | [Roadmap](roadmap.md) |

Example scripts live in the repository under `examples/`. Links in
[Examples](examples.md) point at GitHub source files.

## Scope

torchfits is not a full replacement for Astropy, fitsio, or CFITSIO. The
supported surface is documented in [Parity](parity.md). WCS, sky-coordinate
models, HEALPix, and photometric physics are out of scope for this package.
