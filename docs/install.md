# Installation

## Quick install

```bash
pip install torchfits
```

Pre-built wheels for **Linux x86_64** and **macOS arm64**. No system
libraries needed — CFITSIO is bundled. Other architectures build from
source.

**Requires:** Python 3.10+, PyTorch 2.0+

**Next steps:** [Quick start](quickstart.md) or
[I want to…](index.md#i-want-to)

---

## From source

### Prerequisites

- Python 3.10+
- C++17 compiler (GCC 10+, Clang 14+, or MSVC 2019+)
- [CMake](https://cmake.org/) 3.21+
- [Ninja](https://ninja-build.org/) (recommended)
- [PyTorch](https://pytorch.org/) 2.0+
- [NumPy](https://numpy.org/) 1.20+

=== "Linux"

    Ensure build tools are installed:

    ```bash
    sudo apt install build-essential cmake ninja-build
    ```

=== "macOS"

    Install Xcode Command Line Tools if not present:

    ```bash
    xcode-select --install
    ```

=== "Windows"

    Install Visual Studio 2019+ with C++ workload, then:

    ```bash
    pip install cmake ninja
    ```

### Build

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
./extern/vendor.sh      # download vendored CFITSIO sources
pip install -e .        # editable install
```

For a release build:

```bash
pip install .
```

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)
and [nanobind](https://nanobind.readthedocs.io/) for the C++ extension.

### Vendored CFITSIO

torchfits vendors CFITSIO to avoid system-library version mismatches. The
`extern/vendor.sh` script downloads the source into `extern/cfitsio/`.

Pin a specific version:

```bash
./extern/vendor.sh --cfitsio-version cfitsio-4.6.2
```

Link against system CFITSIO instead:

```bash
pip install -e . --config-settings=cmake.args="-DTORCHFITS_USE_VENDORED_CFITSIO=OFF"
```

---

## GPU support

torchfits reads FITS data on the CPU and places the resulting tensor on
the requested device. No GPU-specific build steps are required — just
install PyTorch with CUDA or MPS support:

=== "CUDA"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

=== "MPS (Apple Silicon)"

    ```bash
    pip install torch
    ```

    MPS support is included in the default macOS PyTorch wheel.

Then pass `device="cuda"` or `device="mps"` to `read_tensor()`:

```python
tensor = torchfits.read_tensor("image.fits", device="cuda")
```

---

## Verify install

```python
import torchfits
print(torchfits.__version__)
_ = torchfits.read_tensor  # extension loaded
```

---

## Development setup (pixi)

The project uses [pixi](https://pixi.sh/) for reproducible environments:

```bash
pixi install
pixi run test           # run tests
pixi run lint           # ruff lint
pixi run bench-all      # exhaustive benchmarks
```

---

## Optional dependencies

| Extra | Installs | Use |
|---|---|---|
| `torchfits[dev]` | pytest, ruff, mypy, ipykernel | Development |
| `torchfits[bench]` | astropy, fitsio, pandas, matplotlib | Benchmarking |
| `torchfits[test]` | pytest, pytest-cov | Testing |
| `torchfits[examples]` | matplotlib | Running examples |

PyArrow is a core dependency (`torchfits.table` is Arrow-native). Pandas,
Polars, and DuckDB remain optional integrations:

```bash
pip install polars duckdb  # optional table interop
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torchfits._C'`** (or `'torchfits.cpp'`)

The compiled extension (`torchfits._C`) did not build. `torchfits.cpp` is a
pure-Python compatibility shim that re-exports it, so either import failure
points at the missing extension. Check that CMake, a C++17 compiler, and
Ninja are installed. Re-run with verbose output:

```bash
pip install -e . -v
```

**`./extern/vendor.sh` fails**

Ensure `curl` or `wget` is available. Behind a proxy? Set `HTTPS_PROXY`.

**`ImportError: ... symbol not found`**

Version mismatch between the compiled extension and PyTorch. Rebuild:

```bash
pip install -e . --no-build-isolation --force-reinstall
```

**Slow first read**

The first call initializes file and metadata caches. Call
`torchfits.cache.configure_for_environment()` at startup for auto-tuning
before the first read.
