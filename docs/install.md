# Installation

torchfits needs **Python 3.10+** and **PyTorch ≥ 2.10**.

- **PyPI wheels** are ABI-matched to **PyTorch 2.10.x** (Linux x86_64, macOS
  arm64). No system CFITSIO — CFITSIO is vendored into the wheel.
- **Other torch minors (≥ 2.10):** build from source against the torch already
  installed (see [From source](#from-source) and [Compatibility](compatibility.md)).

## Quick install (wheels)

```bash
# 1. Install matching PyTorch (pick one accelerator story — see below)
pip install "torch>=2.10,<2.11"

# 2. Install torchfits
pip install torchfits
```

This also installs the `torchfits` CLI (`torchfits --help`). See the
[CLI guide](cli.md).

**Next steps:** [Quick start](quickstart.md), [Examples](examples.md),
[API reference](api.md)

### GPU / accelerator (one recipe)

PyTorch provides CPU, CUDA, and MPS in the **same** `torch` package — choose
the wheel index once, then install torchfits:

```bash
# CUDA (pick the current CUDA build from pytorch.org; example: cu128)
pip install "torch>=2.10,<2.11" --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon MPS — default macOS torch wheel includes MPS
pip install "torch>=2.10,<2.11"

# CPU-only
pip install "torch>=2.10,<2.11" --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install torchfits
```

torchfits reads FITS on the CPU and places the resulting tensor on the
requested device (`device="cuda"` / `device="mps"`). No GPU-specific build of
torchfits is required.

At import, torchfits calls `cache.configure_for_environment()` once so mmap /
prefetch defaults match CPU vs CUDA vs MPS.

**Disk cache** (HTTP remotes + example samples only): default
`$XDG_CACHE_HOME/torchfits` or `~/.cache/torchfits`. Override with
`TORCHFITS_CACHE_DIR` (or `TORCHFITS_REMOTE_CACHE` / `TORCHFITS_SAMPLE_CACHE`).
This is separate from the in-memory handle/`configure_for_environment` path.

!!! tip "Verify accelerator"
    ```python
    import torch
    print(torch.cuda.is_available(), getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    ```

---

## From source

Use this when you need torchfits against a **non-2.10** torch (≥ 2.10), or when
contributing.

### Prerequisites

- Python 3.10+
- C++17 compiler (GCC 10+, Clang 14+, or MSVC 2019+)
- [CMake](https://cmake.org/) 3.21+
- [Ninja](https://ninja-build.org/) (recommended)
- [PyTorch](https://pytorch.org/) **≥ 2.10** already installed (the build
  embeds that torch's major.minor as the ABI tag)
- [NumPy](https://numpy.org/) 1.20+

=== "Linux"

    ```bash
    sudo apt install build-essential cmake ninja-build
    ```

=== "macOS"

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
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
./extern/vendor.sh      # download vendored CFITSIO sources

# Install the torch minor you want first, then build against it
pip install "torch>=2.10"   # or a specific 2.11+ build from pytorch.org
pip install --no-build-isolation -e .
```

For a release build:

```bash
pip install --no-build-isolation .
```

The extension records the torch **major.minor** it was built with. Import fails
if the running torch minor differs — rebuild after upgrading torch.

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)
and [nanobind](https://nanobind.readthedocs.io/) for the C++ extension.

### Vendored CFITSIO

torchfits vendors CFITSIO to avoid system-library version mismatches. The
`extern/vendor.sh` script downloads the source into `extern/cfitsio/`.

Pin a specific version:

```bash
./extern/vendor.sh --cfitsio-version cfitsio-4.6.4
```

Link against system CFITSIO instead:

```bash
pip install -e . --no-build-isolation --config-settings=cmake.args="-DTORCHFITS_USE_VENDORED_CFITSIO=OFF"
```

---

## Verify install

```python
import torchfits
print(torchfits.__version__)
_ = torchfits.read_tensor  # extension loaded
```

```bash
torchfits --help
torchfits info --help
```

---

## Development setup (pixi)

The project uses [pixi](https://pixi.sh/) for reproducible environments
(dev pixi stays on the 2.10 ABI lane for wheel parity):

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
| `torchfits[dev]` | pytest, ruff, mypy, astropy, fitsio, pandas, matplotlib | Development (test + bench + examples deps) |
| `torchfits[bench]` | astropy, fitsio, pandas, matplotlib | Benchmarking |
| `torchfits[test]` | pytest, pytest-cov | Testing |
| `torchfits[examples]` | matplotlib | Running examples |

Notebooks: `_repr_html_` works with any Jupyter kernel — **ipykernel is not**
a torchfits dependency.

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
pip install -e . --no-build-isolation -v
```

**`./extern/vendor.sh` fails**

Ensure `curl` or `wget` are available. Behind a proxy? Set `HTTPS_PROXY`.

**`ImportError: ... ABI mismatch` / symbol not found**

The extension was built for a different torch major.minor than the one
imported. Rebuild against the active torch:

```bash
pip install -e . --no-build-isolation --force-reinstall
```

**Slow first read**

Import already runs `torchfits.cache.configure_for_environment()`. Call it
again only if you change devices or want to re-tune after changing hardware
visibility.
