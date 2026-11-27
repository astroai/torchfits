# Installation

`torchfits` is distributed as prebuilt binary wheels for **Linux (x86_64,
aarch64, glibc 2.28+)** and **macOS 14+ (Apple Silicon arm64)** supporting
**Python 3.11–3.15** and **PyTorch 2.12 or newer**. CFITSIO is vendored
directly into the wheels, so no C++ compiler, system libraries, or manual build
steps are required.

## Quick install

### 1. Standard install (CUDA + CPU auto-detection) — Recommended

For standard environments (Linux with NVIDIA GPU, CPU-only Linux, or macOS Apple Silicon):

```bash
pip install torchfits
```

- **On Linux with an NVIDIA GPU:** `device="cuda"` copies the host-decoded tensor onto CUDA.
- **On Linux without a GPU:** falls back to CPU automatically (`device="cpu"`).
- **On macOS (Apple Silicon):** `device="mps"` copies the host-decoded tensor onto Metal.

### 2. CPU-only install (Minimal footprint)

On headless servers, CI runners, or containers where CUDA runtime libraries are
not needed, install a CPU build of PyTorch from its CPU index, then torchfits:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchfits
```

### 3. Specific CUDA toolkit versions

To match a specific CUDA version installed on your system (e.g. `cu126`, `cu130`),
install that PyTorch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install torchfits
```

The same torchfits wheel serves every PyTorch build flavor (CPU, any CUDA, MPS).

---

## PyTorch Version Compatibility

torchfits supports **PyTorch 2.12 and newer** (`torch>=2.12`, no upper bound).
The native extension does not link libtorch: decoded buffers are handed to
PyTorch through DLPack, its public zero-copy exchange protocol, so a torchfits
wheel keeps working when you upgrade PyTorch. CI tests the floor (2.12) and the
newest tested minor (2.15), which `scripts/torch_support.json` records, and a
nightly job runs the suite against the newest PyTorch release.

The extension also targets the CPython stable ABI: one `cp312-abi3` wheel per
platform serves Python 3.12 and every later CPython, and Python 3.11 has its
own wheel.

!!! warning "Python 3.15 on macOS"
    PyTorch 2.14's cp315 macOS wheels were built against CPython 3.15.0b3 headers and segfault on the first tensor operation outside the main thread ([pytorch/pytorch#200779](https://github.com/pytorch/pytorch/issues/200779)). On macOS with Python 3.15, use PyTorch 2.15 or newer.

---

## Verify your installation

```python
import torchfits
import torch

print(f"torchfits {torchfits.__version__} with torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

```bash
torchfits info --help
```

---

## Building from source

Builds link libbz2 when available (conda prefix or system via CMake), which enables `BZIP2_1` tile compression and transparent reading of whole-file `.bz2` FITS. Without it, `.bz2` inputs raise an actionable error. Likewise, CFITSIO opens `https://` and `ftps://` URLs through libcurl when the build finds it; without it those URLs raise an actionable `ValueError` instead of a generic open failure.

To check what a build was compiled with, print the two build flags. They live on
the private compiled module, so treat this as a diagnostic, not an API:

```bash
python -c "import torchfits._C as c; print('bzip2:', c.HAS_BZIP2, 'https/ftps:', c.HAS_CURL)"
```

Building from source is only needed when developing torchfits or using a
platform without a published wheel. The build itself needs no PyTorch.

### Prerequisites
- Python 3.11+
- C++20 compiler (GCC 10+, Clang 14+, MSVC 2019+)
- [CMake](https://cmake.org/) 3.21+ and [Ninja](https://ninja-build.org/)
- At runtime: [PyTorch](https://pytorch.org/) ≥ 2.12, [NumPy](https://numpy.org/) ≥ 2.4.6, and [PyArrow](https://arrow.apache.org/docs/python/) ≥ 25

=== "Linux"

    ```bash
    sudo apt install build-essential cmake ninja-build
    ```

=== "macOS"

    ```bash
    xcode-select --install
    brew install cmake ninja
    ```

=== "Windows"

    Windows is **unsupported**. There are no wheels, and the native extension
    is not tested with MSVC. Use WSL2, Linux, or macOS.

### Build steps

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt   # downloads the pinned CFITSIO

# Runtime dependencies, then the build backend (the build needs no PyTorch)
pip install torch numpy pyarrow scikit-build-core nanobind
pip install --no-deps --no-build-isolation -e .
```

To link against a system-installed CFITSIO instead of the vendored source:

```bash
pip install -e . --no-deps --no-build-isolation --config-settings=cmake.args="-DTORCHFITS_USE_VENDORED_CFITSIO=OFF"
```

---

## Development setup (Pixi)

For local development and running the test suite, use [pixi](https://pixi.sh/) `0.82.0` (`requires-pixi` in `pixi.toml`):

```bash
pixi install --locked
pixi run test           # build the extension, then the full suite
pixi run lint           # ruff linter and formatter
```

`pixi run test` rebuilds the extension in the default environment. Astropy and fitsio are not installed here; comparisons against them run in [torchfits-harness](https://github.com/astroai/torchfits-harness).

---

## Optional dependencies

| Extra | Included packages | Purpose |
|---|---|---|
| `torchfits[dev]` | pytest, ruff, mypy, pandas, matplotlib | Library development toolchain |
| `torchfits[test]` | pytest, pytest-cov, psutil | Unit testing extras (`pip install torchfits[test]`); PyArrow comes from the base package |
| `torchfits[docs]` | zensical | Building the documentation site |
| `torchfits[examples]` | matplotlib | Running tutorial scripts |

PyArrow is installed automatically for tabular operations (`torchfits.table`). Interoperability with [Pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), and [DuckDB](https://duckdb.org/) is supported seamlessly:

```bash
pip install pandas polars duckdb
```

---

## Troubleshooting

**`OMP: Error #15: Initializing libomp.dylib`**

Two OpenMP libraries are loaded in the same process — typical on macOS when PyTorch's `libomp` meets Homebrew or conda `llvm-openmp`. Importing `torchfits` sets `KMP_DUPLICATE_LIB_OK=TRUE` before PyTorch loads. Import `torchfits` first, or export that variable in the shell before `import torch`:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**`No matching distribution found for torchfits`**

Prebuilt binary wheels are available for Linux (x86_64 and aarch64, glibc
2.28+) and macOS 14+ (Apple Silicon arm64) on CPython 3.11–3.15. Windows, musl
Linux, older glibc, older macOS, and x86_64 macOS are unsupported.

**`ImportError: torchfits was built for PyTorch 2.x.x but found PyTorch ...`**

Releases before 1.3.3 linked libtorch and only imported under the PyTorch minor
they were built for. Upgrade torchfits: 1.3.3 and newer work with every
PyTorch from 2.12 on.

**`./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt` fails**

The script requires `--cfitsio-version`; run with no arguments it prints its
usage and exits 1. Beyond that, ensure `curl` and `tar` are installed and
reachable (a sha256 mismatch against the pinned CFITSIO tarball also fails
here, by design). If behind a proxy, set `HTTPS_PROXY`.
