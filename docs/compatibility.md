# Environment & Platform Compatibility

Supported Python versions, PyTorch versions, CUDA flavors, and operating systems for **torchfits**.

For supported FITS formats, HDU types, tile compression, and catalog features, see [Evaluation](evaluation.md).

---

## Supported Environments

| Component | Prebuilt Wheels | Source Builds |
|---|---|---|
| **Python** | **3.11** (`cp311` wheel), **3.12–3.15** (one `cp312-abi3` wheel) | **3.11+** |
| **PyTorch** | **2.12 or newer** (`torch>=2.12`, no upper bound) | Same; the build needs no PyTorch |
| **Hardware & CUDA** | **CPU**, **any CUDA build of a supported PyTorch**, **Apple Silicon MPS** | All PyTorch-supported compute devices |
| **Operating Systems** | **Linux** (`x86_64` with SSSE3, `aarch64`, glibc 2.28+)<br>**macOS 14+** (`arm64` Apple Silicon) | Linux with glibc 2.28+, macOS 14+; `x86_64` builds require an SSSE3 CPU |
| **Core Libraries** | **NumPy ≥ 2.4.6**, **PyArrow ≥ 25.0.0** | Same |

---

## PyTorch Version Support

The native extension includes and links no libtorch. It decodes FITS data into
its own buffers and hands them to PyTorch through DLPack (zero copy), using
only PyTorch's public Python API. A wheel is therefore not tied to the PyTorch
C++ ABI of any one minor release, and the same wheel works with every PyTorch
from the floor up.

| PyTorch Version | Status |
|---|---|
| **2.12 – 2.15** | Tested in CI (floor and newest), `pip install torchfits` |
| **Newer releases** | Supported (no upper bound); report regressions |
| **Older than 2.12** | Unsupported (below the tested floor) |

The floor and the newest tested minor live in `scripts/torch_support.json`.

---

## CUDA & Accelerator Compatibility

- **Universal CUDA / CPU Wheels:** A single `torchfits` wheel functions with every CUDA flavor of every supported PyTorch (`cu126`, `cu128`, `cu130`, ...) as well as CPU-only (`+cpu`) installations. Decoding runs on the CPU; `device=` moves the result with PyTorch.
- **Apple Silicon (MPS):** Native `arm64` wheels for macOS leverage Metal Performance Shaders (`device="mps"`).
- **Graceful Fallback:** CUDA-built environments run seamlessly on CPU-only machines via automatic CPU fallback.
- **MPS dtype handling:** `device="mps"` (and `mps:N`) downcasts `float64 → float32` and `complex128 → complex64` before the host-to-device transfer because MPS has no native 64-bit float/complex. Each downcast emits a `UserWarning` (Python's default filter shows it once per call site): `MPS does not support float64; downcasting to float32 (precision loss)` and `MPS does not support complex128; downcasting to complex64 (precision loss)`. The result keeps its shape and lands on the requested device; CPU and CUDA paths keep 64-bit.
- **Scaled images:** A real `BSCALE`/`BZERO` on BITPIX=8 or 16 is read as float32. The same on BITPIX=32 or 64 is read as float64, so integers above 2^24 are not rounded. `BLANK` on a plain integer image uses that same float dtype so the blank pixel can be NaN. Signed-byte and unsigned integer conventions (`BZERO` of -128, 32768, or 2147483648 with `BSCALE=1`) stay integers. Native `BITPIX=-32` images stay float32. Table `TSCAL`/`TZERO` on real columns is float64.

### Known limitations

- **Python 3.15 on macOS needs PyTorch 2.15+:** PyTorch 2.14's cp315 macOS and Windows wheels were built against CPython 3.15.0b3 headers, whose `PyThreadState` layout predates 3.15.0b4. The first Python-level tensor operation outside the main thread then segfaults inside pybind11's `gil_scoped_acquire` ([pytorch/pytorch#200779](https://github.com/pytorch/pytorch/issues/200779)); the Linux cp315 wheels are unaffected. The torchfits `cp312-abi3` wheel installs on macOS Python 3.15; pair it with PyTorch 2.15 or newer, which is the pairing CI and the release wheel tests run on macOS Python 3.15.
- **No Windows support:** Prebuilt wheels and CI are Linux and macOS only; Windows is documented as unsupported.
- **Runtime env knobs:** every `TORCHFITS_*` variable the library reads, with its default and effect, is listed under [Environment variables](architecture.md#environment-variables). Those knobs are not a stable public API and may change without notice.
- **`KMP_DUPLICATE_LIB_OK`:** importing `torchfits` runs `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` before PyTorch loads. macOS otherwise aborts when PyTorch's `libomp` and a second OpenMP runtime (Homebrew or conda) are both mapped. An existing value is left alone. See [Install](install.md) if you import `torch` first.
- **CFITSIO DNS:** `http`/`https`/`ftp`/`ftps` paths (and CFITSIO's `httpmem://`, `httpfile://`, `root://`, `gsiftp://` and similar driver variants) are classified at guard time: any address that is not globally routable (private, loopback, link-local, shared `100.64.0.0/10`, reserved) is blocked, and a numeric host must be canonical dotted-decimal IPv4 (`0177.0.0.1`, `127.1` or `2130706433` are refused because resolvers disagree on them). Python fetches (`http_open`) dial the addresses from that lookup, or, when `http_proxy`/`https_proxy` applies, connect to the proxy, which resolves the (still guarded) target itself. A URL that CFITSIO opens itself is only checked then; the driver re-resolves the host afterwards (residual TOCTOU, DNS rebinding). CFITSIO follows an `http` 301/302 redirect only when it stays on the same host; a redirect to another host fails the open instead of reaching an unchecked target (libcurl `https` opens never follow redirects). Pinning those connections would need a custom CFITSIO driver and is not implemented.
- **Removed knobs (1.3.0):** `handle_cache_capacity` on a table read, `handles=` on `clear_file_cache()`, `torchfits.cache.configure_cache()`, `CacheManager.configure_cpp_cache()`, and `TableHDU(col_stats=...)` are gone. `cache_capacity` still controls the read cache and `fast_header` still affects the unified-reader fallback.

---

## Upgrading from 1.3

The next release after 1.3.3 (on `main` until it is tagged) changes a few
results that 1.3 returned silently. Most code needs no change; check the items
below if you read non-ASCII tables, catch column errors by type, or script the
CLI.

**Breaking: string columns decode strictly.** 1.3 decoded character (`A`)
columns with `errors="ignore"`, so bytes the encoding could not decode
vanished (`b'caf\xe9'` read as `'caf'`, and `where="NAME == 'caf'"` matched
it). Every read path -- `table.read` / `scan` / `reader` / `schema`, `where=`
string comparisons, `TableHDU.get_string_column` and the converters that read
a path -- now raises `UnicodeDecodeError` naming the column. Pass the encoding
the file really uses (`encoding="latin-1"`), or `decode_errors="ignore"` for
the 1.3 result (`"replace"` keeps a U+FFFD per bad byte); see
[`table.read()`](api-tables.md#tableread). Tensor reads (`read_torch`,
`scan_torch`, the table datasets) return raw `uint8` bytes and are unaffected,
except for a `where=` predicate that compares a string column.

**Tables**

- A column whose `TTYPE` repeats an earlier column's name is named `COL<n>` on
  every path (1.3 dropped it from `read`/`scan` results while `schema()`
  listed it); see [Column names](api-tables.md#column-names).
- An unknown column name raises `torchfits.UnknownColumnError` and a new name
  that is already in use raises `torchfits.ColumnExistsError`. Both subclass
  `KeyError` and `ValueError`, so existing `except` clauses still catch them.
- `TableHDU.num_rows` / `TableHDURef.num_rows` raise on a malformed `NAXIS2`
  or an unreadable header instead of reporting 0 rows.
- `append_rows` / `insert_rows` fill an omitted float or complex column with
  NaN (was 0) and store an omitted integer column's `TNULL` unscaled, so it
  reads back as null.
- A string clipped to its column width by `table.write(schema=...)`,
  `insert_column` / `replace_column` or a streamed batch now emits a
  `RuntimeWarning`, as the row mutations already did.
- A column with an explicit repeat of 1 (`TFORM = '1E'`) reads as a scalar
  Arrow array, as `schema()` describes it, instead of `FixedSizeList[1]`.

**Images and compression**

- NaN pixels of a float image written with a quantizing codec (`RICE_1`,
  `GZIP_2`, `HCOMPRESS_1`, `BZIP2_1`, or any `quantize_level`) are kept as
  NaN. 1.3 stored them as arbitrary integers that every reader decoded as
  garbage, so rewrite such files from their source data. NaN-free and
  lossless (`GZIP_1`) images are written exactly as before.

**Datasets and transforms**

- `FitsTableDataset` rows of a variable-length column are 1-D tensors, as
  `FitsTableIterableDataset` yields them (1.3 returned Python lists). Only the
  native reader's own I/O errors fall back to pyarrow now, logged at
  `WARNING`; any other error is raised.
- `FitsStagedCutoutIterableDataset` raises `ValueError` when a file's flux
  bands or `ivar_hdu` / `mask_hdu` companions differ in shape (1.3 cut them at
  the same window anyway).
- A normalizer's `forward()` on a typed `Payload` records its inverse
  statistics in `meta["torchfits.inverse_stats"]`; see
  [Transforms](api-transforms.md).
- `read_batch(strict=True)` raises the failing file's own error, the one a
  single `read()` of it raises.

**CLI** (see [CLI](cli.md))

- Every output (FITS, Parquet, CSV, TSV, Arrow, PNG, `copy`, `setkey -o`) is
  written to a temporary file and renamed into place: a failure leaves an
  existing OUTPUT untouched. An OUTPUT that is a hard link to an INPUT is
  refused (exit `2`).
- `convert --to fits` keeps the table HDU in its stored form (`TFORM`,
  `TSCAL`/`TZERO`, `TNULL`, `TDIM`, `TUNIT`, `EXTNAME` and other cards) and
  streams it; 1.3 renamed the HDU and dropped those cards.
- `arith` is exact for 64-bit integer images and for float operands on
  integer images, saturating instead of wrapping; a NaN result into an integer
  image is a usage error (exit `2`). `arith` and `cutout` drop the stale
  `DATAMIN` / `DATAMAX`; `arith` and `transform` drop `BUNIT` when the unit
  changes, and `arith` refuses to add or subtract images with different
  `BUNIT` (exit `2`).
- `verify` reports a truncated uncompressed file as an I/O error (exit `3`),
  even when it carries no checksum keywords.
- `table` previews are strict JSON: complex values are `{"re", "im"}` objects
  and undecodable bytes become U+FFFD.

---

## Verification

To verify that your installation matches your current Python and PyTorch runtime:

```python
import torch
import torchfits

print("torchfits version:", torchfits.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA GPU available:", torch.cuda.is_available())
print("Apple MPS available:", torch.backends.mps.is_available())
```
