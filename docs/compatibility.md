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

## Scale limits

Supported platforms are 64-bit Linux and macOS: file offsets, row counts and
element counts are 64-bit throughout. These numbers come from
`pixi run test-scale` (`tests/test_scale_large.py`, sparse files) and probes
on a shared 4-core Linux host; treat them as orders of magnitude.

| Case | Tested | Measured |
|---|---|---|
| Image HDU larger than 4 GiB | 4.46 GB int16 primary + extension past the 4 GiB offset | extension read, cutout at the last pixel: < 1 ms; `write_checksums` 3.7 s, `verify_checksums` 1.9 s |
| Image with more than 2^31 pixels | 2.15 GB uint8, `mmap=True` and `False` | full read 1.6 s, peak RSS ≈ image size |
| Tile compression of a > 2 GiB image | `torchfits compress --algorithm RICE_1` / `decompress` | ~6 s each, peak RSS ≈ image size + 0.2 GB |
| Table with more than 2^31 rows | 2^31 + 1000 one-byte rows | `row_slice` / `rows=` / `start_row` at the end: < 0.2 s; `where=` over every row 4.7 s; `scan(where=)` 4.6 s at 0.3 GB RSS |
| Table larger than 4 GiB | 600 M rows × 12 bytes (7.2 GB) | last-row reads < 1 ms; full `scan` / `scan_torch` 3–4 s at 0.2–0.3 GB RSS |
| Streaming write across 2^31 rows | `table.writer()` in 16 Mi-row batches, then `append_rows`, checksums | peak RSS 0.3 GB |
| Many HDUs | 10,000-extension file | `torchfits.open` 0.34 s; read of any HDU after the first visit ~0.04 ms; `torchfits info` 0.7 s, `verify` 1.1 s, `stats` 3.7 s; `insert_hdu` / `replace_hdu` / `delete_hdu` rewrite the file: ~0.8 s |
| Many files | 10,000 files under `ulimit -n 256` | `read`, `read_batch`, datasets and a 4-worker `DataLoader` keep a handful of descriptors open |
| Repeated calls | 10,000 reads / writes / `where=` / `scan` / streaming writes / compressed reads | no RSS or descriptor growth |

Memory follows the result, not the file, except where a whole object is
materialized:

- `table.scan()` / `scan_torch()` (with or without `where=`) and
  `table.writer()` hold one batch at a time; use them for tables larger
  than RAM.
- `table.read(where=)` keeps the matching rows only. `table.read_torch(where=)`
  reads the predicate columns whole and builds a boolean mask (a 2.1 GB
  one-byte column peaked at 4.4 GB).
- `FitsTableDataset` loads the table at construction; `lazy=True` keeps
  `cache_blocks × block_rows` rows per process (per `DataLoader` worker). With
  4 workers over a 5 M-row table: ~100 MB PSS per worker eager (copy-on-write
  shares the table), ~80 MB lazy.
- `torchfits stats` peaks at about twice the image; `compress` /
  `decompress` hold the decoded image once.
- `mmap=True` reads count mapped file pages in RSS; the kernel reclaims them
  like page cache.

Open descriptors are bounded by `TORCHFITS_MAX_CACHED_FDS` (default 32 raw
descriptors, see [Environment variables](architecture.md#environment-variables))
plus up to 8 cached table readers per thread (two descriptors each).
`clear_all_caches()` closes all of them.

Threads: reads, cutouts, table reads, checksums and file opens run without
the GIL. Image and table *writes* (`write`, `table.write`, `append_rows`)
hold it while CFITSIO writes. Calls under ~100 µs (headers, tiny images) are
bound by Python overhead and do not speed up with more threads.

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
