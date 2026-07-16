# Architecture

Internal design of torchfits for users who need to understand the C++/Python
layering, I/O paths, caching, and threading model without reading source.

---

## Layer structure

```
┌─────────────────────────────────────────────────┐
│  Python API  (torchfits.read, table.read, ...)  │
├─────────────────────────────────────────────────┤
│  _io_engine  (dispatch, caching, pipeline)      │
├─────────────────────────────────────────────────┤
│  torchfits._C  (nanobind C++ extension)         │
│    ├── fits_bindings.cpp   (image / header I/O) │
│    ├── table_bindings.cpp  (table I/O)          │
│    ├── cache.cpp           (handle cache)       │
│    └── fits_detail.h       (read strategies)    │
├─────────────────────────────────────────────────┤
│  Vendored CFITSIO  (extern/cfitsio/)            │
│  Raw fd pread/mmap  (bypasses CFITSIO for data) │
└─────────────────────────────────────────────────┘
```

`io.py` is a thin re-export layer. All real work happens in `_io_engine/`
(submodules: `_read_pipeline.py`, `table_api.py`, `caches.py`) which call
into `torchfits._C`. The C++ extension is compiled via scikit-build-core +
nanobind; it vendors CFITSIO statically.

---

## Image read paths

Three distinct paths, selected automatically:

### 1. mmap fast path (default for uncompressed images)

Used when the image is uncompressed and `mmap=True`.

1. Open raw fd (`O_RDONLY | O_CLOEXEC`)
2. Call `fits_get_hduaddrll` to get the data offset within the file
3. `mmap()` the data region directly — CFITSIO is **not** used for pixel reads
4. Byte-swap in parallel via `at::parallel_for` using `__builtin_bswap16/32/64`
5. Apply BSCALE/BZERO in the same parallel pass or on-device (controlled by
   `scale_on_device`)
6. Return a `torch::Tensor` backed by the mmap region (zero-copy until first
   mutation)

This path bypasses `fits_read_img` entirely. The performance advantage comes
from avoiding CFITSIO's per-row overhead and enabling parallel byte-swapping.

### 2. Signed-byte mmap path

Special case for `BYTE_IMG` with `BSCALE=1, BZERO=-128` (signed byte
convention). After the raw mmap read, applies `_xor_sign_bit_u8` (XOR 0x80
on every byte) in parallel. Faster than arithmetic negation.

### 3. CFITSIO fallback

Used when:

- Image is compressed (`fits_is_compressed_image` returns true)
- File path contains `[` (multi-extension syntax)
- mmap fails or is disabled
- `read_subset` on a compressed image

Calls `fits_read_img` in 128 MB chunks for large images. For compressed
images, CFITSIO handles decompression internally.

---

## Table read paths

### Buffered full-row path

When all selected columns span the full row width, uses `fits_read_tblbytes`
to read entire rows in one call, then unpacks columns in-memory. Fewer I/O
syscalls than per-column reads. Controlled by `TORCHFITS_TABLE_BUFFERED`.

### Column-by-column path

Uses `fits_read_col` per column. Handles VLA, BIT, logical, string, and
complex types with type-specific dispatch. Activated when column selection
doesn't cover the full row, or for types that require special handling.

### mmap table path

Opens raw fd, mmaps the file, uses `fits_get_hduaddrll` for the data offset,
then does parallel byte-swapping on the mmap region. Uses
`posix_madvise(POSIX_MADV_SEQUENTIAL)`. Creates tensor copies (not views).
Activated when `mmap=True` and the table layout permits it.

### Filtered read (predicate pushdown)

`read_columns_mmap_filtered` implements row filtering entirely in C++:

1. mmap the file
2. Pre-byte-swap filter target values to match raw FITS bytes
3. Scan rows in parallel via `at::parallel_for`
4. For EQ/NE on integers: compare raw bytes directly (zero bswap per row)
5. For GT/LT/GE/LE: byte-swap per row then compare

The Python layer (`where.py`) parses the SQL-like `where=` string into
`(column, operator, value)` tuples, which the C++ function consumes.

---

## Caching

Three tiers, all in-process:

### L0 — Handle cache (`UnifiedCache`)

LRU cache of open `fitsfile*` handles with reference counting. Validates
staleness via `stat()` (mtime/size/inode) at configurable intervals. Default
capacity: 100 files. Controlled by:

| Env var | Default | Description |
|---|---|---|
| `TORCHFITS_CACHE_VALIDATE` | `1` | Enable stat()-based validation |
| `TORCHFITS_CACHE_VALIDATE_INTERVAL_MS` | `1000` | Validation interval |

### L1 — SharedReadMeta

Per-path shared metadata: image_info, compressed status, scale info, HDU name
resolution, raw fd. Validated via `stat()`. Lives in a global map protected
by per-entry mutexes. Avoids repeated `open()` + header parsing for hot files.

| Env var | Default | Description |
|---|---|---|
| `TORCHFITS_SHARED_META_VALIDATE` | `1` | Enable validation |
| `TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS` | `1000` | Validation interval |

### L2 — Thread-local metadata

`static thread_local` `LocalHduMeta` map in `read_full_cached`, keyed by
`(meta_uid, hdu)`. Eliminates cross-thread mutex contention in hot loops.
Cleared when exceeding 4096 entries.

---

## Threading model

### GIL release

Every major I/O operation (`read_full`, `read_fits_table`,
`read_fits_table_filtered`, `read_hdus_batch`) releases the GIL during the
CFITSIO/raw-fd call and re-acquires before returning to Python. Safe for
multi-worker DataLoader usage.

### Batch image reads

`read_images_batch` spawns one `std::thread` per file (after the first).
Adaptive: if the first read is faster than thread-creation overhead, falls
back to sequential. Each thread opens its own `FITSFile` independently.

### Parallel byte-swapping

All mmap byte-swapping and table column decoding use `at::parallel_for`
(PyTorch's intra-op thread pool). Threshold for parallel sign-bit XOR:
256 KB (`TORCHFITS_XOR_PARALLEL_MIN_BYTES`).

### Handle cache thread safety

`UnifiedCache` uses `std::mutex`. `SharedReadMeta` uses per-entry
`std::mutex`. Thread-local `LocalHduMeta` avoids cross-thread contention.

---

## CFITSIO function mapping

### File open/close

`fits_open_file`, `fits_create_file`, `fits_close_file`

### HDU navigation

`fits_movabs_hdu`, `fits_movnam_hdu`, `fits_get_hdu_num`,
`fits_get_num_hdus`, `fits_get_hdu_type`

### Image I/O

`fits_get_img_paramll`, `fits_get_img_dim`, `fits_get_img_size`,
`fits_get_img_type`, `fits_get_img_equivtype`,
`fits_read_img`, `fits_read_pixll`, `fits_read_subset`,
`fits_create_img`, `fits_write_img`

### Table I/O

`fits_get_num_rows`, `fits_get_num_cols`, `fits_get_coltype`,
`fits_read_col`, `fits_read_tblbytes`,
`fits_create_tbl`, `fits_write_col`

### Header I/O

`fits_get_hdrspace`, `fits_read_keyn`, `fits_read_key`,
`fits_update_key`, `fits_write_history`, `fits_write_comment`,
`fits_hdr2str`

### Compression

`fits_is_compressed_image`, `fits_is_compressed_with_nulls` (via `dlsym`),
`fits_set_compression_type`, `fits_set_tile_dim`

### Checksums

`ffpcks` (write), `ffvcks` (verify)

### Metadata / mmap support

`fits_get_hduaddrll` — data offset for raw fd reads,
`fits_set_bscale`, `fits_free_memory`

---

## Data type mapping

| FITS BITPIX | torch dtype | Notes |
|---|---|---|
| 8 (BYTE_IMG) | `int8` | Signed byte convention via BZERO=-128 → XOR 0x80 |
| 16 | `int16` | Byte-swapped from big-endian |
| 32 | `int32` | Byte-swapped |
| 64 | `int64` | Byte-swapped |
| -32 | `float32` | IEEE 754, byte-swapped |
| -64 | `float64` | IEEE 754, byte-swapped |

Unsigned integers (BZERO=32768 for uint16, BZERO=2147483648 for uint32) are
handled by applying the offset rather than promoting to a wider signed type.
This preserves the narrow dtype for GPU transfer.

BSCALE/BZERO and TSCAL/TZERO scaling is applied by torchfits' own C++ code,
not by CFITSIO's internal scaling. This gives control over the exact output
dtype and avoids CFITSIO's float promotion.

---

## Security

`security.h` rejects filenames starting or ending with `|` (pipe character)
to prevent CFITSIO command execution via shell pipes. Leading `!` (overwrite
prefix) is stripped.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TORCHFITS_TABLE_SCANNER_THRESHOLD` | `10000` | Row count above which mmap scanner is used |
| `TORCHFITS_TABLE_HANDLE_CACHE` | `16` | Handle cache capacity for table reads |
| `TORCHFITS_TABLE_READER_CACHE` | `4` | TableReader instance cache size |
| `TORCHFITS_TABLE_BUFFERED` | `1` | Enable buffered full-row reads |
| `TORCHFITS_CACHE_VALIDATE` | `1` | Enable stat()-based handle cache validation |
| `TORCHFITS_CACHE_VALIDATE_INTERVAL_MS` | `1000` | Handle cache validation interval |
| `TORCHFITS_SHARED_META_VALIDATE` | `1` | Enable SharedReadMeta validation |
| `TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS` | `1000` | SharedReadMeta validation interval |
| `TORCHFITS_COLD_NOMMAP` | `0` | Force non-mmap reads |
| `TORCHFITS_COLD_NOCACHE` | `0` | Disable caching |
| `TORCHFITS_DEBUG_SCALE` | `0` | Debug scaling output |
| `TORCHFITS_XOR_PARALLEL_MIN_BYTES` | `262144` | Threshold for parallel sign-bit XOR |

---

## C++ extension surface (`torchfits._C`)

The native module exports the following classes and functions. This surface
is the stability contract — new symbols are private until promoted to
`torchfits.cpp.__all__`.

### Classes

| Class | Purpose |
|---|---|
| `FITSFile` | RAII wrapper around `fitsfile*`. Holds per-handle scale/compressed/image_info caches, raw fd, shared metadata. |
| `SubsetReader` | Reusable RAII reader for repeated cutout access from a single file. |
| `HDUInfo` | Struct: HDU index, type string, header dict. |
| `TableReader` | Column-oriented reader with mmap, buffered, and filtered paths. |

### Image functions

`read_full` (3 overloads), `read_full_cached`, `read_full_nocache`,
`read_full_unmapped`, `read_full_unmapped_raw`, `read_full_raw`,
`read_full_raw_with_scale`, `read_full_scaled_cpu`,
`read_full_numpy`, `read_full_numpy_cached`,
`read_images_batch`, `read_hdus_batch`, `read_hdus_sequence_last`

### Header functions

`read_header`, `read_header_string`, `read_header_dict`,
`resolve_hdu_name_cached`, `get_num_hdus`, `get_hdu_type`

### Table functions

`read_fits_table` (2 overloads), `read_fits_table_from_handle`,
`read_fits_table_rows`, `read_fits_table_rows_from_handle`,
`read_fits_table_rows_numpy`, `read_fits_table_rows_numpy_from_handle`,
`read_fits_table_filtered`,
`write_fits_table`, `append_fits_table_rows`, `insert_fits_table_rows`,
`update_fits_table_rows`, `update_fits_table_rows_mmap`,
`rename_fits_table_columns`, `drop_fits_table_columns`,
`delete_fits_table_rows`

### Write functions

`write_fits_file`, `write_fits_file_compressed_images`,
`write_hdu_checksums`, `verify_hdu_checksums`, `write_hdu_header_cards`

### Cache functions

`configure_cache`, `clear_file_cache`, `invalidate_file_cache`,
`clear_shared_read_meta_cache`, `get_cache_size`

### Utilities

`open_and_read_headers`, `open_fits_file`, `read_tensor_from_handle`,
`echo_tensor`

---

## PyTorch ABI compatibility

The extension performs a runtime check that the PyTorch version prefix
matches `TORCHFITS_TORCH_ABI` (a compile-time constant). Wheels built
against a different PyTorch ABI will fail to load with a clear error
message.
