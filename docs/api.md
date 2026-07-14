# API Reference

`torchfits` covers FITS file I/O (images, tables, headers, compression) and
ML helpers (`torchfits.data`, `torchfits.transforms`). Sky-domain modelling
(WCS, coordinates, simulation) is out of scope — use downstream packages.

## How to use this page

1. **Find your task** in [Quick paths](#quick-paths) below (grouped by I/O, tables, ML).
2. **Skim [Core I/O](#core-io-reference)** for read/write patterns and GPU notes.
3. **Tables with filters** → [Table module](#table-module-reference) and [Predicate pushdown](#predicate-pushdown-syntax).
4. **Training loops** → [Data module](#data-module) and [Transforms](#transforms).

Two table surfaces exist and return different types:

- Root helpers (`read_table`, `read_table_rows`, `stream_table`, and `read`
  on a table HDU) return **`dict[str, torch.Tensor]`** — tensor-first, no
  PyArrow needed.
- The `torchfits.table` module is **Arrow-native**: `table.read`/`table.scan`
  return `pyarrow.Table` / `pyarrow.RecordBatch` and support `where=` predicate
  pushdown, dataframe/SQL interop, and mutations.

## Quick paths

### Images and files

| Goal | Entry point |
|---|---|
| Read N-D array/tensor | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| Read image or table (auto) | `torchfits.read(path, hdu=..., return_header=True)` |
| Cutout | `torchfits.read_subset(path, hdu, x1, y1, x2, y2)` |
| Multi-HDU arrays | `torchfits.read_hdus(path, hdus=[0, 1, 2])` |
| Repeated cutouts | `torchfits.open_subset_reader(path, hdu)` |
| Batch same HDU across files | `torchfits.read_batch(paths, hdu=0)` |
| Write tensor | `torchfits.write_tensor(path, tensor, overwrite=True)` |
| Header only | `torchfits.get_header(path, hdu=0)` |
| Multi-HDU handle | `with torchfits.open(path) as hdul: ...` |

### Tables

| Goal | Entry point |
|---|---|
| Read table (tensor dict) | `torchfits.read_table(path, hdu=1, columns=[...])` |
| Row slice (tensor dict) | `torchfits.read_table_rows(path, hdu=1, start_row=1, num_rows=N)` |
| Stream chunks (tensor dicts) | `torchfits.stream_table(path, hdu=1, chunk_rows=10000)` |
| Read table (Arrow) | `torchfits.table.read(path, hdu=1)` |
| Filter + project (`where=`) | `torchfits.table.read(..., where=...)` or `torchfits.table.scan(...)` |
| Arrow / Polars / DuckDB | `torchfits.table.read_polars(...)`, `scan_polars(...)`, `to_polars_lazy(...)`, `to_duckdb(...)` |
| In-place mutations | `torchfits.table.append_rows()`, `update_rows()`, `insert_column()`, `rename_columns()`, `drop_columns()` |

### ML training

| Goal | Entry point |
|---|---|
| Image map-style DataLoader | `torchfits.data.FitsImageDataset(paths)` |
| Image streaming DataLoader | `torchfits.data.FitsImageIterableDataset(paths)` |
| Table map-style (small catalog) | `torchfits.data.FitsTableDataset(path)` |
| Table streaming (large catalog) | `torchfits.data.FitsTableIterableDataset(path)` |
| Patch / cutout training | `torchfits.data.FitsCutoutDataset(cutouts)` |
| Sensible DataLoader factory | `torchfits.data.make_loader(ds)` |

## Core I/O Reference

### `torchfits.read`

```python
def read(
    path,
    hdu=None,
    device="cpu",
    mmap="auto",
    mode="auto",
    options=None,
    return_header=False,
    **kwargs,
): ...
```

Unified high-level reader for both images and tables.

* `hdu`: HDU index (0-based int), extension name (str), or `None` (auto-detects
  the first HDU with data).
* `device`: target device for image tensors (`"cpu"`, `"cuda"`, `"mps"`).
* `mmap`: `True`, `False`, or `"auto"`.
* `mode`: `"auto"` (detect from header), `"image"`, or `"table"`.
* `return_header`: if `True`, returns `(data, Header)`.
* **Returns**: a `torch.Tensor` for image HDUs, or `dict[str, torch.Tensor]`
  for table HDUs. (For an Arrow `Table` with `where=` pushdown, use
  [`torchfits.table.read`](#torchfitstableread).)

```python
import torchfits

image = torchfits.read("frame.fits", hdu=0, device="cpu")
data, header = torchfits.read("frame.fits", hdu=0, return_header=True)
columns = torchfits.read("catalog.fits", hdu=1, mode="table")  # dict[str, Tensor]
```

### `torchfits.read_tensor`

```python
def read_tensor(
    path,
    hdu=0,
    device="cpu",
    mmap=True,
    handle_cache=True,
    fp16=False,
    bf16=False,
    raw_scale=False,
    return_header=False,
    fallback_get_header=None,
): ...
```

Directly reads a FITS image HDU as a contiguous `torch.Tensor`.

* `hdu`: explicit integer HDU index (0-based). String/`"auto"` is not supported here.
* `fp16` / `bf16`: cast the result to half precision.
* `raw_scale`: if `True`, return raw storage values (e.g. `int16`) without
  applying `BSCALE`/`BZERO`. If `False` (default), values are scaled.
* **Returns**: a `torch.Tensor` (or `(Tensor, Header)` when `return_header=True`).

```python
import torchfits

x = torchfits.read_tensor("frame.fits", hdu=0, device="cuda")
raw = torchfits.read_tensor("counts.fits", hdu=0, raw_scale=True)  # keeps int dtype
```

> **Scaling on device.** The low-level fast path (`torchfits.read_fast`) accepts
> `scale_on_device=True` to apply `BSCALE`/`BZERO` in device registers. `read`
> forwards scaling policy through `options`/`**kwargs` to the pipeline;
> `read_tensor` itself has no `scale_on_device` parameter.

### `torchfits.read_subset`

```python
def read_subset(path, hdu, x1, y1, x2, y2, handle_cache_capacity=16): ...
```

Extracts a rectangular cutout from an image HDU, loading only the needed strides.

* `x1, y1`: lower corner (0-based, inclusive).
* `x2, y2`: upper corner (0-based, **exclusive** — half-open `[x1, x2) × [y1, y2)`).
  A `(0, 0, 2, 2)` request returns a `2 × 2` tensor.
* No `device` parameter — the cutout is returned on CPU; move it yourself.

```python
import torchfits

stamp = torchfits.read_subset("mosaic.fits", 0, 100, 100, 164, 164)  # 64x64
stamp = stamp.to("cuda")
```

### `torchfits.open_subset_reader`

```python
def open_subset_reader(path, hdu=0, device="cpu"): ...
```

Context manager returning a reusable callable `reader(x1, y1, x2, y2) -> Tensor`.
Reuses the file handle across many cutouts — ideal for training loops over one
large mosaic.

```python
import torchfits

with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    a = reader(0, 0, 64, 64)
    b = reader(64, 64, 128, 128)
```

### `torchfits.read_hdus`

```python
def read_hdus(path, hdus, *, device="cpu", mmap=True, return_header=False): ...
```

Reads several image extensions in one pass. `hdus` is a list of indexes and/or
EXTNAME strings. Returns a list of tensors.

```python
import torchfits

sci, err, dq = torchfits.read_hdus("mef.fits", ["SCI", "ERR", "DQ"])
```

### `torchfits.read_table`

```python
def read_table(
    path,
    hdu=1,
    columns=None,
    start_row=1,
    num_rows=-1,
    device="cpu",
    mmap="auto",
    cache_capacity=10,
    handle_cache_capacity=16,
    fast_header=True,
    return_header=False,
): ...
```

Reads a table HDU into a **`dict[str, torch.Tensor]`** (column name → tensor).
`start_row` is 1-based; `num_rows=-1` reads to the end.

```python
import torchfits

cols = torchfits.read_table("catalog.fits", hdu=1, columns=["RA", "DEC", "MAG"])
print(cols["RA"].shape)
```

### `torchfits.read_table_rows`

```python
def read_table_rows(
    path, hdu=1, start_row=1, num_rows=1000, columns=None,
    device="cpu", mmap=True, cache_capacity=10,
    handle_cache_capacity=16, fast_header=True, return_header=False,
): ...
```

Convenience wrapper for a contiguous row range. `num_rows` must be `> 0`.
Returns a `dict[str, torch.Tensor]`.

```python
import torchfits

first_1k = torchfits.read_table_rows("catalog.fits", hdu=1, start_row=1, num_rows=1000)
```

### `torchfits.stream_table`

```python
def stream_table(
    file_path,
    hdu=1,
    columns=None,
    start_row=1,
    num_rows=-1,
    chunk_rows=65536,
    mmap=False,
    max_chunks=None,
): ...
```

Generator that yields `dict[str, torch.Tensor]` chunks of `chunk_rows` rows,
keeping memory bounded for large catalogs. Note the first argument is
`file_path`.

```python
import torchfits

for chunk in torchfits.stream_table("huge.fits", hdu=1, chunk_rows=100_000):
    process(chunk["FLUX"])  # dict[str, Tensor]
```

### `torchfits.read_batch` / `torchfits.get_batch_info`

```python
def read_batch(file_paths, hdu=0, device="cpu", *, strict=False): ...
def get_batch_info(file_paths): ...
```

`read_batch` reads the same HDU from many files and stacks them into one batched
tensor (files must share shape/dtype). `get_batch_info` inspects shape/dtype
consistency across files before batching.

```python
import glob, torchfits

files = sorted(glob.glob("stamps/*.fits"))
info = torchfits.get_batch_info(files)
batch = torchfits.read_batch(files, hdu=0, device="cuda")  # [N, ...]
```

### Slicing N-dimensional data (3D & 4D)

FITS arrays are arbitrary N-D (from `NAXIS`/`NAXISn`). Reads return standard
tensors, so slice with normal indexing:

```python
import torchfits

cube = torchfits.read_tensor("cube.fits", hdu=0)   # [wavelength, y, x]
sub = cube[1:3, :, :]                               # sub-cube
hyper = torchfits.read_tensor("stokes.fits", hdu=0) # [pol, vel, y, x]
stokes_i = hyper[0]                                 # 3D cube
```

### `torchfits.write` / `torchfits.write_tensor`

```python
def write(path, data, header=None, overwrite=False, compress=False): ...
def write_tensor(path, tensor, header=None, overwrite=False, compress=False): ...
```

`write` accepts an image `Tensor`, a table dict, or an `HDUList`.
`write_tensor` is the image shortcut and requires a `torch.Tensor`.
`compress=True` or a codec string (e.g. `"RICE"`) enables tile compression for
tensor image payloads.

```python
import torch, torchfits

torchfits.write_tensor("out.fits", torch.randn(256, 256), overwrite=True)
torchfits.write("rice.fits", torch.randn(512, 512), overwrite=True, compress="RICE")
```

### MEF (Multi-Extension FITS) workflows

`torchfits.open` returns an `HDUList` for low-level access. HDU mutation helpers
edit files in place:

```python
def insert_hdu(path, data, index=1, header=None, compress=False): ...
def replace_hdu(path, hdu, data, header=None, compress=False): ...
def delete_hdu(path, hdu, compress=False): ...
```

```python
import torch, torchfits

with torchfits.open("mef.fits") as hdul:
    n = len(hdul)

torchfits.insert_hdu("mef.fits", torch.zeros(64, 64), index=1, header={"EXTNAME": "MASK"})
torchfits.replace_hdu("mef.fits", "SCI", torch.ones(64, 64))
torchfits.delete_hdu("mef.fits", "MASK")
```

### Checksums & integrity

```python
def write_checksums(path, hdu=0): ...
def verify_checksums(path, hdu=0): ...   # -> {"datastatus": int, "hdustatus": int, "ok": bool}
```

```python
import torchfits

torchfits.write_checksums("frame.fits", hdu=0)
status = torchfits.verify_checksums("frame.fits", hdu=0)
assert status["ok"]
```

### HDU & header classes

Low-level object model, re-exported at the package root and from `torchfits.hdu`.

| Class | Role |
|---|---|
| `Header` | Dict-like FITS header; `header["NAXIS1"]`, `.get(key, default)`, card iteration. |
| `Card` | A single header card (keyword, value, comment). |
| `HDUList` | Sequence of HDUs from `torchfits.open(path)`; index by position or EXTNAME. |
| `TensorHDU` | Image/cube HDU with lazy tensor access. |
| `TableHDU` | Tensor-backed table HDU (`hdu[name]`, `get_string_column`, `get_vla_column`). |
| `TableHDURef` | Lazy, file-backed table handle for on-demand reads. |

```python
import torchfits

with torchfits.open("mef.fits") as hdul:
    sci = hdul["SCI"]
    header = sci.header
    print(header["NAXIS1"], header.get("BUNIT", "counts"))
```

## Table Module Reference

`torchfits.table` is the **Arrow-native** table surface: predicate pushdown,
streaming, dataframe/SQL interop, and in-place mutations. Reads return
`pyarrow.Table` / `pyarrow.RecordBatch`. PyArrow is a core dependency.

### Readers & scanners

#### `torchfits.table.read`

```python
def read(
    path, hdu=1, columns=None, row_slice=None, rows=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=False, apply_fits_nulls=True,
    backend="auto",
): ...
```

Reads a table HDU as a `pyarrow.Table` with optional projection, row selection,
and `where=` filtering.

```python
import torchfits

t = torchfits.table.read("catalog.fits", hdu=1, columns=["RA", "DEC", "MAG"])
bright = torchfits.table.read("catalog.fits", hdu=1, where="MAG < 20 AND DEC > 0")
some = torchfits.table.read("catalog.fits", hdu=1, rows=[0, 5, 42])
```

#### `torchfits.table.scan`

```python
def scan(
    path, hdu=1, columns=None, row_slice=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=False, apply_fits_nulls=True,
    backend="auto",
): ...
```

Generator of `pyarrow.RecordBatch` — constant-memory streaming over big tables.

```python
import torchfits

for batch in torchfits.table.scan("huge.fits", hdu=1, where="Z > 1.0", batch_size=50_000):
    process(batch)
```

#### `torchfits.table.scan_torch`

```python
def scan_torch(
    path, hdu=1, columns=None, row_slice=None, batch_size=65536,
    mmap=True, device="cpu", non_blocking=True, pin_memory=False,
): ...
```

Streams `dict[str, torch.Tensor]` batches, optionally moved to `device`.
No `where=` (use `scan` for filtering).

```python
import torchfits

for chunk in torchfits.table.scan_torch("huge.fits", hdu=1, device="cuda"):
    train_step(chunk["FLUX"])
```

#### `torchfits.table.schema`

```python
def schema(path, hdu=1, columns=None, where=None, ...): ...
```

Returns the `pyarrow.Schema`. With `where=None` it is inferred from header cards
only (no row read).

```python
import torchfits

sch = torchfits.table.schema("catalog.fits", hdu=1)
print(sch.names)
```

#### `torchfits.table.reader` / `scanner` / `dataset`

* `reader(path, hdu=1, ...)` → `pyarrow.RecordBatchReader` (streaming, includes
  FITS metadata by default).
* `dataset(data, **kwargs)` → `pyarrow.dataset.Dataset` from a path or Arrow object.
* `scanner(data, *, columns=None, where=None, filter=None, batch_size=65536, use_threads=True, **kwargs)`
  → a `pyarrow.dataset.Scanner`.

```python
import torchfits

rdr = torchfits.table.reader("catalog.fits", hdu=1, columns=["RA", "DEC"])
tbl = rdr.read_all()
```

### Backends

Reads accept `backend=` from `torchfits.table.TABLE_BACKENDS`, which is
`frozenset({"auto", "cpp", "torch"})`:

* `"auto"` (default) — pick the fastest safe path.
* `"cpp"` — force the native C++ reader.
* `"torch"` — force the Python/torch streaming path.

```python
import torchfits

assert torchfits.table.TABLE_BACKENDS == frozenset({"auto", "cpp", "torch"})
t = torchfits.table.read("catalog.fits", hdu=1, backend="cpp")
```

### Writing

```python
def write(path, data, *, schema=None, header=None, overwrite=False,
          extname=None, table_type="binary"): ...
def write_parquet(where, data, *, stream=False, compression="zstd",
                  row_group_size=None, **kwargs): ...
```

`table.write` writes a `dict` of columns (arrays/tensors/lists) to a FITS binary
or ASCII table. `write_parquet` exports a FITS path or Arrow data to Parquet.

```python
import numpy as np, torchfits

torchfits.table.write(
    "out.fits",
    {"RA": np.array([1.0, 2.0]), "MAG": np.array([20.0, 21.0])},
    overwrite=True,
)
torchfits.table.write_parquet("out.parquet", "catalog.fits", compression="zstd")
```

### In-place mutations

All mutations invalidate Python-side and C++ handle/metadata caches to prevent
stale reads. `hdu` accepts an index or EXTNAME.

| Function | Signature (trailing keyword-only args after `*`) |
|---|---|
| `append_rows` | `append_rows(path, rows, hdu=1)` |
| `insert_rows` | `insert_rows(path, rows, *, row, hdu=1)` |
| `delete_rows` | `delete_rows(path, row_slice, *, hdu=1)` |
| `update_rows` | `update_rows(path, rows, row_slice, hdu=1, *, mmap="auto")` |
| `insert_column` | `insert_column(path, name, values, *, hdu=1, index=None, format=None, unit=None, dim=None, tnull=None, tscal=None, tzero=None)` |
| `replace_column` | `replace_column(path, name, values, *, hdu=1, format=None, unit=None, dim=None, tnull=None, tscal=None, tzero=None)` |
| `rename_columns` | `rename_columns(path, mapping, hdu=1)` |
| `drop_columns` | `drop_columns(path, columns, hdu=1)` |

`rows` is a `dict[str, values]`; `row_slice` is a `slice` or `(start, num)`
tuple (0-based start).

```python
import numpy as np, torchfits

torchfits.table.append_rows("cat.fits", {"RA": np.array([9.0]), "MAG": np.array([18.0])})
torchfits.table.insert_rows("cat.fits", {"RA": np.array([0.0]), "MAG": np.array([0.0])}, row=0)
torchfits.table.delete_rows("cat.fits", slice(10, 20))
torchfits.table.update_rows("cat.fits", {"MAG": np.array([15.0])}, row_slice=(0, 1))
torchfits.table.insert_column("cat.fits", "FLAG", np.zeros(3, dtype=np.int16), unit="")
torchfits.table.rename_columns("cat.fits", {"MAG": "MAG_G"})
torchfits.table.drop_columns("cat.fits", ["FLAG"])
```

### Dataframe & SQL interop

Convert whole FITS tables to dataframe engines (optional dependencies):

```python
def read_polars(path, *, rechunk=False, **kwargs): ...       # -> FITSPolarsFrame
def scan_polars(path, *, batch_size=65536, rechunk=False, **kwargs): ...  # -> iterator[pl.DataFrame]
def to_polars_lazy(data, *, rechunk=False, **kwargs): ...    # materializes then -> pl.LazyFrame
def to_duckdb(data, relation_name="fits_table", connection=None, **kwargs): ...
def duckdb_query(data, query, relation_name="fits_table", connection=None, return_arrow=True, **kwargs): ...
```

* `read_polars` returns a `FITSPolarsFrame` wrapping a `pl.DataFrame` and keeping
  FITS metadata (TFORM, TUNIT, TDIM, TNULL, TSCAL, TZERO, HDU identity).
* `scan_polars` is a genuine streaming path (one `pl.DataFrame` per batch).
* `to_polars_lazy` materializes the whole table before wrapping it in a
  `LazyFrame`; use `scan_polars` for real streaming.
* `duckdb_query` accepts exactly one `SELECT`/`EXPLAIN` statement.

```python
import torchfits

frame = torchfits.table.read_polars("catalog.fits", hdu=1)
res = torchfits.table.duckdb_query("catalog.fits", "SELECT COUNT(*) FROM fits_table WHERE MAG < 20")
```

### Tensor-dict interop

Convert a `dict[str, Tensor]` (from `read_table`/`stream_table`) to Arrow-backed
frames. Available at the package root and in `torchfits.interop`. **`decode_bytes`
defaults to `False`** (byte columns stay as `uint8` tensors unless you opt in).

```python
def to_arrow(data, decode_bytes=False, encoding="ascii", strip=True, vla_policy="list"): ...
def to_pandas(data, decode_bytes=False, encoding="ascii", strip=True, vla_policy="object"): ...
def to_polars(data, decode_bytes=False, encoding="ascii", strip=True, vla_policy="list", *, rechunk=False): ...
```

```python
import torchfits

cols = torchfits.read_table("catalog.fits", hdu=1)
arrow_tbl = torchfits.to_arrow(cols, decode_bytes=True)
df = torchfits.to_pandas(cols, decode_bytes=True)
```

### Predicate pushdown syntax

The `where=` parameter accepts SQL-like predicates:

- Comparison: `=`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `AND`, `OR`, `NOT`
- Lists: `IN (val1, val2)`, `NOT IN (...)`
- Ranges: `BETWEEN min AND max`
- Null checks: `IS NULL`, `IS NOT NULL`

```python
import torchfits

t = torchfits.table.read(
    "catalog.fits", hdu=1,
    where="MAG_G < 20 AND DEC > 0 AND FLAGS IN (0, 1)",
)
```

`torchfits` chooses C++ pushdown when the predicate reduces to simple
comparisons/`BETWEEN` on supported column types; otherwise it reads the needed
columns and filters with PyArrow compute. Predicates over VLA/vector columns are
rejected.

### Environment variables

Verified in `src/torchfits/_table/cache.py` and `src/torchfits/__init__.py`:

| Variable | Default | Effect |
|---|---|---|
| `TORCHFITS_TABLE_HANDLE_CACHE` | `1` | Enable the LRU cache of C++ file handles (`0`/`false`/`no`/`off` disables). |
| `TORCHFITS_TABLE_HANDLE_CACHE_SIZE` | `8` | Max cached file handles. |
| `TORCHFITS_TABLE_READER_CACHE` | `1` | Enable the LRU cache of C++ `TableReader`s. |
| `TORCHFITS_TABLE_READER_CACHE_SIZE` | `8` | Max cached readers. |
| `TORCHFITS_CFITSIO_CACHE_FILES` | `32` | CFITSIO cache file-count limit (applied when set). |
| `TORCHFITS_CFITSIO_CACHE_MB` | `256` | CFITSIO cache size in MB (applied when set). |

## Cache & performance

Root I/O cache helpers plus tuning under `torchfits.cache`:

```python
def clear_file_cache(*, data=True, handles=True, meta=True,
                     hdu_types=True, stats=True, cpp=True, cpp_module=None): ...
def get_cache_performance(): ...  # hit/miss stats for handle & metadata caches
```

`torchfits.cache` (auto-tuning and dataset warm-up):

* `cache.configure_for_environment()` — size caches for the detected
  environment (local / HPC / cloud / GPU workstation).
* `cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)` — pre-warm
  caches for a training file list. Handle/reader caches are guarded by locks, so
  warming them once up front reduces lock contention across DataLoader workers.
* `cache.stats()` / `cache.clear()` — inspect and clear all torchfits caches.

```python
import torchfits.cache
from torchfits.data import FitsImageDataset, make_loader

torchfits.cache.optimize_for_dataset(training_files)   # warm caches once
ds = FitsImageDataset(training_files)
loader = make_loader(ds, batch_size=64, num_workers=4, pin_memory=True)
```

> **DataLoader workers.** There is no `TORCHFITS_WORKER_HANDLE` feature — do not
> rely on it. For multi-process loading, use `torchfits.data` datasets with
> `make_loader` (it calls `cache.optimize_for_dataset` when the dataset exposes a
> `files` attribute), and call `cache.optimize_for_dataset` /
> `cache.configure_for_environment` yourself for custom datasets. Map-style
> datasets are worker-safe; iterable datasets shard work per worker (see
> [Worker sharding](#worker-sharding)).

## `torchfits.where`

Helpers for parsing and evaluating table predicate strings (the same grammar
used by `where=`). Useful for validating or introspecting predicates.

| Symbol | Description |
|---|---|
| `parse_where_expression(where)` | Parse a predicate string into an AST tuple. |
| `where_columns_from_ast(ast)` | List column names referenced by an AST. |
| `evaluate_where(ast, data)` | Evaluate an AST against a mapping of NumPy arrays → boolean mask. |
| `tokenize_where_expression(where)` | Tokenize a predicate string. |
| `parse_where_literal(text)` | Parse a single literal value. |
| `normalize_where_syntax(where)` | Normalize predicate syntax. |
| `where_identifier_re` | Compiled regex for valid column identifiers. |

```python
import numpy as np
from torchfits.where import parse_where_expression, evaluate_where, where_columns_from_ast

ast = parse_where_expression("MAG < 20 AND DEC > 0")
print(where_columns_from_ast(ast))          # ['MAG', 'DEC']
mask = evaluate_where(ast, {"MAG": np.array([19.0, 21.0]), "DEC": np.array([1.0, 1.0])})
```

## Transforms

`torchfits.transforms` provides PyTorch-native, gradient-safe transforms for
astronomical images, spectra, and time series. All classes inherit from
`FITSTransform`, which defines `forward(x, mask=None)` (via `__call__`) and, where
meaningful, an `inverse()` method for reversible operations. Transforms are
`torch.nn.Module`s, so `.to(device)` works.

### Full catalog

| Transform | Group | Notes |
|---|---|---|
| `Compose(transforms)` | utility | Chain transforms; inverse runs in reverse order. |
| `FITSTransform` | base | Base class; subclass to add custom transforms. |
| `ArcsinhStretch(a=1.0)` | stretch | Invertible high-dynamic-range stretch. |
| `LogStretch(a=1000.0, eps=1e-9)` | stretch | Log stretch for non-negative data. |
| `SqrtStretch()` | stretch | Variance-stabilizing sqrt. |
| `ZScaleNormalize(contrast=0.25, dim=(-2,-1))` | normalize | IRAF zscale contrast mapping. |
| `RobustNormalize(dim=(-2,-1))` | normalize | Median/MAD standardization. |
| `BackgroundSubtract(...)` | normalize | Subtract median background. |
| `PercentileClipNormalize(lower_pct=1.0, upper_pct=99.0, dim=(-2,-1))` | normalize | Clip to percentiles, scale to [0,1]. |
| `MinMaxNormalize(dim=(-2,-1))` | normalize | Linear min–max to [0,1]. |
| `GlobalScalarNorm(...)` | normalize | Normalize by cached global scalar stats. |
| `FITSHeaderScale(...)` | header-aware | Apply `BSCALE`/`BZERO` from a header. |
| `FITSHeaderNormalize(...)` | header-aware | Header-driven normalization. |
| `FITSScaleColumns(scales)` | table | Apply per-column `(scale, zero)` to table dicts. |
| `TNullToNan(nulls)` | table | Map per-column TNULL sentinels to NaN. |
| `ContinuumNormalize(...)` | spectral | Divide by fitted continuum. |
| `ContinuumRemoval(...)` | spectral | Subtract fitted/spline baseline. |
| `DopplerShift(z=0.0)` | spectral | Resample spectrum to redshift `z`. |
| `SpectralBinning(...)` | spectral | Average adjacent channels. |
| `BandMath(func, band_dim=0)` | spectral | Arbitrary per-band arithmetic. |
| `SavitzkyGolayFilter(...)` | baseline | Local polynomial smoothing. |
| `AsymmetricLeastSquares(...)` | baseline | ALS baseline (C++-accelerated). |
| `RunningPercentile(...)` | baseline | Rolling-percentile baseline. |
| `UpperEnvelopeContinuum(...)` | baseline | Upper-envelope continuum estimate. |
| `AlphaShapeContinuum(...)` | baseline | Alpha-shape continuum estimate. |
| `WaveletDecompose(...)` | baseline | Haar wavelet decomposition. |
| `PhaseFold(period=1.0, n_bins=64, t0=0.0)` | time | Fold a light curve on a period. |
| `SigmaClip(n_sigma=3.0, max_iter=5, dim=(-2,-1), fill="mean")` | outliers | Iterative sigma clipping. |
| `AsymmetricSigmaClip(...)` | outliers | Separate low/high sigma thresholds. |

Helper functions: `safe_arcsinh`, `safe_log`, `estimate_background`,
`zscale_limits`.

```python
import torch
from torchfits.transforms import Compose, RobustNormalize, ArcsinhStretch

pipeline = Compose([RobustNormalize(), ArcsinhStretch(a=5.0)])
x = torch.randn(1, 128, 128)
y = pipeline(x)
```

### Detailed formulations

#### Stretches (stateless, exact roundtrip)

##### `ArcsinhStretch`
* $$y = \frac{\operatorname{arcsinh}(a \cdot x)}{\operatorname{arcsinh}(a)}$$
  where $a$ controls the soft threshold.
* **Useful for**: high dynamic range images (galaxies beside bright stars);
  preserves color ratios across bands.
* **Avoid when**: noise is high — near-zero values are amplified.
* Invertible via $\sinh(y \cdot \operatorname{arcsinh}(a)) / a$. CPU/GPU, any dim.

##### `LogStretch`
* $$y = \frac{\log(1 + a \cdot \max(0, x))}{\log(1 + a)}$$
* **Useful for**: high dynamic range structure (radio maps, diffraction patterns).
* **Avoid when**: negatives are physical (they clamp to zero).
* Invertible via $(\exp(y \cdot \log(1 + a)) - 1)/a$ for non-negative values.

##### `SqrtStretch`
* $$y = \sqrt{x}$$
* **Useful for**: variance stabilization of Poisson-dominated images.
* **Avoid when**: inputs contain negatives.
* Invertible via $y^2$.

#### Normalizers (data-dependent, invertible)

##### `ZScaleNormalize`
* Fits the IRAF zscale linear mapping to the pixel dispersion to find
  $[z_1, z_2]$ display thresholds. Defaults to spatial dims `dim=(-2, -1)`.
* **Useful for**: contrast-stretching to $[0, 1]$ for NN inputs/display.
* **Avoid when**: absolute photometry must be preserved.
* Invertible only if scale/offset are cached.

##### `RobustNormalize`
* $$y = \frac{x - \operatorname{median}(x)}{1.4826 \cdot \operatorname{MAD}(x)}$$
* **Useful for**: ML standardization; robust to cosmic rays / bright stars.
* **Avoid when**: absolute zero flux must be preserved.
* Invertible; stats computed along `dim`.

##### `BackgroundSubtract`
* $$y = x - \operatorname{median}(x)$$
* **Useful for**: uniform sky/continuum removal.
* **Avoid when**: background is highly structured. Fully invertible.

##### `PercentileClipNormalize`
* Clips to $[P_\text{low}, P_\text{high}]$ then scales linearly to $[0, 1]$.
* **Useful for**: highly variable brightness (transient surveys).
* Lossy outside the clipped range.

#### Spectral transforms (1D)

##### `ContinuumNormalize`
* $$y = \frac{x}{C(x)}$$ with $C(x)$ a sigma-clipped polynomial continuum.
* **Useful for**: isolating absorption lines in stellar/quasar spectra.
* Invertible by multiplying the cached continuum; operates along `dim=-1`.

##### `ContinuumRemoval`
* $$y = x - C(x)$$ with $C(x)$ a polynomial or cubic-spline baseline.
* **Useful for**: subtracting broad instrument response / extinction.
* Fully invertible via addition.

##### `DopplerShift`
* Resamples a spectrum from coordinate $\nu$ to $\nu(1+z)$ by linear interpolation.
* **Avoid when**: sharp features need high accuracy (minor smoothing).
* Invertible via shift by $1/(1+z)$.

##### `SpectralBinning`
* Averages adjacent channels by an integer factor.
* Lossy; inverse uses nearest-neighbor upsampling.

#### Baseline / continuum estimators (additive, invertible)

These split a 1D signal into a smooth baseline and residual: $x = B(x) + R(x)$.

##### `SavitzkyGolayFilter`
* Local low-degree polynomial smoothing via 1D convolution. Fully invertible via
  residual addition; avoid on sharp step functions.

##### `AsymmetricLeastSquares`
* Baseline $y$ minimizing
  $$\sum_i w_i (x_i - y_i)^2 + \lambda \sum_i (\Delta^2 y_i)^2,\quad
    w_i = p \text{ if } x_i > y_i \text{ else } 1-p\ (p \ll 1).$$
* **Useful for**: baselines under positive emission features. C++-accelerated,
  fully invertible via residual addition.

#### Time-domain

##### `PhaseFold`
* Maps time $t$ to phase $\phi = ((t - t_0)/P) \bmod 1$ and resamples onto $N$ bins.
* **Useful for**: transits, pulsations, orbital phase. Lossy (no inverse).

#### Outlier rejection

##### `SigmaClip`
* Iteratively clips pixels outside $[-n\sigma, n\sigma]$ and fills with group
  mean/median. Lossy.

##### `AsymmetricSigmaClip`
* Separate low/high sigma thresholds (e.g. keep positive flares). Lossy.

### Device placement

Split transforms between CPU dataset prep and GPU batch processing:

* **CPU (in `Dataset.__getitem__`)** — slow/iterative fits (`AsymmetricLeastSquares`,
  spline fits, iterative `SigmaClip`). Runs in parallel across DataLoader workers.
* **GPU (after batch transfer)** — fast vectorized ops (`ArcsinhStretch`,
  `LogStretch`, `RobustNormalize`, `DopplerShift`):

```python
from torchfits.transforms import ArcsinhStretch

device = "cuda"
stretch = ArcsinhStretch(a=5.0).to(device)

for batch in dataloader:
    x = batch["image"].to(device)
    output = model(stretch(x))
```

## Data Module

Map-style and iterable `torch.utils.data.Dataset` implementations plus
`make_loader`. **Chooser:** small image list → `FitsImageDataset`; large table
→ `FitsTableIterableDataset`; fixed cutouts → `FitsCutoutDataset`. See
[Examples → Which dataset class?](examples.md#which-dataset-class).

Map-style datasets are worker-safe (each worker calls `__getitem__`
independently). Iterable datasets shard across workers (see below).

```python
from torchfits.data import FitsImageIterableDataset, make_loader

ds = FitsImageIterableDataset("observations/*.fits", shuffle=True, seed=42)
loader = make_loader(ds, batch_size=32, num_workers=4, pin_memory=True)

for batch in loader:        # torch.Tensor [B, 1, H, W] (with add_channel_dim=True)
    ...
```

### Datasets

| Class | Mode | Use case |
|---|---|---|
| `FitsImageDataset(paths, hdu=0, label_key=None, labels=None, transform=None, device="cpu", mmap=True, add_channel_dim=True)` | map | Small-to-medium labelled image catalogs. Reads once per `__getitem__`. Labels from `labels=` or a header `label_key`. |
| `FitsImageIterableDataset(paths, hdu=0, transform=None, device="cpu", mmap=True, shuffle=False, seed=0, add_channel_dim=True)` | iterable | Multi-worker sharded image loading; partitions file indices by `worker_id`. |
| `FitsTableDataset(path, hdu=1, columns=None, where=None, transform=None, device="cpu", mmap="auto")` | map | Row-indexable catalog. **Loads the full filtered table at init** — small/medium only. Supports `columns=` and `where=`. |
| `FitsTableIterableDataset(path, hdu=1, columns=None, where=None, batch_size=65536, transform=None, device="cpu", mmap="auto")` | iterable | Streams rows via `torchfits.table.scan`; constant memory. Workers take alternating scan batches. |
| `FitsCutoutDataset(cutouts, transform=None, device="cpu", add_channel_dim=True)` | map | Fixed windows: `(path, hdu, x, y, size)` or `(path, hdu, x1, y1, x2, y2)` with **half-open** `[x1,x2)×[y1,y2)` bounds. |

```python
from torchfits.data import FitsTableIterableDataset, make_loader

ds = FitsTableIterableDataset("huge.fits", hdu=1, where="MAG < 22", batch_size=50_000)
loader = make_loader(ds, batch_size=256, num_workers=4)
```

### Helpers

| Symbol | Description |
|---|---|
| `fits_collate_fn(batch)` | Stacks `(image, label)` tuples and `dict[str, Tensor]` rows. Raises `ValueError` on non-tensor columns (strings, VLA lists) — drop those columns or pass a custom `collate_fn`. |
| `make_loader(ds, batch_size=32, shuffle=None, num_workers=0, pin_memory=False, prefetch_factor=2, drop_last=False, *, optimize_cache=True, avg_file_size_mb=10.0, **loader_kwargs)` | Wraps `DataLoader`. When `optimize_cache=True` and the dataset exposes a `files` attribute, calls `cache.optimize_for_dataset(...)` once. Map-style datasets default to `shuffle=True`; iterable datasets to `shuffle=False`. |

### Worker sharding

**Images** — `FitsImageIterableDataset.__iter__` partitions file indices by
`worker_id`:

```text
per_worker = total // num_workers
remainder  = total %  num_workers
start      = worker_id * per_worker + min(worker_id, remainder)
size       = per_worker + (1 if worker_id < remainder else 0)
```

Verified in `tests/test_data.py::TestMultiWorkerDataLoader` (subprocess,
`num_workers=2`, 8 files).

**Tables** — `FitsTableIterableDataset.__iter__` assigns scan batches to workers
via `batch_idx % num_workers == worker_id`. Row order within a batch is
preserved; workers do not interleave individual rows. Verified in
`tests/test_data.py::TestMultiWorkerDataLoader::test_multiprocess_table_iterable`.

## Deprecated & removed

| Symbol / feature | Status | Replacement |
|---|---|---|
| `read_image` | deprecated | `read_tensor` |
| `read_fast` | low-level fast path | prefer `read` / `read_tensor` |
| `read_large_table` | removed | `stream_table` or `read_table` |
| `"cpp_numpy"` table backend | removed (now `ValueError`) | `backend="cpp"` |
| `torchfits.FITSDataset`, `torchfits.IterableFITSDataset` | removed | `torchfits.data` typed datasets |
| `TensorHDU.stats()` | removed (never implemented) | compute stats from the tensor directly |
| `torchfits.hdu.TensorFrame` / `torch_frame` inheritance | removed | tensor/list column mappings + Arrow/Polars interop |

## Limitations

- VLA columns are read via buffered I/O; mmap reads and in-place updates are not
  supported for VLA.
- Scaled table columns are not supported for mmap updates; use the buffered path.
- Non-CPU tensors are copied to host before FITS writes.
- Compressed writes support tensor image payloads; dict HDU payloads for
  compressed writes must contain tensor image data.
- `torchfits` intentionally does not expose WCS/sphere/domain modelling APIs.
