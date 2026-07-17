# API Reference

`torchfits` covers FITS file I/O (images, tables, headers, compression) and
ML helpers (`torchfits.data`, `torchfits.transforms`). Sky-domain modelling
(WCS, coordinates, simulation) is out of scope.

---

## Quick Paths

### Images and Files

| Goal | Entry point | Reference |
|---|---|---|
| Read N-D array as tensor | `read_tensor(path, hdu=0, device="cpu", mmap=True)` | [Core I/O](api-core-io.md#read_tensor) |
| Read image or table (auto-detect) | `read(path, hdu=None, return_header=False)` | [Core I/O](api-core-io.md#read) |
| Rectangular cutout | `read_subset(path, hdu, x1, y1, x2, y2)` | [Core I/O](api-core-io.md#read_subset) |
| Repeated cutouts from one file | `open_subset_reader(path, hdu=0)` | [Core I/O](api-core-io.md#open_subset_reader) |
| Multiple HDUs at once | `read_hdus(path, hdus=[0, 1, 2])` | [Core I/O](api-core-io.md#read_hdus) |
| Write a tensor | `write_tensor(path, tensor, header=None, overwrite=False)` | [Core I/O](api-core-io.md#write_tensor) |
| Read header only | `get_header(path, hdu=0)` | [Core I/O](api-core-io.md#get_header) |
| Multi-HDU context manager | `open(path, mode="r")` | [Core I/O](api-core-io.md#open) |
| Batch-read many files | `read_batch(file_paths, hdu=0)` | [Core I/O](api-core-io.md#read_batch) |

### Tables

| Goal | Entry point | Reference |
|---|---|---|
| Read table (returns `pyarrow.Table`) | `table.read(path, hdu=1, columns=None, where=None)` | [Tables](api-tables.md#tableread) |
| Stream table in chunks | `table.scan(path, hdu=1, chunk_rows=65536)` | [Tables](api-tables.md#tablescan) |
| Read as tensor dict | `read_table(path, hdu=1, columns=None)` | [Core I/O](api-core-io.md#read_table) |
| Stream as tensor dicts | `stream_table(path, hdu=1, chunk_rows=65536)` | [Core I/O](api-core-io.md#stream_table) |
| Read row range | `read_table_rows(path, hdu=1, start_row=1, num_rows=1000)` | [Core I/O](api-core-io.md#read_table_rows) |
| Direct to Polars | `table.read_polars(path, hdu=1)` | [Tables](api-tables.md#polars) |
| Streaming Polars batches | `table.scan_polars(path, hdu=1)` | [Tables](api-tables.md#polars) |
| DuckDB SQL | `table.duckdb_query(path, sql, hdu=1)` | [Tables](api-tables.md#duckdb) |

### ML Training

| Goal | Entry point | Reference |
|---|---|---|
| Image map-style dataset | `FitsImageDataset(paths, hdu=0, label_key=None)` | [Data](api-data.md#fitsimagedataset) |
| Image iterable (multi-worker) | `FitsImageIterableDataset(paths, shuffle=False)` | [Data](api-data.md#fitsimageiterabledataset) |
| Table map-style (fits in RAM) | `FitsTableDataset(path, hdu=1)` | [Data](api-data.md#fitstabledataset) |
| Table streaming (large) | `FitsTableIterableDataset(path, hdu=1, batch_size=65536)` | [Data](api-data.md#fitstableiterabledataset) |
| Cutout patches | `FitsCutoutDataset(cutouts)` | [Data](api-data.md#fitscutoutdataset) |
| DataLoader with defaults | `make_loader(dataset, batch_size=32)` | [Data](api-data.md#make_loader) |
| Image stretches, normalizers | `from torchfits.transforms import …` | [Transforms](api-transforms.md) |
| Spectral preprocessing | `ContinuumNormalize`, `DopplerShift`, ... | [Transforms](api-transforms.md) |
| Shell inspect / convert | `torchfits info|header|verify|…` | [CLI](cli.md) |

---

## Reference Pages

| Page | What it covers |
|---|---|
| [CLI](cli.md) | `torchfits` command-line tools, exit codes, MEF defaults |
| [Core I/O](api-core-io.md) | `read`, `read_tensor`, `read_subset`, `read_hdus`, `write_tensor`, `write`, `open`, headers, HDU mutation, checksums, batch reads, cache |
| [Tables](api-tables.md) | `table.read`, `table.scan`, predicate pushdown, backend selection, mutations, Polars/DuckDB/Pandas/Arrow interop, schema |
| [Data](api-data.md) | `FitsImageDataset`, `FitsImageIterableDataset`, `FitsTableDataset`, `FitsTableIterableDataset`, `FitsCutoutDataset`, `make_loader`, worker sharding |
| [Transforms](api-transforms.md) | Transform classes (callable protocol, not `nn.Module`) with verified math, parameters, invertibility, and when-to-use guidance |
| [Architecture](architecture.md) | C++/Python layering, I/O paths, caching, threading, CFITSIO mapping, environment variables |

---

## Package Namespaces

| Namespace | Purpose |
|---|---|
| `torchfits` (root) | I/O functions and HDU classes |
| `torchfits.hdu` | HDU/header types (`Header`, `Card`, `HDUList`, …) |
| `torchfits.table` | Table I/O, mutation, interop |
| `torchfits.data` | Dataset classes and loader factory |
| `torchfits.transforms` | Transform classes |
| `torchfits.cache` | Cache configuration and management |
| `torchfits.where` | Predicate parser and evaluator |
| `torchfits.cpp` | Low-level native compatibility surface |

`torchfits.cpp` is the low-level native compatibility surface used by
performance-sensitive downstream packages. Its `__all__` is the
function-level compatibility contract; new compiled-extension symbols are
private until promoted there.

---

## HDU Types

| Class | Description |
|---|---|
| `TensorHDU` | Image HDU with lazy `.data` (returns `DataView`) and `.header` |
| `TableHDU` | In-memory table HDU with tensor columns |
| `TableHDURef` | Lazy file-backed table handle |
| `Header` | Dict-like FITS header preserving card order and semantics |
| `Card` | Single FITS header card: `Card(key, value, comment)` |

---

## Removed names

| Old | Use instead |
|---|---|
| `read_fast(...)` | `read(...)` or `read_tensor(...)` |
| `read_image(...)` | `read_tensor(...)` |
| Root transform classes (e.g. `torchfits.SpectralBinning`) | `torchfits.transforms.*` |

Transforms are not re-exported at the package root. Import them from
:mod:`torchfits.transforms`. HDU helpers are available as root names and via
:mod:`torchfits.hdu`.

---

## Limitations

- VLA columns use buffered I/O; mmap reads and in-place updates are not supported.
- Scaled table columns do not support mmap updates; use the buffered path.
- Non-CPU tensors are copied to host before FITS writes.
- Compressed writes accept tensor image payloads; dict payloads must contain tensor image data.
- `torchfits` intentionally does not expose WCS, sphere, or domain modelling APIs.
