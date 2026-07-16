# Data Module

`torchfits.data` provides `torch.utils.data.Dataset` and `IterableDataset`
implementations for FITS images and tables, plus a `make_loader` factory.

---

## Choosing a Dataset

| Your data | Catalog size | Use | Why |
|---|---|---|---|
| Image files (paths list) | Any | `FitsImageDataset` (map) | Random access, shuffle, simple training loops |
| Image files (paths list) | Any, many workers | `FitsImageIterableDataset` | Deterministic sharding, no duplication across workers |
| One table HDU | Fits in RAM | `FitsTableDataset` | Row-indexed `dict[str, Tensor]` with predicate pushdown |
| One table HDU | Too large for RAM | `FitsTableIterableDataset` | Constant-memory streaming via `table.scan` |
| Fixed `(path, hdu, x, y, size)` cutouts | Any | `FitsCutoutDataset` | Patch training from a mosaic |

---

## `FitsImageDataset`

Map-style dataset for image catalogs. Reads one tensor per `__getitem__` call.

```python
from torchfits.data import FitsImageDataset, make_loader

ds = FitsImageDataset(
    "observations/*.fits",
    hdu=0,
    label_key="CLASS",        # header keyword → int label
    transform=None,            # optional callable
    device="cpu",
    mmap=True,
    add_channel_dim=True,      # [H, W] → [1, H, W]
)
loader = make_loader(ds, batch_size=32, num_workers=4)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `str \| list[str]` | *(required)* | File paths or glob pattern |
| `hdu` | `int` | `0` | HDU index to read |
| `label_key` | `str \| None` | `None` | Header keyword for classification labels |
| `labels` | `list[int] \| None` | `None` | Explicit per-file labels (overrides `label_key`) |
| `transform` | `callable \| None` | `None` | Applied to each image tensor |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool \| str` | `True` | Memory-mapped reads |
| `add_channel_dim` | `bool` | `True` | Prepend channel dimension for 2D images |

**Returns per item:** `(image: Tensor, label: Tensor)`

!!! info "When to use"
    Use `FitsImageDataset` for small-to-medium image catalogs where the file
    list fits in memory. Each worker reads independently — no GIL contention.
    For very large catalogs (100k+ files), prefer `FitsImageIterableDataset`
    which shards file indices deterministically.

---

## `FitsImageIterableDataset`

Iterable dataset for multi-worker sharded image loading. Each worker processes
a deterministic subset — every file is seen exactly once per epoch.

```python
from torchfits.data import FitsImageIterableDataset

ds = FitsImageIterableDataset(
    "observations/*.fits",
    hdu=0,
    shuffle=True,
    seed=42,
    add_channel_dim=True,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `str \| list[str]` | *(required)* | File paths or glob pattern |
| `hdu` | `int` | `0` | HDU index |
| `transform` | `callable \| None` | `None` | Applied to each image tensor |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool \| str` | `True` | Memory-mapped reads |
| `shuffle` | `bool` | `False` | Shuffle file order per epoch |
| `seed` | `int` | `0` | Base seed for shuffling |
| `add_channel_dim` | `bool` | `True` | Prepend channel dimension |

**Returns per item:** `Tensor` (no label — use this for self-supervised or when labels come from elsewhere).

!!! info "When to use"
    Use when you have many files and want deterministic multi-worker loading
    without file duplication. Unlike map-style + `num_workers`, each worker
    gets a disjoint shard of files.

---

## `FitsTableDataset`

Map-style dataset for row-indexable FITS catalogs. Loads the full filtered
table at `__init__` — use only when the catalog fits in RAM.

```python
from torchfits.data import FitsTableDataset

ds = FitsTableDataset(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",        # predicate pushdown at load time
    transform=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | `1` | Table HDU index |
| `columns` | `list[str] \| None` | `None` | Column names (None = all) |
| `where` | `str \| None` | `None` | SQL-like predicate for row filtering |
| `transform` | `callable \| None` | `None` | Applied to each row dict |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool \| str` | `"auto"` | Memory-mapped reads |

**Returns per item:** `dict[str, Tensor | Any]` — one row as a column-name-keyed dict.

!!! info "When to use"
    Use for catalogs up to a few million rows that fit in memory. The
    `where=` predicate is pushed down so only matching rows are loaded. For
    larger catalogs, use `FitsTableIterableDataset`.

---

## `FitsTableIterableDataset`

Streams table rows in constant memory. Each `__getitem__` yields one
`dict[str, Tensor]` row.

```python
from torchfits.data import FitsTableIterableDataset

ds = FitsTableIterableDataset(
    "survey.fits",
    hdu=1,
    columns=["RA", "DEC"],
    where="DEC > 0",
    batch_size=65536,
    mmap="auto",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | `1` | Table HDU index |
| `columns` | `list[str] \| None` | `None` | Column projection |
| `where` | `str \| None` | `None` | SQL-like predicate |
| `batch_size` | `int` | `65536` | Rows per internal scan batch |
| `transform` | `callable \| None` | `None` | Applied to each row dict |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool \| str` | `"auto"` | Memory-mapped reads |

**Returns per item:** `dict[str, Tensor | Any]` — one row.

!!! info "When to use"
    Use for large catalogs that don't fit in memory. Workers shard by scan
    batch index (`batch_idx % num_workers == worker_id`), so each row is
    seen exactly once. Row order within a batch is preserved.

---

## `FitsCutoutDataset`

Map-style dataset for fixed cutout windows from one or more FITS images.

```python
from torchfits.data import FitsCutoutDataset

cutouts = [
    ("mosaic.fits", 0, 100, 200, 164),   # (path, hdu, x, y, size)
    ("mosaic.fits", 0, 300, 400, 164),
]
ds = FitsCutoutDataset(cutouts, transform=None, device="cpu")
```

Accepts `(path, hdu, x, y, size)` or `(path, hdu, x1, y1, x2, y2)` tuples.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cutouts` | `Sequence` | *(required)* | List of cutout specs |
| `transform` | `callable \| None` | `None` | Applied to each cutout tensor |
| `device` | `str` | `"cpu"` | Torch device |
| `add_channel_dim` | `bool` | `True` | Prepend channel dimension |

**Returns per item:** `Tensor`

!!! info "When to use"
    Use for patch training from a mosaic. Each cutout is read independently
    via `read_subset` using **pixel coordinates** — torchfits does not
    interpret WCS, so you must supply pixel offsets directly. If most
    cutouts come from the same file, consider `open_subset_reader`
    directly for better performance.

---

## `make_loader()`

Wraps a dataset in a `DataLoader` with sensible defaults and optional cache
warm-up.

```python
from torchfits.data import make_loader

loader = make_loader(
    ds,
    batch_size=32,
    num_workers=4,
    pin_memory=True,       # faster CPU → GPU transfer
    optimize_cache=True,   # warm handle caches before iteration
    shuffle=True,          # default for map-style; False for iterable
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `Dataset \| IterableDataset` | *(required)* | A torchfits dataset |
| `batch_size` | `int` | `32` | Batch size |
| `shuffle` | `bool \| None` | `None` | Auto: `True` for map-style, `False` for iterable |
| `num_workers` | `int` | `0` | Worker processes |
| `pin_memory` | `bool` | `False` | Pin for GPU transfers |
| `prefetch_factor` | `int` | `2` | Prefetch per worker |
| `drop_last` | `bool` | `False` | Drop incomplete batches |
| `optimize_cache` | `bool` | `True` | Call `cache.optimize_for_dataset` |
| `avg_file_size_mb` | `float` | `10.0` | For cache sizing |
| `collate_fn` | `callable` | `fits_collate_fn` | Batch collation |
| `**loader_kwargs` | | | Passed to `DataLoader` |

**Returns:** `torch.utils.data.DataLoader`

---

## `fits_collate_fn`

Default collation function for torchfits datasets:

- `list[Tensor]` → stacked `Tensor` (all must have the same shape)
- `list[(Tensor, Tensor)]` → `(stacked_images, stacked_labels)`
- `list[dict[str, Tensor]]` → `dict[str, stacked_Tensor]`

All tensors in a batch must have identical shapes — `torch.stack` is used
under the hood. Raises `ValueError` on non-tensor columns (strings, VLA
lists). For variable-size images, either pad externally before collation or
supply a custom `collate_fn`.

---

## Worker Sharding

### Images (`FitsImageIterableDataset`)

File indices are partitioned by `worker_id`:

```
per_worker = total // num_workers
remainder  = total %  num_workers
start      = worker_id * per_worker + min(worker_id, remainder)
size       = per_worker + (1 if worker_id < remainder else 0)
```

Every file is seen exactly once per epoch. Shuffling permutes within each
worker's shard (epoch-seeded).

### Tables (`FitsTableIterableDataset`)

Scan batches are assigned to workers via `batch_idx % num_workers == worker_id`.
Row order within a batch is preserved. Workers do not interleave individual
rows.

!!! tip "Multi-worker tip"
    Use `num_workers=2` or more for parallel reads. Each worker independently
    opens files and reads data — no GIL contention on the I/O path. Add
    `persistent_workers=True` (passed via `**loader_kwargs`) to keep workers
    alive between epochs — this avoids re-opening file handles each epoch and
    significantly speeds up epoch startup for mmap-backed datasets.
