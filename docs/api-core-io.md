# Core I/O Reference

Fundamental read and write operations for FITS images, tables, headers,
multi-extension files, checksums, and caching.

---

## `read()`

Unified FITS reader. Auto-detects image or table HDUs.

```python
torchfits.read(path, hdu=None, device="cpu", mmap="auto", mode="auto",
               options=None, return_header=False, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| PathLike` | *(required)* | FITS file path |
| `hdu` | `int \| str \| None` | `None` | HDU index, EXTNAME, or `None` for auto |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool \| str` | `"auto"` | `True`, `False`, or `"auto"` |
| `mode` | `str` | `"auto"` | `"auto"`, `"image"`, or `"table"` |
| `return_header` | `bool` | `False` | Return `(data, Header)` tuple |

**Returns:** `torch.Tensor` (images), `dict[str, torch.Tensor]` (tables), or
tuple if `return_header=True`.

!!! info "When to use"
    Use `read()` for quick exploration or when you want auto-detection. For
    explicit tensor reads, prefer `read_tensor()`. For dataframe workflows
    (predicate pushdown `where=`, Arrow/Polars), prefer `table.read()` —
    root `read()` does not accept `where=` and does **not** return Arrow.

```python
# Image with header
data, hdr = torchfits.read("image.fits", hdu=0, return_header=True)

# Auto-detect table → tensor-column dict (not a pyarrow.Table)
columns = torchfits.read("catalog.fits", hdu=1)
```

---

## `read_tensor()`

Read any N-dimensional FITS array directly as a PyTorch Tensor.

```python
torchfits.read_tensor(path, hdu=0, device="cpu", mmap=True, handle_cache=True,
                      fp16=False, bf16=False, raw_scale=False,
                      return_header=False, fallback_get_header=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | `0` | Explicit integer HDU index (required) |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool` | `True` | Memory-mapped reads (faster for repeated access) |
| `handle_cache` | `bool` | `True` | Cache the open FITS handle |
| `fp16` | `bool` | `False` | Read as float16 |
| `bf16` | `bool` | `False` | Read as bfloat16 |
| `raw_scale` | `bool` | `False` | Skip BSCALE/BZERO, return native storage dtype |
| `return_header` | `bool` | `False` | Return `(tensor, Header)` |

**Returns:** `torch.Tensor` (or tuple if `return_header=True`).

!!! info "When to use"
    This is the primary function for reading FITS images as tensors. Use it
    when you want a single tensor (1D spectra, 2D images, 3D cubes). For
    multi-extension files, use `read_hdus()`. For cutouts, use
    `read_subset()`.

!!! tip "GPU reads"
    Pass `device="cuda"` or `device="mps"` to place the result on device.
    Generic BSCALE/BZERO scaling still yields `float32` unless you opt into
    storage dtypes with `raw_scale=True` (e.g. int8 / uint16 parity with fitsio).

```python
# Read to GPU
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cuda")

# Native integer dtype (matches fitsio)
tensor = torchfits.read_tensor("image.fits", hdu=0, raw_scale=True)

# 3D cube
cube = torchfits.read_tensor("cube.fits", hdu=0)
# cube.shape = (nz, ny, nx)
```

---

## `read_subset()`

Read a rectangular pixel subset from an image HDU.

```python
torchfits.read_subset(path, hdu, x1, y1, x2, y2, handle_cache_capacity=16)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | *(required)* | HDU index |
| `x1, y1, x2, y2` | `int` | *(required)* | Half-open pixel window `[x1,x2)×[y1,y2)` |

**Returns:** `torch.Tensor`

```python
stamp = torchfits.read_subset("mosaic.fits", hdu=0, x1=0, y1=0, x2=256, y2=256)
```

---

## `open_subset_reader()`

Reusable reader for repeated cutout access on a single image HDU. Opens the
file once; each call reads a region without re-opening.

```python
torchfits.open_subset_reader(path, hdu=0, device="cpu")
```

**Returns:** Context manager yielding a `SubsetReader`. Call `reader(x1, y1, x2, y2)` for each cutout.

```python
with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    stamp1 = reader(0, 0, 256, 256)
    stamp2 = reader(128, 128, 256, 256)
```

!!! info "When to use"
    Use `open_subset_reader` when you need many cutouts from the same large
    file (e.g., training on patches from a mosaic). It avoids repeatedly
    opening and closing the FITS handle.

---

## `read_hdus()`

Read multiple HDUs from a single FITS file as a list of tensors.

```python
torchfits.read_hdus(path, hdus, *, device="cpu", mmap=True, return_header=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdus` | `list[int \| str]` | *(required)* | HDU indices or EXTNAME strings |
| `device` | `str` | `"cpu"` | Target device |
| `mmap` | `bool` | `True` | Memory-mapped reads |
| `return_header` | `bool` | `False` | Return list of `(tensor, Header)` tuples |

**Returns:** `list[torch.Tensor]`

```python
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
```

---

## `read_table()`

Root alias of [`table.read_torch`](api-tables.md#tableread_torch): read a FITS
table (dataframe on disk) as column tensors for training.

```python
torchfits.read_table(path, hdu=1, columns=None, start_row=1, num_rows=-1,
                     device="cpu", mmap="auto", cache_capacity=10,
                     handle_cache_capacity=16, fast_header=True,
                     return_header=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | `1` | Table HDU index |
| `columns` | `list[str] \| None` | `None` | Column names (None = all) |
| `start_row` | `int` | `1` | First row (1-indexed) |
| `num_rows` | `int` | `-1` | Number of rows (-1 = all) |
| `device` | `str` | `"cpu"` | Target device |

**Returns:** `dict[str, torch.Tensor]`

!!! info "When to use"
    Prefer `table.read_torch()` in new code. Use this root alias when you want
    dataframe columns as tensors. For Arrow/Polars dataframes with `where=`,
    use `table.read()` / `table.read_polars()`.

---

## `read_table_rows()`

Sugar on `read_table` / `table.read_torch` for a contiguous row range
(`num_rows` must be `> 0`).

```python
torchfits.read_table_rows(path, hdu=1, start_row=1, num_rows=1000,
                          columns=None, device="cpu", mmap=True)
```

**Returns:** `dict[str, torch.Tensor]`

---

## `stream_table()`

Iterate over dataframe rows in fixed-size tensor chunks. Prefer
[`table.scan_torch`](api-tables.md#tablescan_torch) in new code (same idea;
`scan_torch` adds device/pin_memory options and uses `batch_size` /
`row_slice`).

```python
torchfits.stream_table(file_path, hdu=1, columns=None, start_row=1,
                       num_rows=-1, chunk_rows=65536, mmap=False,
                       max_chunks=None)
```

**Yields:** `dict[str, Tensor | list]` per chunk.

```python
for chunk in torchfits.stream_table("survey.fits", hdu=1, chunk_rows=100_000):
    process(chunk)
```

---

## `read_batch()`

Read the same HDU from multiple FITS files as a batched tensor.

```python
torchfits.read_batch(file_paths, hdu=0, device="cpu", *, strict=False)
```

**Returns:** `torch.Tensor` (stacked batch).

```python
tensors = torchfits.read_batch(["img1.fits", "img2.fits"], hdu=0)
```

---

## `get_batch_info()`

Inspect shape and dtype consistency across files before batch reading.

```python
torchfits.get_batch_info(file_paths)
```

**Returns:** `dict` with shape, dtype, and file info.

---

## `open()`

Multi-HDU context manager for low-level HDU/header access.

```python
torchfits.open(path, mode="r")
```

**Returns:** `HDUList` context manager.

```python
with torchfits.open("mef.fits") as hdul:
    primary = hdul[0]          # TensorHDU
    sci = hdul["SCI"]          # TensorHDU by EXTNAME
    data = sci.data            # DataView (lazy)
    header = sci.header        # Header (dict-like)
```

!!! warning "mmap writes require flush"
    When writing via `open()` with mmap-backed HDUs, changes are held in
    memory-mapped pages and **not persisted to disk** until you call
    `hdul.flush()`. Always flush before closing the context manager if
    you modified data in-place.

---

## `get_header()`

Read only the FITS header from an HDU.

```python
torchfits.get_header(path, hdu=None)
```

**Returns:** `Header` (dict-like).

```python
header = torchfits.get_header("image.fits", hdu=0)
print(header["EXPTIME"])  # e.g. 300.0
```

---

## Writes

### `write()`

Write a tensor, numpy array, dict table, or HDUList to FITS.

```python
torchfits.write(path, data, header=None, overwrite=False, compress=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| PathLike` | *(required)* | Output path |
| `data` | `Tensor \| ndarray \| dict \| HDUList` | *(required)* | Data to write |
| `header` | `dict \| Header \| None` | `None` | FITS header key-value pairs |
| `overwrite` | `bool` | `False` | Overwrite existing file |
| `compress` | `bool \| str` | `False` | `True`, `"gzip"`, `"rice"`, etc. |

### `write_tensor()`

Write a single PyTorch Tensor to a FITS image extension.

```python
torchfits.write_tensor(path, tensor, header=None, overwrite=False, compress=False)
```

```python
torchfits.write_tensor("out.fits", tensor, header={"OBJECT": "M31"}, overwrite=True)
```

---

## HDU Mutation

```python
torchfits.insert_hdu(path, data, index=1, header=None, compress=False)
torchfits.replace_hdu(path, hdu, data, header=None, compress=False)
torchfits.delete_hdu(path, hdu, compress=False)
```

---

## Checksums

```python
torchfits.write_checksums(path, hdu=0)
result = torchfits.verify_checksums(path, hdu=0)
# result: dict with "datastatus", "hdustatus", "ok"
```

---

## Cache Utilities

```python
# Root I/O cache
torchfits.get_cache_performance()
torchfits.clear_file_cache(data=True, handles=True, meta=True, cpp=True)

# Higher-level cache manager
torchfits.cache.configure_for_environment()
torchfits.cache.get_cache_stats()
torchfits.cache.clear_cache()
torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)
```

!!! tip "Training loops"
    Call `torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=...)`
    before `DataLoader` epochs to warm handle caches. When using
    `make_loader(..., optimize_cache=True)` (the default), this happens
    automatically.
