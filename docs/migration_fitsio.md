# Migration from fitsio to torchfits

Side-by-side replacements for common **FITS I/O** tasks. fitsio remains the
right tool for some metadata workflows; torchfits targets tensor pipelines and
PyTorch training. See [Benchmarks](benchmarks.md#performance-deficits) for cases
where fitsio still wins on narrow table predicates.

## Reading an image

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Read image | `fitsio.read(path)` | `torchfits.read(path)` |
| Read image as tensor | `torch.from_numpy(fitsio.read(path))` | `torchfits.read_tensor(path, hdu=0)` |
| Read image with mmap | *(fitsio has no mmap mode; use slice reads)* | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| Read image to GPU | `torch.from_numpy(fitsio.read(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Read header | `fitsio.read_header(path)` | `torchfits.get_header(path, hdu=0)` |

## Reading a table

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Read all rows | `fitsio.read(path, ext=1)` | `torchfits.table.read(path, hdu=1)` |
| Read with WHERE | `fitsio.FITS(path)[1].where("RA > 0")` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| Read subset of columns | `fitsio.read(path, ext=1, columns=['RA','DEC'])` | `torchfits.table.read(path, hdu=1, columns=["RA","DEC"])` |
| Read tensor dict | `{n: torch.from_numpy(fitsio.read(path, ext=1, columns=n)) for n in names}` | `torchfits.read_table(path, hdu=1)` |
| Stream rows | `for row in fitsio.FITS(path)[1]: ...` | `for chunk in torchfits.stream_table(path, hdu=1, chunk_rows=10000): ...` |

## Writing

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Write tensor | `fitsio.write(path, tensor.numpy())` | `torchfits.write_tensor(path, tensor)` |
| Write table | `fitsio.write(path, table_dict)` | `torchfits.table.write(path, table_dict)` |
| Append rows | `f.append(table_dict)` | `torchfits.table.append_rows(path, rows, hdu=1)` |
| Update rows | `f[1].update(rows, row_slice)` | `torchfits.table.update_rows(path, rows, row_slice, hdu=1)` |
| Insert column | `f[1].insert_column(name, values)` | `torchfits.table.insert_column(path, name, values, hdu=1)` |
| Rename column | `f[1].rename_column(old, new)` | `torchfits.table.rename_columns(path, {old: new}, hdu=1)` |
| Drop column | `f[1].delete_column(name)` | `torchfits.table.drop_columns(path, [name], hdu=1)` |

## Predicate pushdown

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Column filter | `fitsio.FITS(path)[1].where("MAG_G < 20")` | `torchfits.table.read(path, hdu=1, where="MAG_G < 20")` |
| Compound predicate | `fitsio.FITS(path)[1].where("MAG_G < 20 AND DEC > 0")` | `torchfits.table.read(path, hdu=1, where="MAG_G < 20 AND DEC > 0")` |
| IN list | `fitsio.FITS(path)[1].where("id IN (1,2,3)")` | `torchfits.table.read(path, hdu=1, where="id IN (1, 2, 3)")` |

## GPU transfer

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Image to GPU | `torch.from_numpy(fitsio.read(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Narrow dtype GPU (uint16) | `torch.from_numpy(fitsio.read(path)).to(torch.uint16).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda", raw_scale=True)` |

## Performance notes

| Metric | fitsio | torchfits |
|--------|--------|-----------|
| Large float32 image (16 MB, CPU) | 5.89 ms | 3.85 ms (**1.53× faster**) |
| Same read @ CUDA | 5.50 ms | 3.42 ms (**1.61× faster**) |
| Compressed Rice image (CPU) | 9.43 ms | 9.06 ms (**1.04× faster**) |
| 50× repeated 100×100 cutouts (CPU) | 4.94 ms | 4.68 ms (**1.09× faster**) |
| Table read (100k rows, 8 cols) | 59.84 ms | 95.3 μs (**627.6× faster**) |

*Benchmarks from `exhaustive_cuda_20260716_191255` (CANFAR staging, mmap on+off
matrix, 0 TorchFits deficits). See [benchmarks.md](benchmarks.md) for methodology.*

## Key Behavioral Differences

### 1. Multi-Processing Fork Safety
* **fitsio**: High-performance CFITSIO reads into NumPy; there is no true OS
  `mmap` toggle (`memmap=` is ignored). Long-lived `FITS` handles across
  `DataLoader` forks can still share CFITSIO state poorly — prefer reopen-per-
  worker or torchfits datasets.
* **torchfits**: Use the `torchfits.data` datasets with `make_loader` for multi-worker loading. Map-style datasets are worker-safe (each worker reads independently); iterable datasets shard work per `worker_id`. To reduce lock contention on the shared handle/reader caches, call `torchfits.cache.optimize_for_dataset(paths)` (also invoked by `make_loader` when the dataset exposes a `files` attribute) or `torchfits.cache.configure_for_environment()` before training.

### 2. Table Mutations
* **fitsio**: In-place updates can corrupt FITS tables if not handled carefully, and do not invalidate read buffers automatically.
* **torchfits**: Functions under `torchfits.table` (like `append_rows`, `update_rows`, `insert_column`, `rename_columns`, `drop_columns`) perform parallel columns reconstruction and automatically invalidate all Python-side and C++ handle/meta caches, preventing stale reads.

### 3. Variable Length Arrays (VLAs)
* **fitsio**: Reads VLA columns as NumPy arrays of object pointers (`object` dtype).
* **torchfits**: Translates VLAs to standard PyArrow `ListArray` types, allowing high-performance, memory-contiguous vectorization on the CPU.

