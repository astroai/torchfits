# Migration from astropy to torchfits

Side-by-side replacements for common **FITS I/O** tasks. torchfits does not
mirror all of Astropy — see [Parity](parity.md) for scope. For runnable
workflows, start with [Examples](examples.md).

## Reading an image

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read image | `astropy.io.fits.getdata(path)` | `torchfits.read(path)` |
| Read image as tensor | `torch.from_numpy(astropy.io.fits.getdata(path))` | `torchfits.read_tensor(path, hdu=0)` |
| Read image with mmap | `astropy.io.fits.getdata(path, use_mmap=True)` | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| Read image to GPU | `torch.from_numpy(astropy.io.fits.getdata(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Read image + header | `hdul = astropy.io.fits.open(path); data = hdul[0].data; hdr = hdul[0].header` | `data, header = torchfits.read(path, hdu=0, return_header=True)` |

## Reading a table

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read all rows | `astropy.io.fits.getdata(path, ext=1)` | `torchfits.table.read(path, hdu=1)` |
| Read with WHERE | `t = astropy.io.fits.getdata(path, ext=1); mask = t['RA'] > 0; t[mask]` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| Read subset of columns | `astropy.io.fits.getdata(path, ext=1)[['RA','DEC']]` | `torchfits.table.read(path, hdu=1, columns=["RA","DEC"])` |
| Read tensor dict | `t = astropy.io.fits.getdata(path, ext=1); {n: torch.from_numpy(t[n]) for n in t.names}` | `torchfits.read_table(path, hdu=1)` |

## Writing

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Write tensor | `astropy.io.fits.PrimaryHDU(tensor.numpy()).writeto(path)` | `torchfits.write_tensor(path, tensor)` |
| Write table | `astropy.io.fits.BinTableHDU(table).writeto(path)` | `torchfits.table.write(path, table_dict)` |
| Write with header | `hdu = astropy.io.fits.PrimaryHDU(data); hdu.header['KEY'] = val; hdu.writeto(path)` | `torchfits.write(path, data, header={'KEY': val})` |

## Multi-HDU access

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Open MEF | `hdul = astropy.io.fits.open(path)` | `hdul = torchfits.open(path)` |
| Read by EXTNAME | `hdul['SCI'].data` | `torchfits.read_hdus(path, hdus=['SCI'])` |
| Read multiple HDUs | `[hdul[i].data for i in range(3)]` | `torchfits.read_hdus(path, hdus=[0, 1, 2])` |

## GPU transfer

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Image to GPU | `torch.from_numpy(astropy.io.fits.getdata(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Unsigned integer GPU (correct) | `torch.from_numpy(astropy.io.fits.getdata(path)).to(torch.int32).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` narrow H2D |

## Compression & checksums

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read compressed | `astropy.io.fits.open(path)[1].data` | `torchfits.read(path, hdu=1)` (auto-detected) |
| Verify checksums | manual | `torchfits.verify_checksums(path)` |
| Write checksums | `hdul.writeto(path, checksum=True)` | `torchfits.write_checksums(path)` |

## Performance notes

| Metric | astropy | torchfits |
|--------|---------|-----------|
| Large float32 image (16 MB, CPU) | 16.67 ms | 3.85 ms (**4.3× faster**) |
| Same read @ CUDA | 17.67 ms | 3.42 ms (**5.2× faster**) |
| Compressed Rice image (CPU) | 27.77 ms | 9.06 ms (**3.1× faster**) |
| 50× repeated 100×100 cutouts (CPU) | 75.36 ms | 4.68 ms (**16.7× faster**) |
| Table read (100k rows, 8 cols) | 6.74 ms | 95.3 μs (**70.6× faster**) |

*Benchmarks from `exhaustive_cuda_0.9.0_20260714_065950` (CANFAR staging, mmap on+off matrix). See [benchmarks.md](benchmarks.md) for methodology.*

## Key Behavioral Differences

### 1. Data Scaling & Type Promotion
* **Astropy**: Scaling (applying `BSCALE` and `BZERO` keywords) is applied on the CPU when the HDU data is initialized. Integer types (like `uint16` or `int32`) are promoted to double-precision `float64` in memory if the scaling yields floating-point numbers.
* **torchfits**: Defer scaling to the device with `torchfits.read(..., scale_on_device=True)` (forwarded via `**kwargs` into the read pipeline) or the low-level `torchfits.read_fast(..., scale_on_device=True)`. This transfers raw integers to GPU/MPS and applies `BSCALE`/`BZERO` in device registers, keeping host transfers small and returning `float32` instead of `float64`. (`read_tensor` does not take `scale_on_device`.)

### 2. Table Representation
* **Astropy**: Tables are represented as `astropy.table.Table` or `numpy.recarray`.
* **torchfits**: Tables are represented either as a Python dictionary of PyTorch Tensors (for `read_table`) or a PyArrow `Table` (for `torchfits.table.read`). Column types like variable-length arrays (VLAs) are translated to native Arrow list columns.

### 3. Thread-Safety & Multi-Processing
* **Astropy**: HDU handles (`HDUList`) are not thread-safe. Opening the same file in multiple background threads can lead to file descriptor and read-pointer conflicts.
* **torchfits**: C++ file handles and table readers are pooled in lock-guarded, global LRU caches, so concurrent reads are coordinated rather than racing on a shared descriptor. For PyTorch `DataLoader` workers, use the `torchfits.data` datasets with `make_loader`: map-style datasets read independently per worker, and iterable datasets shard work per `worker_id`. Pre-warm the caches with `torchfits.cache.optimize_for_dataset(paths)` to reduce lock contention.

