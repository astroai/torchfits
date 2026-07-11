# Migration from astropy to torchfits

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
| Large float32 image (16 MB, CPU) | 7.60 ms | 3.93 ms (**1.9× faster**) |
| Same read @ CUDA | 8.18 ms | 3.19 ms (**2.6× faster**) |
| Compressed Rice image (CPU) | 28.14 ms | 8.99 ms (**3.1× faster**) |
| 50× repeated 100×100 cutouts (CPU) | 76.04 ms | 4.63 ms (**16× faster**) |
| Table read (100k rows, 8 cols) | 6.32 ms | 86.9 μs (**73× faster**) |

*Benchmarks from `exhaustive_cuda_0.7.0_20260711_055635` (CANFAR staging, mmap on+off matrix). See [benchmarks.md](benchmarks.md) for methodology.*
