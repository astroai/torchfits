# Migration from fitsio to torchfits

## Reading an image

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Read image | `fitsio.read(path)` | `torchfits.read(path)` |
| Read image as tensor | `torch.from_numpy(fitsio.read(path))` | `torchfits.read_tensor(path, hdu=0)` |
| Read image with mmap | `fitsio.read(path, memmap=True)` | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
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
| Large float32 image (16 MB, CPU) | ~1.8 ms | ~1.8 ms (**parity**) |
| Compressed Rice image (CPU) | ~10.3 ms | ~9.2 ms (**1.12× faster**) |
| Table read (100k rows, mixed) | ~4.3 ms | ~0.05 ms (**94× faster**) |
| Table predicate (1M narrow) | ~6.2 ms | ~10.1 ms (**0.6× — fitsio faster on narrow predicates**) |
| 50× repeated 100×100 cutouts (CPU) | ~3.3 ms | ~3.1 ms (**1.09× faster**) |

*Benchmarks from `exhaustive_mmap_v060b2_20260708_232039` (lab, H100 CUDA). torchfits dominates table I/O and is competitive on image reads. Narrow predicate_filter lags fitsio by ~1.5×–2.1× on small tables; see [benchmarks.md](benchmarks.md) for details.*
