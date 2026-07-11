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
| Large float32 image (16 MB, CPU) | 6.09 ms | 3.93 ms (**1.6× faster**) |
| Compressed Rice image (CPU) | 9.36 ms | 8.99 ms (**1.04× faster**) |
| Table read (100k rows, 8 cols) | 59.41 ms | 86.9 μs (**~680× faster**) |
| Table predicate (1M narrow) | 13.07 ms | 14.10 ms (**1.08× slower — `predicate_filter` deficit**) |
| 50× repeated 100×100 cutouts (CPU) | 4.76 ms | 4.63 ms (**1.03× faster**) |

*Benchmarks from `exhaustive_cuda_0.7.0_20260711_055635` (CANFAR staging, mmap on+off matrix). Narrow `predicate_filter` cases lag fitsio by up to ~1.17× on CPU — see [benchmarks.md](benchmarks.md) deficit table.*
