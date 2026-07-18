# Dataset API migration (0.7+)

**Read this if** you used `torchfits.FITSDataset` or `IterableFITSDataset` before
0.7.0. New code should start from [Examples → Dataset + make_loader](examples.md#dataset-make_loader).

Legacy classes were removed. Use `torchfits.data` instead.

## Image catalogs

| Before | After |
|---|---|
| `FITSDataset(paths, hdu=0, mode="image")` | `FitsImageDataset(paths, hdu=0)` |
| `IterableFITSDataset(paths, hdu=0, shuffle=True)` | `FitsImageIterableDataset(paths, shuffle=True)` |
| `FITSDataset(paths, transform=fn)` | `FitsImageDataset(paths, transform=fn)` |
| `FITSDataset(paths, target_transform=fn)` | `FitsImageDataset(paths, transform=fn)` — compose both in one callable |
| `FITSDataset(paths, preload=True)` | No preload; map-style datasets read lazily per `__getitem__` |

## Table catalogs

| Before | After |
|---|---|
| `FITSDataset(path, hdu=1, mode="table", columns=[...])` | `FitsTableDataset(path, hdu=1, columns=[...])` |
| `FITSDataset(path, hdu=1, mode="table", columns=[...], where="...")` | `FitsTableDataset(..., where="...")` |
| Large table, constant memory | `FitsTableIterableDataset(path, batch_size=50_000)` |

## Dropped kwargs

| Legacy kwarg | Replacement |
|---|---|
| `mode="auto"` | Pick explicitly: `FitsImageDataset` vs `FitsTableDataset` / `FitsTableIterableDataset` |
| `target_transform` | Single `transform=` on the new dataset classes |
| `preload=True` | Not supported — use lazy map-style reads or `FitsTableIterableDataset` |
| `cache_capacity` on dataset | `make_loader(..., optimize_cache=True)` or `torchfits.cache.optimize_for_dataset` |

## `mode="auto"`

Pick explicitly:

- Primary HDU image → `FitsImageDataset`
- Binary table HDU → `FitsTableDataset` or `FitsTableIterableDataset` for large files

## Cutouts / patches

| Goal | API |
|---|---|
| Fixed window list | `FitsCutoutDataset([(path, hdu, x, y, size), ...])` |
| Many cutouts, one mosaic | `open_subset_reader(path, hdu)` in a tight loop |

## DataLoader

```python
from torchfits.data import FitsImageDataset, make_loader

loader = make_loader(FitsImageDataset("*.fits", label_key="CLASS"), batch_size=32, num_workers=4)
```
