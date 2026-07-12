# torchfits

**FITS → PyTorch tensors**, without NumPy glue. Images, tables, compression,
multi-extension files, and ML-ready `Dataset` / `transforms` on top of a
vendored CFITSIO engine.

## Start here

1. **[Install](install.md)** — `pip install torchfits` (+ PyTorch with CUDA/MPS if needed).
2. **Run one example** — `pixi run python examples/example_image.py` (or copy the snippet below).
3. **Pick your workflow** — use the table below, then open the linked page.

```python
import torchfits

tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")
print(tensor.shape, tensor.dtype)
```

## I want to...

| Goal | Start with | Then read |
|---|---|---|
| Read a single image or cutout | `read_tensor`, `read_subset` | [API → Core I/O](api.md#core-io), [`example_image.py`](../examples/example_image.py) |
| Filter a catalog (`WHERE`, columns) | `torchfits.table.read(..., where=...)` | [API → Predicate pushdown](api.md#predicate-pushdown), [`example_table.py`](../examples/example_table.py) |
| Train a model (images) | `FitsImageDataset` + `make_loader` | [API → Data module](api.md#data-module), [`example_image_dataset.py`](../examples/example_image_dataset.py) |
| Train on table rows (large file) | `FitsTableIterableDataset` | [`example_data_catalogs.py`](../examples/example_data_catalogs.py) |
| Train on patches / cutouts | `FitsCutoutDataset` | [Examples → PyTorch training](examples.md#pytorch-training) |
| SQL / Polars / DuckDB on FITS | `table.read_polars`, `scan_polars`, `to_polars_lazy`, `to_duckdb` | [`example_polars.py`](../examples/example_polars.py), [`example_table_recipes.py`](../examples/example_table_recipes.py) |
| Switch from Astropy or fitsio | Side-by-side tables | [migration_astropy.md](migration_astropy.md), [migration_fitsio.md](migration_fitsio.md) |
| Upgrading from pre-0.7 `FITSDataset` | Replacement classes | [migration_datasets.md](migration_datasets.md) |
| See if a feature exists | Supported / partial / out of scope | [Parity matrix](parity.md) |
| Compare speed vs astropy/fitsio | Highlights + deficit table | [Benchmarks → Performance highlights](benchmarks.md#performance-highlights) |

## How the docs are organized

| Section | What you get |
|---|---|
| [Installation](install.md) | Wheels, source build, GPU, pixi dev setup |
| [API Reference](api.md) | Entry points grouped by task (I/O, tables, ML) |
| [Examples](examples.md) | Runnable scripts in learning order |
| [Benchmarks](benchmarks.md) | Methodology, highlights, full CSV-derived tables |
| [Parity](parity.md) | Feature contract vs Astropy/fitsio |
| [Migration](migration_astropy.md) | Cookbook replacements from other libraries |
| [Roadmap](roadmap.md) | Release path toward 1.0 |

## Scope

torchfits is **not** a full Astropy replacement. WCS, sky coordinates, HEALPix,
and simulation frameworks belong in downstream sky-domain packages. See
[Parity](parity.md) for what is in and out of scope.
