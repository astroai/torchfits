# Examples

Runnable scripts under `examples/`. Each creates temporary FITS files,
prints results, and cleans up. Run the full smoke suite:

```bash
pixi run python examples/test_examples.py
```

## Suggested order

Read and run in this sequence the first time through:

| Step | Script | You learn |
|:---:|---|---|
| 1 | [`example_image.py`](../examples/example_image.py) | `read_tensor`, headers, `write_tensor` |
| 2 | [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | `read_subset`, `open_subset_reader` |
| 3 | [`example_table.py`](../examples/example_table.py) | `table.read` + `where=`, `stream_table` |
| 4 | [`example_image_dataset.py`](../examples/example_image_dataset.py) | `FitsImageDataset`, `make_loader`, cache warmup |
| 5 | [`example_transforms.py`](../examples/example_transforms.py) | `torchfits.transforms` pipeline |
| 6 | [`example_data_catalogs.py`](../examples/example_data_catalogs.py) | Table + cutout datasets |

Then explore by interest:

- **3D / MEF** — [`example_image_cube.py`](../examples/example_image_cube.py), [`example_image_mef.py`](../examples/example_image_mef.py)
- **Table interop** — [`example_table_interop.py`](../examples/example_table_interop.py), [`example_polars.py`](../examples/example_polars.py), [`example_table_recipes.py`](../examples/example_table_recipes.py)
- **Spectral** — [`example_hyperspectral.py`](../examples/example_hyperspectral.py)

## Which dataset class?

| Your data | Catalog size | Class |
|---|---|---|
| Image files (paths list) | Any | `FitsImageDataset` (map) or `FitsImageIterableDataset` (shuffle / many workers) |
| One table HDU | Fits in RAM (under a few million rows) | `FitsTableDataset` |
| One table HDU | Too large to load at once | `FitsTableIterableDataset` |
| Fixed (path, x, y) cutouts | Any | `FitsCutoutDataset` |

Use `make_loader(dataset, ...)` for sensible `num_workers`, `pin_memory`, and optional cache warmup.

## By topic

### Arrays and tensors

| Script | What it demonstrates |
|---|---|
| [`example_image.py`](../examples/example_image.py) | `read_tensor`, `read`, `get_header`, and `write_tensor` round-trip |
| [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | `read_subset`, tensor slicing, and `open_subset_reader` |
| [`example_image_cube.py`](../examples/example_image_cube.py) | 3D cubes with `read_tensor` and tensor slicing |
| [`example_image_mef.py`](../examples/example_image_mef.py) | Multi-extension files with `open`, `read_hdus`, and table `filter` |

### Tables

| Script | What it demonstrates |
|---|---|
| [`example_table.py`](../examples/example_table.py) | `read_table`, `table.read` with `where=`, `stream_table`, and `table.write` |
| [`example_table_interop.py`](../examples/example_table_interop.py) | VLA columns and `to_pandas` / `to_arrow` / `to_polars` conversion |
| [`example_polars.py`](../examples/example_polars.py) | Direct FITS → Polars via `read_polars`, `scan_polars`, `to_polars`, and `to_polars_lazy` |
| [`example_table_recipes.py`](../examples/example_table_recipes.py) | Arrow scanner, Polars lazy frames, and DuckDB SQL on FITS tables |

### PyTorch training

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | `FitsImageDataset` + `make_loader` |
| [`example_data_catalogs.py`](../examples/example_data_catalogs.py) | `FitsTableDataset`, `FitsTableIterableDataset`, `FitsCutoutDataset` |
| [`example_transforms.py`](../examples/example_transforms.py) | `torchfits.transforms` pipeline + `FitsImageDataset` |
| [`example_hyperspectral.py`](../examples/example_hyperspectral.py) | Spectral/hyperspectral transforms on tensor cubes |

## Optional dependencies

Polars, DuckDB, and some interop paths need extra packages. Examples print a
skip message and exit 0 when a dependency is missing:

```bash
pip install polars duckdb
```
