# Examples

Runnable scripts under `examples/`. Each prints results (galleries also write
PNGs to `examples/output/`). Smoke suite:

```bash
pixi run python examples/test_examples.py
```

---

## Start here

| Step | Script | You learn |
|:---:|---|---|
| 1 | [`gallery_images.py`](../examples/gallery_images.py) | Real HorseHead + stretch/normalize **before/after** figures |
| 2 | [`gallery_spectra.py`](../examples/gallery_spectra.py) | Continuum normalize/removal (specutils-shaped plots) |
| 3 | [`example_image.py`](../examples/example_image.py) | `read_tensor`, headers, `write_tensor` |
| 4 | [`example_table.py`](../examples/example_table.py) | FITS tables as dataframes: `table.read` / `read_torch` / `scan_torch` |
| 5 | CLI recipes | [cli-recipes.md](cli-recipes.md) — `imstat` / `imarith` / fitsort analogues |

Transform figure gallery (embedded PNGs): [examples-transforms.md](examples-transforms.md).

### Real public FITS

```bash
pixi run python -c "from examples._sample_data import ensure_sample; print(ensure_sample('horsehead'))"
```

Cached under `~/.cache/torchfits/samples/` (`horsehead`, `chandra_events`,
`sdss_spectrum`). CI sets `TORCHFITS_EXAMPLE_FAST=1` to skip downloads.

---

## By topic

### Transform galleries (with plots)

| Script | What it demonstrates |
|---|---|
| [`gallery_images.py`](../examples/gallery_images.py) | All image stretches/norms/clips + FITS header scale |
| [`gallery_spectra.py`](../examples/gallery_spectra.py) | Continuum suite, Doppler, binning, BandMath |
| [`gallery_tables_lc.py`](../examples/gallery_tables_lc.py) | `FITSScaleColumns`, `TNullToNan`, phase-fold LC |

### Arrays and tensors

| Script | What it demonstrates |
|---|---|
| [`example_image.py`](../examples/example_image.py) | `read_tensor`, `read`, `get_header`, and `write_tensor` round-trip |
| [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | `read_subset`, tensor slicing, and `open_subset_reader` |
| [`example_image_cube.py`](../examples/example_image_cube.py) | 3D cubes with `read_tensor` and tensor slicing |
| [`example_image_mef.py`](../examples/example_image_mef.py) | Multi-extension files with `open`, `read_hdus`, and table `filter` |

### Tables as dataframes

| Script | What it demonstrates |
|---|---|
| [`example_table.py`](../examples/example_table.py) | `table.read` (Arrow dataframe), `table.read_torch`, `table.scan_torch`, `table.write` |
| [`example_table_interop.py`](../examples/example_table_interop.py) | VLA columns and `to_pandas` / `to_arrow` / `to_polars` conversion |
| [`example_polars.py`](../examples/example_polars.py) | Direct FITS → Polars dataframe via `read_polars`, `scan_polars`, `to_polars`, and `to_polars_lazy` |
| [`example_table_recipes.py`](../examples/example_table_recipes.py) | Arrow scanner, Polars lazy frames, and DuckDB SQL on FITS tables |

### PyTorch training

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | `FitsImageDataset` + `make_loader` |
| [`example_data_catalogs.py`](../examples/example_data_catalogs.py) | `FitsTableDataset`, `FitsTableIterableDataset`, `FitsCutoutDataset` |
| [`example_transforms.py`](../examples/example_transforms.py) | `torchfits.transforms` pipeline + `FitsImageDataset` |
| [`example_hyperspectral.py`](../examples/example_hyperspectral.py) | Spectral/hyperspectral transforms on tensor cubes |

### Time series

| Script | What it demonstrates |
|---|---|
| [`example_time_series.py`](../examples/example_time_series.py) | Exoplanet transit light curve, FITS table write/read, `AsymmetricSigmaClip`, `PhaseFold`, `SavitzkyGolayFilter` |

### Shell (CLI)

| Script | What it demonstrates |
|---|---|
| [`cli/imstat_imarith.sh`](../examples/cli/imstat_imarith.sh) | info/stats/arith/cutout/transform/png on HorseHead |

See the [CLI guide](cli.md) and [CLI recipes](cli-recipes.md).

---

## Which dataset class?

| Your data | Catalog size | Class |
|---|---|---|
| Image files (paths list) | Any | `FitsImageDataset` (map) or `FitsImageIterableDataset` (many workers) |
| One table HDU | Fits in RAM | `FitsTableDataset` |
| One table HDU | Too large for RAM | `FitsTableIterableDataset` |
| Catalog + cutouts from images | Any | `FitsCutoutDataset` |

Out of scope for these demos: WCS overlays, photometry, interactive GUIs
(jdaviz). Static matplotlib PNGs only.
