# Quick Start

Get up and running with torchfits in minutes.

## Install

```bash
pip install torchfits
```

Pre-built wheels for Linux x86_64 and macOS arm64. No system CFITSIO needed — it's
vendored. Requires Python 3.10+ and PyTorch 2.0+.

!!! note "GPU support"
    For CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
    For MPS (Apple Silicon): the default PyTorch wheel includes MPS support.

## Your First Read

```python
import torchfits

# Read a FITS image as a PyTorch tensor
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")
print(tensor.shape, tensor.dtype)
# torch.Size([4096, 4096]) torch.float32
```

`hdu=0` selects the **HDU** (Header Data Unit) — FITS files are structured as
a stack of numbered sections, each with a header and data block. HDU 0 is
typically the primary image; higher-numbered HDUs hold tables or additional
images.

!!! tip "Memory-mapped reads"
    By default, `read_tensor` uses **mmap** (memory mapping) — the file stays
    on disk and pages are loaded on demand, so a 10 GB image uses minimal RAM.
    Pass `mmap=False` to read the entire file into memory instead.

!!! tip "GPU transfer"
    Pass `device="cuda"` or `device="mps"` to read directly to GPU:
    ```python
    tensor = torchfits.read_tensor("image.fits", hdu=0, device="cuda")
    ```

## Read an Image with Header

```python
data, header = torchfits.read("image.fits", hdu=0, return_header=True)
print(header["OBJECT"])  # e.g. "M31"
```

## Filter a Table

```python
# Filtering happens in C++, only matching rows reach Python
table = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.9",
)
# table: pyarrow.Table
print(table.num_rows)
```

## Stream a Large Table

```python
# Stream 100M+ rows in constant memory
for batch in torchfits.table.scan("survey.fits", hdu=1, chunk_rows=50_000):
    process(batch)  # batch: pyarrow.RecordBatch
```

## PyTorch DataLoader

```python
from torchfits.data import FitsImageDataset, make_loader

ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
loader = make_loader(ds, batch_size=32, num_workers=4)

for images, labels in loader:
    # images: torch.Tensor [B, 1, H, W]
    # labels: torch.Tensor [B]
    pass
```

## Write Back

```python
torchfits.write("output.fits", tensor, header={"OBJECT": "M31"}, overwrite=True)

# Table write
torchfits.table.write("catalog_out.fits", table_dict, overwrite=True)
```

## Multi-HDU Files

```python
with torchfits.open("multi_ext.fits") as hdul:
    img = hdul[0].data       # image tensor
    tbl = hdul[1].data       # table accessor
    filtered = hdul[1].filter("FLUX > 100")
```

HDUs can also be addressed by **EXTNAME** labels (e.g., `'SCI'`, `'EVENTS'`)
instead of integer indices. Use `torchfits.read_hdus(path, hdus=['SCI'])` to
read by name.

## What's Next?

<div class="grid cards" markdown>

-   :material-image-multiple: __Core I/O Reference__

    `read_tensor`, `read`, `read_subset`, `write_tensor`, headers, HDUs.

    [:octicons-arrow-right-24: Core I/O](api-core-io.md)

-   :material-table-large: __Table Reference__

    `table.read`, predicate pushdown, Polars/DuckDB interop, mutations.

    [:octicons-arrow-right-24: Tables](api-tables.md)

-   :material-brain: __Datasets & DataLoaders__

    `FitsImageDataset`, `FitsTableIterableDataset`, `make_loader`.

    [:octicons-arrow-right-24: Data Module](api-data.md)

-   :material-axis-arrow: __Transforms__

    Stretches, normalizers, spectral transforms, continuum estimators.

    [:octicons-arrow-right-24: Transforms](api-transforms.md)

</div>
