"""ML-native FITS datasets and DataLoader integration.

Provides :class:`torch.utils.data.Dataset` and
:class:`~torch.utils.data.IterableDataset` implementations that wrap
``read_tensor`` and ``read_table``, plus a ``fits_collate_fn`` and
``make_loader`` helper.

Basic usage::

    from torchfits.data import FitsImageDataset, make_loader

    ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
    loader = make_loader(ds, batch_size=32, num_workers=4)
    for images, labels in loader:
        ...

See the :ref:`data loading documentation <data>` for multi-worker setup and
cache tuning.
"""

from __future__ import annotations

import glob as _glob
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Re-export the generic FITSDataset classes from torchfits.datasets
from torchfits.datasets import FITSDataset, IterableFITSDataset  # noqa: F401


# ---------------------------------------------------------------------------
# FitsImageDataset — file-list image dataset with label-from-header
# ---------------------------------------------------------------------------


class FitsImageDataset(Dataset):
    """Map-style dataset that reads FITS images via ``read_tensor``.

    Each ``__getitem__`` returns ``(image, label)`` where *image* is a
    ``torch.Tensor`` and *label* is extracted from a FITS header keyword
    (or a user-supplied list).

    Parameters
    ----------
    paths : str or list[str]
        File path, glob pattern, or list of file paths.
    hdu : int
        HDU index to read (default 0).
    label_key : str or None
        Header keyword to extract as the integer label.  When *None*, you
        must supply *labels* explicitly.
    labels : list[int] or None
        Explicit per-file labels (same length as the resolved file list).
        Overrides *label_key*.
    transform : callable or None
        Optional transform applied to the image tensor before returning.
    device : str
        Torch device for the read tensor (default ``"cpu"``).
    mmap : bool or str
        Mmap policy passed to ``read_tensor`` (default ``True``).
    add_channel_dim : bool
        If True, 2D images are unsqueezed to ``[1, H, W]`` (default True).
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: int = 0,
        label_key: str | None = None,
        labels: list[int] | None = None,
        transform: Callable | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        add_channel_dim: bool = True,
    ) -> None:
        if isinstance(paths, str):
            paths = sorted(_glob.glob(paths)) or [paths]
        self.files = list(paths)
        self.hdu = hdu
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.add_channel_dim = add_channel_dim

        if labels is not None:
            if len(labels) != len(self.files):
                raise ValueError(
                    f"labels length {len(labels)} != files length {len(self.files)}"
                )
            self._labels = list(labels)
        elif label_key is not None:
            from torchfits import get_header

            self._labels = [
                int(get_header(f, hdu=self.hdu)[label_key]) for f in self.files
            ]
        else:
            self._labels = [0] * len(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from torchfits import read_tensor

        image = read_tensor(
            self.files[idx], hdu=self.hdu, device=self.device, mmap=self.mmap
        )
        if self.add_channel_dim and image.ndim == 2:
            image = image.unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return image, label

    def __repr__(self) -> str:
        return (
            f"FitsImageDataset(n={len(self.files)}, hdu={self.hdu}, "
            f"device={self.device!r})"
        )


# ---------------------------------------------------------------------------
# FitsImageIterableDataset — sharded for multi-worker DataLoader
# ---------------------------------------------------------------------------


class FitsImageIterableDataset(IterableDataset):
    """Iterable dataset for sharded multi-worker image loading.

    Each worker processes a deterministic subset of the file list so every
    sample is seen exactly once per epoch regardless of ``num_workers``.

    Parameters
    ----------
    paths : str or list[str]
        File path, glob pattern, or list of file paths.
    hdu : int
        HDU index to read (default 0).
    transform : callable or None
        Optional transform applied to the image tensor.
    device : str
        Torch device.
    mmap : bool or str
        Mmap policy.
    shuffle : bool
        Shuffle file order per epoch (uses epoch-based seed).
    seed : int
        Base seed for shuffling.
    add_channel_dim : bool
        If True, 2D images are unsqueezed to ``[1, H, W]``.
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: int = 0,
        transform: Callable | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        shuffle: bool = False,
        seed: int = 0,
        add_channel_dim: bool = True,
    ) -> None:
        if isinstance(paths, str):
            paths = sorted(_glob.glob(paths)) or [paths]
        self.files = list(paths)
        self.hdu = hdu
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.shuffle = shuffle
        self.seed = seed
        self.add_channel_dim = add_channel_dim

    def __iter__(self) -> Iterator[torch.Tensor]:
        from torchfits import read_tensor

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            indices = list(range(len(self.files)))
        else:
            total = len(self.files)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = total // num_workers
            remainder = total % num_workers
            start = worker_id * per_worker + min(worker_id, remainder)
            size = per_worker + (1 if worker_id < remainder else 0)
            indices = list(range(start, start + size))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        for idx in indices:
            image = read_tensor(
                self.files[idx], hdu=self.hdu, device=self.device, mmap=self.mmap
            )
            if self.add_channel_dim and image.ndim == 2:
                image = image.unsqueeze(0)
            if self.transform is not None:
                image = self.transform(image)
            yield image

    def __repr__(self) -> str:
        return (
            f"FitsImageIterableDataset(n={len(self.files)}, hdu={self.hdu}, "
            f"device={self.device!r})"
        )


# ---------------------------------------------------------------------------
# FitsTableDataset — row-indexable table catalog
# ---------------------------------------------------------------------------


class FitsTableDataset(Dataset):
    """Map-style dataset for row-indexable FITS binary tables.

    Each ``__getitem__`` returns a ``dict[str, Tensor]`` for one row.
    Supports column projection and ``where=`` pushdown for filtered access.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    hdu : int
        Table HDU index (default 1).
    columns : list[str] or None
        Column names to read.  None reads all columns.
    where : str or None
        FITS WHERE expression for row filtering (e.g. ``"MAG < 20"``).
    transform : callable or None
        Optional transform applied to the row dict.
    device : str
        Torch device for tensors.
    mmap : bool or str
        Mmap policy.
    """

    def __init__(
        self,
        path: str,
        hdu: int = 1,
        columns: list[str] | None = None,
        where: str | None = None,
        transform: Callable | None = None,
        device: str = "cpu",
        mmap: bool | str = "auto",
    ) -> None:
        self.path = path
        self.hdu = hdu
        self.columns = columns
        self.where = where
        self.transform = transform
        self.device = device
        self.mmap = mmap

        # Read all data once and index by row.
        # Small-to-medium catalogs only; see roadmap for streaming variant.
        import torchfits.table

        pa_table = torchfits.table.read(
            self.path,
            hdu=self.hdu,
            columns=self.columns,
            where=self.where,
        )
        # torchfits.table.read returns a pyarrow.Table; convert to dict
        # Numeric/boolean columns → torch.Tensor; string/VLA → list
        import pyarrow.types as pt

        result: dict[str, Any] = {}
        for col_name in pa_table.column_names:
            col = pa_table.column(col_name)
            t = col.type
            if pt.is_integer(t) or pt.is_floating(t) or pt.is_boolean(t):
                tensor = torch.from_numpy(col.to_numpy())
                if self.device != "cpu":
                    tensor = tensor.to(self.device)
                result[col_name] = tensor
            else:
                result[col_name] = col.to_pylist()
        self._data = result
        # Determine row count from the first tensor or list column
        self._n_rows = 0
        for v in self._data.values():
            if isinstance(v, (torch.Tensor, list)):
                self._n_rows = len(v)
                break

    def __len__(self) -> int:
        return self._n_rows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = {k: v[idx] for k, v in self._data.items()}
        if self.transform is not None:
            row = self.transform(row)
        return row

    def __repr__(self) -> str:
        return (
            f"FitsTableDataset(path={self.path!r}, hdu={self.hdu}, "
            f"n_rows={self._n_rows})"
        )


# ---------------------------------------------------------------------------
# fits_collate_fn — stack homogeneous tensors from batch of dicts/tensors
# ---------------------------------------------------------------------------


def fits_collate_fn(
    batch: list[Any],
) -> Any:
    """Collate a list of samples into a batch.

    - Lists of ``dict[str, Tensor]`` are stacked per key.
    - Lists of ``Tensor`` are stacked with ``torch.stack``.
    - Lists of ``(Tensor, Tensor)`` (image, label) pairs are stacked into
      ``(images, labels)`` tuples.
    - Ragged / VLA columns raise ``ValueError``.

    Parameters
    ----------
    batch : list
        List of samples from a torchfits dataset.

    Returns
    -------
    Collated batch (torch.Tensor, tuple, or dict).
    """
    if not batch:
        return batch

    first = batch[0]

    if isinstance(first, dict):
        out: dict[str, torch.Tensor] = {}
        for key in first:
            values = [sample[key] for sample in batch]
            if not all(isinstance(v, torch.Tensor) for v in values):
                raise ValueError(
                    f"Cannot collate non-tensor column {key!r}. "
                    f"Use vla_policy= to handle variable-length arrays."
                )
            out[key] = torch.stack(values)
        return out

    if isinstance(first, tuple) and len(first) == 2:
        images = torch.stack([s[0] for s in batch])
        labels = torch.stack([s[1] for s in batch])
        return images, labels

    if isinstance(first, torch.Tensor):
        return torch.stack(batch)

    raise TypeError(f"Unsupported sample type for collation: {type(first)}")


# ---------------------------------------------------------------------------
# make_loader — DataLoader with cache optimisation
# ---------------------------------------------------------------------------


def make_loader(
    dataset: Dataset | IterableDataset,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    *,
    optimize_cache: bool = True,
    avg_file_size_mb: float = 10.0,
    **loader_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader with sensible defaults and optional cache warm-up.

    When *optimize_cache* is True and the dataset exposes a ``files``
    attribute, :func:`torchfits.cache.optimize_for_dataset` is called to
    pre-warm handle and file caches.

    Parameters
    ----------
    dataset : Dataset or IterableDataset
        A torchfits dataset instance.
    batch_size : int
        Batch size.
    shuffle : bool or None
        Whether to shuffle.  Defaults to True for map-style datasets.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Pin memory for faster CPU→GPU transfers.
    prefetch_factor : int
        Prefetch factor for multi-worker loading.
    drop_last : bool
        Drop the last incomplete batch.
    optimize_cache : bool
        Call ``cache.optimize_for_dataset`` before creating the loader.
    avg_file_size_mb : float
        Average file size in MB used for cache sizing.
    **loader_kwargs :
        Passed through to :class:`torch.utils.data.DataLoader`.

    Returns
    -------
    DataLoader
    """
    collate_fn = loader_kwargs.pop("collate_fn", fits_collate_fn)

    if shuffle is None:
        # Map-style datasets shuffle by default; IterableDataset must not
        shuffle = not isinstance(dataset, IterableDataset)

    # prefetch_factor is only valid with num_workers > 0
    if num_workers == 0:
        prefetch_factor = None  # type: ignore[assignment]

    if optimize_cache:
        file_list = getattr(dataset, "files", None)
        if file_list:
            from torchfits.cache import optimize_for_dataset

            optimize_for_dataset(file_list, avg_file_size_mb=avg_file_size_mb)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **loader_kwargs,
    )


__all__ = [
    "FITSDataset",
    "FitsImageDataset",
    "FitsImageIterableDataset",
    "FitsTableDataset",
    "IterableFITSDataset",
    "fits_collate_fn",
    "make_loader",
]
