from __future__ import annotations

from typing import Any, Callable, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from torchfits.io import read as _read


class FITSDataset(Dataset):
    def __init__(
        self,
        paths: str | list[str],
        hdu: int | str | None = None,
        columns: list[str] | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        mode: str = "auto",
        device: str = "cpu",
        mmap: bool | str = "auto",
        cache_capacity: int = 10,
        preload: bool = False,
    ):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.hdu = hdu
        self.columns = columns
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.device = device
        self.mmap = mmap
        self.cache_capacity = cache_capacity
        self._data = None
        if preload:
            self._data = [self._load(i) for i in range(len(self.paths))]

    def _resolve_hdu_and_mode(self) -> tuple[int | str, str]:
        mode = self.mode
        hdu = self.hdu
        if mode == "auto":
            if hdu is None:
                return 0, "image"
            if isinstance(hdu, str):
                return hdu, "table"
            return hdu, "image" if hdu == 0 else "table"
        if hdu is None:
            return 0 if mode == "image" else 1, mode
        return hdu, mode

    def _load(self, idx: int) -> Any:
        path = self.paths[idx]
        hdu, mode = self._resolve_hdu_and_mode()
        kwargs: dict[str, Any] = dict(
            path=path,
            hdu=hdu,
            device=self.device,
            mmap=self.mmap,
            mode=mode,
        )
        if mode == "table":
            kwargs["columns"] = self.columns
            kwargs["cache_capacity"] = self.cache_capacity
        return _read(**kwargs)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        if self._data is not None:
            sample = self._data[idx]
        else:
            sample = self._load(idx)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class IterableFITSDataset(IterableDataset):
    def __init__(
        self,
        paths: str | list[str],
        hdu: int | str | None = None,
        columns: list[str] | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        mode: str = "auto",
        device: str = "cpu",
        mmap: bool | str = "auto",
        cache_capacity: int = 10,
        preload: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.hdu = hdu
        self.columns = columns
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.device = device
        self.mmap = mmap
        self.cache_capacity = cache_capacity
        self.shuffle = shuffle
        self.seed = seed
        self._data = None
        if preload:
            self._data = [self._load(i) for i in range(len(self.paths))]

    def _resolve_hdu_and_mode(self) -> tuple[int | str, str]:
        mode = self.mode
        hdu = self.hdu
        if mode == "auto":
            if hdu is None:
                return 0, "image"
            if isinstance(hdu, str):
                return hdu, "table"
            return hdu, "image" if hdu == 0 else "table"
        if hdu is None:
            return 0 if mode == "image" else 1, mode
        return hdu, mode

    def _load(self, idx: int) -> Any:
        path = self.paths[idx]
        hdu, mode = self._resolve_hdu_and_mode()
        kwargs: dict[str, Any] = dict(
            path=path,
            hdu=hdu,
            device=self.device,
            mmap=self.mmap,
            mode=mode,
        )
        if mode == "table":
            kwargs["columns"] = self.columns
            kwargs["cache_capacity"] = self.cache_capacity
        return _read(**kwargs)

    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = list(range(len(self.paths)))
        else:
            total = len(self.paths)
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
            if self._data is not None:
                sample = self._data[idx]
            else:
                sample = self._load(idx)
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample
