"""labels= / label_key= on FitsImageIterableDataset and FitsCubeIterableDataset.

Each label must follow its own file through rank sharding, DataLoader worker
sharding, per-epoch shuffling and the reservoir shuffle buffer. File ``i``
holds pixels equal to ``i`` and label ``100 + i``, so a pairing mistake shows.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

import torchfits
from torchfits.data import FitsCubeIterableDataset, FitsImageIterableDataset

N = 7


@pytest.fixture()
def files(tmp_path):
    paths = []
    for i in range(N):
        p = tmp_path / f"f{i}.fits"
        torchfits.write(str(p), torch.full((4, 4), float(i)), header={"CLS": 100 + i})
        paths.append(str(p))
    return paths


def _check_pairs(items):
    seen = []
    for payload, label in items:
        value = int(payload.flatten()[0].item())
        assert label.dtype == torch.long
        assert int(label) == 100 + value
        seen.append(value)
    return seen


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("buffer", [None, 3])
def test_labels_follow_files_across_ranks(files, shuffle, buffer):
    labels = [100 + i for i in range(N)]
    seen = []
    for rank in range(3):
        ds = FitsImageIterableDataset(
            files,
            labels=labels,
            rank=rank,
            world_size=3,
            shuffle=shuffle,
            shuffle_buffer_size=buffer,
            seed=5,
        )
        for _epoch in range(2):
            seen_epoch = _check_pairs(iter(ds))
        seen.extend(seen_epoch)
    assert sorted(seen) == list(range(N))


def test_labels_follow_files_across_dataloader_workers(files):
    ds = FitsImageIterableDataset(
        files, labels=torch.arange(100, 100 + N), shuffle=True, seed=1
    )
    loader = DataLoader(ds, batch_size=None, num_workers=2)
    assert sorted(_check_pairs(loader)) == list(range(N))


def test_label_key_reads_header(files):
    ds = FitsImageIterableDataset(files, label_key="CLS")
    payload, label = next(iter(ds))
    assert payload.shape == (1, 4, 4)
    assert sorted(_check_pairs(iter(ds))) == list(range(N))


def test_duplicate_paths_keep_their_own_labels(files):
    ds = FitsImageIterableDataset([files[0], files[0]], labels=[1, 2])
    assert [int(lbl) for _x, lbl in ds] == [1, 2]


def test_cube_iterable_labels(files):
    ds = FitsCubeIterableDataset(
        files, labels=[100 + i for i in range(N)], slice_index=0
    )
    items = list(ds)
    assert items[0][0].shape == (4,)
    assert sorted(_check_pairs(items)) == list(range(N))


def test_no_labels_still_yields_payload_alone(files):
    item = next(iter(FitsImageIterableDataset(files)))
    assert isinstance(item, torch.Tensor)
    item = next(iter(FitsCubeIterableDataset(files, slice_index=0)))
    assert isinstance(item, torch.Tensor)


def test_labels_length_mismatch_raises(files):
    with pytest.raises(ValueError, match="labels length"):
        FitsImageIterableDataset(files, labels=[0, 1])
