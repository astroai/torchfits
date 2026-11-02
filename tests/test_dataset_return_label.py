"""return_label= on the map-style datasets that always pair a label.

FitsTensorDataset / FitsImageDataset / FitsCubeDataset / FitsTableDataset
return ``(x, tensor(0))`` without labels; that stays the default. Passing
``return_label=False`` returns ``x`` alone, like the other datasets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits.data import (
    FitsCubeDataset,
    FitsImageDataset,
    FitsTableDataset,
    FitsTensorDataset,
)


@pytest.fixture()
def image(tmp_path):
    p = tmp_path / "img.fits"
    torchfits.write(str(p), torch.arange(12.0).reshape(3, 2, 2))
    return str(p)


@pytest.fixture()
def tab(tmp_path):
    p = tmp_path / "tab.fits"
    torchfits.table.write(str(p), {"x": np.arange(3.0)})
    return str(p)


@pytest.mark.parametrize("cls", [FitsTensorDataset, FitsImageDataset])
def test_default_still_pairs_zero_label(image, cls):
    x, label = cls([image])[0]
    assert isinstance(x, torch.Tensor)
    assert label.dtype == torch.long and int(label) == 0


@pytest.mark.parametrize("cls", [FitsTensorDataset, FitsImageDataset])
def test_return_label_false_returns_payload_alone(image, cls):
    item = cls([image], return_label=False)[0]
    assert isinstance(item, torch.Tensor)
    assert torch.equal(item, cls([image])[0][0])


def test_cube_return_label(image):
    x, label = FitsCubeDataset([image], slice_index=1)[0]
    assert torch.equal(x, torch.tensor([[4.0, 5.0], [6.0, 7.0]]))
    item = FitsCubeDataset([image], slice_index=1, return_label=False)[0]
    assert torch.equal(item, x)
    x2, label2 = FitsCubeDataset([image], slice_index=1, labels=[3])[0]
    assert torch.equal(x2, x) and int(label2) == 3


def test_table_return_label(tab):
    row, label = FitsTableDataset(tab)[1]
    assert int(label) == 0 and float(row["x"]) == 1.0
    row = FitsTableDataset(tab, return_label=False)[1]
    assert isinstance(row, dict) and float(row["x"]) == 1.0
    _row, label = FitsTableDataset(tab, labels=[4, 5, 6], return_label=True)[1]
    assert int(label) == 5


def test_return_label_false_with_labels_is_refused(image, tab):
    with pytest.raises(ValueError, match="return_label=False"):
        FitsImageDataset([image], labels=[1], return_label=False)
    with pytest.raises(ValueError, match="return_label=False"):
        FitsImageDataset([image], label_key="NAXIS", return_label=False)
    with pytest.raises(ValueError, match="return_label=False"):
        FitsTableDataset(tab, labels=[1, 2, 3], return_label=False)
