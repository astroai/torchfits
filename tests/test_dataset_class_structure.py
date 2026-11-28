"""The dataset classes share their machinery instead of copying it.

Image and cube datasets are presets of the general N-D tensor datasets, and
every iterable dataset takes its epoch-seed / shuffle-buffer ``__iter__``
from one mixin (it used to be copied into four classes).
"""

from __future__ import annotations

from torchfits import data
from torchfits.data.datasets import _EpochStream


def test_image_and_cube_datasets_are_presets_of_the_tensor_datasets():
    assert issubclass(data.FitsImageDataset, data.FitsTensorDataset)
    assert issubclass(data.FitsCubeDataset, data.FitsTensorDataset)
    assert issubclass(data.FitsImageIterableDataset, data.FitsTensorIterableDataset)
    assert issubclass(data.FitsCubeIterableDataset, data.FitsTensorIterableDataset)


def test_iterable_datasets_share_one_epoch_iterator():
    for cls in (
        data.FitsTensorIterableDataset,
        data.FitsImageIterableDataset,
        data.FitsSpectrumIterableDataset,
        data.FitsStagedCutoutIterableDataset,
    ):
        assert issubclass(cls, _EpochStream)
        assert cls.__iter__ is _EpochStream.__iter__, cls
    # The table stream validates its shuffle options, then defers to it.
    assert issubclass(data.FitsTableIterableDataset, _EpochStream)
