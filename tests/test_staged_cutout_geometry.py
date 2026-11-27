"""``FitsStagedCutoutIterableDataset`` refuses spatially mismatched HDUs.

Cutout windows come from the first flux band's geometry. A second band or an
ivar/mask companion of another shape used to be cut at the same coordinates
and stacked silently (or, for a smaller mask, yield an empty ``(1, 0, 0)``
window); the map-style readers already reject both.
"""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits.data import FitsStagedCutoutIterableDataset


@pytest.fixture
def mismatched(tmp_path):
    path = tmp_path / "m.fits"
    flux = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    ivar_big = torch.full((128, 128), 2.0)
    mask_small = torch.zeros((16, 16), dtype=torch.int16)
    ivar_ok = torch.ones((64, 64))
    torchfits.write(str(path), [flux, ivar_big, mask_small, ivar_ok], overwrite=True)
    return str(path)


def _first(ds):
    return next(iter(ds))


def _fixed(window):
    return lambda h, w, ch, cw: window


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ivar_hdu": 1}, r"ivar companion \[1\] has shape \(128, 128\)"),
        ({"mask_hdu": 2}, r"mask companion \[2\] has shape \(16, 16\)"),
        ({"hdu": [0, 1]}, r"identical shapes"),
    ],
)
def test_mismatched_hdus_are_rejected(mismatched, kwargs, match):
    kwargs = {"hdu": 0, **kwargs}
    ds = FitsStagedCutoutIterableDataset(
        [mismatched],
        cutouts_per_file=1,
        cutout_size=8,
        cutout_generator=_fixed((0, 0, 8, 8)),
        cleanup=False,
        **kwargs,
    )
    with pytest.raises(ValueError, match=match):
        _first(ds)


def test_window_outside_smaller_mask_is_rejected(mismatched):
    ds = FitsStagedCutoutIterableDataset(
        [mismatched],
        cutouts_per_file=1,
        cutout_size=8,
        hdu=0,
        mask_hdu=2,
        cutout_generator=_fixed((40, 40, 48, 48)),
        cleanup=False,
    )
    with pytest.raises(ValueError, match="pixel-aligned"):
        _first(ds)


def test_aligned_companion_is_accepted(mismatched):
    ds = FitsStagedCutoutIterableDataset(
        [mismatched],
        cutouts_per_file=1,
        cutout_size=8,
        hdu=0,
        ivar_hdu=3,
        cutout_generator=_fixed((0, 0, 8, 8)),
        cleanup=False,
    )
    out = _first(ds)
    assert out["flux"].shape == (1, 8, 8)
    assert out["ivar"].shape == (1, 8, 8)
