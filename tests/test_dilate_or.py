"""_dilate_or is the zscale bad-pixel dilation; it must equal numpy's
``np.convolve(mask, ones(w), "same") > 0`` (the astropy reference), including
even widths (left-biased centre), width 1, width == row length, rows with a
shorter valid prefix, and masks hitting both edges.
"""

import numpy as np
import pytest
import torch

from torchfits.transforms.helpers import _dilate_or


def _reference(row: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(row.astype(np.int64), np.ones(w, dtype=np.int64), "same") > 0


@pytest.mark.parametrize("seed", range(8))
def test_matches_numpy_convolve_same(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    groups = 12
    widths = rng.integers(1, n + 1, size=groups)
    widths[:3] = [1, n, max(1, n - (n % 2))]  # unit, full-row and an even width
    lengths = rng.integers(widths, n + 1)  # valid prefix, never shorter than w
    mask = rng.random((groups, n)) < rng.uniform(0.02, 0.4)
    mask[0, 0] = mask[1, -1] = True  # edges
    for g, length in enumerate(lengths):
        mask[g, length:] = False

    got = _dilate_or(torch.from_numpy(mask), torch.from_numpy(widths)).numpy()

    for g in range(groups):
        w, length = int(widths[g]), int(lengths[g])
        np.testing.assert_array_equal(got[g], _reference(mask[g], w), err_msg=f"{g=}")
        np.testing.assert_array_equal(
            got[g, :length], _reference(mask[g, :length], w), err_msg=f"{g=} prefix"
        )


@pytest.mark.parametrize("w", [1, 2, 3, 4, 5, 6])
def test_single_pixel_spreads_like_numpy(w):
    n = 9
    for pos in (0, n // 2, n - 1):
        row = np.zeros(n, dtype=bool)
        row[pos] = True
        got = _dilate_or(torch.from_numpy(row[None]), torch.tensor([w]))[0].numpy()
        np.testing.assert_array_equal(got, _reference(row, w), err_msg=f"{pos=}")
