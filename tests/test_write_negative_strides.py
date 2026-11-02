"""numpy images with negative strides (a[::-1]) can be written.

torch.as_tensor cannot view negative strides, so every numpy image route
raised (or reported the array as unconvertible).
"""

import numpy as np
import pytest

import torchfits

A = np.arange(12, dtype=np.float32).reshape(3, 4)[::-1, ::-1]


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize(
    "wrap", [lambda a: a, lambda a: {"data": a}, lambda a: [{"data": a}]]
)
def test_negative_stride_image(tmp_path, compress, wrap):
    path = str(tmp_path / "neg.fits")
    torchfits.write(path, wrap(A), compress=compress)
    got = torchfits.read(path, hdu=1 if compress else 0).numpy()
    np.testing.assert_array_equal(got, A)
