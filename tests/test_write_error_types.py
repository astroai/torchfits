"""write() keeps the documented ValueError for invalid input.

ValueError (e.g. non-ASCII header text) used to be wrapped in RuntimeError,
contradicting docs/api-core-io.md; other failures must name the caller's path.
"""

import pytest
import torch

import torchfits


@pytest.mark.parametrize(
    "data, kwargs",
    [
        (torch.ones(2, 2), {"header": {"UNI": "λ-cold"}}),
        ({"a": torch.arange(2)}, {"header": {"UNI": "λ-cold"}}),
        ({"s": ["λ", "a"]}, {}),
    ],
)
def test_invalid_text_raises_value_error(tmp_path, data, kwargs):
    path = tmp_path / "x.fits"
    with pytest.raises(ValueError) as info:
        torchfits.write(str(path), data, **kwargs)
    assert not isinstance(info.value, RuntimeError)
    assert list(tmp_path.iterdir()) == []


def test_other_failures_name_the_callers_path(tmp_path):
    path = str(tmp_path / "x.fits")
    with pytest.raises(RuntimeError, match="Failed to write FITS file '" + path):
        torchfits.write(path, [{"data": "not a tensor"}])
