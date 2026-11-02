"""write() keeps the documented ValueError / TypeError for invalid input.

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


@pytest.mark.parametrize(
    "data, kwargs",
    [
        ([{"data": "not a tensor"}], {}),
        (torch.ones(2, 2), {"compress": 3}),
    ],
)
def test_argument_type_errors_raise_type_error(tmp_path, data, kwargs):
    # TypeError used to be wrapped in RuntimeError while ValueError passed.
    with pytest.raises(TypeError) as info:
        torchfits.write(str(tmp_path / "x.fits"), data, **kwargs)
    assert not isinstance(info.value, RuntimeError)
    assert list(tmp_path.iterdir()) == []


def test_other_failures_name_the_callers_path(tmp_path):
    path = str(tmp_path / "x.fits")
    # PLIO_1 cannot hold floating-point pixels; the writer refuses them.
    with pytest.raises(RuntimeError, match="Failed to write FITS file '" + path):
        torchfits.write(path, torch.ones(2, 2), compress="PLIO_1")
    # 64-bit integer tile compression is refused as an argument error.
    with pytest.raises(ValueError, match="64-bit integer"):
        torchfits.write(path, torch.ones(2, 2, dtype=torch.int64), compress=True)
