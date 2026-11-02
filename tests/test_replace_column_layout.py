"""replace_column with TZERO-convention dtypes replaces the column layout.

A 1K column replaced by uint16 values used to keep TFORM=1K and gain
TZERO=32768, which reads back as float64. Without format= the values' dtype
now picks the layout (as table.write would); with format= the TZERO convention
of that format applies, or the values must fit its plain range.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits import table


def _fresh(path, a=None):
    table.write(
        str(path),
        {"a": np.arange(4, dtype=np.int64) if a is None else a, "b": np.ones(4)},
        overwrite=True,
    )


@pytest.mark.parametrize(
    "values, tform, tzero",
    [
        (np.array([0, 1, 40000, 65535], dtype=np.uint16), "1I", 32768),
        (torch.tensor([0, 1, 40000, 65535], dtype=torch.uint16), "1I", 32768),
        (np.array([0, 1, 2**32 - 1, 7], dtype=np.uint32), "1J", 2**31),
        (np.array([0, 1, 2**64 - 1, 7], dtype=np.uint64), "1K", 2**63),
        (np.array([-128, 0, 5, 127], dtype=np.int8), "1B", -128),
    ],
)
def test_replace_column_takes_layout_from_values(tmp_path, values, tform, tzero):
    path = tmp_path / "t.fits"
    _fresh(path)
    table.replace_column(str(path), "a", values)
    header = torchfits.read_header(str(path), 1)
    assert header["TFORM1"] == tform
    assert float(header["TZERO1"]) == float(tzero)
    out = torchfits.read(str(path), hdu=1)["a"]
    expected = torch.as_tensor(np.asarray(values))
    assert out.dtype == expected.dtype
    assert torch.equal(out, expected)
    # The other column is untouched.
    assert torch.equal(
        torchfits.read(str(path), hdu=1)["b"], torch.ones(4, dtype=torch.float64)
    )


def test_replace_unsigned_column_with_same_dtype_keeps_layout(tmp_path):
    path = tmp_path / "t.fits"
    _fresh(path, np.array([1, 2, 3, 4], dtype=np.uint16))
    vals = np.array([9, 8, 60000, 0], dtype=np.uint16)
    table.replace_column(str(path), "a", vals)
    header = torchfits.read_header(str(path), 1)
    assert header["TFORM1"] in ("I", "1I") and header["TZERO1"] == 32768
    out = torchfits.read(str(path), hdu=1)["a"]
    assert torch.equal(out, torch.from_numpy(vals))


def test_replace_column_signed_values_keep_existing_tform(tmp_path):
    path = tmp_path / "t.fits"
    _fresh(path)
    table.replace_column(str(path), "a", np.array([0, 1, -5, 7], dtype=np.int32))
    header = torchfits.read_header(str(path), 1)
    assert header["TFORM1"] == "1K" and "TZERO1" not in header
    assert torchfits.read(str(path), hdu=1)["a"].tolist() == [0, 1, -5, 7]


def test_replace_column_format_with_matching_convention(tmp_path):
    path = tmp_path / "t.fits"
    _fresh(path)
    vals = np.array([0, 1, 40000, 65535], dtype=np.uint16)
    table.replace_column(str(path), "a", vals, format="1I")
    header = torchfits.read_header(str(path), 1)
    assert header["TFORM1"] in ("I", "1I") and header["TZERO1"] == 32768
    assert torch.equal(torchfits.read(str(path), hdu=1)["a"], torch.from_numpy(vals))


def test_replace_column_format_wider_plain_storage(tmp_path):
    path = tmp_path / "t.fits"
    _fresh(path)
    vals = np.array([0, 1, 40000, 65535], dtype=np.uint16)
    table.replace_column(str(path), "a", vals, format="1J")
    header = torchfits.read_header(str(path), 1)
    assert header["TFORM1"] == "1J" and "TZERO1" not in header
    out = torchfits.read(str(path), hdu=1)["a"]
    assert out.dtype == torch.int32
    assert out.tolist() == [0, 1, 40000, 65535]


@pytest.mark.parametrize("fmt", ["1B", "1L", "4A"])
def test_replace_column_format_incompatible_raises(tmp_path, fmt):
    path = tmp_path / "t.fits"
    _fresh(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="cannot"):
        table.replace_column(
            str(path), "a", np.array([0, 1, 40000, 65535], dtype=np.uint16), format=fmt
        )
    assert path.read_bytes() == before
