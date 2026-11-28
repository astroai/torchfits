"""A zero-width vector column is refused with a clear ValueError.

A ``(N, 0)`` column maps to TFORM ``0E`` (valid FITS, no bytes per row), but
CFITSIO refuses the zero-length column write and torchfits' readers refuse a
repeat of 0, so the write failed with CFITSIO's "bad first element number"
wrapped in a RuntimeError. It is now a ValueError naming the column, raised
before anything is written.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import torchfits


def _write(path, data):
    torchfits.write(path, data)


def _table_write(path, data):
    torchfits.table.write(path, data)


def _writer(path, data):
    with torchfits.table.writer(path) as w:
        w.write(data)


@pytest.mark.parametrize("writer", [_write, _table_write, _writer])
@pytest.mark.parametrize(
    "column",
    [
        np.zeros((3, 0), np.float32),
        np.zeros((3, 0), np.int16),
        np.zeros((3, 0), np.bool_),
        torch.zeros(3, 0),
    ],
    ids=["float32", "int16", "bool", "torch"],
)
def test_zero_width_vector_column_raises_value_error(tmp_path, writer, column):
    path = str(tmp_path / "z.fits")
    with pytest.raises(ValueError, match="'Z' has zero elements per row"):
        writer(path, {"A": np.arange(3, dtype=np.int32), "Z": column})
    assert not os.path.exists(path)
