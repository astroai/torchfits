"""Test that TableHDURef to_arrow and scan_arrow support kwargs collisions.

TableHDURef wrapped kwargs dict passed to torchfits.table.read/scan, which
caused `TypeError: multiple values for keyword argument 'columns'` when
users explicitly passed `columns` or `row_slice` as kwargs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
from astropy.io import fits  # noqa: E402


def test_table_hdu_ref_to_arrow_kwargs_collision(tmp_path):
    path = str(tmp_path / "test.fits")
    col1 = fits.Column(name="A", format="J", array=np.array([1, 2, 3], dtype=np.int32))
    col2 = fits.Column(name="B", format="J", array=np.array([4, 5, 6], dtype=np.int32))
    fits.BinTableHDU.from_columns([col1, col2]).writeto(path, overwrite=True)

    hdul = torchfits.open(path)
    ref = hdul[1]

    # Providing `columns` explicitly in kwargs must override the internal default
    # without throwing a TypeError.
    table = ref.to_arrow(columns=["A"])
    assert table is not None
    assert table.column_names == ["A"]
    assert table["A"].to_pylist() == [1, 2, 3]

    # Providing `row_slice` explicitly in kwargs must also override safely.
    table2 = ref.to_arrow(row_slice=slice(1, 3))
    assert table2 is not None
    assert table2.column_names == ["A", "B"]
    assert table2["A"].to_pylist() == [2, 3]

def test_table_hdu_ref_scan_arrow_kwargs_collision(tmp_path):
    path = str(tmp_path / "test.fits")
    col1 = fits.Column(name="A", format="J", array=np.array([1, 2, 3], dtype=np.int32))
    col2 = fits.Column(name="B", format="J", array=np.array([4, 5, 6], dtype=np.int32))
    fits.BinTableHDU.from_columns([col1, col2]).writeto(path, overwrite=True)

    hdul = torchfits.open(path)
    ref = hdul[1]

    # Providing `columns` explicitly in kwargs must override the internal default
    # without throwing a TypeError.
    batches = list(ref.scan_arrow(columns=["A"], batch_size=2))
    assert len(batches) == 2
    assert batches[0].column_names == ["A"]
    assert batches[0]["A"].to_pylist() == [1, 2]
    assert batches[1]["A"].to_pylist() == [3]
