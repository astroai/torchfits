"""Structure tests for the native read strategies behind ``table.read``.

``_read_scan._read_cpp_table_chunk`` tries ``read_full_raw`` /
``read_selected_rows`` / ``read_row_window`` (``_table/_read_strategies.py``);
each falls back from the mmap reader to the buffered reader, and an invalid
geometry error always propagates instead of degrading to a slower path.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits
import torchfits._C as cpp
from torchfits._table import _read_scan, _read_strategies


@pytest.fixture()
def table_path(tmp_path):
    path = str(tmp_path / "t.fits")
    torchfits.table.write(
        path,
        {"A": np.arange(6, dtype=np.int32), "B": np.linspace(0, 1, 6)},
        overwrite=True,
    )
    return path


def _plain_read(path, **kwargs):
    return torchfits.table.read(
        path, decode_bytes=False, apply_fits_nulls=False, **kwargs
    )


def test_old_private_names_stay_importable():
    assert _read_scan._reraise_invalid_geometry is (
        _read_strategies._reraise_invalid_geometry
    )


def test_full_read_falls_back_from_mmap_to_buffered(table_path, monkeypatch):
    real = cpp.read_fits_table_raw
    calls = []

    def flaky(path, hdu, cols, use_mmap):
        calls.append(use_mmap)
        if use_mmap:
            raise RuntimeError("mmap layout unsupported")
        return real(path, hdu, cols, use_mmap)

    monkeypatch.setattr(cpp, "read_fits_table_raw", flaky)
    table = _plain_read(table_path, columns=["A", "B"])
    assert table.column("A").to_pylist() == list(range(6))
    assert calls == [True, False]


def test_row_window_falls_back_from_mmap_to_buffered(table_path, monkeypatch):
    def mmap_fails(*args):
        raise RuntimeError("mmap layout unsupported")

    monkeypatch.setattr(cpp, "read_fits_table_rows_raw", mmap_fails)
    table = _plain_read(table_path, columns=["A"], row_slice=(2, 5))
    assert table.column("A").to_pylist() == [2, 3, 4]


def test_invalid_geometry_is_never_masked_by_a_fallback(table_path, monkeypatch):
    def broken(*args):
        raise RuntimeError("Invalid table geometry: NAXIS1 too small")

    monkeypatch.setattr(cpp, "read_fits_table_rows_raw", broken)
    with pytest.raises(RuntimeError, match="Invalid table geometry"):
        _plain_read(table_path, columns=["A"], row_slice=(2, 5))


def test_selected_rows_strategy(table_path):
    table = _plain_read(table_path, columns=["A"], rows=[5, 0, 3])
    assert table.column("A").to_pylist() == [5, 0, 3]
