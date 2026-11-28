"""A string longer than its schema width is clipped with a RuntimeWarning.

The row mutations warned when they clipped a value to an ``A`` column's
width, while ``table.write(schema=...)``, ``insert_column`` /
``replace_column`` and the first streamed batch clipped it silently, in binary
and ASCII tables alike (r5b-06). Every whole-table writer now warns the same
way; the value is still clipped, as CFITSIO (and astropy) store it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import torchfits

T = torchfits.table


def _fmt(table_type: str) -> str:
    return "3A" if table_type == "binary" else "A3"


@pytest.mark.parametrize("table_type", ["binary", "ascii"])
def test_write_with_narrow_schema_warns_and_clips(tmp_path, table_type):
    path = str(tmp_path / "s.fits")
    with pytest.warns(RuntimeWarning, match="Column 's' has width 3: 1 value"):
        T.write(
            path,
            {"s": ["abcdef", "x"]},
            schema={"s": {"format": _fmt(table_type)}},
            overwrite=True,
            table_type=table_type,
        )
    assert T.read(path)["s"].to_pylist() == ["abc", "x"]


@pytest.mark.parametrize("table_type", ["binary", "ascii"])
def test_replace_column_warns_and_clips(tmp_path, table_type):
    path = str(tmp_path / "s.fits")
    T.write(
        path,
        {"s": ["ab", "x"]},
        schema={"s": {"format": _fmt(table_type)}},
        overwrite=True,
        table_type=table_type,
    )
    with pytest.warns(RuntimeWarning, match="width 3"):
        T.replace_column(path, "s", ["rrrrrr", "r"])
    assert T.read(path)["s"].to_pylist() == ["rrr", "r"]


@pytest.mark.parametrize("table_type", ["binary", "ascii"])
def test_first_streamed_batch_warns_and_clips(tmp_path, table_type):
    path = str(tmp_path / "s.fits")
    with pytest.warns(RuntimeWarning, match="width 3"):
        with T.writer(
            path,
            schema={"s": {"format": _fmt(table_type)}},
            overwrite=True,
            table_type=table_type,
        ) as w:
            w.write({"s": ["abcdef"]})
    assert T.read(path)["s"].to_pylist() == ["abc"]


def test_values_that_fit_do_not_warn(tmp_path):
    path = str(tmp_path / "s.fits")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        T.write(path, {"s": ["abc"], "i": np.arange(1)}, schema={"s": {"format": "3A"}})
        T.write(str(tmp_path / "t.fits"), {"s": ["abcdef", "x"]})
