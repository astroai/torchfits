"""Columns omitted from append_rows / insert_rows are stored as undefined.

The omitted-column fill was TNULL handed to the native writer as data, which
then applied TSCAL / TZERO to it: on a column with TZERO = 5, TNULL = 7 the
new row stored 2 and read back as the valid value 7.0 instead of null (only
the unsigned conventions, whose TNULL was moved into the value domain first,
came out right). The fill is now written as the column's undefined value,
stored as-is: TNULL for an integer column, NaN for a floating-point one.
"""

import numpy as np
import pytest

import torchfits


def _make(path, tzero=None, tscal=None, tnull=7):
    schema = {"a": {"format": "J"}}
    if tnull is not None:
        schema["a"]["tnull"] = tnull
    if tzero is not None:
        schema["a"]["bzero"] = tzero
    if tscal is not None:
        schema["a"]["bscale"] = tscal
    torchfits.table.write(
        path,
        {
            "a": np.array([7, 2, 0], dtype=np.int32),
            "f": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "b": np.array([1.0, 2.0, 3.0]),
        },
        schema=schema,
    )


def _add(op, path):
    if op == "append":
        torchfits.table.append_rows(path, {"b": np.array([4.0])})
        return 3
    torchfits.table.insert_rows(path, {"b": np.array([4.0])}, row=1)
    return 1


@pytest.mark.parametrize("op", ["append", "insert"])
@pytest.mark.parametrize(
    "tzero,tscal",
    [(None, None), (5, None), (5, 2.0), (None, 0.5), (2**31, None)],
)
def test_omitted_tnull_column_reads_back_null(tmp_path, op, tzero, tscal):
    path = str(tmp_path / "t.fits")
    _make(path, tzero=tzero, tscal=tscal)
    before = torchfits.table.read(path).column("a").to_pylist()
    assert before[0] is None
    row = _add(op, path)
    after = torchfits.table.read(path).column("a").to_pylist()
    assert after[row] is None
    assert [v for i, v in enumerate(after) if i != row] == before


@pytest.mark.parametrize("op", ["append", "insert"])
def test_omitted_column_without_tnull_is_zero(tmp_path, op):
    path = str(tmp_path / "z.fits")
    _make(path, tnull=None)
    row = _add(op, path)
    assert torchfits.table.read(path).column("a").to_pylist()[row] == 0


@pytest.mark.parametrize("op", ["append", "insert"])
def test_omitted_float_column_is_nan(tmp_path, op):
    path = str(tmp_path / "f.fits")
    _make(path)
    row = _add(op, path)
    f = torchfits.table.read(path).column("f").to_numpy(zero_copy_only=False)
    assert np.isnan(f[row])
    assert not np.isnan(np.delete(f, row)).any()
