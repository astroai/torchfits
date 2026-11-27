"""Float payloads into integer columns are range-checked before any write.

update_rows checked only integer payloads: a float such as 2.0**40 into an
int32 column reached CFITSIO, which wrote the clipped value 2147483647 and
then reported the overflow, so update_rows raised RuntimeError *after*
modifying the file (mmap="auto" and mmap=False). append_rows / insert_rows
rolled back but raised an opaque RuntimeError. Every path now refuses such a
value with ValueError before anything is written, judging it the way CFITSIO
converts it (truncation for plain columns, scaling + rounding for TSCAL /
TZERO and the unsigned conventions).
"""

import warnings

import numpy as np
import pytest

import torchfits


def _make(path, column, schema=None):
    torchfits.table.write(
        path, {"a": column, "b": np.arange(3, dtype=np.float64)}, schema=schema
    )
    return open(path, "rb").read()


def _write(op, path, payload, mmap):
    if op == "update":
        torchfits.table.update_rows(path, {"a": payload}, (0, len(payload)), mmap=mmap)
    elif op == "append":
        torchfits.table.append_rows(path, {"a": payload})
    else:
        torchfits.table.insert_rows(path, {"a": payload}, row=1)


OPS = [
    ("update", True),
    ("update", False),
    ("update", "auto"),
    ("append", None),
    ("insert", None),
]
# mmap=True writes only payloads of the column's own dtype (a float payload
# into an integer column is refused as a dtype mismatch), so the paths that
# accept a float payload are the others.
WRITING_OPS = [o for o in OPS if o[1] is not True]


@pytest.mark.parametrize("op,mmap", OPS)
@pytest.mark.parametrize(
    "dtype,bad",
    [
        (np.int32, float(2**40)),
        (np.int32, -float(2**40)),
        (np.int32, 2147483647.9),  # CFITSIO refuses > 2147483647.49
        (np.int16, 40000.0),
        (np.uint8, -0.6),
        (np.uint8, 256.0),
        (np.int64, 2.0**63),
        (np.uint16, 65535.6),  # unsigned convention: rounds to 65536
        (np.uint16, -0.6),
    ],
)
def test_out_of_range_float_leaves_file_unchanged(tmp_path, op, mmap, dtype, bad):
    path = str(tmp_path / "t.fits")
    before = _make(path, np.zeros(3, dtype=dtype))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="nothing was written"):
            _write(op, path, np.array([bad]), mmap)
    assert open(path, "rb").read() == before


@pytest.mark.parametrize("op,mmap", WRITING_OPS)
def test_scaled_column_is_checked_in_stored_domain(tmp_path, op, mmap):
    path = str(tmp_path / "s.fits")
    schema = {"a": {"format": "I", "bscale": 0.5, "bzero": 10.0}}
    before = _make(path, np.zeros(3, dtype=np.int16), schema)
    # (20000 - 10) / 0.5 = 39980 does not fit int16.
    for payload in (np.array([20000.0]), np.array([20000])):
        with pytest.raises(ValueError, match="nothing was written"):
            _write(op, path, payload, mmap)
        assert open(path, "rb").read() == before

    # In range: physical values, no truncation warning (scaling rounds).
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _write(op, path, np.array([11.5]), mmap)
    assert 11.5 in torchfits.table.read(path).column("a").to_pylist()


@pytest.mark.parametrize("op,mmap", WRITING_OPS)
def test_in_range_float_is_truncated_with_warning(tmp_path, op, mmap):
    path = str(tmp_path / "w.fits")
    _make(path, np.zeros(3, dtype=np.int32))
    with pytest.warns(RuntimeWarning, match="truncated toward zero"):
        _write(op, path, np.array([2147483647.4]), mmap)
    assert 2147483647 in torchfits.table.read(path).column("a").to_pylist()


def test_int64_lower_bound_float_is_accepted(tmp_path):
    path = str(tmp_path / "k.fits")
    _make(path, np.zeros(3, dtype=np.int64))
    torchfits.table.update_rows(path, {"a": np.array([-(2.0**63)])}, (0, 1))
    assert torchfits.table.read(path).column("a").to_pylist()[0] == -(2**63)
