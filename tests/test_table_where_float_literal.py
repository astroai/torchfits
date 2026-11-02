"""A float literal compared with an integer column must not be truncated.

The native pushdown scan compared integer cells against the literal truncated
toward zero, so ``X >= 2.5`` matched 2 and ``X == 2.5`` matched 2.
"""

import operator

import numpy as np
import pytest

import torchfits.table as table
from torchfits._table.write import write as write_table

OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint8, np.uint16])
@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("literal", [2.5, -1.5, 3.0])
def test_integer_column_float_literal(tmp_path, dtype, op, literal):
    values = np.array([0, 1, 2, 3, 4], dtype=dtype)
    if np.issubdtype(dtype, np.signedinteger):
        values = np.array([-3, -2, -1, 0, 1, 2, 3, 4], dtype=dtype)
    path = tmp_path / "ints.fits"
    write_table(str(path), {"X": values}, overwrite=True)

    expected = values[OPS[op](values.astype(np.float64), literal)].tolist()
    for kwargs in ({}, {"backend": "torch"}):
        got = table.read(str(path), hdu=1, where=f"X {op} {literal}", **kwargs)
        col = got["X"]
        col = col.to_pylist() if hasattr(col, "to_pylist") else col.tolist()
        assert col == expected, (op, literal, kwargs)


def test_native_filtered_read_float_literal(tmp_path):
    import torchfits._C as C

    path = tmp_path / "ints.fits"
    write_table(
        str(path), {"X": np.array([1, 2, 3, 4], dtype=np.int32)}, overwrite=True
    )
    for op, literal, expected in [
        (">=", 2.5, [3, 4]),
        (">", -2.5, [1, 2, 3, 4]),
        ("<=", 2.5, [1, 2]),
        ("<", 2.5, [1, 2]),
        ("==", 2.5, []),
        ("!=", 2.5, [1, 2, 3, 4]),
    ]:
        out = C.read_fits_table_filtered(str(path), 1, ["X"], [("X", op, literal)])
        assert out["X"].tolist() == expected, (op, literal)
        # Two filters exercise the multi-filter scan.
        out = C.read_fits_table_filtered(
            str(path), 1, ["X"], [("X", op, literal), ("X", ">=", -10)]
        )
        assert out["X"].tolist() == expected, (op, literal, "multi")
