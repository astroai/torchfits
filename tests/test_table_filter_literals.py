"""Literal types accepted by the native filtered-table binding.

``read_fits_table_filtered`` is the fast path behind ``where=`` pushdown.  Its
literal contract is float/int: Python scalars and the NumPy scalar types users
actually pass.  A NumPy integer scalar used to be rejected because
``nb::isinstance<int>`` is an exact-type test, which made the same predicate
work or fail depending on how the threshold had been produced.

Booleans and complex scalars are not filter numbers: they must raise rather
than be coerced to 1/0 or have their imaginary part discarded.
"""

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as cpp


@pytest.fixture
def table_path(tmp_path):
    path = str(tmp_path / "filter_literals.fits")
    torchfits.table.write(
        path,
        {
            "A": torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype=torch.int32),
            "F": torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
        },
    )
    return path


def _filtered(path, column, op, literal):
    data = cpp.read_fits_table_filtered(path, 1, [column], [(column, op, literal)])
    return data[column].tolist()


@pytest.mark.parametrize(
    "literal",
    [
        np.int8(1),
        np.int16(1),
        np.int32(1),
        np.int64(1),
        np.uint8(1),
        np.uint16(1),
        np.uint32(1),
        np.uint64(1),
    ],
)
def test_numpy_integer_literals_are_accepted(table_path, literal):
    assert _filtered(table_path, "A", ">", literal) == [2, 3]


@pytest.mark.parametrize(
    "literal",
    [
        1.5,
        np.float16(1.5),
        np.float32(1.5),
        np.float64(1.5),
        np.longdouble(1.5),
    ],
)
def test_numpy_float_literals_are_accepted(table_path, literal):
    assert _filtered(table_path, "A", ">", literal) == [2, 3]


@pytest.mark.parametrize("literal", [np.int32(1), np.int64(1), np.uint32(1)])
def test_integer_literals_compare_against_float_columns(table_path, literal):
    assert _filtered(table_path, "F", ">", literal) == [2.0, 3.0]


@pytest.mark.parametrize(
    "literal",
    [
        True,
        False,
        np.bool_(True),
        np.complex128(1 + 2j),
        "1",
        None,
        np.array([1]),
        np.array(1.5),
        2**64,
        -(2**63) - 1,
    ],
)
def test_non_numeric_literals_are_rejected(table_path, literal):
    with pytest.raises(RuntimeError, match="Unsupported filter value type"):
        _filtered(table_path, "A", ">", literal)


@pytest.mark.parametrize("literal", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_literals_are_rejected(table_path, literal):
    with pytest.raises(RuntimeError, match="must be finite"):
        _filtered(table_path, "A", ">", literal)


U64 = [0, 5, 2**63 - 1, 2**63, 2**63 + 5, 10**19, 2**64 - 2, 2**64 - 1]
I64 = [-(2**63), -5, 0, 5, 2**63 - 1]


@pytest.fixture
def wide_path(tmp_path):
    path = str(tmp_path / "wide.fits")
    torchfits.table.write(
        path,
        {"U": np.array(U64, dtype=np.uint64)},
    )
    torchfits.table.write(
        str(tmp_path / "i64.fits"), {"I": np.array(I64, dtype=np.int64)}
    )
    return path


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize(
    "literal",
    [2**63, 2**63 + 5, 2**64 - 2, 2**64 - 1, np.uint64(2**64 - 1), 2.0**63, -(2.0**64)],
)
def test_literals_beyond_int64_compare_exactly(wide_path, tmp_path, op, literal):
    """A literal outside int64 was clamped to INT64_MAX (or refused): uint64
    columns now compare it exactly and other integer columns treat it as a
    bound above (or below) every value."""
    exact = int(literal)
    expected_u = [v for v in U64 if _OPS[op](v, exact)]
    assert _filtered(wide_path, "U", op, literal) == expected_u
    expected_i = [v for v in I64 if _OPS[op](v, exact)]
    assert _filtered(str(tmp_path / "i64.fits"), "I", op, literal) == expected_i


@pytest.mark.parametrize(
    "where",
    [
        "U > 1.8e19",
        "U >= 1.0e19",
        "U == 18446744073709551615",
        "U != 18446744073709551615",
        "U > 9223372036854775808",
        "U <= 9223372036854775807",
    ],
)
@pytest.mark.parametrize("backend", ["auto", "cpp", "torch"])
def test_uint64_where_beyond_int64_agrees_across_backends(wide_path, where, backend):
    col, op, lit = where.split()
    literal = float(lit) if "e" in lit else int(lit)
    expected = [v for v in U64 if _OPS[op](v, literal)]
    result = torchfits.table.read(wide_path, where=where, backend=backend)
    assert result.column("U").to_pylist() == expected
