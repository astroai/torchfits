import numpy as np
import pytest

from torchfits import where


def test_public_where_evaluates_common_predicates():
    ast = where.parse_where_expression("A > 1 AND B IS NOT NULL")
    mask = where.evaluate_where(
        ast,
        {"A": np.array([0, 2, 3]), "B": np.array([1, None, 4], dtype=object)},
    )
    np.testing.assert_array_equal(mask, np.array([False, False, True]))


def test_public_where_rejects_null_cmp_on_numeric():
    ast = where.parse_where_expression("A == NULL")
    with pytest.raises(ValueError, match="not supported on numeric"):
        where.evaluate_where(ast, {"A": np.array([1, 2, 3], dtype=np.int32)})


def test_public_where_exports_parser_helpers():
    assert where.parse_where_literal("42") == 42


# NaN / None rows are nulls: every negation must exclude them (SQL 3VL).
_F = np.array([1.0, 2.0, np.nan, 3.0, 5.0])
_S = np.array(["a", "b", None, "c", "a"], dtype=object)


def _mask(expr, data):
    return where.evaluate_where(where.parse_where_expression(expr), data).tolist()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("X == 1", [True, False, False, False, False]),
        ("NOT (X == 1)", [False, True, False, True, True]),
        ("X != 1", [False, True, False, True, True]),
        ("X IN (1, 3)", [True, False, False, True, False]),
        ("X NOT IN (1, 3)", [False, True, False, False, True]),
        ("X BETWEEN 2 AND 3", [False, True, False, True, False]),
        ("X NOT BETWEEN 2 AND 3", [True, False, False, False, True]),
        ("X IS NULL", [False, False, True, False, False]),
        ("X IS NOT NULL", [True, True, False, True, True]),
    ],
)
def test_where_float_nan_masks(expr, expected):
    assert _mask(expr, {"X": _F}) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("NOT (S == 'a')", [False, True, False, True, False]),
        ("S != 'a'", [False, True, False, True, False]),
        ("S IN ('a', 'c')", [True, False, False, True, True]),
        ("S NOT IN ('a', 'c')", [False, True, False, False, False]),
        ("S IS NULL", [False, False, True, False, False]),
        ("S IS NOT NULL", [True, True, False, True, True]),
        ("S == NULL", [False, False, True, False, False]),
        ("S != NULL", [True, True, False, True, True]),
    ],
)
def test_where_object_none_masks(expr, expected):
    assert _mask(expr, {"S": _S}) == expected


def test_where_not_of_compound_excludes_nulls_of_every_column():
    data = {"X": _F, "S": _S}
    expected = [False, False, False, True, True]
    assert _mask("NOT (X == 1 OR S == 'b')", data) == expected


def test_where_unknown_column_raises():
    with pytest.raises(ValueError, match="Unknown column"):
        _mask("Y == 1", {"X": _F})
