import numpy as np

from torchfits import where


def test_public_where_evaluates_common_predicates():
    ast = where.parse_where_expression("A > 1 AND B IS NOT NULL")
    mask = where.evaluate_where(
        ast,
        {"A": np.array([0, 2, 3]), "B": np.array([1, None, 4], dtype=object)},
    )
    np.testing.assert_array_equal(mask, np.array([False, False, True]))


def test_public_where_exports_parser_helpers():
    assert where.parse_where_literal("42") == 42
    assert where.where_identifier_re.match("column_1")
