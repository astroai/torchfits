import numpy as np

import torchfits
from torchfits import hdu, table, where


def test_hdu_and_table_public_surfaces_are_importable():
    assert hdu.DataView is not None
    assert hdu.TableDataAccessor is not None
    assert hdu.TensorFrame is not None
    assert callable(table.read)
    assert callable(table.write)
    assert callable(table.clear_cache)


def test_where_public_surface_matches_table_predicate_semantics():
    ast = where.parse_where_expression("A > 1 AND B IS NOT NULL")
    mask = where.evaluate_where(
        ast, {"A": np.array([0, 2, 3]), "B": np.array([1, None, 4], dtype=object)}
    )
    np.testing.assert_array_equal(mask, np.array([False, False, True]))
    assert torchfits.where.parse_where_literal("42") == 42
    assert where.where_columns_from_ast(ast) == ["A", "B"]
