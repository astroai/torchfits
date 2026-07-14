"""Public table-predicate parsing and evaluation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

if TYPE_CHECKING:
    import numpy as np

from ._where import (
    _WHERE_IDENT_RE as where_identifier_re,
    _normalize_where_syntax as normalize_where_syntax,
    _parse_where_expression as parse_where_expression,
    _parse_where_literal as parse_where_literal,
    _tokenize_where_expression as tokenize_where_expression,
    _where_columns_from_ast as where_columns_from_ast,
)


def evaluate_where(ast: tuple[Any, ...], data: Mapping[str, Any]) -> np.ndarray:
    """Evaluate a parsed predicate against mapping values as NumPy arrays.

    The ``numpy`` import is lazy to avoid a mandatory dependency at
    the package level (torchfits itself only requires PyTorch).
    """
    import numpy as np

    kind = ast[0]
    if kind == "and":
        return cast(
            np.ndarray, evaluate_where(ast[1], data) & evaluate_where(ast[2], data)
        )
    if kind == "or":
        return cast(
            np.ndarray, evaluate_where(ast[1], data) | evaluate_where(ast[2], data)
        )
    if kind == "not":
        return ~evaluate_where(ast[1], data)

    column = ast[1]
    if column not in data:
        raise ValueError(f"Unknown column: {column}")
    values = np.asarray(data[column])
    if kind == "cmp":
        _, _, operator, literal = ast
        if literal is None:
            if operator == "==":
                return np.asarray([value is None for value in values], dtype=bool)
            if operator == "!=":
                return np.asarray([value is not None for value in values], dtype=bool)
            raise ValueError("NULL comparisons only support == and !=")
        operators = {
            "==": np.equal,
            "!=": np.not_equal,
            ">": np.greater,
            ">=": np.greater_equal,
            "<": np.less,
            "<=": np.less_equal,
        }
        try:
            return cast(np.ndarray, operators[operator](values, literal))
        except KeyError as exc:
            raise ValueError(f"Unsupported operator: {operator}") from exc
    if kind == "in":
        _, _, literals, negate = ast
        mask = cast(np.ndarray, np.isin(values, literals))
        return ~mask if negate else mask
    if kind == "between":
        _, _, low, high, negate = ast
        mask = cast(np.ndarray, (values >= low) & (values <= high))
        return ~mask if negate else mask
    if kind == "isnull":
        _, _, negate = ast
        if np.issubdtype(values.dtype, np.floating):
            mask = np.isnan(values)
        else:
            mask = np.asarray([value is None for value in values], dtype=bool)
        return ~mask if negate else mask
    raise ValueError(f"Invalid AST node: {kind}")


__all__ = [
    "evaluate_where",
    "normalize_where_syntax",
    "parse_where_expression",
    "parse_where_literal",
    "tokenize_where_expression",
    "where_identifier_re",
    "where_columns_from_ast",
]
