"""``torchfits diff`` — compare FITS headers, image pixels, and table data."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torchfits

from .common import EXIT_DIFF, EXIT_OK, IoError, hdu_type_name


_SKIP_HEADER_KEYS = frozenset({"CHECKSUM", "DATASUM"})


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("diff", help="compare two FITS files")
    parser.add_argument("path_a", help="first FITS path")
    parser.add_argument("path_b", help="second FITS path")
    parser.set_defaults(func=run)


def _header_cards(header: Any) -> list[tuple[str, Any, str]]:
    return [
        (card.key, card.value, card.comment)
        for card in header.cards
        if card.key not in _SKIP_HEADER_KEYS
    ]


def _read_image(path: str, index: int) -> Any:
    import torch

    tensor = torchfits.read_tensor(path, hdu=index)
    if not isinstance(tensor, torch.Tensor):
        raise IoError(f"{path}:{index} read_tensor did not return a tensor")
    return tensor


def _image_diffs(path_a: str, path_b: str, index: int) -> list[str]:
    """Exact pixel comparison; NaN (and BLANK, read as NaN) equals NaN.

    Summary statistics are not enough: a permuted image, or integers that
    differ beyond float32 precision, share min/max/mean with the original.
    """
    import numpy as np

    tensor_a = _read_image(path_a, index)
    tensor_b = _read_image(path_b, index)
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        return [
            f"HDU {index} image shape: {list(tensor_a.shape)!r} vs "
            f"{list(tensor_b.shape)!r}"
        ]
    diffs: list[str] = []
    if tensor_a.dtype != tensor_b.dtype:
        diffs.append(
            f"HDU {index} image dtype: {str(tensor_a.dtype).replace('torch.', '')}"
            f" vs {str(tensor_b.dtype).replace('torch.', '')}"
        )
    array_a = tensor_a.numpy()
    array_b = tensor_b.numpy()
    if np.array_equal(array_a, array_b, equal_nan=True):
        return diffs
    differ = array_a != array_b
    if array_a.dtype.kind == "f" or array_b.dtype.kind == "f":
        differ &= ~(np.isnan(array_a) & np.isnan(array_b))
    diffs.append(
        f"HDU {index} image data: {int(differ.sum())} of {differ.size} pixels differ"
    )
    return diffs


def _py_equal(a: Any, b: Any) -> bool:
    """Exact, NaN-aware equality for Arrow ``as_py`` values (incl. lists)."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_py_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_py_equal(a[k], b[k]) for k in a)
    return bool(a == b)


def _table_diffs(path_a: str, path_b: str, index: int) -> list[str]:
    from torchfits import table as tf_table

    table_a = tf_table.read(path_a, hdu=index)
    table_b = tf_table.read(path_b, hdu=index)
    diffs: list[str] = []
    if table_a.num_rows != table_b.num_rows:
        diffs.append(
            f"HDU {index} table rows: {table_a.num_rows} vs {table_b.num_rows}"
        )
    names_a = list(table_a.column_names)
    names_b = list(table_b.column_names)
    if names_a != names_b:
        diffs.append(f"HDU {index} table columns: {names_a!r} vs {names_b!r}")
    if diffs:
        return diffs
    for name in names_a:
        col_a = table_a.column(name)
        col_b = table_b.column(name)
        if col_a.type != col_b.type:
            diffs.append(
                f"HDU {index} table column {name!r} type: {col_a.type} vs {col_b.type}"
            )
            continue
        # Fast path; Arrow's equals treats NaN != NaN, so an unequal result
        # falls back to an exact NaN-aware per-row comparison.
        if col_a.equals(col_b):
            continue
        rows = sum(
            1
            for value_a, value_b in zip(col_a.to_pylist(), col_b.to_pylist())
            if not _py_equal(value_a, value_b)
        )
        if rows:
            diffs.append(
                f"HDU {index} table column {name!r}: {rows} of {table_a.num_rows} "
                "rows differ"
            )
    return diffs


def _values_equal(a: Any, b: Any) -> bool:
    """NaN-aware equality: identical files containing NaN must not diff."""
    try:
        both_nan = isinstance(a, float) and isinstance(b, float) and a != a and b != b
        if both_nan:
            return True
        return bool(a == b)
    except TypeError:
        return bool(a == b)


def _cards_equal(
    card_a: tuple[str, Any, str] | None, card_b: tuple[str, Any, str] | None
) -> bool:
    if card_a is None or card_b is None:
        return card_a is card_b
    return (
        card_a[0] == card_b[0]
        and _values_equal(card_a[1], card_b[1])
        and card_a[2] == card_b[2]
    )


def _diff_pair(path_a: str, path_b: str) -> list[str]:
    diffs: list[str] = []
    try:
        with torchfits.open(path_a) as hdul_a, torchfits.open(path_b) as hdul_b:
            if len(hdul_a) != len(hdul_b):
                diffs.append(f"HDU count: {len(hdul_a)} vs {len(hdul_b)}")
            for index in range(min(len(hdul_a), len(hdul_b))):
                header_a = hdul_a[index].header
                header_b = hdul_b[index].header
                type_a = hdu_type_name(header_a, hdul_a[index])
                type_b = hdu_type_name(header_b, hdul_b[index])
                if type_a != type_b:
                    diffs.append(f"HDU {index} type: {type_a} vs {type_b}")
                cards_a = _header_cards(header_a)
                cards_b = _header_cards(header_b)
                for position in range(max(len(cards_a), len(cards_b))):
                    card_a = cards_a[position] if position < len(cards_a) else None
                    card_b = cards_b[position] if position < len(cards_b) else None
                    if not _cards_equal(card_a, card_b):
                        diffs.append(
                            f"HDU {index} card {position + 1}: {card_a!r} vs {card_b!r}"
                        )
                if type_a == "IMAGE" and type_b == "IMAGE":
                    diffs.extend(_image_diffs(path_a, path_b, index))
                elif type_a == "TABLE" and type_b == "TABLE":
                    diffs.extend(_table_diffs(path_a, path_b, index))
    except IoError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return diffs


def run(args: argparse.Namespace) -> int:
    diffs = _diff_pair(args.path_a, args.path_b)
    if not diffs:
        return EXIT_OK
    for line in diffs:
        print(line, file=sys.stderr)
    return EXIT_DIFF
