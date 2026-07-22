"""``torchfits header`` — dump FITS header cards or keyword tables."""

from __future__ import annotations

import argparse
import fnmatch
from typing import Any

import torchfits

from .common import (
    EXIT_OK,
    UsageError,
    add_emit_format_args,
    add_hdu_arg,
    add_keyword_arg,
    emit_records,
    header_extname,
    iter_file_hdu_pairs,
    resolve_emit_format,
    resolve_paths,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "header",
        help="dump FITS header cards (all HDUs by default)",
        description=(
            "Dump header cards for every HDU (fitsheader / listhead style). "
            "Use -e/--hdu to narrow; -k/--keyword to filter cards."
        ),
    )
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_keyword_arg(
        parser,
        help=(
            "filter to keyword(s); supports wildcards (e.g. NAXIS*); "
            "repeat for multiple; required with --keyword-table"
        ),
    )
    parser.add_argument(
        "--keyword-table",
        action="store_true",
        dest="keyword_table",
        help="print a table of selected keywords across files",
    )
    add_emit_format_args(parser)
    parser.set_defaults(func=run)


def _card_dict(card: Any) -> dict[str, Any]:
    return {"keyword": card.key, "value": card.value, "comment": card.comment}


def _header_lookup(header: Any) -> dict[str, Any]:
    return {str(card.key).upper(): card.value for card in header.cards}


def _pattern_has_wildcards(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _keyword_matches(card_key: str, patterns: list[str]) -> bool:
    upper = str(card_key).upper()
    for pattern in patterns:
        pat = pattern.upper()
        if _pattern_has_wildcards(pat):
            if fnmatch.fnmatchcase(upper, pat):
                return True
        elif upper == pat:
            return True
    return False


def _expand_keyword_columns(headers: list[Any], patterns: list[str]) -> list[str]:
    """Resolve -k patterns to concrete column names (wildcards → matched keys)."""
    columns: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        pat = pattern.upper()
        if not _pattern_has_wildcards(pat):
            if pat not in seen:
                columns.append(pat)
                seen.add(pat)
            continue
        matched: list[str] = []
        for header in headers:
            for card in header.cards:
                key = str(card.key).upper()
                if key not in seen and fnmatch.fnmatchcase(key, pat):
                    matched.append(key)
                    seen.add(key)
        columns.extend(sorted(matched))
    return columns


def _card_records(
    path: str, index: int, header: Any, *, keywords: list[str] | None
) -> list[dict[str, Any]]:
    cards = list(header.cards)
    if keywords:
        cards = [card for card in cards if _keyword_matches(str(card.key), keywords)]
    return [
        {
            "file": path,
            "hdu": index,
            "name": header_extname(header, index),
            **_card_dict(card),
        }
        for card in cards
    ]


def _keyword_table_record(
    path: str, index: int, header: Any, *, keywords: list[str]
) -> dict[str, Any]:
    lookup = _header_lookup(header)
    record: dict[str, Any] = {
        "file": path,
        "hdu": index,
        "name": header_extname(header, index),
    }
    for key in keywords:
        record[key.upper()] = lookup.get(key.upper())
    return record


def _print_keyword_table(records: list[dict[str, Any]], keywords: list[str]) -> None:
    keys = ["file", "hdu", "name", *[key.upper() for key in keywords]]
    widths = {key: len(key) for key in keys}
    rows: list[dict[str, str]] = []
    for record in records:
        row = {
            key: "" if record.get(key) is None else str(record.get(key)) for key in keys
        }
        rows.append(row)
        for key in keys:
            widths[key] = max(widths[key], len(row[key]))
    print("  ".join(key.ljust(widths[key]) for key in keys))
    print("  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  ".join(row[key].ljust(widths[key]) for key in keys))


def _format_value(value: Any) -> tuple[str, bool]:
    """Return ``(rendered, quote_like_string)`` for fitsheader-ish alignment."""
    if isinstance(value, bool):
        return ("T" if value else "F"), False
    if isinstance(value, (int, float)):
        return str(value), False
    if value is None:
        return "", False
    text = str(value)
    if text in {"T", "F"}:
        return text, False
    if text in {"True", "False"}:
        return ("T" if text == "True" else "F"), False
    try:
        int(text)
        return text, False
    except ValueError:
        pass
    try:
        float(text)
        return text, False
    except ValueError:
        pass
    return f"'{text}'", True


def _format_card_line(card: Any) -> str:
    """fitsheader-like single-line card (not a strict 80-byte image)."""
    key = str(card.key)
    value = card.value
    comment = str(card.comment or "").rstrip()
    upper = key.upper()
    if upper in {"HISTORY", "COMMENT", "END"} or key.strip() == "":
        text = f"{key:8} {value}" if value not in (None, "") else f"{key:8}"
        return text.rstrip()
    rendered, as_string = _format_value(value)
    if rendered == "":
        body = f"{key:8}="
    elif as_string:
        body = f"{key:8}= {rendered}"
    else:
        body = f"{key:8}= {rendered:>20}"
    if comment:
        return f"{body} / {comment}"
    return body


def _print_headers_text(
    paths: list[str],
    hdu: str | None,
    *,
    keywords: list[str] | None,
) -> None:
    first_block = True
    for path, index, hdu_obj in iter_file_hdu_pairs(paths, hdu):
        header = (
            hdu_obj.header
            if hasattr(hdu_obj, "header")
            else torchfits.read_header(path, index)
        )
        cards = list(header.cards)
        if keywords:
            cards = [
                card for card in cards if _keyword_matches(str(card.key), keywords)
            ]
        if not first_block:
            print()
        first_block = False
        name = header_extname(header, index)
        print(f"# HDU {index} ({name}) in {path}:")
        for card in cards:
            print(_format_card_line(card))


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    keywords = list(args.keywords or [])
    if args.keyword_table and not keywords:
        raise UsageError("--keyword-table requires at least one --keyword")

    fmt = resolve_emit_format(args)
    if fmt == "text" and not args.keyword_table:
        _print_headers_text(paths, args.hdu, keywords=keywords or None)
        return EXIT_OK

    pairs: list[tuple[str, int, Any]] = []
    for path, index, hdu in iter_file_hdu_pairs(paths, args.hdu):
        header = (
            hdu.header if hasattr(hdu, "header") else torchfits.read_header(path, index)
        )
        pairs.append((path, index, header))

    columns = (
        _expand_keyword_columns([header for _, _, header in pairs], keywords)
        if args.keyword_table
        else keywords
    )

    records: list[dict[str, Any]] = []
    for path, index, header in pairs:
        if args.keyword_table:
            records.append(_keyword_table_record(path, index, header, keywords=columns))
        else:
            records.extend(
                _card_records(path, index, header, keywords=keywords or None)
            )

    if args.keyword_table and fmt == "text":
        _print_keyword_table(records, columns)
        return EXIT_OK

    emit_records(records, format=fmt)
    return EXIT_OK
