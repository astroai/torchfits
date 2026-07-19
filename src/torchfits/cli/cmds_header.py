"""``torchfits header`` — dump FITS header cards or keyword tables."""

from __future__ import annotations

import argparse
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
    parser = subparsers.add_parser("header", help="dump FITS header cards")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_keyword_arg(
        parser,
        help="filter to keyword(s); repeat for multiple; required with --keyword-table",
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


def _card_records(
    path: str, index: int, header: Any, *, keywords: list[str] | None
) -> list[dict[str, Any]]:
    cards = list(header.cards)
    if keywords:
        wanted = {key.upper() for key in keywords}
        cards = [card for card in cards if str(card.key).upper() in wanted]
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


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    keywords = list(args.keywords or [])
    if args.keyword_table and not keywords:
        raise UsageError("--keyword-table requires at least one --keyword")

    records: list[dict[str, Any]] = []
    for path, index, hdu in iter_file_hdu_pairs(paths, args.hdu):
        header = (
            hdu.header if hasattr(hdu, "header") else torchfits.read_header(path, index)
        )
        if args.keyword_table:
            records.append(
                _keyword_table_record(path, index, header, keywords=keywords)
            )
        else:
            records.extend(
                _card_records(path, index, header, keywords=keywords or None)
            )

    fmt = resolve_emit_format(args)
    if args.keyword_table and fmt == "text":
        _print_keyword_table(records, keywords)
        return EXIT_OK

    if fmt == "text":
        for record in records:
            comment = record.get("comment") or ""
            suffix = f" / {comment}" if comment else ""
            print(
                f"{record['file']}:{record['hdu']} "
                f"{record['keyword']} = {record['value']!r}{suffix}"
            )
        return EXIT_OK

    emit_records(records, format=fmt)
    return EXIT_OK
