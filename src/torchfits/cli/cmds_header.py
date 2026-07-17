"""``torchfits header`` — dump FITS header cards."""

from __future__ import annotations

import argparse
from typing import Any

import torchfits

from .common import (
    EXIT_OK,
    emit_records,
    header_extname,
    iter_file_hdu_pairs,
    resolve_paths,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("header", help="dump FITS header cards")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    parser.add_argument("--hdu", help="comma-separated HDU indices (default: all)")
    parser.add_argument("--keyword", help="filter to a single keyword")
    parser.add_argument("--json", action="store_true", help="emit JSON array")
    parser.add_argument("--jsonl", action="store_true", help="emit JSONL records")
    parser.set_defaults(func=run)


def _card_dict(card: Any) -> dict[str, Any]:
    return {"keyword": card.key, "value": card.value, "comment": card.comment}


def _header_records(
    path: str, index: int, header: Any, *, keyword: str | None
) -> list[dict[str, Any]]:
    cards = header.cards
    if keyword is not None:
        key = keyword.upper()
        cards = [card for card in cards if card.key == key]
    return [
        {
            "file": path,
            "hdu": index,
            "name": header_extname(header, index),
            **_card_dict(card),
        }
        for card in cards
    ]


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    records: list[dict[str, Any]] = []
    for path, index, hdu in iter_file_hdu_pairs(paths, args.hdu):
        header = (
            hdu.header if hasattr(hdu, "header") else torchfits.get_header(path, index)
        )
        records.extend(_header_records(path, index, header, keyword=args.keyword))
    if not (args.json or args.jsonl):
        for record in records:
            comment = record.get("comment") or ""
            suffix = f" / {comment}" if comment else ""
            print(
                f"{record['file']}:{record['hdu']} "
                f"{record['keyword']} = {record['value']!r}{suffix}"
            )
        return EXIT_OK
    emit_records(records, json_mode=args.json, jsonl=args.jsonl)
    return EXIT_OK
