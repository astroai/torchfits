"""``torchfits info`` — HDU inventory."""

from __future__ import annotations

import argparse
from typing import Any

from .common import (
    EXIT_OK,
    emit_records,
    header_extname,
    hdu_type_name,
    iter_file_hdu_pairs,
    resolve_paths,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("info", help="list HDUs in FITS files")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    parser.add_argument("--hdu", help="comma-separated HDU indices (default: all)")
    parser.add_argument("--json", action="store_true", help="emit JSON array")
    parser.add_argument("--jsonl", action="store_true", help="emit JSONL records")
    parser.set_defaults(func=run)


def _info_record(path: str, index: int, hdu: Any) -> dict[str, Any]:
    header = hdu.header
    record: dict[str, Any] = {
        "file": path,
        "hdu": index,
        "name": header_extname(header, index),
        "type": hdu_type_name(header, hdu),
    }
    if record["type"] == "IMAGE":
        shape = getattr(hdu, "_get_shape_str", lambda: None)()
        dtype = getattr(hdu, "_get_dtype_str", lambda: None)()
        if shape is not None:
            record["shape"] = shape
        if dtype is not None:
            record["dtype"] = dtype
    elif record["type"] == "TABLE":
        nrows = header.get("NAXIS2")
        if nrows is not None:
            record["nrows"] = int(nrows)
        tfields = header.get("TFIELDS")
        if tfields is not None:
            record["ncols"] = int(tfields)
    return record


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    records = [
        _info_record(path, index, hdu)
        for path, index, hdu in iter_file_hdu_pairs(paths, args.hdu)
    ]
    emit_records(records, json_mode=args.json, jsonl=args.jsonl)
    return EXIT_OK
