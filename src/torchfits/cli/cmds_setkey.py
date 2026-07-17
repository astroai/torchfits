"""``torchfits setkey`` — set one FITS header keyword."""

from __future__ import annotations

import argparse
from typing import Any

import torchfits
from torchfits._hdu.header import Header

from .common import EXIT_OK, IoError, UsageError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("setkey", help="set a FITS header keyword")
    parser.add_argument("input", help="input FITS path")
    parser.add_argument("--key", required=True, help="header keyword")
    parser.add_argument("--value", required=True, help="header value")
    parser.add_argument("--hdu", type=int, default=0, help="target HDU index")
    parser.add_argument(
        "--out",
        help="output FITS path (default: update input in place)",
    )
    parser.set_defaults(func=run)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def run(args: argparse.Namespace) -> int:
    target = args.out or args.input
    keyword = args.key.upper()
    if keyword in {"", "HISTORY", "COMMENT"}:
        raise UsageError("--key must be a normal FITS keyword")
    try:
        if args.out and args.out != args.input:
            with torchfits.open(args.input) as hdul:
                hdul.write(target, overwrite=True)
        header = Header(torchfits.get_header(target, args.hdu))
        header[keyword] = _parse_value(args.value)
        torchfits.io._write_header_cards_if_supported(target, args.hdu, header)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
