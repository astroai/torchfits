"""``torchfits cutout`` — read a pixel subset and write FITS."""

from __future__ import annotations

import argparse

import torchfits

from .common import EXIT_OK, UsageError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("cutout", help="extract a pixel box to a FITS file")
    parser.add_argument("input", help="input FITS image path")
    parser.add_argument("output", help="output FITS path")
    parser.add_argument("--hdu", type=int, default=0, help="source HDU index")
    parser.add_argument(
        "--box",
        required=True,
        help="pixel box as x1,y1,x2,y2 (Python slice end, exclusive)",
    )
    parser.set_defaults(func=run)


def _parse_box(raw: str) -> tuple[int, int, int, int]:
    parts = [piece.strip() for piece in raw.split(",")]
    if len(parts) != 4:
        raise UsageError("--box requires x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as exc:
        raise UsageError("--box values must be integers") from exc
    return x1, y1, x2, y2


def run(args: argparse.Namespace) -> int:
    x1, y1, x2, y2 = _parse_box(args.box)
    tensor = torchfits.read_subset(args.input, args.hdu, x1, y1, x2, y2)
    header = torchfits.get_header(args.input, args.hdu)
    torchfits.write_tensor(args.output, tensor, header=header, overwrite=True)
    return EXIT_OK
