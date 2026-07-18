"""``torchfits cutout`` — read a pixel subset and write FITS."""

from __future__ import annotations

import argparse

import torchfits

from torchfits._io_engine.paths import has_cfitsio_filter

from .common import EXIT_OK, UsageError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("cutout", help="extract a pixel box to a FITS file")
    parser.add_argument(
        "input",
        help="input FITS image path (optional CFITSIO section, e.g. img.fits[10:100,20:200])",
    )
    parser.add_argument("output", help="output FITS path")
    parser.add_argument("--hdu", type=int, default=0, help="source HDU index")
    parser.add_argument(
        "--box",
        default=None,
        help="pixel box as x1,y1,x2,y2 (Python slice end, exclusive); "
        "omit when input uses a CFITSIO image section",
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
    sectioned = has_cfitsio_filter(args.input)
    if sectioned and args.box is not None:
        raise UsageError(
            "pass either a CFITSIO image section in the path or --box, not both"
        )
    if not sectioned and args.box is None:
        raise UsageError(
            "cutout needs --box x1,y1,x2,y2 or a CFITSIO section "
            "(e.g. image.fits[10:100,20:200])"
        )

    if sectioned:
        # CFITSIO parses the image section at open; pass path through unchanged.
        tensor = torchfits.read_tensor(args.input, hdu=args.hdu)
    else:
        x1, y1, x2, y2 = _parse_box(args.box)
        tensor = torchfits.read_subset(args.input, args.hdu, x1, y1, x2, y2)
    header = torchfits.get_header(args.input, args.hdu)
    torchfits.write_tensor(args.output, tensor, header=header, overwrite=True)
    return EXIT_OK
