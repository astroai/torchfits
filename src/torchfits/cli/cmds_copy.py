"""``torchfits copy`` — FITS→FITS copy preserving MEF structure."""

from __future__ import annotations

import argparse

import torchfits

from .common import EXIT_OK, IoError, UsageError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("copy", help="copy FITS file(s) preserving HDUs")
    parser.add_argument("input", help="input FITS path")
    parser.add_argument("output", help="output FITS path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        with torchfits.open(args.input) as hdul:
            hdul.write(args.output, overwrite=True)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{args.input}: {exc}") from exc
    return EXIT_OK
