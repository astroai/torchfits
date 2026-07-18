"""``torchfits copy`` — FITS→FITS copy preserving MEF structure."""

from __future__ import annotations

import argparse

import torchfits

from .common import EXIT_OK, IoError, UsageError, add_out_arg, resolve_out_path


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("copy", help="copy FITS file(s) preserving HDUs")
    parser.add_argument("input", help="input FITS path")
    add_out_arg(parser, help="output FITS path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    output = resolve_out_path(args)
    try:
        with torchfits.open(args.input) as hdul:
            hdul.write(output, overwrite=True)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{args.input}: {exc}") from exc
    return EXIT_OK
