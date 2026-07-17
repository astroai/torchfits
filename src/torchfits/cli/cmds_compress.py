"""``torchfits compress`` / ``decompress`` — tile-compressed FITS I/O."""

from __future__ import annotations

import argparse

import torchfits

from .common import EXIT_OK, IoError, UsageError


def add_compress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("compress", help="write tile-compressed FITS")
    parser.add_argument("input", help="input FITS path")
    parser.add_argument("output", help="output FITS path")
    parser.set_defaults(func=run_compress)


def add_decompress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("decompress", help="write uncompressed FITS")
    parser.add_argument("input", help="input FITS path")
    parser.add_argument("output", help="output FITS path")
    parser.set_defaults(func=run_decompress)


def _rewrite(input_path: str, output_path: str, *, compress: bool) -> int:
    try:
        with torchfits.open(input_path) as hdul:
            torchfits.write(output_path, hdul, overwrite=True, compress=compress)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc
    return EXIT_OK


def run_compress(args: argparse.Namespace) -> int:
    return _rewrite(args.input, args.output, compress=True)


def run_decompress(args: argparse.Namespace) -> int:
    return _rewrite(args.input, args.output, compress=False)
