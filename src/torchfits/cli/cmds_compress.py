"""``torchfits compress`` / ``decompress`` — tile-compressed FITS I/O."""

from __future__ import annotations

import argparse

import torchfits

from .common import EXIT_OK, IoError, UsageError, add_out_arg, resolve_out_path


def add_compress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("compress", help="write tile-compressed FITS")
    parser.add_argument("input", help="input FITS path")
    add_out_arg(parser, help="output FITS path")
    parser.set_defaults(func=run_compress)


def add_decompress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("decompress", help="write uncompressed FITS")
    parser.add_argument("input", help="input FITS path")
    add_out_arg(parser, help="output FITS path")
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
    return _rewrite(args.input, resolve_out_path(args), compress=True)


def run_decompress(args: argparse.Namespace) -> int:
    return _rewrite(args.input, resolve_out_path(args), compress=False)
