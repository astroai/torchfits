"""``torchfits copy`` — FITS→FITS copy preserving MEF structure."""

from __future__ import annotations

import argparse

import torchfits

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    resolve_batch_io_pairs,
    resolve_file_jobs,
    run_file_jobs,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "copy",
        help="copy FITS file(s) preserving HDUs",
        description=(
            "MEF-preserving FITS→FITS copy. "
            "Multiple inputs need --out-dir; -J fans out across files."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="INPUT [OUTPUT], or multiple INPUTs with --out-dir",
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs when copying multiple inputs",
    )
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _copy_one(pair: tuple[str, str]) -> None:
    input_path, output_path = pair
    try:
        with torchfits.open(input_path) as hdul:
            hdul.write(output_path, overwrite=True)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc


def run(args: argparse.Namespace) -> int:
    pairs = resolve_batch_io_pairs(
        [str(p) for p in args.paths],
        out=args.out,
        out_dir=args.out_dir,
    )
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(pairs))
    run_file_jobs(pairs, _copy_one, file_jobs)
    return EXIT_OK
