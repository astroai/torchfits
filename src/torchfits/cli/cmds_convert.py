"""``torchfits convert`` — table export and Lupton RGB→PNG."""

from __future__ import annotations

import argparse

import torchfits
from torchfits import table as tf_table

from .common import EXIT_OK, IoError, UsageError
from .rgb import lupton_rgb, write_rgb_image

_TABLE_FORMATS = ("parquet", "csv", "tsv", "arrow")
_ALL_FORMATS = (*_TABLE_FORMATS, "png")


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="convert FITS tables (parquet/csv/tsv/arrow) or RGB→PNG",
    )
    parser.add_argument("inputs", nargs="+", help="input FITS path(s)")
    parser.add_argument("output", help="output path")
    parser.add_argument(
        "--to",
        required=True,
        choices=_ALL_FORMATS,
        help="parquet|csv|tsv|arrow (tables) or png (Lupton RGB)",
    )
    parser.add_argument("--hdu", type=int, default=1, help="table HDU (default: 1)")
    parser.add_argument(
        "--bands",
        help="comma-separated HDU indices for png (default: 0,1,2 on one file)",
    )
    parser.add_argument("--q", type=float, default=8.0, help="Lupton Q parameter")
    parser.add_argument("--stretch", type=float, default=0.5, help="Lupton stretch")
    parser.set_defaults(func=run)


def _band_indices(raw: str | None, num_inputs: int) -> list[int]:
    if raw is None:
        if num_inputs == 1:
            return [0, 1, 2]
        if num_inputs == 3:
            return [0, 0, 0]
        raise UsageError("png convert needs one file plus --bands or three band files")
    try:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise UsageError("--bands must be comma-separated integers, e.g. 0,1,2")
    if len(indices) != 3:
        raise UsageError("--bands requires exactly three HDU indices")
    return indices


def _convert_table(args: argparse.Namespace) -> int:
    if len(args.inputs) != 1:
        raise UsageError("table convert accepts one input FITS file")
    path = args.inputs[0]
    hdu = args.hdu
    fmt = args.to
    if fmt == "parquet":
        tf_table.write_parquet(args.output, path, hdu=hdu, stream=True)
    elif fmt == "csv":
        tf_table.write_csv(args.output, path, hdu=hdu, stream=True, delimiter=",")
    elif fmt == "tsv":
        tf_table.write_csv(args.output, path, hdu=hdu, stream=True, delimiter="\t")
    elif fmt == "arrow":
        tf_table.write_ipc(args.output, path, hdu=hdu, stream=True)
    else:
        raise UsageError(f"unsupported table format: {fmt}")
    return EXIT_OK


def _read_band(path: str, hdu: int) -> object:
    return torchfits.read_tensor(path, hdu=hdu).detach().cpu()


def _convert_png(args: argparse.Namespace) -> int:
    band_indices = _band_indices(args.bands, len(args.inputs))
    if len(args.inputs) == 1:
        path = args.inputs[0]
        bands = [_read_band(path, index) for index in band_indices]
    elif len(args.inputs) == 3:
        bands = [
            _read_band(path, band_indices[idx]) for idx, path in enumerate(args.inputs)
        ]
    else:
        raise UsageError("png convert accepts one FITS or three band FITS files")
    rgb = lupton_rgb(*bands, Q=args.q, stretch=args.stretch)
    write_rgb_image(args.output, rgb)
    return EXIT_OK


def run(args: argparse.Namespace) -> int:
    try:
        if args.to in _TABLE_FORMATS:
            return _convert_table(args)
        return _convert_png(args)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
