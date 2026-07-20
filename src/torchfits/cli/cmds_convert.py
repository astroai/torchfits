"""``torchfits convert`` — table export and Lupton RGB→PNG."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torchfits
from torchfits import table as tf_table

from .common import EXIT_OK, IoError, UsageError, add_hdu_arg
from torchfits.transforms.rgb import lupton_rgb, write_rgb_image

_TABLE_FORMATS = ("parquet", "csv", "tsv", "arrow", "fits")
_ALL_FORMATS = (*_TABLE_FORMATS, "png")
_EXT_TO_FORMAT = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".arrow": "arrow",
    ".feather": "arrow",
    ".ipc": "arrow",
    ".fits": "fits",
    ".fit": "fits",
    ".png": "png",
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="convert FITS tables (parquet/csv/tsv/arrow/fits) or RGB→PNG",
    )
    # nargs='+' would swallow a trailing positional output; resolve in run().
    parser.add_argument(
        "paths",
        nargs="+",
        help="input FITS path(s); trailing path is output unless -o/--out",
    )
    parser.add_argument("-o", "--out", default=None, help="output path")
    parser.add_argument(
        "--to",
        choices=_ALL_FORMATS,
        default=None,
        help="output format (default: infer from output extension)",
    )
    add_hdu_arg(parser, type=int, default=1, help="table HDU (default: 1)")
    parser.add_argument(
        "-w",
        "--where",
        help="row filter expression (table convert; same syntax as table.read)",
    )
    parser.add_argument(
        "-c",
        "--columns",
        help="comma-separated column names to keep (table convert)",
    )
    parser.add_argument(
        "--bands",
        help="comma-separated HDU indices for png (default: 0,1,2 on one file)",
    )
    parser.add_argument("--q", type=float, default=8.0, help="Lupton Q parameter")
    parser.add_argument("--stretch", type=float, default=0.5, help="Lupton stretch")
    parser.set_defaults(func=run)


def _infer_format(output: str, to: str | None) -> str:
    if to is not None:
        return to
    suffix = Path(output).suffix.lower()
    fmt = _EXT_TO_FORMAT.get(suffix)
    if fmt is None:
        raise UsageError(
            "cannot infer convert format from output path; "
            "pass --to parquet|csv|tsv|arrow|fits|png"
        )
    return fmt


def _parse_columns(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    cols = [part.strip() for part in raw.split(",") if part.strip()]
    if not cols:
        raise UsageError("--columns requires at least one column name")
    return cols


def _band_indices(raw: str | None, num_inputs: int) -> list[int]:
    if raw is None:
        if num_inputs == 1:
            return [0, 1, 2]
        if num_inputs == 3:
            return [0, 0, 0]
        raise UsageError(
            f"png convert got {num_inputs} input path(s); need one FITS file "
            "(optionally with --bands 0,1,2) or exactly three band files"
        )
    try:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise UsageError("--bands must be comma-separated integers, e.g. 0,1,2")
    if len(indices) != 3:
        raise UsageError("--bands requires exactly three HDU indices")
    return indices


def _arrow_to_column_dict(table: Any) -> dict[str, Any]:
    """Materialize an Arrow table as a dict of NumPy columns for FITS write."""
    import numpy as np

    out: dict[str, Any] = {}
    for name in table.column_names:
        col = table[name]
        try:
            out[name] = col.to_numpy(zero_copy_only=False)
        except Exception:
            out[name] = np.asarray(col.to_pylist(), dtype=object)
    return out


def _convert_table(args: argparse.Namespace, fmt: str) -> int:
    if len(args.inputs) != 1:
        raise UsageError("table convert accepts one input FITS file")
    path = args.inputs[0]
    hdu = args.hdu
    columns = _parse_columns(args.columns)
    where = args.where
    if where or columns or fmt == "fits":
        # Filter / column-select / FITS out: materialize via table.read.
        arrow = tf_table.read(path, hdu=hdu, columns=columns, where=where)
        if fmt == "parquet":
            tf_table.write_parquet(args.output, arrow)
        elif fmt == "csv":
            tf_table.write_csv(args.output, arrow, delimiter=",")
        elif fmt == "tsv":
            tf_table.write_csv(args.output, arrow, delimiter="\t")
        elif fmt == "arrow":
            tf_table.write_ipc(args.output, arrow)
        elif fmt == "fits":
            tf_table.write(args.output, _arrow_to_column_dict(arrow), overwrite=True)
        else:
            raise UsageError(f"unsupported table format: {fmt}")
        return EXIT_OK

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
    if args.where or args.columns:
        raise UsageError("--where / --columns apply only to table convert")
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
        if args.out:
            args.inputs = list(args.paths)
            args.output = args.out
        else:
            if len(args.paths) < 2:
                raise UsageError(
                    "output path required (-o/--out or trailing positional)"
                )
            args.inputs = list(args.paths[:-1])
            args.output = args.paths[-1]
        fmt = _infer_format(args.output, args.to)
        if fmt in _TABLE_FORMATS:
            return _convert_table(args, fmt)
        return _convert_png(args)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
