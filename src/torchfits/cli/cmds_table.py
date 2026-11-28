"""``torchfits table`` — table schema and preview."""

from __future__ import annotations

import argparse
from typing import Any

from torchfits import table as tf_table

import json

import torchfits
from torchfits.fits_schema import complex_column_names, iter_table_columns

from .common import (
    EXIT_OK,
    CliError,
    IoError,
    _json_safe,
    UsageError,
    add_emit_format_args,
    add_hdu_arg,
    emit_records,
    header_extname,
    hdu_type_name,
    json_default,
    resolve_emit_format,
    resolve_paths,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("table", help="table schema and preview rows")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser, help="comma-separated table HDU indices (default: all)")
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=5,
        dest="preview",
        help="preview row count (default: 5; 0 = schema only)",
    )
    add_emit_format_args(parser)
    parser.set_defaults(func=run)


# Preview strings decode like ``table.read`` (ASCII) but never abort the
# listing: undecodable bytes become U+FFFD replacement characters.
_PREVIEW_DECODE_ERRORS = "replace"


def _complex_type_name(info: Any) -> str:
    base = "complex64" if info.code == "C" else "complex128"
    if info.vla:
        return f"list<item: {base}>"
    if info.repeat != 1:
        return f"fixed_size_list<item: {base}>[{info.repeat}]"
    return base


def _schema_fields(
    path: str, hdu: int, header: Any, complex_names: set[str]
) -> list[dict[str, str]]:
    """Column name/type list in file order.

    Arrow has no complex type, so complex columns (TFORM ``C``/``M``) are
    described from the header and every other column from the Arrow schema.
    """
    columns = list(iter_table_columns(header))
    arrow_names = [col.name for col in columns if col.name not in complex_names]
    arrow_types: dict[str, str] = {}
    if arrow_names:
        schema = tf_table.schema(
            path,
            hdu=hdu,
            columns=arrow_names if complex_names else None,
            decode_errors=_PREVIEW_DECODE_ERRORS,
        )
        arrow_types = {field.name: str(field.type) for field in schema}
    fields = []
    for col in columns:
        if col.name in complex_names:
            fields.append(
                {"name": col.name, "type": _complex_type_name(col.tform_info)}
            )
        elif col.name in arrow_types:
            fields.append({"name": col.name, "type": arrow_types[col.name]})
    return fields


def _preview_rows(
    path: str, hdu: int, limit: int, header: Any, complex_names: set[str]
) -> list[dict[str, Any]]:
    """The first ``limit`` rows as JSON-ready dicts (complex via read_torch)."""
    if limit <= 0:
        return []
    names = [col.name for col in iter_table_columns(header)]
    columns: dict[str, list[Any]] = {}
    num_rows = 0
    arrow_names = [name for name in names if name not in complex_names]
    if arrow_names or not complex_names:
        arrow_table = tf_table.read(
            path,
            hdu=hdu,
            columns=arrow_names if complex_names else None,
            row_slice=slice(0, limit),
            decode_errors=_PREVIEW_DECODE_ERRORS,
        )
        num_rows = arrow_table.num_rows
        for name in arrow_table.column_names:
            columns[name] = arrow_table.column(name).to_pylist()
    if complex_names:
        total = int(header.get("NAXIS2", 0) or 0)
        count = min(limit, total)
        if count > 0:
            tensors = tf_table.read_torch(
                path,
                hdu=hdu,
                columns=[name for name in names if name in complex_names],
                start_row=1,
                num_rows=count,
            )
            for name, values in tensors.items():
                columns[name] = (
                    values.tolist()
                    if hasattr(values, "tolist")
                    else [v.tolist() for v in values]
                )
        num_rows = max(num_rows, count)
    order = [name for name in names if name in columns] or list(columns)
    return [{name: columns[name][row] for name in order} for row in range(num_rows)]


def run(args: argparse.Namespace) -> int:
    if args.preview < 0:
        # Every other numeric flag (-j, -J) rejects negatives; -n silently
        # rendering zero rows would read as "this table has no rows".
        raise UsageError(f"--rows must be >= 0 (0 = schema only), got {args.preview}")
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            with torchfits.open(path) as hdul:
                indices = selected_hdu_indices(len(hdul), args.hdu, path=path)
                for index in indices:
                    hdu = hdul[index]
                    header = hdu.header
                    if hdu_type_name(header, hdu) != "TABLE":
                        continue
                    complex_names = complex_column_names(header)
                    record: dict[str, Any] = {
                        "file": path,
                        "hdu": index,
                        "name": header_extname(header, index),
                        "schema": _schema_fields(path, index, header, complex_names),
                        "nrows": int(header.get("NAXIS2", 0)),
                    }
                    record["preview"] = _preview_rows(
                        path, index, args.preview, header, complex_names
                    )
                    records.append(record)
        except CliError:
            raise
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
    fmt = resolve_emit_format(args)
    if fmt != "text":
        emit_records(records, format=fmt)
        return EXIT_OK
    for record in records:
        print(
            f"{record['file']}:{record['hdu']} {record['name']} nrows={record['nrows']}"
        )
        for field in record["schema"]:
            print(f"  {field['name']}: {field['type']}")
        if record["preview"]:
            print("preview:")
            # Same JSON contract as -f json: non-finite floats serialize as
            # null so the preview block is valid JSON (no bare NaN/Infinity).
            print(
                json.dumps(
                    _json_safe(record["preview"]),
                    default=json_default,
                    indent=2,
                    allow_nan=False,
                )
            )
    return EXIT_OK
