"""``torchfits convert`` — table export and RGB→PNG."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torchfits

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_hdu_arg,
    reject_same_path,
    staged_output,
)

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
    add_hdu_arg(
        parser,
        type=int,
        default=None,
        help="table HDU (table only; default: 1)",
    )
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
        help=(
            "comma-separated HDU indices for png "
            "(auto: 1–7 per input file; default: HDUs 0,1,2 of one file "
            "when present, else HDU 0)"
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=("auto", "lupton"),
        default=None,
        help="png mapping (png only): auto (default, blue→red) or lupton (reddest first)",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="auto rgb sky+noise display value (default: 0.15)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=None,
        help="auto rgb chroma boost (default: 2; 1 = photometric)",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="read AB zeropoints from MAGZP/PHOTZP/FLUXMAG0/ZP (or use --zeropoints)",
    )
    parser.add_argument(
        "--zeropoints",
        default=None,
        help="comma-separated AB mag of 1 count per band (implies calibrated)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Lupton Q (--recipe lupton; default: 8)",
    )
    parser.add_argument(
        "--stretch",
        type=float,
        default=None,
        help="Lupton stretch (--recipe lupton; default: 0.5)",
    )
    parser.add_argument(
        "--origin",
        choices=("lower", "upper"),
        default="lower",
        help=(
            "png row order: lower (default; FITS pixel (1,1) at bottom-left, "
            "north up for standard WCS, as in ds9) or upper (array row 0 on top)"
        ),
    )
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


def _parse_hdu_list(raw: str, *, flag: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise UsageError(f"{flag} must be comma-separated integers") from exc
    if not indices:
        raise UsageError(f"{flag} requires at least one integer")
    return indices


def _lupton_band_indices(raw: str | None, num_inputs: int) -> list[int]:
    if raw is None:
        if num_inputs == 1:
            return [0, 1, 2]
        if num_inputs == 3:
            return [0, 0, 0]
        raise UsageError(
            f"--recipe lupton got {num_inputs} input path(s); need one FITS file "
            "(optionally with --bands 0,1,2) or exactly three band files"
        )
    indices = _parse_hdu_list(raw, flag="--bands")
    if len(indices) != 3:
        raise UsageError("--recipe lupton: --bands requires exactly three HDU indices")
    if num_inputs not in (1, 3):
        raise UsageError("--recipe lupton accepts one FITS or three band FITS files")
    return indices


def _auto_band_indices(
    raw: str | None, num_inputs: int, inputs: list[str] | None = None
) -> list[int]:
    if not 1 <= num_inputs <= 7:
        raise UsageError(f"png convert accepts 1–7 FITS files, got {num_inputs}")
    if raw is None:
        if num_inputs == 1:
            # Keep the historical convenience: a single multi-HDU file
            # defaults to RGB from HDUs 0,1,2; grey otherwise. A failure to
            # read the source must not silently render a wrong-band PNG.
            first = (inputs or [""])[0]
            n_hdus = torchfits.read_num_hdus(first) if first else 1
            return [0, 1, 2][: min(3, n_hdus)] or [0]
        return [0] * num_inputs
    indices = _parse_hdu_list(raw, flag="--bands")
    if not 1 <= len(indices) <= 7:
        raise UsageError("--bands requires 1–7 HDU indices")
    if num_inputs == 1:
        return indices
    if len(indices) != num_inputs:
        raise UsageError(
            f"--bands length ({len(indices)}) must match input files ({num_inputs})"
        )
    return indices


def _parse_zeropoints(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise UsageError("--zeropoints must be comma-separated numbers") from exc
    if not values:
        raise UsageError("--zeropoints requires at least one value")
    return values


def _ab_zeropoint_from_header(path: str, hdu: int) -> float:
    header = torchfits.read_header(path, hdu)
    for key in ("MAGZP", "PHOTZP", "ZP"):
        value = header[key] if key in header else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    fluxmag0 = header["FLUXMAG0"] if "FLUXMAG0" in header else None
    if fluxmag0 is not None:
        try:
            flux = float(fluxmag0)
        except (TypeError, ValueError) as exc:
            raise UsageError(f"{path}: FLUXMAG0 is not a number") from exc
        if flux <= 0.0:
            raise UsageError(f"{path}: FLUXMAG0 must be positive")
        return 2.5 * math.log10(flux)
    raise UsageError(
        f"{path} HDU {hdu}: --calibrated needs MAGZP, PHOTZP, FLUXMAG0, or ZP "
        "(or pass --zeropoints)"
    )


def _select_rows(values: Any, rows: Any) -> Any:
    """Rows ``rows`` (an int index array) of one rewrite column."""
    import numpy as np

    if isinstance(values, list):
        return [values[int(i)] for i in rows]
    return np.ascontiguousarray(np.asarray(values)[rows])


def _where_row_indices(path: str, hdu: int, where: str) -> Any:
    """Indices of the rows ``where`` selects (same predicate as table.read)."""
    import numpy as np

    from torchfits import table as tf_table
    from torchfits._table._read_where import _where_mask_for_table
    from torchfits._where import parse_where_expression, where_columns_from_ast

    ast = parse_where_expression(where)
    referenced = sorted(set(where_columns_from_ast(ast)))
    predicate_table = tf_table.read(path, hdu=hdu, columns=referenced or None)
    mask = _where_mask_for_table(predicate_table, where, parsed_ast=ast)
    flags: Any = mask
    return np.flatnonzero(np.asarray(flags.to_numpy(zero_copy_only=False), dtype=bool))


# Source bytes read per chunk by the streamed FITS -> FITS convert.
_RAW_CHUNK_BYTES = 8 << 20


def _convert_table_to_fits(
    path: str, output: str, hdu: int, columns: list[str] | None, where: str | None
) -> None:
    """FITS table -> FITS table, keeping the column layout and metadata.

    Columns are copied in their stored form (TFORM, TSCAL/TZERO, TNULL
    sentinels, TDIM, unsigned conventions) with their TUNIT and other
    per-column cards renumbered for a ``--columns`` projection; EXTNAME and
    the other non-structural keywords of the source HDU are carried over.

    Fixed-width columns stream through ``table.writer``: the stored row bytes
    are copied chunk by chunk (projected and filtered), so memory is bounded
    by the chunk size and every cell is byte-identical. A selection with a
    variable-length column (whose cells live in the heap) is rewritten from
    the materialized table instead.
    """
    from torchfits._table.write import _resolve_table_hdu_index_and_columns

    target_hdu, header_map, all_columns, _ = _resolve_table_hdu_index_and_columns(
        path, hdu
    )
    if columns is None:
        selected = list(all_columns)
    else:
        missing = [name for name in columns if name not in all_columns]
        if missing:
            raise UsageError(f"--columns: unknown column(s): {', '.join(missing)}")
        selected = list(dict.fromkeys(columns))
    if _raw_copy_spans(header_map, selected) is None:
        _convert_table_to_fits_materialized(
            path, output, target_hdu, header_map, selected, where
        )
        return
    _convert_table_to_fits_streamed(
        path, output, target_hdu, header_map, selected, where
    )


def _raw_copy_spans(
    header_map: dict[str, Any], selected: list[str]
) -> dict[str, tuple[int, int]] | None:
    """Stored-row byte spans of ``selected``, or None when they cannot be
    copied as bytes (variable-length or unknown columns, compressed tables)."""
    from torchfits import fits_schema
    from torchfits._table.write import _table_column_spans

    if header_map.get("ZTABLE") or header_map.get("ZIMAGE"):
        return None
    wanted = set(selected)
    for col in fits_schema.iter_table_columns(header_map):
        if col.name in wanted and (col.tform_info.vla or not col.tform_info.code):
            return None
    spans = _table_column_spans(header_map)
    if spans is None or any(name not in spans for name in selected):
        return None
    return {name: spans[name] for name in selected}


def _stored_column_schema(
    header_map: dict[str, Any], selected: list[str]
) -> dict[str, dict[str, Any]]:
    """Column layout cards of ``selected`` for a writer fed stored bytes:
    TNULL stays the raw stored sentinel (no unsigned-domain conversion)."""
    from torchfits._table.write import (
        _column_name_index_map,
        _extract_table_schema_from_header,
    )

    schema = _extract_table_schema_from_header(header_map, selected)
    index = {name: idx for idx, name in _column_name_index_map(header_map).items()}
    for name in selected:
        raw = header_map.get(f"TNULL{index[name]}")
        if raw is not None:
            schema[name]["tnull"] = raw
    return schema


def _empty_stored_column(fmt: str) -> Any:
    """A zero-row column the native writer types by ``fmt`` alone."""
    import numpy as np

    from torchfits import fits_schema

    code = fits_schema.parse_tform(fmt).code
    if code == "A":
        return np.array([], dtype="S1")
    if code == "X":
        return np.zeros(0, dtype=bool)
    return np.zeros(0, dtype=np.float64)


def _chunk_where_mask(
    path: str, hdu: int, start: int, stop: int, where: str, ast: Any, refs: list[str]
) -> Any:
    """Rows ``start:stop`` that ``where`` keeps (same predicate as table.read)."""
    import numpy as np

    from torchfits import table as tf_table
    from torchfits._table._read_where import _where_mask_for_table

    try:
        predicate_table = tf_table.read(
            path, hdu=hdu, columns=refs or None, row_slice=(start, stop)
        )
        mask = _where_mask_for_table(predicate_table, where, parsed_ast=ast)
    except (KeyError, ValueError) as exc:
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise UsageError(f"--where: {detail}") from exc
    flags: Any = mask
    return np.asarray(flags.to_numpy(zero_copy_only=False), dtype=bool)


def _convert_table_to_fits_streamed(
    path: str,
    output: str,
    hdu: int,
    header_map: dict[str, Any],
    selected: list[str],
    where: str | None,
) -> None:
    import numpy as np

    from torchfits._io_engine._hdu_splice import hdu_byte_spans, open_fits_bytes
    from torchfits._table.write import (
        _resolve_table_hdu_index_and_columns,
        _rewrite_table_header,
        _table_column_spans,
    )
    from torchfits._table.write_stream import TableWriter
    from torchfits._where import parse_where_expression, where_columns_from_ast

    src_spans = _raw_copy_spans(header_map, selected) or {}
    schema = _stored_column_schema(header_map, selected)
    header = _rewrite_table_header(path, hdu, header_map, selected)
    ascii_table = str(header_map.get("XTENSION", "")).strip().upper() == "TABLE"
    nrows = int(header_map.get("NAXIS2", 0) or 0)
    width = int(header_map.get("NAXIS1", 0) or 0)
    ast = parse_where_expression(where) if where else None
    refs = sorted(set(where_columns_from_ast(ast))) if ast is not None else []

    with TableWriter(
        output,
        schema=schema,
        overwrite=True,
        table_type="ascii" if ascii_table else "binary",
    ) as writer:
        empty = {
            name: _empty_stored_column(schema[name]["format"]) for name in selected
        }
        writer._begin(empty, replay_header=header)
        out_map = _resolve_table_hdu_index_and_columns(writer._temp, 1)[1]
        dst_spans = _table_column_spans(out_map) or {}
        out_width = int(out_map.get("NAXIS1", 0) or 0)
        if any(src_spans[n][1] != dst_spans.get(n, (0, -1))[1] for n in selected):
            raise IoError(f"{path}: cannot map the selected columns onto the output")
        identity = out_width == width and all(
            src_spans[n][0] == dst_spans[n][0] for n in selected
        )
        if nrows == 0 or width == 0:
            return
        chunk = max(1, _RAW_CHUNK_BYTES // width)
        data_start = hdu_byte_spans(path)[hdu][1]
        with open_fits_bytes(path) as fh:
            fh.seek(data_start)
            for start in range(0, nrows, chunk):
                count = min(chunk, nrows - start)
                raw = fh.read(count * width)
                if len(raw) != count * width:
                    raise IoError(f"{path}: table HDU {hdu} data is truncated")
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(count, width)
                if ast is not None:
                    rows = rows[
                        _chunk_where_mask(
                            path, hdu, start, start + count, where or "", ast, refs
                        )
                    ]
                if not len(rows):
                    continue
                if identity:
                    out = np.ascontiguousarray(rows)
                else:
                    out = np.full(
                        (len(rows), out_width), 0x20 if ascii_table else 0, np.uint8
                    )
                    for name in selected:
                        s_off, nbytes = src_spans[name]
                        d_off = dst_spans[name][0]
                        out[:, d_off : d_off + nbytes] = rows[:, s_off : s_off + nbytes]
                writer._write_raw_rows(out)


def _convert_table_to_fits_materialized(
    path: str,
    output: str,
    target_hdu: int,
    header_map: dict[str, Any],
    selected: list[str],
    where: str | None,
) -> None:
    """Whole-table rewrite for selections the byte copy cannot stream."""
    import numpy as np

    from torchfits.io import _write_header_cards_if_supported
    from torchfits._table._mutation_coerce import _read_table_for_rewrite
    from torchfits._table.write import (
        _extract_table_schema_from_header,
        _rewrite_table_header,
        _scaled_columns_as_stored,
        write,
    )

    rows = None
    if where:
        try:
            rows = _where_row_indices(path, target_hdu, where)
        except (KeyError, ValueError) as exc:
            detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            raise UsageError(f"--where: {detail}") from exc

    data = _read_table_for_rewrite(path, target_hdu, selected)
    stored = _scaled_columns_as_stored(path, target_hdu, header_map)
    data.update({name: stored[name] for name in selected if name in stored})
    if rows is not None:
        data = {name: _select_rows(values, rows) for name, values in data.items()}
    schema = _extract_table_schema_from_header(header_map, selected)
    for name in selected:
        if isinstance(data[name], list) and not data[name]:
            fmt = str(schema.get(name, {}).get("format", "")).upper()
            if "A" in fmt and "P" not in fmt and "Q" not in fmt:
                # The native writer cannot type an empty list of strings.
                data[name] = np.array([], dtype="S1")
    header = _rewrite_table_header(path, target_hdu, header_map, selected)
    table_type = (
        "ascii"
        if str(header_map.get("XTENSION", "")).strip().upper() == "TABLE"
        else "binary"
    )
    with staged_output(output) as temp_path:
        write(
            temp_path,
            {name: data[name] for name in selected},
            schema=schema,
            overwrite=True,
            table_type=table_type,
        )
        # Card by card, as the in-place column rewrites do: keeps every
        # HISTORY/COMMENT line and typed values.
        _write_header_cards_if_supported(temp_path, 1, header)


def _convert_table(args: argparse.Namespace, fmt: str) -> int:
    from torchfits import table as tf_table

    if len(args.inputs) != 1:
        raise UsageError("table convert accepts one input FITS file")
    path = args.inputs[0]
    for flag, given in (
        ("--bands", args.bands is not None),
        ("--recipe", args.recipe is not None),
        ("--brightness", args.brightness is not None),
        ("--saturation", args.saturation is not None),
        ("--calibrated", args.calibrated),
        ("--zeropoints", args.zeropoints is not None),
        ("--q", args.q is not None),
        ("--stretch", args.stretch is not None),
    ):
        if given:
            raise UsageError(f"{flag} applies only to png convert")
    hdu = 1 if args.hdu is None else args.hdu
    columns = _parse_columns(args.columns)
    where = args.where
    if columns or where:
        # A bad --columns / --where is a usage error (exit 2), raised before
        # the output is touched; an unreadable file or HDU stays exit 3.
        _validate_selection(path, hdu, columns, where)
    if fmt == "fits":
        _convert_table_to_fits(path, args.output, hdu, columns, where)
        return EXIT_OK
    selection: dict[str, Any] = {"hdu": hdu, "columns": columns, "where": where}
    try:
        # Streamed batch by batch: --columns/--where never materialize the
        # whole table.
        if fmt == "parquet":
            tf_table.write_parquet(args.output, path, stream=True, **selection)
        elif fmt == "csv":
            tf_table.write_csv(
                args.output, path, stream=True, delimiter=",", **selection
            )
        elif fmt == "tsv":
            tf_table.write_csv(
                args.output, path, stream=True, delimiter="\t", **selection
            )
        elif fmt == "arrow":
            tf_table.write_ipc(args.output, path, stream=True, **selection)
        else:
            raise UsageError(f"unsupported table format: {fmt}")
    except ValueError as exc:
        if not where:
            raise
        # Validated up front, so this is a predicate error only the data
        # could reveal; the atomic writers left the output untouched.
        raise UsageError(f"--where: {exc}") from exc
    return EXIT_OK


def _validate_selection(
    path: str, hdu: int, columns: list[str] | None, where: str | None
) -> None:
    """Check --columns / --where against the table schema without reading rows."""
    from torchfits import table as tf_table
    from torchfits._table._read_where import _where_mask_for_table
    from torchfits._where import parse_where_expression

    # Reading the schema first keeps an unreadable file or HDU an I/O error.
    arrow_schema = tf_table.schema(path, hdu=hdu)
    if columns:
        try:
            tf_table.schema(path, hdu=hdu, columns=columns)
        except KeyError as exc:
            raise UsageError(f"--columns: {exc.args[0] if exc.args else exc}") from exc
    if where:
        try:
            ast = parse_where_expression(where)
            # Evaluating on a zero-row table of the same schema surfaces
            # unknown columns and type mismatches as well as syntax errors.
            _where_mask_for_table(arrow_schema.empty_table(), where, parsed_ast=ast)
        except (KeyError, ValueError) as exc:
            detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            raise UsageError(f"--where: {detail}") from exc


def _read_band(path: str, hdu: int) -> object:
    """Read one PNG band, naming the file and HDU in every failure.

    CFITSIO's own text ("Could not move to HDU", "Could not open FITS file:
    <path>") is unattributable in a multi-band invocation: the caller cannot
    tell which of the inputs failed nor that the index came from ``--bands``.
    """
    try:
        return torchfits.read_tensor(path, hdu=hdu).detach().cpu()
    except (UsageError, IoError):
        raise
    except Exception as exc:
        raise IoError(f"{path}: HDU {hdu}: {exc}") from exc


def _load_png_bands(inputs: list[str], hdus: list[int]) -> list[Any]:
    if len(inputs) == 1:
        path = inputs[0]
        return [_read_band(path, index) for index in hdus]
    return [_read_band(path, hdu) for path, hdu in zip(inputs, hdus, strict=True)]


def _orient(image: Any, args: argparse.Namespace) -> Any:
    """FITS rows run bottom-up; PNG rows run top-down."""
    if getattr(args, "origin", "lower") == "lower":
        return image.flip(0)
    return image


def _convert_png(args: argparse.Namespace) -> int:
    from torchfits.transforms.rgb import (
        lupton_rgb,
        rgb as auto_rgb,
        write_rgb_image,
    )

    if args.hdu is not None:
        raise UsageError("--hdu applies only to table convert; use --bands for png")
    recipe = "auto" if args.recipe is None else args.recipe
    if args.where or args.columns:
        raise UsageError("--where / --columns apply only to table convert")
    if recipe == "lupton":
        for flag, given in (
            ("--zeropoints", args.zeropoints is not None),
            ("--calibrated", args.calibrated),
            ("--brightness", args.brightness is not None),
            ("--saturation", args.saturation is not None),
        ):
            if given:
                raise UsageError(f"{flag} applies only to --recipe auto")
        hdus = _lupton_band_indices(args.bands, len(args.inputs))
        bands = _load_png_bands(args.inputs, hdus)
        image = lupton_rgb(
            *bands,
            Q=8.0 if args.q is None else args.q,
            stretch=0.5 if args.stretch is None else args.stretch,
        )
        write_rgb_image(args.output, _orient(image, args))
        return EXIT_OK

    if args.q is not None or args.stretch is not None:
        raise UsageError("--q / --stretch apply only to --recipe lupton")
    hdus = _auto_band_indices(args.bands, len(args.inputs), args.inputs)
    bands = _load_png_bands(args.inputs, hdus)
    rgb_args: tuple[Any, ...]
    if len(args.inputs) == 1 and len(bands) == 1:
        rgb_args = (bands[0],)
    else:
        rgb_args = tuple(bands)
    zps = _parse_zeropoints(args.zeropoints)
    if args.calibrated and zps is None:
        if len(args.inputs) == 1:
            zps = [_ab_zeropoint_from_header(args.inputs[0], hdu) for hdu in hdus]
        else:
            zps = [
                _ab_zeropoint_from_header(path, hdu)
                for path, hdu in zip(args.inputs, hdus, strict=True)
            ]
    image = auto_rgb(
        *rgb_args,
        brightness=0.15 if args.brightness is None else args.brightness,
        saturation=2.0 if args.saturation is None else args.saturation,
        zeropoints=zps,
    )
    write_rgb_image(args.output, _orient(image, args))
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
        for src in args.inputs:
            reject_same_path(src, args.output)
        fmt = _infer_format(args.output, args.to)
        if fmt in _TABLE_FORMATS:
            return _convert_table(args, fmt)
        return _convert_png(args)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
