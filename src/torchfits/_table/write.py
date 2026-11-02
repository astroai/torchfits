"""Table write path: CFITSIO-backed table writes, header management, schema rewrite."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from .._io_engine.caches import invalidate_path_caches as _invalidate_path_caches

# Internal write/mutation helpers re-exported on torchfits.io (not a public API).
# Crossing _table → io → _io_engine is intentional package-internal wiring.
from ..io import _normalize_cpp_table_data, _write_header_cards_if_supported

_log = logging.getLogger(__name__)


def write(
    path: str,
    data: dict[str, Any],
    *,
    schema: dict[str, dict[str, Any]] | None = None,
    header: dict[str, Any] | None = None,
    overwrite: bool = False,
    extname: str | None = None,
    table_type: str = "binary",
    quantize: Any = None,
) -> None:
    """Write a columnar dict as a FITS table.

    ``quantize=\"robust\"`` packs all float columns to ``TFORM=I`` with
    ``TSCAL``/``TZERO``. Pass ``{\"col\": \"robust\"}`` (or per-column option
    dicts) to select columns. Default keeps native float ``TFORM``.  When
    ``quantize`` is set but no column qualifies (e.g. an integer-only table),
    a ``QuantizeError`` is raised rather than silently writing unpacked.
    """
    from .._io_engine._write_helpers import _normalize_table_input
    from .._io_engine.paths import guard_write_path

    guard_write_path(path)
    if str(path).lower().endswith(".bz2"):
        # CFITSIO would create a plain, uncompressed FITS under this name.
        raise ValueError(
            "Writing bzip2-wrapped FITS files ('.bz2') is not supported; "
            "use compress='BZIP2_1' for tile compression inside the file."
        )
    data = _normalize_table_input(data)
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "data must be a non-empty dictionary or table/dataframe object"
        )
    table_kind = str(table_type).lower().strip()
    if table_kind not in {"binary", "ascii"}:
        raise ValueError("table_type must be 'binary' or 'ascii'")
    if schema is not None and not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary when provided")
    if quantize is not None and table_kind == "ascii":
        raise ValueError("quantize= is only supported for binary tables")
    hdr: dict[str, Any] | None = None
    if extname is not None:
        hdr = dict(header or {})
        hdr["EXTNAME"] = str(extname)
    else:
        hdr = header
    import torchfits
    from .._io_engine._write_helpers import QuantizeError
    from .._io_engine._write_helpers import (
        _prepare_quantized_table_data_for_write,
        _prepare_unsigned_table_data_for_write,
    )

    data, schema, unsigned_converted = _prepare_unsigned_table_data_for_write(
        data, schema, ascii_table=table_kind == "ascii"
    )
    data, schema, quantized = _prepare_quantized_table_data_for_write(
        data, quantize, schema
    )
    if quantize is not None and quantize is not False and not quantized:
        raise QuantizeError(
            f"quantize={quantize!r} requested but no table column qualifies for "
            "packing (only floating-point columns can be quantized)"
        )

    if schema or unsigned_converted or quantized or table_kind == "ascii":
        import torchfits._C as cpp
        from .._io_engine.write_api import atomic_write_target

        _invalidate_path_caches(path)
        data = _normalize_cpp_table_data(data)
        # This branch talks to cpp.write_fits_table directly, so it never passes
        # through write()'s temp-file-and-rename wrapper -- and the C++ writer
        # reaches fits_create_file("!path"), which unlinks the target before the
        # payload is validated. A rejected overwrite therefore destroyed the
        # caller's file (R2-013). Same wrapper, same protection.
        with atomic_write_target(path, overwrite) as write_path:
            cpp.write_fits_table(
                write_path,
                data,
                hdr if hdr else {},
                # The temp path does not exist; asking CFITSIO to overwrite it
                # would be meaningless, and False fails loudly if one ever does.
                False,
                schema if schema else None,
                table_kind,
            )
            if hdr:
                _write_header_cards_if_supported(write_path, 1, hdr)
        _invalidate_path_caches(path)
        return

    torchfits.write(
        path,
        data,
        header=hdr if hdr else None,
        overwrite=overwrite,
        quantize=quantize,
    )


def _header_cards_to_mapping(header_cards: Any) -> dict[str, Any]:
    if isinstance(header_cards, dict):
        return {str(k): v for k, v in header_cards.items()}
    out: dict[str, Any] = {}
    if isinstance(header_cards, (list, tuple)):
        for card in header_cards:
            if hasattr(card, "key") and hasattr(card, "value"):
                out[str(card.key)] = card.value
                continue
            if not isinstance(card, (list, tuple)) or len(card) < 2:
                continue
            out[str(card[0])] = card[1]
    return out


def _column_tform_map(header_map: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    name_by_idx = _column_name_index_map(header_map)
    tform_by_idx: dict[int, str] = {}
    for key, value in header_map.items():
        key_u = str(key).upper()
        if key_u.startswith("TFORM"):
            suffix = key_u[5:]
            if suffix.isdigit():
                tform_by_idx[int(suffix)] = str(value)

    for idx, name in name_by_idx.items():
        out[name] = tform_by_idx.get(idx, "")
    return out


def _column_name_index_map(header_map: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}

    try:
        tfields = int(header_map.get("TFIELDS", 0))
    except (ValueError, TypeError):
        tfields = 0

    if tfields > 0:
        for i in range(1, tfields + 1):
            val = header_map.get(f"TTYPE{i}")
            if val is not None:
                out[i] = str(val)
        return out

    for key, value in header_map.items():
        key_u = str(key).upper()
        if not key_u.startswith("TTYPE"):
            continue
        suffix = key_u[5:]
        if suffix.isdigit():
            out[int(suffix)] = str(value)
    return out


_UNSIGNED_OFFSETS = {"int8": -128, "uint16": 32768, "uint32": 2**31, "uint64": 2**63}


def _extract_table_schema_from_header(
    header_map: dict[str, Any], columns: list[str]
) -> dict[str, dict[str, Any]]:
    from ..fits_schema import unsigned_column_dtype_names_from_header

    name_by_idx = _column_name_index_map(header_map)
    index_by_name = {name: idx for idx, name in name_by_idx.items()}
    unsigned = unsigned_column_dtype_names_from_header(header_map)
    schema: dict[str, dict[str, Any]] = {}
    for name in columns:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        meta: dict[str, Any] = {}
        tform = header_map.get(f"TFORM{idx}")
        if tform is not None:
            meta["format"] = str(tform)
        tunit = header_map.get(f"TUNIT{idx}")
        if tunit is not None:
            meta["unit"] = str(tunit)
        tdim = header_map.get(f"TDIM{idx}")
        if tdim is not None:
            meta["dim"] = str(tdim)
        tnull = header_map.get(f"TNULL{idx}")
        if tnull is not None:
            # Unsigned columns are read back as values (raw + TZERO); the
            # writer moves their TNULL by the same offset, so hand it over in
            # that domain too.
            offset = _UNSIGNED_OFFSETS.get(unsigned.get(name, ""), 0)
            meta["tnull"] = int(tnull) + offset if offset else tnull
        tscal = header_map.get(f"TSCAL{idx}")
        if tscal is not None:
            meta["bscale"] = tscal
        tzero = header_map.get(f"TZERO{idx}")
        if tzero is not None:
            meta["bzero"] = tzero
        schema[name] = meta
    return schema


# Structural FITS keywords that must not be copied into rewritten table headers.
_TABLE_STRUCTURAL_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "CHECKSUM",
        "DATASUM",
        "EXTEND",
        "THEAP",
    }
)
# Column cards the writer re-derives from the schema; TBCOL is recomputed for
# the new ASCII-table layout.
_TABLE_COLUMN_REDERIVED: frozenset[str] = frozenset(
    {"TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO", "TBCOL"}
)


def _rewrite_table_header(
    path: str, hdu: int, header_map: dict[str, Any], new_columns: list[str]
) -> Any:
    """Header cards to carry onto a table rewritten with ``new_columns``.

    Typed cards from ``read_header`` (numbers stay numbers, every HISTORY /
    COMMENT line survives), minus structural keywords and the column cards
    the writer re-derives; other column-indexed cards (TDISPn, TLMINn,
    TCTYPn, ...) are renumbered by column name.
    """
    import torchfits
    from ..fits_schema import remap_table_column_cards

    source = torchfits.read_header(path, hdu)
    cards = [
        card
        for card in source.cards
        if str(card.key).upper() not in _TABLE_STRUCTURAL_SKIP_KEYS
    ]
    return torchfits.Header(
        remap_table_column_cards(
            cards,
            _column_name_index_map(header_map),
            new_columns,
            drop=_TABLE_COLUMN_REDERIVED,
        )
    )


def _ordered_dict_for_columns(
    columns: list[str], data_by_name: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in columns:
        out[name] = data_by_name[name]
    return out


_BINTABLE_CELL_BYTES = {
    "L": 1,
    "B": 1,
    "I": 2,
    "J": 4,
    "K": 8,
    "A": 1,
    "E": 4,
    "D": 8,
    "C": 8,
    "M": 16,
    "P": 8,
    "Q": 16,
}


def _scaled_columns_as_stored(
    path: str, hdu: int, header_map: dict[str, Any]
) -> dict[str, Any]:
    """Stored (pre-TSCAL/TZERO) cells of every scaled numeric column.

    A rewrite hands its columns back to ``write``, whose schema
    ``bscale``/``bzero`` describe stored values; passing the decoded physical
    values scaled them a second time. Binary-table cells are read from the
    file as-is; ASCII-table cells are recovered by inverting the scaling.
    Unsigned-convention columns are left to the unsigned write path.
    """
    import numpy as np

    from .. import fits_schema

    unsigned = fits_schema.unsigned_column_dtype_names_from_header(header_map)
    ascii_table = str(header_map.get("XTENSION", "")).strip().upper() == "TABLE"
    columns = list(fits_schema.iter_table_columns(header_map))
    scaled = [
        col
        for col in columns
        if col.name not in unsigned
        and not col.tform_info.vla
        and col.tform_info.code in {"B", "I", "J", "K", "E", "D", "F"}
        and ((col.tscal or 1.0) != 1.0 or (col.tzero or 0.0) != 0.0)
    ]
    if not scaled:
        return {}

    out: dict[str, Any] = {}
    if ascii_table:
        import torchfits

        for col in scaled:
            phys = np.asarray(
                torchfits.table.read_torch(path, hdu=hdu, columns=[col.name])[
                    col.name
                ].numpy(),
                dtype=np.float64,
            )
            stored = (phys - (col.tzero or 0.0)) / (col.tscal or 1.0)
            if col.tform_info.code == "I":
                if col.tnull is not None:
                    stored = np.where(np.isnan(stored), float(col.tnull), stored)
                stored = np.rint(stored).astype(np.int64)
            out[col.name] = stored
        return out

    nrows = int(header_map.get("NAXIS2", 0) or 0)
    width = int(header_map.get("NAXIS1", 0) or 0)
    offsets: dict[str, int] = {}
    offset = 0
    for col in columns:
        offsets[col.name] = offset
        code = col.tform_info.code or ""
        repeat = col.tform_info.repeat
        if col.tform_info.vla:
            offset += _BINTABLE_CELL_BYTES[col.tform_info.vla_descriptor or "P"]
        elif code == "X":
            offset += (repeat + 7) // 8
        else:
            offset += repeat * _BINTABLE_CELL_BYTES.get(code, 0)
    from .._io_engine._hdu_splice import hdu_byte_spans, open_fits_bytes

    # Offsets are into the FITS byte stream, which for x.fits.gz is the
    # decompressed one: read through the same opener, not the raw file.
    data_start = hdu_byte_spans(path)[hdu][1]
    nbytes = nrows * width
    with open_fits_bytes(path) as fh:
        fh.seek(data_start)
        raw = fh.read(nbytes)
    if len(raw) != nbytes:
        raise OSError(f"{path}: table HDU {hdu} data is truncated")
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(nrows, width)
    big_endian = {"B": "u1", "I": ">i2", "J": ">i4", "K": ">i8", "E": ">f4", "D": ">f8"}
    for col in scaled:
        dtype = np.dtype(big_endian[col.tform_info.code or ""])
        repeat = col.tform_info.repeat
        start = offsets[col.name]
        cells = np.ascontiguousarray(rows[:, start : start + repeat * dtype.itemsize])
        values = cells.view(dtype).astype(dtype.newbyteorder("="))
        out[col.name] = values.reshape(nrows) if repeat == 1 else values
    return out


def _rewrite_table_hdu_with_schema(
    path: str,
    target_hdu: int,
    data: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    header: dict[str, Any],
    table_type: str,
    *,
    restamp_checksums: bool = False,
) -> None:
    from .._io_engine._hdu_splice import Piece, hdu_byte_spans, publish_spliced

    # Stage the single-HDU file in a temp path so building it never races with
    # CFITSIO's internal file-locking on the real file when another handle
    # (e.g. torchfits.open()) is open on the same path. The new HDU's bytes are
    # then spliced in place of the old ones: decoding the table back into a
    # TableHDU for replace_hdu stored scaled columns as float and dropped
    # their TSCAL/TZERO, and re-encoded every other HDU of the file.
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        write(
            tmp_path,
            data=data,
            schema=schema,
            overwrite=True,
            table_type=table_type,
        )
        # Replayed card by card: the mapping write path re-emits HISTORY and
        # COMMENT as valued cards and collapses repeated lines.
        _write_header_cards_if_supported(tmp_path, 1, header)
        new_start, _new_data, new_end = hdu_byte_spans(tmp_path)[1]
        spans = hdu_byte_spans(path)
        old_start, _old_data, old_end = spans[target_hdu]
        pieces: list[Piece] = [
            (path, 0, old_start),
            (tmp_path, new_start, new_end),
            (path, old_end, spans[-1][2]),
        ]
        publish_spliced(
            path,
            [p for p in pieces if isinstance(p, bytes) or p[2] > p[1]],
            restamp=[target_hdu] if restamp_checksums else [],
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _resolve_table_hdu_index_and_columns(
    path: str, hdu: int | str
) -> tuple[int, dict[str, Any], list[str], dict[str, str]]:
    """Resolve table HDU index + column metadata for write/mutation/read helpers.

    Lives in ``_table.write`` (not ``_table.read``) so ``read`` / ``mutation``
    can import it without a write↔read cycle. Keep it here; do not move to
    ``read.py``.
    """
    import torchfits._C as cpp
    from .._io_engine.paths import guard_fits_path

    guard_fits_path(path)
    handle = cpp.open_fits_file(path, "r")
    try:
        num_hdus = int(cpp.get_num_hdus(handle))
        if num_hdus <= 0:
            raise RuntimeError(f"No HDUs found in '{path}'")

        target_idx: int | None = None
        if isinstance(hdu, int):
            target_idx = hdu
        elif isinstance(hdu, str):
            wanted = hdu.strip().upper()
            if not wanted:
                raise ValueError("hdu name cannot be empty")
            for i in range(num_hdus):
                hdu_type = str(cpp.get_hdu_type(handle, i))
                if hdu_type not in {"BINARY_TABLE", "ASCII_TABLE"}:
                    continue
                header_map = _header_cards_to_mapping(cpp.read_header(handle, i))
                extname = str(header_map.get("EXTNAME", "")).strip().upper()
                if extname == wanted:
                    target_idx = i
                    break
            if target_idx is None:
                raise KeyError(f"Table HDU named '{hdu}' not found in '{path}'")
        else:
            raise TypeError("hdu must be an int index or EXTNAME string")

        if target_idx is None or target_idx < 0 or target_idx >= num_hdus:
            raise IndexError(
                f"hdu index {hdu} out of range for '{path}' (num_hdus={num_hdus})"
            )

        hdu_type = str(cpp.get_hdu_type(handle, target_idx))
        if hdu_type not in {"BINARY_TABLE", "ASCII_TABLE"}:
            raise ValueError(f"HDU {target_idx} is not a table (type={hdu_type})")

        header_map = _header_cards_to_mapping(cpp.read_header(handle, target_idx))
        col_map = _column_name_index_map(header_map)
        columns = [col_map[idx] for idx in sorted(col_map)]
        tform_map = _column_tform_map(header_map)
        return target_idx, header_map, columns, tform_map
    finally:
        try:
            handle.close()
        except Exception as exc:
            _log.warning("table write: failed to close handle for %r: %s", path, exc)
