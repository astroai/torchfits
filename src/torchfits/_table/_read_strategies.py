"""Native read strategies behind ``table.read``'s C++ path.

``_read_scan._read_cpp_table_chunk`` resolves the request (row selection,
``where=`` rows, header-derived decode metadata) and then asks these
strategies, in order, for the raw column chunk the C++ engine returns. Each
returns the chunk, or ``None`` when it cannot serve the request so the caller
tries the next one; an "Invalid table/VLA geometry" error always propagates.

* :func:`read_full_raw` -- the whole table with no decode, null or metadata
  work: the mmap reader first, then the buffered reader.
* :func:`read_selected_rows` -- a ``rows=`` scatter selection. It has no
  fallback: the scan path would silently return the full table (r5a-01).
* :func:`read_row_window` -- a contiguous ``(start_row, num_rows)`` window:
  the mmap reader first, then the buffered reader.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .._table.cache import _acquire_cpp_reader
from ._read_schema import _can_use_mmap_row_path_for_full_read

# Same logger as the read body that used to inline these strategies.
logger = logging.getLogger("torchfits._table._read_scan")


def _reraise_invalid_geometry(exc: BaseException) -> None:
    message = str(exc)
    if "Invalid table geometry" in message or "Invalid VLA geometry" in message:
        raise exc


def read_full_raw(
    cpp: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    *,
    mmap: bool,
    get_header: Callable[[], Any],
) -> Any:
    """Whole-table raw read (caller checked the full-read capability)."""
    col_list = columns if columns else []
    chunk: Any = None
    if mmap and _can_use_mmap_row_path_for_full_read(
        path, hdu, columns, header=get_header()
    ):
        try:
            chunk = cpp.read_fits_table_raw(path, hdu, col_list, True)
        except (RuntimeError, OSError) as exc:
            _reraise_invalid_geometry(exc)
            logger.debug("mmap full table read failed; retrying: %s", exc)
            chunk = None
    if chunk is None:
        try:
            if not col_list:
                reader = _acquire_cpp_reader(path, hdu, cpp)
                chunk = reader.read_rows_raw([], 1, -1)
            else:
                chunk = cpp.read_fits_table_raw(path, hdu, col_list, False)
        except (RuntimeError, OSError) as exc:
            _reraise_invalid_geometry(exc)
            logger.debug("full table read failed; falling back: %s", exc)
            chunk = None
    return chunk


def read_selected_rows(
    cpp: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    rows_arr: Any,
    header: Any,
) -> Any:
    """``rows=`` scatter read of 0-based row indices; errors propagate."""
    from .engine import _read_selected_rows

    try:
        row_bytes = int(header.get("NAXIS1", 0)) if header is not None else 0
    except (TypeError, ValueError):
        row_bytes = 0
    reader = _acquire_cpp_reader(path, hdu, cpp)
    return _read_selected_rows(
        reader, columns if columns else [], rows_arr, row_bytes=row_bytes
    )


def read_row_window(
    cpp: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    start_row: int,
    num_rows: int,
    *,
    mmap: bool,
) -> Any:
    """Contiguous row-window read (1-based ``start_row``, ``-1`` = to the end)."""
    col_list = columns if columns else []
    chunk: Any = None
    if mmap:
        try:
            chunk = cpp.read_fits_table_rows_raw(
                path, hdu, col_list, start_row, num_rows, True
            )
        except (RuntimeError, OSError) as exc:
            _reraise_invalid_geometry(exc)
            # VLA, scaled, and ASCII tables cannot use the mmap layout.
            # Retry the buffered reader before falling back to the
            # row-by-row Python scan, which is an order of magnitude slower.
            logger.debug("mmap row-slice read failed; retrying buffered: %s", exc)
            chunk = None
    if chunk is None:
        try:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk = reader.read_rows_raw(col_list, start_row, num_rows)
        except (RuntimeError, OSError) as exc:
            _reraise_invalid_geometry(exc)
            logger.debug("row-slice table read failed: %s", exc)
            chunk = None
    return chunk
