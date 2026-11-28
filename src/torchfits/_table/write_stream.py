"""Incremental FITS table writes: ``table.writer()`` and ``table.write()`` of batches.

A stream of record batches (Arrow batches or tables, column dicts, or
dataframes) is written into one table HDU of a new file without ever holding
more than one batch. The file is built under a temp name next to the target
and published only when the stream completes; any failure (a bad batch, an
exception in the producer, an IO error) leaves an existing destination
byte-identical and no partial file behind.

The first batch with rows fixes the FITS layout -- column names and order,
TFORM (including string widths), TSCAL/TZERO/TNULL -- exactly as the one-shot
``table.write`` would lay it out for that batch, unless ``schema=`` pins it.
Later batches must match: same names in the same order, values safely
castable to the first batch's dtypes, and strings that fit the fixed width.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, Optional

__all__ = ["TableWriter", "is_batch_stream", "write_batches"]

_EMPTY_STREAM = (
    "no batches were written: an empty stream with no schema cannot define the "
    "table layout"
)


class _Abandoned(Exception):
    """Internal: the stream was abandoned, so the temp file is discarded."""


def is_batch_stream(data: Any) -> bool:
    """True for inputs ``table.write`` streams instead of materializing.

    A ``pyarrow.RecordBatchReader``, a generator/iterator, or a list/tuple of
    batches. Mappings, table/dataframe objects and arrays are single tables.
    """
    if isinstance(data, (Mapping, str, bytes)):
        return False
    if hasattr(data, "read_next_batch"):
        return True
    if any(hasattr(data, a) for a in ("column_names", "colnames", "columns")):
        return False
    if hasattr(data, "__array__") or hasattr(data, "__dlpack__"):
        return False
    return isinstance(data, Iterable)


def _is_string_list(values: Any) -> bool:
    return any(isinstance(v, (str, bytes)) for v in values) and all(
        v is None or isinstance(v, (str, bytes)) for v in values
    )


def _canonical_column(value: Any) -> Any:
    """One batch column as a numpy array, or a list for string / VLA cells."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.device.type != "cpu":
            value = value.cpu()
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        if any(isinstance(v, (list, tuple, np.ndarray, torch.Tensor)) for v in items):
            # A sequence of arrays is a VLA column for table.write even when
            # every cell has the same length; lay it out the same way.
            return items
        if items and not _is_string_list(items):
            try:
                arr = np.asarray(items)
            except ValueError:
                return items
            if arr.dtype != np.object_:
                return arr
        return items
    if isinstance(value, (str, bytes)):
        return [value]
    return np.asarray(value)


def _column_kind(value: Any) -> tuple[Any, ...]:
    """``("num", dtype, cell_shape)``, ``("str",)`` or ``("vla",)``."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            return ("str",)
        if value.dtype == np.object_:
            return ("str",) if _is_string_list(value.tolist()) else ("vla",)
        return ("num", value.dtype, tuple(value.shape[1:]))
    return ("str",) if _is_string_list(value) else ("vla",)


def _column_len(value: Any) -> int:
    import numpy as np

    if isinstance(value, np.ndarray):
        return 1 if value.ndim == 0 else int(value.shape[0])
    return len(value)


def _string_width(tform: str, ascii_table: bool) -> int:
    """Characters a string column's TFORM holds ('20A' binary, 'A20' ASCII)."""
    text = str(tform).strip().upper()
    match = re.match(r"^A(\d+)", text) if ascii_table else re.match(r"^(\d*)A", text)
    if match is None:
        return 1
    return int(match.group(1) or 1)


class TableWriter:
    """Incremental FITS table writer returned by :func:`torchfits.table.writer`.

    Call :meth:`write` once per batch, then :meth:`close` (or use it as a
    context manager). Nothing appears at ``path`` until :meth:`close`
    succeeds; :meth:`abort`, an exception inside the ``with`` block, or a
    failed :meth:`write` discard the stream and leave ``path`` untouched.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        schema: Optional[dict[str, dict[str, Any]]] = None,
        header: Optional[Mapping[str, Any]] = None,
        extname: Optional[str] = None,
        overwrite: bool = False,
        checksum: bool = False,
        table_type: str = "binary",
    ) -> None:
        from .._io_engine._file_ops import atomic_write_target
        from .._io_engine._write_helpers import _write_boundary_header
        from .._io_engine.paths import guard_write_path

        self._closed = True  # until the temp target is set up
        path = os.fspath(path)
        guard_write_path(path)
        if str(path).lower().endswith(".bz2"):
            raise ValueError(
                "Writing bzip2-wrapped FITS files ('.bz2') is not supported; "
                "use compress='BZIP2_1' for tile compression inside the file."
            )
        kind = str(table_type).lower().strip()
        if kind not in {"binary", "ascii"}:
            raise ValueError("table_type must be 'binary' or 'ascii'")
        if schema is not None and not isinstance(schema, dict):
            raise TypeError("schema must be a dictionary when provided")
        if os.path.isdir(path):
            raise IsADirectoryError(path)
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to overwrite."
            )
        hdr: Any = header
        if extname is not None:
            hdr = dict(header or {})
            hdr["EXTNAME"] = str(extname)
        self._path = path
        self._ascii = kind == "ascii"
        self._table_type = kind
        self._schema = (
            {str(k): dict(v or {}) for k, v in schema.items()} if schema else None
        )
        self._header = _write_boundary_header(hdr) if hdr else None
        self._checksum = bool(checksum)
        self._native: Any = None
        self._columns: Optional[list[str]] = None
        self._kinds: dict[str, tuple[Any, ...]] = {}
        self._widths: dict[str, int] = {}
        self._pending_empty: Optional[dict[str, Any]] = None
        self._target = atomic_write_target(path, overwrite)
        self._temp = self._target.__enter__()
        self._closed = False

    # -- public API ----------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_rows(self) -> int:
        """Rows written so far."""
        return int(self._native.num_rows) if self._native is not None else 0

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, batch: Any) -> None:
        """Append one batch: a ``pyarrow.RecordBatch`` / ``Table``, a column
        dict as ``table.write`` accepts, or a pandas/polars/astropy table.

        A batch that cannot be written raises and abandons the whole stream.
        """
        self._check_open()
        try:
            self._write(batch)
        except BaseException:
            self.abort()
            raise

    def close(self) -> None:
        """Finish the table (NAXIS2, checksums) and publish it at ``path``."""
        if self._closed:
            return
        from .._io_engine._file_ops import (
            _invalidate_path_caches,
            _invalidate_written_target,
        )

        try:
            if self._native is None:
                if self._pending_empty is None:
                    raise ValueError(_EMPTY_STREAM)
                self._begin(self._pending_empty)
            self._native.close()
            if self._checksum:
                from .._io_engine.write_api import _write_all_checksums

                _write_all_checksums(self._temp)
            _invalidate_path_caches(self._temp)
        except BaseException:
            self.abort()
            raise
        self._closed = True
        self._native = None
        self._target.__exit__(None, None, None)
        # After the rename, never before: see atomic_write_target.
        _invalidate_written_target(self._path)

    def abort(self) -> None:
        """Discard the stream: nothing is published and ``path`` is untouched."""
        if self._closed:
            return
        self._closed = True
        self._native = None  # closes the handle; the file is discarded anyway
        try:
            from .._io_engine._file_ops import _invalidate_path_caches

            _invalidate_path_caches(self._temp)
        finally:
            self._target.__exit__(_Abandoned, _Abandoned(), None)

    def __enter__(self) -> "TableWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def __del__(self) -> None:
        try:
            self.abort()
        except Exception:
            pass  # finalizer: exceptions here are only printed as noise

    # -- internals -------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("TableWriter is closed")

    def _user_tnull(self, name: str) -> Any:
        meta = (self._schema or {}).get(name) or {}
        for key in ("tnull", "null"):
            if meta.get(key) is not None:
                return meta[key]
        return None

    def _batch_columns(self, batch: Any) -> dict[str, Any]:
        if isinstance(batch, Mapping):
            return {str(k): _canonical_column(v) for k, v in batch.items()}
        if hasattr(batch, "column_names") and hasattr(batch, "column"):
            return self._arrow_columns(batch)
        from .._io_engine._write_helpers import _normalize_table_input

        cols = _normalize_table_input(batch)
        if not isinstance(cols, dict):
            raise TypeError(
                "each batch must be a pyarrow RecordBatch/Table, a column "
                f"dictionary, or a dataframe; got {type(batch).__name__}"
            )
        return {str(k): _canonical_column(v) for k, v in cols.items()}

    def _arrow_columns(self, batch: Any) -> dict[str, Any]:
        """Arrow batch -> columns, with the one-shot writer's null handling.

        Float nulls become NaN and string nulls blank, as ``table.write``
        writes them. An integer column cannot widen to float mid-stream the
        way the one-shot writer widens it, so its nulls need a TNULL sentinel
        from ``schema=``; boolean nulls have no stored form at all.
        """
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc

        from .._io_engine._write_helpers import _arrow_column_values

        names = [str(n) for n in batch.column_names]
        if len(set(names)) != len(names):
            raise ValueError(f"batch has duplicate column names: {names}")
        out: dict[str, Any] = {}
        for i, name in enumerate(names):
            col = batch.column(i)
            typ = col.type
            if col.null_count:
                if pa.types.is_integer(typ):
                    tnull = self._user_tnull(name)
                    if tnull is None:
                        raise ValueError(
                            f"column {name!r} has null values but no TNULL: pass "
                            f"schema={{{name!r}: {{'tnull': <sentinel>}}}} to store "
                            "them in a streamed integer column"
                        )
                    col = pc.fill_null(col, pa.scalar(int(tnull), type=typ))
                elif pa.types.is_boolean(typ):
                    raise ValueError(
                        f"column {name!r} has null values; FITS logical columns "
                        "are written without nulls"
                    )
            if len(col) == 0 and (
                pa.types.is_string(typ)
                or pa.types.is_large_string(typ)
                or pa.types.is_binary(typ)
                or pa.types.is_large_binary(typ)
            ):
                out[name] = np.array([], dtype="S1")
                continue
            out[name] = _canonical_column(
                _arrow_column_values(col, batch.schema.field(i))
            )
        return out

    def _write(self, batch: Any) -> None:
        cols = self._batch_columns(batch)
        names = list(cols)
        if not names:
            raise ValueError("a batch must have at least one column")
        expected = self._columns or (
            list(self._pending_empty) if self._pending_empty is not None else None
        )
        if expected is not None and names != expected:
            raise ValueError(
                f"batch columns {names} do not match the table's columns {expected}"
            )
        lengths = {_column_len(v) for v in cols.values()}
        if len(lengths) != 1:
            raise ValueError("batch columns have different lengths")
        nrows = lengths.pop()
        if self._native is None:
            if nrows == 0:
                # Zero-row batches (a filtered scan yields them) do not fix
                # the layout; they only define it if no row ever arrives.
                self._pending_empty = cols
                return
            self._begin(cols)
            return
        if nrows:
            self._append(cols)

    def _begin(self, cols: dict[str, Any], *, replay_header: Any = None) -> None:
        """Create the table HDU from the first batch and open the stream."""
        import torchfits._C as cpp

        from .._io_engine._write_helpers import (
            _can_use_cpp_table_writer,
            _cpp_header_mapping,
            _normalize_cpp_table_data,
            _prepare_unsigned_table_data_for_write,
            _warn_clipped_string_columns,
            _write_header_cards_if_supported,
        )
        from .write import _resolve_table_hdu_index_and_columns

        from ._cell_shape import shape_cells_for_write

        flat, self._schema = shape_cells_for_write(dict(cols), self._schema)
        prepared, schema, _ = _prepare_unsigned_table_data_for_write(
            flat, self._schema, ascii_table=self._ascii
        )
        if self._schema is None and not _can_use_cpp_table_writer(prepared):
            raise ValueError(
                "Dictionary table writes currently require CFITSIO-native column "
                "types (numeric/bool/complex, strings, or VLA lists). Unsupported "
                "object/structure columns should be converted before writing."
            )
        data = _normalize_cpp_table_data(prepared)
        _warn_clipped_string_columns(data, schema, stacklevel=5)
        header = self._header
        cpp.write_fits_table(
            self._temp,
            data,
            _cpp_header_mapping(header) if header else {},
            False,
            schema if schema else None,
            self._table_type,
        )
        for cards in (header, replay_header):
            if cards:
                _write_header_cards_if_supported(self._temp, 1, cards)

        _, header_map, _, tforms = _resolve_table_hdu_index_and_columns(self._temp, 1)
        self._columns = list(cols)
        for name, value in cols.items():
            kind = _column_kind(value)
            self._kinds[name] = kind
            if kind[0] == "str":
                self._widths[name] = _string_width(tforms.get(name, ""), self._ascii)
        self._pending_empty = None
        self._native = cpp.TableStreamWriter(self._temp, 1)

    def _append(self, cols: dict[str, Any]) -> None:
        import numpy as np

        from .._io_engine._write_helpers import (
            _normalize_cpp_table_data,
            _prepare_unsigned_table_data_for_write,
        )

        for name, value in cols.items():
            want = self._kinds[name]
            got = _column_kind(value)
            if want[0] == "num":
                if got[0] != "num":
                    raise TypeError(
                        f"column {name!r}: batch holds {got[0]} values, but the "
                        f"column was laid out as {want[1]} by the first batch"
                    )
                if got[2] != want[2]:
                    raise ValueError(
                        f"column {name!r}: cell shape {got[2]} does not match "
                        f"the column's {want[2]}"
                    )
                if got[1] != want[1]:
                    if not np.can_cast(got[1], want[1], "safe"):
                        raise TypeError(
                            f"column {name!r}: dtype {got[1]} cannot be stored "
                            f"in the column's {want[1]} (fixed by the first "
                            "batch or schema) without loss"
                        )
                    cols[name] = value.astype(want[1])
            elif got[0] != want[0] and not (
                want[0] == "vla" and _column_len(value) == 0
            ):
                raise TypeError(
                    f"column {name!r}: batch holds {got[0]} values, but the "
                    f"column was laid out for {want[0]} values by the first batch"
                )

        from ._cell_shape import flatten_cells

        cols = {name: flatten_cells(name, value)[0] for name, value in cols.items()}
        prepared, _, _ = _prepare_unsigned_table_data_for_write(
            cols, ascii_table=self._ascii
        )
        data = _normalize_cpp_table_data(prepared)
        for name, width in self._widths.items():
            self._check_strings(name, data[name], width)
        self._native.write_rows(data)

    def _check_strings(self, name: str, values: list[str], width: int) -> None:
        longest = 0
        for value in values:
            if not value.isascii():
                raise ValueError(
                    f"column {name!r}: string {value!r} is not ASCII; FITS "
                    "string columns hold ASCII text"
                )
            longest = max(longest, len(value))
        if longest > width:
            spec = f"A{longest}" if self._ascii else f"{longest}A"
            raise ValueError(
                f"column {name!r}: a {longest}-character string does not fit the "
                f"column's {width} characters, fixed by the first batch; pass "
                f"schema={{{name!r}: {{'format': {spec!r}}}}} (or wider) to size it"
            )

    def _write_raw_rows(self, rows: Any) -> None:
        """Append rows already in this table's stored FITS row layout."""
        self._check_open()
        try:
            self._native.write_raw_rows(rows)
        except BaseException:
            self.abort()
            raise


def write_batches(
    path: str | os.PathLike[str],
    batches: Any,
    *,
    schema: Optional[dict[str, dict[str, Any]]] = None,
    header: Optional[Mapping[str, Any]] = None,
    extname: Optional[str] = None,
    overwrite: bool = False,
    checksum: bool = False,
    table_type: str = "binary",
) -> int:
    """Stream ``batches`` into a new FITS table at ``path``; returns rows written."""
    arrow_schema = (
        getattr(batches, "schema", None)
        if hasattr(batches, "read_next_batch")
        else None
    )
    with TableWriter(
        path,
        schema=schema,
        header=header,
        extname=extname,
        overwrite=overwrite,
        checksum=checksum,
        table_type=table_type,
    ) as writer:
        for batch in batches:
            writer.write(batch)
        if (
            writer._native is None
            and writer._pending_empty is None
            and arrow_schema is not None
            and len(arrow_schema)
        ):
            writer.write(arrow_schema.empty_table())
        rows = writer.num_rows
    return rows
