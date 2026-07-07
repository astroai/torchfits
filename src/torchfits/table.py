"""Arrow-native table I/O helpers."""

from __future__ import annotations

from typing import Any, Optional

from . import fits_schema


_TABLE_IO_KEYS = {
    "hdu",
    "columns",
    "row_slice",
    "rows",
    "where",
    "batch_size",
    "mmap",
    "decode_bytes",
    "encoding",
    "strip",
    "include_fits_metadata",
    "apply_fits_nulls",
    "backend",
}


def _normalize_cpp_table_data(data):
    from torchfits.io import _normalize_cpp_table_data as normalize

    return normalize(data)


def _write_header_cards_if_supported(path: str, hdu: int, hdr) -> None:
    from torchfits.io import _write_header_cards_if_supported as write_hdr

    write_hdr(path, hdu, hdr)


def _parse_tform(tform: str) -> tuple[bool, str, int]:
    info = fits_schema.parse_tform(tform)
    return info.vla, info.code or "", info.repeat


def _column_tnull_map(header_map: dict[str, Any]) -> dict[str, Any]:
    return fits_schema.column_tnull_map(header_map)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for torchfits.table APIs. Install pyarrow to use Arrow-native tables."
        ) from exc
    return pa


def _arrow_column_to_python(pa, column, name: str) -> Any:
    import numpy as np

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    if column.null_count:
        raise ValueError(
            f"Arrow column '{name}' contains nulls (not supported for FITS updates)"
        )

    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return column.to_pylist()
    if pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type):
        return column.to_pylist()
    if pa.types.is_fixed_size_list(column.type):
        values = column.values.to_numpy(zero_copy_only=False)
        size = column.type.list_size
        return values.reshape((len(column), size))
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        pylist_values: list[Any] = column.to_pylist()
        out: list[np.ndarray] = []
        for item in pylist_values:
            if item is None:
                out.append([])
            else:
                out.append(np.asarray(item))
        return out

    return column.to_numpy(zero_copy_only=False)


def _normalize_row_slice(
    row_slice: Optional[slice | tuple[int, int]],
) -> tuple[int, int]:
    if row_slice is None:
        return 1, -1

    if isinstance(row_slice, tuple):
        if len(row_slice) != 2:
            raise ValueError("row_slice tuple must be (start, stop)")
        start, stop = row_slice
        step = 1
    elif isinstance(row_slice, slice):
        start = 0 if row_slice.start is None else row_slice.start
        stop = row_slice.stop
        step = 1 if row_slice.step is None else row_slice.step
    else:
        raise ValueError("row_slice must be a slice, (start, stop), or None")

    if step != 1:
        raise ValueError("row_slice step must be 1 for FITS row streaming")
    if start < 0:
        raise ValueError("row_slice start must be >= 0")

    start_row = start + 1
    if stop is None:
        return start_row, -1
    if stop < start:
        return start_row, 0
    return start_row, stop - start


def _fits_tform_is_bit(tform: Any) -> bool:
    return fits_schema.tform_is_bit(tform)


# -- read re-exports (implementations live in _table.read) -----------------------

from ._table.read import (  # noqa: E402,F401
    _build_fits_metadata,
    _can_use_mmap_row_path_for_full_read,
    _can_use_torch_table_path_for_full_read,
    _column_tform_code_and_repeat,
    _column_tforms_for_decode,
    _compile_where_to_simple_predicates,
    _iter_chunks_cpp_numpy,
    _read_cpp_numpy_table,
    _read_table_from_scan_batches,
    _read_table_unfiltered,
    _read_table_with_where,
    _resolve_rows_from_where_cpp,
    _row_slice_from_start_num,
    _try_cpp_where_pushdown,
    _unsigned_column_dtypes,
    _where_mask_for_table,
    dataset,
    read,
    reader,
    scan,
    scan_torch,
    scanner,
    schema,
)


# -- write re-exports (implementations live in _table.write) ---------------------

from ._table.write import (  # noqa: E402,F401
    _apply_hdu_header_cards,
    _column_name_index_map,
    _column_tform_map,
    _extract_table_schema_from_header,
    _header_cards_to_mapping,
    _ordered_dict_for_columns,
    _resolve_table_hdu_index_and_columns,
    _rewrite_table_hdu_with_schema,
    _sanitize_table_header_for_rewrite,
    write,
)


# -- mutation re-exports (implementations live in _table.mutation) ---------------

from ._table.mutation import (  # noqa: E402,F401
    _coerce_rows_from_arrow,
    _coerce_table_column_array,
    _coerce_table_complex_values,
    _coerce_table_string_values,
    _coerce_table_vla_values,
    _default_table_column_values,
    _delete_column_rows,
    _infer_column_format_for_insert,
    _infer_fits_format,
    _infer_fits_scalar_code,
    _merge_insert_column,
    _normalize_column_values_for_format,
    _normalize_mutation_rows,
    _prepare_array_for_column,
    _read_table_for_rewrite,
    append_rows,
    delete_rows,
    drop_columns,
    insert_column,
    insert_rows,
    rename_columns,
    replace_column,
    update_rows,
)


# -- interop re-exports (implementations live in _table.interop) -----------------

__all__ = [
    "duckdb_query",
    "to_duckdb",
    "to_pandas",
    "to_polars",
    "to_polars_lazy",
    "write_parquet",
]

from ._table.interop import (  # noqa: E402,F401
    _materialize_arrow_table,
    _split_io_kwargs,
    duckdb_query,
    to_duckdb,
    to_pandas,
    to_polars,
    to_polars_lazy,
    write_parquet,
)


# -- arrow-convert re-exports (implementations live in _table.arrow_convert) -----

from ._table.arrow_convert import (  # noqa: E402,F401
    _chunk_to_record_batch,
    _coerce_null_sentinel,
    _column_tnull_from_meta,
    _decode_uint8_matrix_to_arrow,
    _is_vla_tuple,
    _numpy_to_arrow_array,
    _pa_array,
    _tensor_to_arrow_array,
    _uint8_matrix_to_fixed_binary,
    _uint8_matrix_to_fixed_bool_list,
    _vla_tuple_to_arrow_array,
)
