"""Arrow-native table I/O helpers."""

from __future__ import annotations

# -- helpers (implementations live in _table.utils) -------------------------

from ._table.utils import (  # noqa: E402,F401
    _TABLE_IO_KEYS,
    _arrow_column_to_python,
    _column_tnull_map,
    _fits_tform_is_bit,
    _normalize_cpp_table_data,
    _normalize_row_slice,
    _parse_tform,
    _require_pyarrow,
    _write_header_cards_if_supported,
)


# -- read re-exports (implementations live in _table.read) -------------------

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


# -- write re-exports (implementations live in _table.write) -----------------

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


# -- mutation re-exports (implementations live in _table.mutation) -----------

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


# -- interop re-exports (implementations live in _table.interop) -------------

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


# -- arrow-convert re-exports (implementations live in _table.arrow_convert) -

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


__all__ = [
    "append_rows",
    "dataset",
    "delete_rows",
    "drop_columns",
    "duckdb_query",
    "insert_column",
    "insert_rows",
    "read",
    "reader",
    "rename_columns",
    "replace_column",
    "scan",
    "scan_torch",
    "scanner",
    "schema",
    "to_duckdb",
    "to_pandas",
    "to_polars",
    "to_polars_lazy",
    "update_rows",
    "write",
    "write_parquet",
]
