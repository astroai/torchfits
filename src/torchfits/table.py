"""Arrow-native table I/O helpers."""

from __future__ import annotations

from ._table.interop import (
    FITSPolarsFrame,
    duckdb_query,
    read_polars,
    scan_polars,
    to_duckdb,
    to_pandas,
    to_polars,
    to_polars_lazy,
    write_parquet,
)
from ._table.cache import _close_all_cached_handles as clear_cache
from ._table.read import (
    _can_use_mmap_row_path_for_full_read as can_use_mmap_row_path_for_full_read,
    _can_use_torch_table_path_for_full_read as can_use_torch_table_path_for_full_read,
)
from ._table.write import _column_name_index_map as column_name_index_map
from ._table.mutation import (
    append_rows,
    delete_rows,
    drop_columns,
    insert_column,
    insert_rows,
    rename_columns,
    replace_column,
    update_rows,
)
from ._table.read import dataset, read, reader, scan, scan_torch, scanner, schema
from ._table.write import write
from ._table_engine import TABLE_BACKENDS

__all__ = [
    "TABLE_BACKENDS",
    "append_rows",
    "can_use_mmap_row_path_for_full_read",
    "can_use_torch_table_path_for_full_read",
    "column_name_index_map",
    "clear_cache",
    "FITSPolarsFrame",
    "dataset",
    "delete_rows",
    "drop_columns",
    "duckdb_query",
    "insert_column",
    "insert_rows",
    "read",
    "read_polars",
    "reader",
    "rename_columns",
    "replace_column",
    "scan",
    "scan_polars",
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
