"""Arrow-native table I/O helpers."""

from __future__ import annotations

from ._table.interop import (
    duckdb_query,
    to_duckdb,
    to_pandas,
    to_polars,
    to_polars_lazy,
    write_parquet,
)
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
