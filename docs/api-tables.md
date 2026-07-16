# Table Reference

FITS table I/O with predicate pushdown, column projection, streaming, in-place
mutations, and interop with Polars, DuckDB, Pandas, and PyArrow.

---

## `table.read()`

Arrow-native FITS table reader with WHERE pushdown.

```python
torchfits.table.read(
    path, hdu=1, columns=None, row_slice=None, rows=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=False, apply_fits_nulls=True,
    backend="auto",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int \| str` | `1` | Table HDU index or EXTNAME |
| `columns` | `list[str] \| None` | `None` | Column projection (None = all) |
| `row_slice` | `slice \| tuple[int,int] \| None` | `None` | Row range filter |
| `rows` | `list[int] \| None` | `None` | Specific row indices |
| `where` | `str \| None` | `None` | SQL-like predicate (pushed to C++) |
| `batch_size` | `int` | `65536` | Internal read batch size |
| `mmap` | `bool` | `True` | Memory-mapped reads |
| `decode_bytes` | `bool` | `True` | Decode byte-string columns |
| `backend` | `str` | `"auto"` | `"auto"`, `"cpp"`, or `"torch"` |
| `include_fits_metadata` | `bool` | `False` | Preserve FITS column metadata |

**Returns:** `pyarrow.Table`

!!! info "When to use"
    This is the primary function for reading FITS tables. Use it when you
    need column projection, row filtering via `where=`, or Arrow/Polars/
    DuckDB interop. For raw tensor dictionaries, use `read_table()` instead.

```python
# Filter and project
table = torchfits.table.read(
    "catalog.fits", hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20 AND DEC > 0",
)
print(table.num_rows, table.column_names)
```

---

## `table.scan()`

Streaming FITS table scanner yielding `pyarrow.RecordBatch` objects without
materializing the entire table.

```python
torchfits.table.scan(
    path, hdu=1, columns=None, row_slice=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=False, apply_fits_nulls=True,
    backend="auto",
)
```

**Yields:** `pyarrow.RecordBatch`

```python
for batch in torchfits.table.scan("survey.fits", hdu=1, chunk_rows=50_000):
    process(batch)  # pyarrow.RecordBatch
```

!!! info "When to use"
    Use `scan()` when the table is too large to fit in memory, or when you
    want to process rows in streaming fashion. For Polars-specific streaming,
    use `scan_polars()`.

---

## `table.scan_torch()`

Streaming scanner yielding dictionaries of `torch.Tensor` per batch.

```python
torchfits.table.scan_torch(
    path, hdu=1, columns=None, row_slice=None, batch_size=65536,
    mmap=True, device="cpu", non_blocking=True, pin_memory=False,
)
```

**Yields:** `dict[str, torch.Tensor]`

```python
for batch in torchfits.table.scan_torch("survey.fits", hdu=1, batch_size=10000):
    # batch: dict[str, torch.Tensor]
    process(batch)
```

---

## `table.reader()`

Open a FITS table as a `pyarrow.RecordBatchReader` for streaming.

```python
torchfits.table.reader(
    path, hdu=1, columns=None, row_slice=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=True, apply_fits_nulls=True,
    backend="auto",
)
```

**Returns:** `pyarrow.RecordBatchReader`

---

## `table.write()`

Write a columnar dictionary as a FITS binary or ASCII table.

```python
torchfits.table.write(path, data, *, schema=None, header=None,
                      overwrite=False, extname=None, table_type="binary")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | Output path |
| `data` | `dict[str, array-like]` | *(required)* | Column name to values |
| `header` | `dict \| None` | `None` | FITS header key-value pairs |
| `overwrite` | `bool` | `False` | Overwrite existing file |
| `table_type` | `str` | `"binary"` | `"binary"` or `"ascii"` |

```python
torchfits.table.write("out.fits", {"RA": ra, "DEC": dec}, overwrite=True)
```

---

## Predicate Pushdown

The `where=` parameter filters rows before data reaches Python. Filtering
happens in C++ for most table sizes.

!!! warning
    Root `torchfits.read()` does **not** accept `where=`. Always use
    `torchfits.table.read()` or `torchfits.table.scan()` for filtered reads.

**Supported operators:**

| Operator | Example |
|---|---|
| `=` / `!=` | `where="CLASS = 'star'"` |
| `<` / `>` / `<=` / `>=` | `where="MAG_G < 20"` |
| `AND` / `OR` | `where="MAG_G < 20 AND DEC > 0"` |
| `NOT` | `where="NOT CLASS = 'star'"` |
| `IN (...)` | `where="id IN (1, 2, 3)"` |
| `NOT IN (...)` | `where="id NOT IN (4, 5)"` |
| `BETWEEN ... AND ...` | `where="MAG_G BETWEEN 15 AND 20"` |
| `IS NULL` / `IS NOT NULL` | `where="DEC IS NOT NULL"` |

### Backend Selection

| Backend | Behavior |
|---|---|
| `"auto"` (default) | C++ pushdown for large tables; Arrow filter for small tables |
| `"cpp"` | C++ row reads as torch tensors, converted to Arrow |
| `"torch"` | `stream_table` chunked path |

### Environment Tuning

| Variable | Default | Effect |
|---|---|---|
| `TORCHFITS_TABLE_SCANNER_THRESHOLD` | `100000` (or `1000` for VLA) | Row count threshold for pushdown vs read-then-filter |
| `TORCHFITS_TABLE_HANDLE_CACHE` | `1` | Set `0` to disable LRU handle cache |
| `TORCHFITS_TABLE_READER_CACHE` | `1` | Set `0` to disable `TableReader` cache |

---

## Predicate Helpers

The `torchfits.where` module provides predicate parsing and evaluation
outside of table reads.

```python
from torchfits.where import evaluate_where, parse_where_expression

ast = parse_where_expression("MAG_G < 20 AND DEC IS NOT NULL")
mask = evaluate_where(ast, {"MAG_G": magnitudes, "DEC": declinations})
# mask: np.ndarray[bool]
```

Additional public names: `parse_where_literal`, `tokenize_where_expression`,
`normalize_where_syntax`, `where_columns_from_ast`.

---

## In-Place Table Mutation

```python
# Row operations
torchfits.table.append_rows(path, rows, hdu=1)
torchfits.table.insert_rows(path, rows, row=0, hdu=1)
torchfits.table.update_rows(path, rows, row_slice, hdu=1)
torchfits.table.delete_rows(path, row_slice, hdu=1)

# Column operations
torchfits.table.insert_column(path, name, values, hdu=1, index=None)
torchfits.table.replace_column(path, name, values, hdu=1)
torchfits.table.rename_columns(path, {"old_name": "new_name"}, hdu=1)
torchfits.table.drop_columns(path, ["col_a", "col_b"], hdu=1)
```

`insert_column` and `replace_column` accept optional `format`, `unit`,
`dim`, `tnull`, `tscal`, `tzero` metadata kwargs.

---

## Interop

### Polars

```python
# One-call FITS to Polars (preserves FITS metadata)
df = torchfits.table.read_polars("catalog.fits", hdu=1)
# df: FITSPolarsFrame — wraps pl.DataFrame with .field_meta, .table_meta

# Streaming FITS to Polars (no full materialization)
for batch in torchfits.table.scan_polars("catalog.fits", hdu=1):
    process(batch)  # pl.DataFrame

# Materialize then wrap as LazyFrame
lazy = torchfits.table.to_polars_lazy("catalog.fits", hdu=1)

# From table dict
polars_df = torchfits.to_polars(table_dict, decode_bytes=True)
```

!!! tip "True streaming"
    `scan_polars()` is the genuine streaming path. `to_polars_lazy()`
    materializes the entire table first, then wraps as `LazyFrame`.

!!! info "rechunk=False default"
    All Polars conversion functions default to `rechunk=False`. Pass
    `rechunk=True` to restore the old chunk-concatenation behavior.

### DuckDB

```python
# Register and query
con = torchfits.table.to_duckdb("catalog.fits", hdu=1, relation_name="tbl")
result = torchfits.table.duckdb_query(
    "catalog.fits",
    "SELECT * FROM tbl WHERE MAG < 20",
    hdu=1,
)
# result: pyarrow.Table (by default)
```

### Arrow and Pandas

```python
arrow_table = torchfits.to_arrow(table_dict, decode_bytes=True)
pandas_df = torchfits.to_pandas(table_dict, decode_bytes=True)
```

---

## Schema

Infer the Arrow schema from FITS TFORM header cards without reading any data.

```python
schema = torchfits.table.schema("catalog.fits", hdu=1)
# schema: pyarrow.Schema
```

---

## Additional Utilities

```python
# PyArrow dataset and scanner
ds = torchfits.table.dataset("catalog.fits", hdu=1)
sc = torchfits.table.scanner("catalog.fits", columns=["RA", "DEC"])

# Parquet export
torchfits.table.write_parquet("out.parquet", "catalog.fits", hdu=1)

# Cache cleanup
torchfits.table.clear_cache()
```

The public constants `torchfits.table.TABLE_BACKENDS` and the decision
helpers `can_use_mmap_row_path_for_full_read` and
`can_use_torch_table_path_for_full_read` are available for downstream
packages that need to inspect table layout compatibility.
