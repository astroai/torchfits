"""Table read path: scans, reads, WHERE filtering, schema inference, torch streaming."""

from __future__ import annotations

from collections.abc import Iterator
import itertools
import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    import numpy as np

from .. import fits_schema
from .._table.cache import acquire_cpp_handle as _acquire_cpp_handle
from .._table.cache import acquire_cpp_reader as _acquire_cpp_reader
from .._where import parse_where_expression, where_columns_from_ast
from .._table_engine import (
    WhereStrategy,
    choose_where_read_plan,
    should_skip_cpp_numpy_for_where,
    validate_table_backend,
)
from ..table import _normalize_row_slice, _require_pyarrow
from .._table.arrow_convert import (
    _chunk_to_record_batch,
    _column_tnull_from_meta,
    _is_vla_tuple,
    _numpy_to_arrow_array,
    _pa_array,
    _vla_tuple_to_arrow_array,
)
from .._table.write import _resolve_table_hdu_index_and_columns

logger = logging.getLogger(__name__)


def _column_tform_code_and_repeat(tform: Any) -> tuple[str, int] | None:
    return fits_schema.tform_code_and_repeat(tform)


def _fits_tform_is_bit(tform: Any) -> bool:
    return fits_schema.tform_is_bit(tform)


def _row_slice_from_start_num(start_row: int, num_rows: int) -> Optional[slice]:
    if start_row == 1 and num_rows == -1:
        return None
    start0 = start_row - 1
    if num_rows == -1:
        return slice(start0, None)
    return slice(start0, start0 + num_rows)


def _compile_where_to_simple_predicates(
    where: str,
) -> Optional[list[tuple[str, str, Any]]]:
    try:
        ast = parse_where_expression(where)
    except Exception:
        return None

    predicates: list[tuple[str, str, Any]] = []

    def _visit(node) -> bool:
        kind = node[0]
        if kind == "cmp":
            _, col, op, literal = node
            if op not in {"==", "!=", ">", ">=", "<", "<="}:
                return False
            if literal is None:
                return False
            predicates.append((col, op, literal))
            return True
        if kind == "between":
            _, col, low, high, negate = node
            if bool(negate) or low is None or high is None:
                return False
            predicates.append((col, ">=", low))
            predicates.append((col, "<=", high))
            return True
        if kind == "and":
            return _visit(node[1]) and _visit(node[2])
        return False

    if not _visit(ast):
        return None
    return predicates


def _where_mask_for_table(table, where: str, parsed_ast=None) -> "np.ndarray":
    pa = _require_pyarrow()
    import pyarrow.compute as pc

    ast = parsed_ast if parsed_ast is not None else parse_where_expression(where)

    def _get_predicate_column(column_name: str):
        if column_name not in table.column_names:
            raise ValueError(f"where references unknown column '{column_name}'")

        column = table[column_name]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            raise ValueError(f"where does not support list/VLA column '{column_name}'")
        if pa.types.is_fixed_size_list(column.type):
            raise ValueError(
                f"where does not support fixed-size vector column '{column_name}'"
            )
        return column

    def _cmp_mask(column_name: str, op: str, literal: Any):
        column = _get_predicate_column(column_name)

        if literal is None:
            if op == "==":
                return pc.is_null(column)
            if op == "!=":
                return pc.invert(pc.is_null(column))
            raise ValueError("where comparisons with null only support == and !=")

        scalar = pa.scalar(literal)
        if op == "==":
            return pc.equal(column, scalar)
        if op == "!=":
            return pc.not_equal(column, scalar)
        if op == ">":
            return pc.greater(column, scalar)
        if op == ">=":
            return pc.greater_equal(column, scalar)
        if op == "<":
            return pc.less(column, scalar)
        if op == "<=":
            return pc.less_equal(column, scalar)
        raise ValueError(f"Unsupported where operator '{op}'")

    def _in_mask(column_name: str, literals: list[Any], negate: bool):
        column = _get_predicate_column(column_name)
        non_null = [v for v in literals if v is not None]
        has_null = any(v is None for v in literals)

        if non_null:
            value_set = _pa_array(pa, non_null)
            mask = pc.is_in(column, value_set=value_set)
        else:
            mask = _pa_array(pa, [False] * int(len(column)))

        if has_null:
            mask = pc.or_(pc.fill_null(mask, False), pc.is_null(column))
        mask = pc.fill_null(mask, False)

        if negate:
            return pc.invert(mask)
        return mask

    def _between_mask(column_name: str, low: Any, high: Any, negate: bool):
        column = _get_predicate_column(column_name)
        if low is None or high is None:
            raise ValueError("where BETWEEN does not support NULL bounds")
        low_s = pa.scalar(low)
        high_s = pa.scalar(high)
        ge = pc.greater_equal(column, low_s)
        le = pc.less_equal(column, high_s)
        mask = pc.and_(pc.fill_null(ge, False), pc.fill_null(le, False))
        mask = pc.fill_null(mask, False)
        if negate:
            return pc.invert(mask)
        return mask

    def _isnull_mask(column_name: str, negate: bool):
        column = _get_predicate_column(column_name)
        mask = pc.is_null(column)
        mask = pc.fill_null(mask, False)
        if negate:
            return pc.invert(mask)
        return mask

    def _eval(node):
        kind = node[0]
        if kind == "cmp":
            return pc.fill_null(_cmp_mask(node[1], node[2], node[3]), False)
        if kind == "in":
            return pc.fill_null(_in_mask(node[1], node[2], bool(node[3])), False)
        if kind == "between":
            return pc.fill_null(
                _between_mask(node[1], node[2], node[3], bool(node[4])), False
            )
        if kind == "isnull":
            return pc.fill_null(_isnull_mask(node[1], bool(node[2])), False)
        if kind == "and":
            left = pc.fill_null(_eval(node[1]), False)
            right = pc.fill_null(_eval(node[2]), False)
            return pc.and_(left, right)
        if kind == "or":
            left = pc.fill_null(_eval(node[1]), False)
            right = pc.fill_null(_eval(node[2]), False)
            return pc.or_(left, right)
        if kind == "not":
            child = pc.fill_null(_eval(node[1]), False)
            return pc.invert(child)
        raise ValueError("Invalid where AST")

    return pc.fill_null(_eval(ast), False)


def _build_fits_metadata(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]] = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    import torchfits

    header = torchfits.get_header(path, hdu)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {
        "fits_hdu": str(hdu),
    }

    try:
        tf_count = int(header.get("TFIELDS", 0))
    except Exception:
        tf_count = 0

    for i in range(1, tf_count + 1):
        si = str(i)
        name = header.get("TTYPE" + si)
        if not isinstance(name, str) or not name:
            continue
        if selected_columns is not None and name not in selected_columns:
            continue

        entry: dict[str, str] = {}

        v = header.get("TFORM" + si)
        if v is not None:
            entry["fits_tform"] = str(v)

        v = header.get("TUNIT" + si)
        if v is not None:
            entry["fits_tunit"] = str(v)

        v = header.get("TDIM" + si)
        if v is not None:
            entry["fits_tdim"] = str(v)

        v = header.get("TNULL" + si)
        if v is not None:
            entry["fits_tnull"] = str(v)

        v = header.get("TSCAL" + si)
        if v is not None:
            entry["fits_tscal"] = str(v)

        v = header.get("TZERO" + si)
        if v is not None:
            entry["fits_tzero"] = str(v)

        if entry:
            field_meta[name] = entry

    return field_meta, table_meta


def _column_tforms_for_decode(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        fm, _ = _build_fits_metadata(path, hdu, selected_columns)
        for col, meta in fm.items():
            tf = meta.get("fits_tform")
            if tf:
                out[col] = tf
    except Exception:
        pass
    return out


def _unsigned_column_dtypes(
    path: str,
    hdu: int,
    selected_columns: Optional[set[str]],
) -> dict[str, str]:
    try:
        fm, _ = _build_fits_metadata(path, hdu, selected_columns)
    except Exception:
        return {}
    targets = {
        ("I", 32768.0): "uint16",
        ("J", 2147483648.0): "uint32",
    }
    out: dict[str, str] = {}
    for col, meta in fm.items():
        parsed = _column_tform_code_and_repeat(meta.get("fits_tform"))
        if parsed is None:
            continue
        code, _repeat = parsed
        try:
            tscal = float(meta.get("fits_tscal", "1"))
            tzero = float(meta.get("fits_tzero", "0"))
        except Exception:
            continue
        target = targets.get((code, tzero))
        if target is not None and tscal == 1.0:
            out[col] = target
    return out


def _can_use_mmap_row_path_for_full_read(
    path: str,
    hdu: int,
    selected_columns: Optional[list[str]],
) -> bool:
    import torchfits

    try:
        header = torchfits.get_header(path, hdu)
        tf_count = int(header.get("TFIELDS", 0))
    except Exception:
        return False
    if tf_count <= 0:
        return False

    selected = set(selected_columns) if selected_columns else None
    supported_codes = {"L", "B", "I", "J", "K", "E", "D"}
    any_selected = False

    for i in range(1, tf_count + 1):
        name = header.get(f"TTYPE{i}")
        if not isinstance(name, str) or not name:
            continue
        if selected is not None and name not in selected:
            continue
        any_selected = True

        if header.get(f"TSCAL{i}") is not None or header.get(f"TZERO{i}") is not None:
            return False

        parsed = _column_tform_code_and_repeat(header.get(f"TFORM{i}"))
        if parsed is None:
            return False
        code, repeat = parsed
        if code not in supported_codes:
            return False
        if repeat != 1:
            return False

    return any_selected


def _can_use_torch_table_path_for_full_read(
    path: str,
    hdu: int,
    selected_columns: Optional[list[str]],
) -> bool:
    import torchfits

    try:
        header = torchfits.get_header(path, hdu)
        tf_count = int(header.get("TFIELDS", 0))
    except Exception:
        return False
    if tf_count <= 0:
        return False

    selected = set(selected_columns) if selected_columns else None
    supported_codes = {"L", "B", "I", "J", "K", "E", "D"}
    any_selected = False

    for i in range(1, tf_count + 1):
        name = header.get(f"TTYPE{i}")
        if not isinstance(name, str) or not name:
            continue
        if selected is not None and name not in selected:
            continue
        any_selected = True

        parsed = _column_tform_code_and_repeat(header.get(f"TFORM{i}"))
        if parsed is None:
            return False
        code, repeat = parsed
        if code not in supported_codes:
            return False
        if repeat != 1:
            return False

    return any_selected


def _iter_chunks_cpp_numpy(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    start_row: int,
    num_rows: int,
    batch_size: int,
    mmap: bool,
):
    import torchfits
    import torchfits._C as cpp

    if not hasattr(cpp, "read_fits_table_rows_numpy_from_handle"):
        return None

    header = torchfits.get_header(path, hdu)
    total_rows = header.get("NAXIS2", 0)
    try:
        total_rows = (
            int(float(total_rows)) if isinstance(total_rows, str) else int(total_rows)
        )
    except Exception:
        total_rows = 0
    if total_rows <= 0:
        return iter(())

    end_row = (
        total_rows if num_rows == -1 else min(total_rows, start_row + num_rows - 1)
    )
    col_list = columns if columns else []

    def _generator():
        can_mmap_rows = mmap and hasattr(cpp, "read_fits_table_rows")
        if can_mmap_rows:
            can_mmap_rows = _can_use_mmap_row_path_for_full_read(path, hdu, columns)
        file_handle = None
        try:
            row = start_row
            while row <= end_row:
                size = min(batch_size, end_row - row + 1)
                if can_mmap_rows:
                    try:
                        yield cpp.read_fits_table_rows(
                            path, hdu, col_list, row, size, True
                        )
                        row += size
                        continue
                    except Exception:
                        can_mmap_rows = False

                if file_handle is None:
                    file_handle = cpp.open_fits_file(path, "r")
                yield cpp.read_fits_table_rows_numpy_from_handle(
                    file_handle, hdu, col_list, row, size
                )
                row += size
        finally:
            if file_handle is not None:
                file_handle.close()

    return _generator()


def _filter_table_with_where(pa, table: Any, where: str) -> Any:
    mask = _where_mask_for_table(table, where)
    if len(mask) == 0 or pa.compute.sum(mask).as_py() == 0:
        return table.slice(0, 0)
    return table.filter(mask)


def _read_table_from_scan_batches(
    *,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
) -> Any:
    pa = _require_pyarrow()
    batches = list(
        scan(
            path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            batch_size=batch_size,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend=backend,
        )
    )
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches)


def _read_table_unfiltered(
    *,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
) -> Any:
    if backend in {"auto", "cpp_numpy"}:
        single = _read_cpp_numpy_table(
            path=path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=None,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
        )
        if single is not None:
            return single
    return _read_table_from_scan_batches(
        path=path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=backend,
    )


def _try_cpp_where_pushdown(
    *,
    pa,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
) -> Any | None:
    import torchfits._C as cpp

    if not hasattr(cpp, "read_fits_table_filtered"):
        return None
    filters = _compile_where_to_simple_predicates(where)
    if filters is None:
        return None
    try:
        target_cols = columns
        if target_cols is None:
            target_cols = list(schema(path, hdu=hdu, backend="cpp_numpy").names)

        data_dict = cpp.read_fits_table_filtered(path, hdu, target_cols, filters)
        pushdown_tforms = (
            _column_tforms_for_decode(path, hdu, set(target_cols))
            if decode_bytes
            else None
        )
        arrays = []
        names_out = []
        for name in target_cols:
            if name not in data_dict:
                continue
            val = data_dict[name]
            if isinstance(val, torch.Tensor):
                if val.device.type != "cpu":
                    val = val.cpu()
                if not val.is_contiguous():
                    val = val.contiguous()
                arr = _numpy_to_arrow_array(
                    pa,
                    val.numpy(),
                    decode_bytes,
                    encoding,
                    strip,
                    fits_tform=pushdown_tforms.get(name) if pushdown_tforms else None,
                )
                arrays.append(arr)
                names_out.append(name)

        if not arrays:
            return pa.table({})
        return pa.Table.from_arrays(arrays, names=names_out)
    except Exception:
        return None


def _read_table_with_where(
    *,
    pa,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    where: str,
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
) -> Any:
    import torchfits

    header_ok = False
    hdr: Any = {}
    n_rows = 0
    try:
        hdr = torchfits.get_header(path, hdu)
        n_rows = int(hdr.get("NAXIS2", 0))
        header_ok = True
    except Exception:
        n_rows = 0

    plan = choose_where_read_plan(
        header=hdr,
        header_ok=header_ok,
        columns=columns,
        backend=backend,
        n_rows=n_rows,
    )

    if plan.strategy == WhereStrategy.CPP_PUSHDOWN:
        pushed = _try_cpp_where_pushdown(
            pa=pa,
            path=path,
            hdu=hdu,
            columns=columns,
            where=where,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
        )
        if pushed is not None:
            return pushed

    base = _read_table_unfiltered(
        path=path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        rows=rows,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=plan.unfiltered_backend,
    )
    return _filter_table_with_where(pa, base, where)


def _resolve_rows_from_where_cpp(
    path: str,
    hdu: int,
    where: str,
    start_row: int,
    num_rows: int,
    mmap: bool,
    apply_fits_nulls: bool,
) -> Optional[list[int]]:
    where_ast = parse_where_expression(where)
    where_columns = where_columns_from_ast(where_ast)
    predicate_table = _read_cpp_numpy_table(
        path=path,
        hdu=hdu,
        columns=where_columns,
        row_slice=_row_slice_from_start_num(start_row, num_rows),
        rows=None,
        where=None,
        mmap=mmap,
        decode_bytes=True,
        encoding="utf-8",
        strip=True,
        include_fits_metadata=False,
        apply_fits_nulls=apply_fits_nulls,
    )
    if predicate_table is None:
        return None
    if predicate_table.num_rows == 0:
        return []

    import pyarrow.compute as pc

    mask = _where_mask_for_table(predicate_table, where, parsed_ast=where_ast)
    if len(mask) == 0 or pc.sum(mask).as_py() == 0:
        return []

    base_row0 = start_row - 1
    selected = pc.indices_nonzero(mask).to_numpy()
    if selected.size == 0:
        return []
    return (selected + base_row0).tolist()


def scan(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
) -> Iterator[Any]:
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    validate_table_backend(backend)

    if where is not None:
        table = read(
            path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            where=where,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend=backend,
        )
        for batch in table.to_batches(max_chunksize=batch_size):
            yield batch
        return

    import torchfits

    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        return
    selected = set(columns) if columns else None
    col_tforms = (
        _column_tforms_for_decode(path, hdu, selected) if decode_bytes else None
    )
    unsigned_dtypes = _unsigned_column_dtypes(path, hdu, selected)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {}
    need_field_meta = include_fits_metadata or apply_fits_nulls
    if need_field_meta:
        try:
            field_meta, table_meta = _build_fits_metadata(path, hdu, selected)
        except Exception:
            field_meta, table_meta = {}, {}
    if columns:
        preferred_order = columns[:]
    elif field_meta:
        preferred_order = list(field_meta.keys())
    else:
        preferred_order = None

    chunk_iter = None
    if backend in {"auto", "cpp_numpy"}:
        chunk_iter = _iter_chunks_cpp_numpy(
            path, hdu, columns, start_row, num_rows, batch_size, mmap
        )
    if chunk_iter is None or backend == "torch":
        chunk_iter = torchfits.stream_table(
            path,
            hdu=hdu,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            chunk_rows=batch_size,
            mmap=mmap,
        )

    for chunk in chunk_iter:
        yield _chunk_to_record_batch(
            chunk,
            decode_bytes,
            encoding,
            strip,
            field_meta=field_meta if include_fits_metadata else None,
            table_meta=table_meta if include_fits_metadata else None,
            preferred_order=preferred_order,
            null_meta=field_meta,
            apply_fits_nulls=apply_fits_nulls,
            column_tforms=col_tforms,
            unsigned_dtypes=unsigned_dtypes,
        )


def read(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    rows: Optional[list[int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
):
    validate_table_backend(backend)
    pa = _require_pyarrow()
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]

    if backend in {"auto", "cpp_numpy"} and not should_skip_cpp_numpy_for_where(
        backend, where
    ):
        single = _read_cpp_numpy_table(
            path=path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=where,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
        )
        if single is not None:
            return single

    if where is not None:
        return _read_table_with_where(
            pa=pa,
            path=path,
            hdu=hdu,
            columns=columns,
            row_slice=row_slice,
            rows=rows,
            where=where,
            batch_size=batch_size,
            mmap=mmap,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            include_fits_metadata=include_fits_metadata,
            apply_fits_nulls=apply_fits_nulls,
            backend=backend,
        )

    return _read_table_from_scan_batches(
        path=path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=backend,
    )


def schema(
    path: str,
    hdu: int | str = 1,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = False,
    apply_fits_nulls: bool = False,
    backend: str = "auto",
):
    pa = _require_pyarrow()
    validate_table_backend(backend)
    if isinstance(hdu, str):
        hdu = _resolve_table_hdu_index_and_columns(path, hdu)[0]
    scan_backend = backend
    iterator = scan(
        path,
        hdu=hdu,
        columns=columns,
        where=where,
        batch_size=1,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=scan_backend,
    )
    first = next(iterator, None)
    if first is None:
        return pa.schema([])
    return first.schema


def _read_cpp_numpy_table(
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    where: Optional[str],
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
):
    import numpy as np
    import torchfits._C as cpp

    has_numpy_row_api = hasattr(
        cpp, "read_fits_table_rows_numpy_from_handle"
    ) or hasattr(cpp, "read_fits_table_rows_numpy")
    has_torch_table_api = hasattr(cpp, "read_fits_table")
    if not has_numpy_row_api and not has_torch_table_api:
        return None

    if rows is not None and row_slice is not None:
        raise ValueError("Only one of rows or row_slice may be provided")

    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        pa = _require_pyarrow()
        return pa.table({})

    if where is not None:
        where_rows = _resolve_rows_from_where_cpp(
            path=path,
            hdu=hdu,
            where=where,
            start_row=start_row,
            num_rows=num_rows,
            mmap=mmap,
            apply_fits_nulls=apply_fits_nulls,
        )
        if where_rows is None:
            return None
        if rows is not None:
            where_set = set(where_rows)
            rows = [int(r) for r in rows if int(r) in where_set]
        else:
            rows = where_rows
        start_row = 1
        num_rows = -1

    selected = set(columns) if columns else None
    col_tforms = (
        _column_tforms_for_decode(path, hdu, selected) if decode_bytes else None
    )
    unsigned_dtypes = _unsigned_column_dtypes(path, hdu, selected)
    field_meta: dict[str, dict[str, str]] = {}
    table_meta: dict[str, str] = {}
    need_field_meta = include_fits_metadata or apply_fits_nulls
    if need_field_meta:
        try:
            field_meta, table_meta = _build_fits_metadata(path, hdu, selected)
        except Exception:
            pass
    if columns:
        preferred_order = columns[:]
    elif field_meta:
        preferred_order = list(field_meta.keys())
    else:
        preferred_order = None

    col_list = columns if columns else []

    def _read_ranges_as_chunk(reader, ranges: list[tuple[int, int]]):
        out_sorted: dict[str, Any] = {}
        n_total = sum(length for _, length in ranges)

        cursor = 0
        for start0, length in ranges:
            seg = reader.read_rows_numpy(col_list, start0 + 1, length)
            if not seg:
                cursor += length
                continue
            for name, value in seg.items():
                buf = out_sorted.get(name)
                if buf is None:
                    if isinstance(value, np.ndarray):
                        buf = np.empty((n_total,) + value.shape[1:], dtype=value.dtype)
                    elif isinstance(value, list):
                        buf = [None] * n_total
                    elif _is_vla_tuple(value):
                        buf = [None] * n_total
                    else:
                        buf = [None] * n_total
                    out_sorted[name] = buf

                if isinstance(value, np.ndarray):
                    buf[cursor : cursor + length] = value
                elif isinstance(value, list):
                    buf[cursor : cursor + length] = value
                elif _is_vla_tuple(value):
                    fixed, offsets = value
                    fixed = np.asarray(fixed)
                    offsets = np.asarray(offsets)
                    items = []
                    for i in range(length):
                        a = int(offsets[i])
                        b = int(offsets[i + 1])
                        items.append(fixed[a:b])
                    buf[cursor : cursor + length] = items
                else:
                    buf[cursor : cursor + length] = [value] * length
            cursor += length
        return out_sorted

    chunk = None
    prefer_torch_full_path = (
        start_row == 1
        and num_rows == -1
        and not decode_bytes
        and not include_fits_metadata
        and not apply_fits_nulls
        and _can_use_torch_table_path_for_full_read(path, hdu, columns)
    )
    if prefer_torch_full_path and has_torch_table_api:
        if mmap and _can_use_mmap_row_path_for_full_read(path, hdu, columns):
            try:
                chunk = cpp.read_fits_table(path, hdu, col_list, True)
            except Exception:
                chunk = None
        if chunk is None:
            try:
                if not col_list and hasattr(cpp, "read_fits_table_from_handle"):
                    file_handle = _acquire_cpp_handle(path, cpp)
                    chunk = cpp.read_fits_table_from_handle(file_handle, hdu)
                else:
                    chunk = cpp.read_fits_table(path, hdu, col_list, False)
            except Exception:
                chunk = None

    if chunk is None and rows is not None:
        if not hasattr(cpp, "TableReader") or not hasattr(
            cpp.TableReader, "read_rows_numpy"
        ):
            return None
        rows_arr = np.asarray(rows, dtype=np.int64)
        if rows_arr.size == 0:
            pa = _require_pyarrow()
            return pa.table({})
        if np.any(rows_arr < 0):
            raise ValueError("rows must be non-negative (0-based)")

        order = np.argsort(rows_arr, kind="stable")
        sorted_rows = rows_arr[order]

        if len(sorted_rows) == 0:
            ranges: list[tuple[int, int]] = []
        else:
            diffs = np.diff(sorted_rows)
            breaks = np.nonzero(diffs != 1)[0]
            start_indices = np.insert(breaks + 1, 0, 0)
            end_indices = np.append(breaks, len(sorted_rows) - 1)

            start0s = sorted_rows[start_indices]
            lengths = end_indices - start_indices + 1

            ranges = list(zip(start0s.tolist(), lengths.tolist()))

        try:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk_sorted = _read_ranges_as_chunk(reader, ranges)
        except Exception:
            chunk_sorted = None
        if chunk_sorted is None:
            return None

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        chunk = {}
        for name, value in chunk_sorted.items():
            if isinstance(value, np.ndarray):
                chunk[name] = value[inv]
            elif isinstance(value, list):
                chunk[name] = [value[i] for i in inv]
            else:
                chunk[name] = value

    if (
        chunk is None
        and hasattr(cpp, "TableReader")
        and hasattr(cpp.TableReader, "read_rows_numpy")
    ):
        try:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk = reader.read_rows_numpy(col_list, start_row, num_rows)
        except Exception:
            chunk = None
    if chunk is None and hasattr(cpp, "read_fits_table_rows_numpy_from_handle"):
        try:
            file_handle = _acquire_cpp_handle(path, cpp)
            chunk = cpp.read_fits_table_rows_numpy_from_handle(
                file_handle, hdu, col_list, start_row, num_rows
            )
        except Exception:
            chunk = None
    if chunk is None and hasattr(cpp, "read_fits_table_rows_numpy"):
        try:
            chunk = cpp.read_fits_table_rows_numpy(
                path, hdu, col_list, start_row, num_rows, False
            )
        except Exception:
            chunk = None
    if chunk is None:
        return None

    pa = _require_pyarrow()
    if not chunk:
        return pa.table({})

    if not field_meta and not table_meta:
        arrays: list[Any] = []
        names_out: list[str] = []
        names = preferred_order[:] if preferred_order else list(chunk.keys())
        for name in names:
            if name not in chunk:
                continue
            value = chunk[name]
            null_sentinel = (
                _column_tnull_from_meta(field_meta, name) if apply_fits_nulls else None
            )
            if isinstance(value, np.ndarray):
                arr = _numpy_to_arrow_array(
                    pa,
                    value,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=col_tforms.get(name) if col_tforms else None,
                    unsigned_dtype=unsigned_dtypes.get(name),
                )
            elif isinstance(value, torch.Tensor):
                t = value.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                arr = _numpy_to_arrow_array(
                    pa,
                    t.numpy(),
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=null_sentinel,
                    fits_tform=col_tforms.get(name) if col_tforms else None,
                    unsigned_dtype=unsigned_dtypes.get(name),
                )
            elif isinstance(value, list):
                converted = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        if not t.is_contiguous():
                            t = t.contiguous()
                        converted.append(t.numpy())
                    else:
                        converted.append(item)
                arr = _pa_array(pa, converted)
            elif _is_vla_tuple(value):
                arr = _vla_tuple_to_arrow_array(pa, value, null_sentinel=null_sentinel)
            else:
                arr = _pa_array(pa, value)
            names_out.append(name)
            arrays.append(arr)
        if not arrays:
            return pa.table({})
        return pa.Table.from_arrays(arrays, names=names_out)

    batch = _chunk_to_record_batch(
        chunk,
        decode_bytes,
        encoding,
        strip,
        field_meta=field_meta if include_fits_metadata else None,
        table_meta=table_meta if include_fits_metadata else None,
        preferred_order=preferred_order,
        null_meta=field_meta,
        apply_fits_nulls=apply_fits_nulls,
        column_tforms=col_tforms,
        unsigned_dtypes=unsigned_dtypes,
    )
    return pa.Table.from_batches([batch])


def scan_torch(
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    device: str = "cpu",
    non_blocking: bool = True,
    pin_memory: bool = False,
) -> Iterator[dict[str, Any]]:
    import torchfits

    start_row, num_rows = _normalize_row_slice(row_slice)
    use_mmap = mmap
    if use_mmap:
        use_mmap = _can_use_mmap_row_path_for_full_read(path, hdu, columns)

    for chunk in torchfits.stream_table(
        path,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        chunk_rows=batch_size,
        mmap=use_mmap,
    ):
        if device == "cpu":
            yield chunk
            continue

        moved: dict[str, Any] = {}
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                t = value
                if pin_memory and t.device.type == "cpu":
                    t = t.pin_memory()
                if device == "mps" and t.dtype == torch.float64:
                    t = t.float()
                moved[key] = t.to(device, non_blocking=non_blocking)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        t = item
                        if pin_memory and t.device.type == "cpu":
                            t = t.pin_memory()
                        if device == "mps" and t.dtype == torch.float64:
                            t = t.float()
                        new_list.append(t.to(device, non_blocking=non_blocking))
                    else:
                        new_list.append(item)
                moved[key] = new_list
            else:
                moved[key] = value
        yield moved


def reader(
    path: str,
    hdu: int = 1,
    columns: Optional[list[str]] = None,
    row_slice: Optional[slice | tuple[int, int]] = None,
    where: Optional[str] = None,
    batch_size: int = 65536,
    mmap: bool = True,
    decode_bytes: bool = True,
    encoding: str = "ascii",
    strip: bool = True,
    include_fits_metadata: bool = True,
    apply_fits_nulls: bool = True,
    backend: str = "auto",
):
    pa = _require_pyarrow()
    validate_table_backend(backend)
    scan_backend = backend
    batches = scan(
        path,
        hdu=hdu,
        columns=columns,
        row_slice=row_slice,
        where=where,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=scan_backend,
    )
    it = iter(batches)
    first = next(it, None)
    if first is None:
        return pa.RecordBatchReader.from_batches(pa.schema([]), [])
    return pa.RecordBatchReader.from_batches(first.schema, itertools.chain([first], it))


def dataset(
    data: str | Any,
    **kwargs,
):
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("pyarrow.dataset is required for dataset conversion") from exc

    if isinstance(data, str):
        return ds.dataset(reader(data, **kwargs))
    return ds.dataset(data)


def scanner(
    data: str | Any,
    *,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    filter: Any = None,
    batch_size: int = 65536,
    use_threads: bool = True,
    **kwargs,
):
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("pyarrow.dataset is required for scanner") from exc

    if where is not None:
        kwargs = dict(kwargs)
        kwargs["where"] = where

    if isinstance(data, str):
        dset = dataset(data, **kwargs)
    elif hasattr(data, "scanner"):
        dset = data
    else:
        dset = ds.dataset(data)
    return dset.scanner(
        columns=columns, filter=filter, batch_size=batch_size, use_threads=use_threads
    )
