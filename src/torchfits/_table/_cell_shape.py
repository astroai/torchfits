"""N-D table cells: ``TDIMn`` on read (``shape_cells=True``) and on write.

FITS ``TDIMn = '(a,b,c)'`` describes a cell of ``a*b*c`` elements in which
``a`` varies fastest (Fortran order). In C order -- numpy, torch, Arrow's
nested lists -- that cell has shape ``(c, b, a)``. A column of such cells
reads as a tensor of shape ``(nrows, c, b, a)`` / an Arrow
``FixedSizeList<FixedSizeList<FixedSizeList<T>[a]>[b]>[c]``, and an array of
shape ``(nrows, c, b, a)`` is written as ``TFORM = 'a*b*c<code>'`` with
``TDIM = '(a,b,c)'``.

Only numeric, logical and bit columns are shaped. A string (``A``) column's
TDIM describes an array of fixed-width strings; it is kept as metadata and
the column still reads as one string per row.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional

from .. import fits_schema

_TDIM_RE = re.compile(r"^\s*\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)\s*$")


def parse_tdim(tdim: Any) -> Optional[tuple[int, ...]]:
    """``'(a,b,c)'`` -> ``(a, b, c)`` (FITS order), or None when malformed."""
    if tdim is None:
        return None
    match = _TDIM_RE.match(str(tdim))
    if match is None:
        return None
    dims = tuple(int(part) for part in match.group(1).split(","))
    if any(dim <= 0 for dim in dims):
        return None
    return dims


def tdim_for_cell_shape(shape: tuple[int, ...]) -> str:
    """C-order cell shape ``(c, b, a)`` -> ``'(a,b,c)'``."""
    return "(" + ",".join(str(int(dim)) for dim in reversed(shape)) + ")"


def cell_shape_from_tdim(tdim: Any, repeat: int) -> Optional[tuple[int, ...]]:
    """The C-order cell shape a ``TDIM`` gives a column of ``repeat`` elements.

    None when the TDIM is missing or malformed, has fewer than two axes (the
    cell is already a vector) or does not describe the whole cell.
    """
    dims = parse_tdim(tdim)
    if dims is None or len(dims) < 2 or math.prod(dims) != int(repeat):
        return None
    return tuple(reversed(dims))


_SHAPED_CODES = frozenset({"L", "X", "B", "I", "J", "K", "E", "D", "C", "M"})


def header_cell_shapes(
    header: Mapping[str, Any], selected: Optional[set[str]] = None
) -> dict[str, tuple[int, ...]]:
    """``{column: C-order cell shape}`` of the header's shapeable TDIM columns."""
    out: dict[str, tuple[int, ...]] = {}
    for col in fits_schema.iter_table_columns(header, selected=selected):
        info = col.tform_info
        if info.vla or info.code not in _SHAPED_CODES or not col.tdim:
            continue
        shape = cell_shape_from_tdim(col.tdim, info.repeat)
        if shape is not None:
            out[col.name] = shape
    return out


# -- read side: Arrow -------------------------------------------------------------


def _nested_type(pa: Any, value_type: Any, shape: tuple[int, ...]) -> Any:
    out = value_type
    for dim in reversed(shape):
        out = pa.list_(out, int(dim))
    return out


def _shape_array(pa: Any, arr: Any, shape: tuple[int, ...]) -> Any:
    """A flat ``FixedSizeList<T>[prod(shape)]`` array as nested fixed-size lists."""
    size = math.prod(shape)
    n = len(arr)
    if arr.null_count:
        values = arr.values.slice(arr.offset * size, n * size)
        mask = arr.is_null()
    else:
        values = arr.flatten()
        mask = None
    for dim in reversed(shape[1:]):
        values = pa.FixedSizeListArray.from_arrays(values, int(dim))
    return pa.FixedSizeListArray.from_arrays(values, int(shape[0]), mask=mask)


def _shapeable(pa: Any, typ: Any, shape: tuple[int, ...]) -> bool:
    return (
        pa.types.is_fixed_size_list(typ)
        and typ.list_size == math.prod(shape)
        and not pa.types.is_fixed_size_list(typ.value_type)
    )


def shape_arrow_schema(
    pa: Any, schema: Any, shapes: Mapping[str, tuple[int, ...]]
) -> Any:
    if not shapes:
        return schema
    fields = []
    for field in schema:
        shape = shapes.get(field.name)
        if shape is not None and _shapeable(pa, field.type, shape):
            field = field.with_type(_nested_type(pa, field.type.value_type, shape))
        fields.append(field)
    return pa.schema(fields, metadata=schema.metadata)


def shape_arrow_data(pa: Any, data: Any, shapes: Mapping[str, tuple[int, ...]]) -> Any:
    """Nest the TDIM columns of a ``pyarrow.Table`` or ``RecordBatch``."""
    if not shapes or data is None:
        return data
    is_batch = isinstance(data, pa.RecordBatch)
    columns = []
    changed = False
    for i, field in enumerate(data.schema):
        col = data.column(i)
        shape = shapes.get(field.name)
        if shape is not None and _shapeable(pa, field.type, shape):
            if isinstance(col, pa.ChunkedArray):
                col = pa.chunked_array(
                    [_shape_array(pa, chunk, shape) for chunk in col.chunks],
                    type=_nested_type(pa, field.type.value_type, shape),
                )
            else:
                col = _shape_array(pa, col, shape)
            changed = True
        columns.append(col)
    if not changed:
        return data
    schema = shape_arrow_schema(pa, data.schema, shapes)
    if is_batch:
        return pa.RecordBatch.from_arrays(columns, schema=schema)
    return pa.Table.from_arrays(columns, schema=schema)


# -- read side: torch -------------------------------------------------------------


def shape_torch_columns(data: Any, shapes: Mapping[str, tuple[int, ...]]) -> Any:
    """Reshape ``(nrows, prod)`` tensors of a column dict to ``(nrows, *shape)``."""
    if not shapes or not isinstance(data, dict):
        return data
    out = dict(data)
    for name, shape in shapes.items():
        value = out.get(name)
        if value is None or not hasattr(value, "reshape") or not hasattr(value, "dim"):
            continue
        if value.dim() == 0:
            continue
        nrows = int(value.shape[0])
        if value.numel() != nrows * math.prod(shape):
            continue
        out[name] = value.reshape(nrows, *shape)
    return out


# -- write side -------------------------------------------------------------------


def _ndim(value: Any) -> int:
    ndim = getattr(value, "ndim", None)
    if ndim is None and hasattr(value, "dim"):
        ndim = value.dim()
    return int(ndim or 0)


def _is_string_array(value: Any) -> bool:
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    return kind in ("U", "S", "O")


def flatten_cells(name: str, value: Any, dim: Any = None) -> tuple[Any, Optional[str]]:
    """``(nrows, *cell)`` -> ``((nrows, prod(cell)), TDIM)`` for N-D cells.

    Values with at most two axes are returned unchanged (TDIM ``None``). A
    ``dim`` given by the caller must describe the same cell, else
    ``ValueError``.
    """
    if _ndim(value) <= 2 or _is_string_array(value):
        return value, None
    shape = tuple(int(d) for d in value.shape[1:])
    tdim = tdim_for_cell_shape(shape)
    if dim is not None:
        given = parse_tdim(dim)
        if given is None or tuple(reversed(given)) != shape:
            raise ValueError(
                f"Column '{name}' has cells of shape {shape} (TDIM {tdim!r}), "
                f"which dim={dim!r} does not describe"
            )
    nrows = int(value.shape[0])
    flat = value.reshape(nrows, math.prod(shape))
    return flat, tdim


def shape_cells_for_write(
    data: dict[str, Any], schema: Optional[dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], Optional[dict[str, dict[str, Any]]]]:
    """Flatten N-D cell columns of a write payload and record their TDIM."""
    out: Optional[dict[str, Any]] = None
    new_schema = schema
    for name, value in data.items():
        meta = (schema or {}).get(name) or {}
        given = meta.get("dim", meta.get("tdim"))
        flat, tdim = flatten_cells(str(name), value, given)
        if tdim is None:
            continue
        if out is None:
            out = dict(data)
            new_schema = {k: dict(v) for k, v in (schema or {}).items()}
        out[name] = flat
        entry = new_schema.setdefault(name, {})  # type: ignore[union-attr]
        if "dim" not in entry and "tdim" not in entry:
            entry["dim"] = tdim
    if out is None:
        return data, schema
    return out, new_schema


def arrow_cell_values(pa: Any, column: Any) -> Any:
    """A nested fixed-size-list or ``fixed_shape_tensor`` column as an N-D array.

    Returns None for any other column, or when the column has nulls (the
    caller's generic conversion handles those).
    """
    import numpy as np

    typ = column.type
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if isinstance(typ, pa.FixedShapeTensorType):
        if arr.null_count or (arr.storage.flatten().null_count):
            return None
        if typ.permutation is not None and list(typ.permutation) != sorted(
            typ.permutation
        ):
            return None
        return np.ascontiguousarray(arr.to_numpy_ndarray())
    if not (
        pa.types.is_fixed_size_list(typ) and pa.types.is_fixed_size_list(typ.value_type)
    ):
        return None
    shape: list[int] = []
    inner = typ
    while pa.types.is_fixed_size_list(inner):
        shape.append(int(inner.list_size))
        inner = inner.value_type
    if not (
        pa.types.is_integer(inner)
        or pa.types.is_floating(inner)
        or pa.types.is_boolean(inner)
    ):
        return None
    flat = arr
    for _ in shape:
        if flat.null_count:
            return None
        flat = flat.flatten()
    if flat.null_count:
        return None
    return flat.to_numpy(zero_copy_only=False).reshape(len(arr), *shape)


def arrow_unnest(pa: Any, column: Any) -> Any:
    """A nested fixed-size-list column as one flat ``FixedSizeList<T>[prod]``.

    Used when :func:`arrow_cell_values` declines (nulls), so the column takes
    the same conversion as a flat vector column would.
    """
    typ = column.type
    if not (
        pa.types.is_fixed_size_list(typ) and pa.types.is_fixed_size_list(typ.value_type)
    ):
        return column
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    size = 1
    values = arr
    while pa.types.is_fixed_size_list(values.type):
        size *= int(values.type.list_size)
        values = values.values
    start = arr.offset * size
    return pa.FixedSizeListArray.from_arrays(
        values.slice(start, len(arr) * size),
        size,
        mask=arr.is_null() if arr.null_count else None,
    )


def arrow_metadata_cells(value: Any, field: Any) -> Any:
    """Reshape a flat vector column by its ``fits_tdim`` field metadata.

    A table read with ``include_fits_metadata=True`` (and without
    ``shape_cells``) carries ``TDIM`` as metadata on a flat vector column;
    writing it back keeps that TDIM.
    """
    metadata = getattr(field, "metadata", None) if field is not None else None
    if not metadata or _ndim(value) != 2 or _is_string_array(value):
        return value
    raw = metadata.get(b"fits_tdim")
    if raw is None:
        return value
    shape = cell_shape_from_tdim(raw.decode("utf-8", "replace"), int(value.shape[1]))
    if shape is None:
        return value
    return value.reshape(int(value.shape[0]), *shape)
