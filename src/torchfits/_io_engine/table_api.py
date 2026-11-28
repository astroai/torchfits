"""Table-shaped wrappers over unified FITS reads."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from ._hdu_lookup import raise_hdu_index_error
from .paths import coerce_fits_path
from .hdu_api import _resolve_hdu_index, autodetect_hdu
from .device import move_table_columns


def _resolve_mmap(mmap: Union[bool, str]) -> bool:
    if isinstance(mmap, bool):
        return mmap
    if isinstance(mmap, str) and mmap.strip().lower() == "auto":
        # "auto" → prefer mmap for tables (CFITSIO column mmap path).
        return True
    raise ValueError("mmap must be bool or 'auto'")


def _move_table_dict(
    data: dict[str, Any], device: str | torch.device
) -> dict[str, Any]:
    return move_table_columns(data, device)


def _apply_row_window(
    data: dict[str, Any], start_row: int, num_rows: int
) -> dict[str, Any]:
    """Apply 1-based FITS row window after a full-table filter."""
    if int(start_row) <= 1 and int(num_rows) < 0:
        return data
    start0 = max(0, int(start_row) - 1)
    stop = None if int(num_rows) < 0 else start0 + int(num_rows)
    out: dict[str, Any] = {}
    for key, value in data.items():
        # Slice every row-aligned payload (tensors *and* VLA/string lists) so
        # a windowed read can never return columns with mismatched lengths.
        if isinstance(value, torch.Tensor):
            out[key] = value[start0:stop]
        elif isinstance(value, list):
            out[key] = value[start0:stop]
        else:
            out[key] = value
    return out


def _squeeze_scalar_columns(data: Any) -> Any:
    # Returns Any by design; callers re-narrow.
    """Normalize FITS scalar columns to rank-1 ``(N,)`` tensors.

    The C++ readers surface repeat==1 table columns as ``(N, 1)`` (FITS
    stores the row-major width); every Python reader boundary normalizes
    them to ``(N,)`` so column shapes never depend on which internal
    binding served the read. Genuine vector columns keep their shape.
    """
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() == 2
                and value.shape[1] == 1
                and value.dtype != torch.uint8
            ):
                # uint8 2D columns are packed FITS strings — never squeezed.
                out[key] = value.reshape(-1)
            else:
                out[key] = value
        return out
    if (
        isinstance(data, torch.Tensor)
        and data.dim() == 2
        and data.shape[1] == 1
        and data.dtype != torch.uint8
    ):
        return data.reshape(-1)
    return data


def _thin_read_table_torch(
    path: str,
    *,
    hdu: int,
    columns: Optional[list[str]],
    start_row: int,
    num_rows: int,
    device: str,
    mmap: Union[bool, str],
) -> dict[str, Any]:
    """CFITSIO → torch dict without going through read_unified image probes."""
    import numpy as np

    import torchfits._C as cpp

    col_names = list(columns) if columns is not None else []
    use_mmap = _resolve_mmap(mmap)
    try:
        data = cpp.read_fits_table_rows(
            path, int(hdu), col_names, int(start_row), int(num_rows), bool(use_mmap)
        )
    except (RuntimeError, OSError, ValueError):
        if not use_mmap:
            raise
        # mmap cannot serve every table (e.g. TSCAL/TZERO-scaled columns).
        # Retry on the buffered path, which applies FITS scaling, so scaled
        # tables read (and where= masks compare) physical values instead of
        # raising.
        data = cpp.read_fits_table_rows(
            path, int(hdu), col_names, int(start_row), int(num_rows), False
        )
        # The buffered binding returns numpy columns; read_torch's contract
        # is torch.Tensor columns (matching the mmap path).
        data = {
            k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }
    result: dict[str, Any] = _squeeze_scalar_columns(
        _move_table_dict(dict(data), device)
    )
    return result


def _thin_read_table_filtered(
    path: str,
    *,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    device: str,
    compile_predicates: Callable[[str], Any],
) -> dict[str, Any] | None:
    """Fused project+predicate via cpp.read_fits_table_filtered (gather path).

    Prefer the project+mask path in :func:`read_table` for typical keep rates;
    this remains as a fallback when the thin full-column read fails.
    """
    import torchfits._C as cpp

    filters = compile_predicates(where)
    if filters is None:
        return None
    target_cols = list(columns) if columns is not None else []
    if not target_cols:
        # Filtered binding needs explicit columns; pull colnames skinny.
        from .hdu_api import read_colnames

        target_cols = read_colnames(path, hdu=hdu)
    try:
        data = cpp.read_fits_table_filtered(path, int(hdu), target_cols, list(filters))
    except (RuntimeError, ValueError, TypeError) as exc:
        # Binding-level failure (bad predicate form, read error surfaced as
        # RuntimeError): keep the established error contract but preserve the
        # cause instead of degrading it to a bare "fallback unavailable" None.
        raise RuntimeError(
            f"Failed to apply where={where!r} on table HDU {hdu!r}"
        ) from exc
    result: dict[str, Any] = _squeeze_scalar_columns(
        _move_table_dict(dict(data), device)
    )
    return result


def _filter_where_rows(
    data: dict[str, Any],
    predicates: Any,
    tnull_map: dict[str, Any],
    columns: Optional[list[str]],
) -> dict[str, Any]:
    """Rows of ``data`` matching every simple predicate (read_torch where=).

    TNULL cells never match. Only ``columns`` are kept (every column when
    None), in ``data``'s order.
    """
    import numpy as np

    from torchfits._table._read_where import _not_null, _torch_cmp_mask

    mask: torch.Tensor | None = None
    for col, op, lit in predicates:
        values = data[col]
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        if values.dim() != 1:
            raise ValueError(f"where does not support fixed-size vector column '{col}'")
        try:
            part = _torch_cmp_mask(values, op, lit)
        except TypeError as exc:
            raise ValueError(
                f"where cannot compare column '{col}' ({values.dtype}) with "
                f"{lit!r} using '{op}': {exc}"
            ) from exc
        sentinel = tnull_map.get(col)
        if sentinel is not None:
            part = part & _not_null(values, sentinel)
        mask = part if mask is None else (mask & part)
    assert mask is not None
    keep_cols = list(columns) if columns is not None else list(data.keys())
    filtered: dict[str, Any] = {}
    # List payloads (VLA / strings) have no boolean index. Build the row
    # list only when one is kept. Doing it for a tensor column threw the
    # list away, and on a dense 1e5-row filter that list cost more than
    # the column read.
    keep_idx: list[int] | None = None
    if any(isinstance(data.get(k), list) for k in keep_cols):
        keep_idx = mask.nonzero().flatten().tolist()
    for k, v in data.items():
        if k not in keep_cols:
            continue
        if isinstance(v, torch.Tensor):
            filtered[k] = v[mask]
        elif isinstance(v, np.ndarray):
            # Buffered reads return numpy columns; gather with the torch
            # mask (numpy fancy indexing is several× slower for the same
            # work) and keep read_torch's tensor contract.
            filtered[k] = torch.as_tensor(v)[mask]
        elif isinstance(v, list):
            assert keep_idx is not None
            filtered[k] = [v[i] for i in keep_idx]
        else:
            filtered[k] = v
    return filtered


# read_torch(where=) evaluates the predicate one row window at a time and keeps
# only the matching rows, so its peak memory is the result plus one window
# rather than every read column plus a whole-table mask (2x a 2 GB column).
_WHERE_WINDOW_BYTES = 64 << 20
_WHERE_WINDOW_MIN_ROWS = 1 << 16
_WHERE_WINDOW_MAX_ROWS = 1 << 22


def _concat_row_parts(parts: list[Any]) -> Any:
    """Join per-window results; frees each part once it is copied."""
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if isinstance(first, torch.Tensor):
        total = sum(int(part.shape[0]) for part in parts)
        out = torch.empty((total, *first.shape[1:]), dtype=first.dtype)
        pos = 0
        for i, part in enumerate(parts):
            parts[i] = None
            out[pos : pos + part.shape[0]] = part
            pos += int(part.shape[0])
            del part
        return out
    if isinstance(first, list):
        return [value for part in parts for value in part]
    return parts[-1]


def _read_where_windowed(
    path: str,
    *,
    hdu: int,
    columns: Optional[list[str]],
    read_cols: Optional[list[str]],
    predicates: Any,
    tnull_map: dict[str, Any],
    header: Any,
    start_row: int,
    num_rows: int,
    mmap: Union[bool, str],
) -> dict[str, Any] | None:
    """read_torch(where=) over row windows; None when a window read fails.

    Each window is read and filtered exactly like the whole-table read (same
    reader, same predicate evaluation), so the result is identical; None hands
    the request to that path, which owns the fallbacks and error messages.
    """
    try:
        total = int(header.get("NAXIS2", 0))
        row_bytes = int(header.get("NAXIS1", 0))
    except (TypeError, ValueError):
        return None
    first = int(start_row)
    last = total if int(num_rows) < 0 else min(total, first + int(num_rows) - 1)
    if first > last:
        return None
    window = _WHERE_WINDOW_BYTES // max(row_bytes, 1)
    window = max(_WHERE_WINDOW_MIN_ROWS, min(_WHERE_WINDOW_MAX_ROWS, window))
    parts: dict[str, list[Any]] = {}
    row = first
    while row <= last:
        size = min(window, last - row + 1)
        try:
            chunk = _thin_read_table_torch(
                path,
                hdu=hdu,
                columns=read_cols or None,
                start_row=row,
                num_rows=size,
                device="cpu",
                mmap=mmap,
            )
        except (RuntimeError, OSError, ValueError, TypeError, MemoryError):
            return None
        kept = _filter_where_rows(chunk, predicates, tnull_map, columns)
        del chunk
        for name, value in kept.items():
            parts.setdefault(name, []).append(value)
        del kept
        row += size
    return {name: _concat_row_parts(values) for name, values in parts.items()}


def read_table(
    read_func: Callable[..., Any],
    path: str,
    hdu: Union[int, str] = 1,
    columns: Optional[list[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    cache_capacity: int = 10,
    fast_header: bool = True,
    return_header: bool = False,
    where: str | None = None,
) -> Any:
    """Read a table HDU as a dictionary of tensors/lists.

    Thin path (default): C++ ``read_fits_table_rows`` / filtered binding.
    Falls back to ``read_func`` (``torchfits.read``) when header is requested
    or the thin path fails. ``hdu`` may be an index or EXTNAME.
    """
    _ = cache_capacity, fast_header
    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        raise ValueError(
            "hdu must be a non-negative integer or EXTNAME string "
            "(not None/'auto'); pass an explicit table HDU"
        )
    if isinstance(hdu, int):
        if hdu < 0:
            raise_hdu_index_error(coerce_fits_path(path), hdu)
    elif isinstance(hdu, str):
        hdu = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    else:
        raise TypeError(f"hdu must be int or str, got {type(hdu)!r}")

    # Validate every per-request parameter *before* the where= branch. That
    # branch wraps its thin read in a broad ``except``, so a parameter
    # ValueError was caught and re-read as "thin path unavailable"; the
    # fallback then re-derived the answer by a different route and returned
    # rows for a request the non-where path rejects (r2-018).
    if int(start_row) < 1:
        raise ValueError("start_row must be >= 1 (FITS uses 1-based indexing)")
    if int(num_rows) < -1:
        raise ValueError("num_rows must be >= 0, or -1 for all rows")
    _resolve_mmap(mmap)  # raises on an unknown mmap mode; the value is unused

    if where is not None and str(where).strip():
        from torchfits._table._read_where import _compile_where_to_simple_predicates

        predicates = _compile_where_to_simple_predicates(str(where))
        if predicates is None:
            raise ValueError(f"Unsupported where expression for read_torch: {where!r}")

        pred_cols = [col for col, _op, _lit in predicates]
        # columns=None means "return every column" (same contract as the Arrow
        # engine): read the full projection instead of just the predicate
        # columns, which used to silently drop all non-predicate output.
        read_cols: Optional[list[str]] = list(columns) if columns is not None else None
        if read_cols is not None:
            for name in pred_cols:
                if name not in read_cols:
                    read_cols.append(name)

        # Filter-within-window contract (matches read(row_slice=..., where=...)
        # and the Arrow engine): clamp the caller's row window FIRST, then
        # evaluate predicates inside it. The old post-filter window returned
        # a different population for the same query depending on API shape.
        window_requested = start_row > 1 or num_rows != -1

        import torchfits
        from torchfits.fits_schema import column_null_sentinel_map

        header = torchfits.read_header(path, hdu=hdu)
        tnull_map = column_null_sentinel_map(header)
        data = _read_where_windowed(
            path,
            hdu=hdu,
            columns=columns,
            read_cols=read_cols,
            predicates=predicates,
            tnull_map=tnull_map,
            header=header,
            start_row=start_row,
            num_rows=num_rows,
            mmap=mmap,
        )
        if data is not None:
            data = _move_table_dict(data, device)
            if return_header:
                return data, header
            return data

        data = None
        if window_requested:
            try:
                data = _thin_read_table_torch(
                    path,
                    hdu=hdu,
                    columns=read_cols or None,
                    start_row=start_row,
                    num_rows=num_rows,
                    device="cpu",
                    mmap=mmap,
                )
            except (RuntimeError, OSError, ValueError, TypeError, MemoryError):
                data = None
            if data is None:
                try:
                    data = _thin_read_table_torch(
                        path,
                        hdu=hdu,
                        columns=read_cols or None,
                        start_row=1,
                        num_rows=-1,
                        device="cpu",
                        mmap=mmap,
                    )
                except (RuntimeError, OSError, ValueError, TypeError, MemoryError):
                    data = None
                if data is not None:
                    data = _apply_row_window(data, start_row, num_rows)
            if (
                data is None
                and columns is not None
                and read_cols is not None
                and len(read_cols) > len(columns)
            ):
                # Retry with the full projection: the failure may come from a
                # predicate-only column rather than the requested window.
                try:
                    data = _thin_read_table_torch(
                        path,
                        hdu=hdu,
                        columns=None,
                        start_row=1,
                        num_rows=-1,
                        device="cpu",
                        mmap=mmap,
                    )
                except (RuntimeError, OSError, ValueError, TypeError, MemoryError):
                    data = None
                if data is not None:
                    data = _apply_row_window(data, start_row, num_rows)
        else:
            # Prefer project-all-needed-cols + torch mask over ``read_fits_table_filtered``.
            # Filtered gather wins only when keep-rate is tiny *and* row assembly is
            # heavy; for narrow/dense predicates (the Round-3 deficit cluster) mask
            # is several× faster and also beats Astropy's numpy mask path.
            try:
                data = _thin_read_table_torch(
                    path,
                    hdu=hdu,
                    columns=read_cols or None,
                    start_row=1,
                    num_rows=-1,
                    device="cpu",
                    mmap=mmap,
                )
            except (RuntimeError, OSError, ValueError, TypeError, MemoryError):
                data = None

        if data is None:
            if window_requested:
                raise RuntimeError(
                    f"Failed to read rows {start_row}..{start_row + max(num_rows, 0) - 1} "
                    f"for where={where!r} on table HDU {hdu!r}"
                )
            pushed = _thin_read_table_filtered(
                path,
                hdu=hdu,
                columns=columns,
                where=str(where),
                device=device,
                compile_predicates=_compile_where_to_simple_predicates,
            )
            if pushed is None:
                raise RuntimeError(
                    f"Failed to apply where={where!r} on table HDU {hdu!r}"
                )
            if return_header:
                import torchfits

                return pushed, torchfits.read_header(path, hdu=hdu)
            return pushed

        data = _move_table_dict(
            _filter_where_rows(data, predicates, tnull_map, columns), device
        )
        if return_header:
            import torchfits

            return data, torchfits.read_header(path, hdu=hdu)
        return data

    if not return_header:
        try:
            data = _thin_read_table_torch(
                path,
                hdu=hdu,
                columns=columns,
                start_row=start_row,
                num_rows=num_rows,
                device=device,
                mmap=mmap,
            )
            if isinstance(data, torch.Tensor):
                raise ValueError(
                    f"HDU {hdu!r} is an image HDU. Use read_tensor(...) or read(...)."
                )
            return data
        except ValueError:
            raise
        except (RuntimeError, OSError, TypeError, MemoryError):
            # Expected read/decode failures fall back to the unified reader
            # (documented above); anything else is a bug and must surface.
            pass

    out = read_func(
        path=path,
        hdu=hdu,
        mode="table",
        device=device,
        mmap=mmap,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
    )
    data = out[0] if return_header else out
    if isinstance(data, torch.Tensor):
        raise ValueError(
            f"HDU {hdu!r} is an image HDU. Use read_tensor(...) or read(...)."
        )
    return out
