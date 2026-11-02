"""Shared C++ table read dispatch engine.

Provides :func:`_read_ranges_as_chunk` — reads multiple row ranges from a
C++ TableReader and assembles them into a single torch-backed dict.
"""

from __future__ import annotations

from typing import Any, Optional

from .arrow_convert import (
    _is_raw_column,
    _is_torch_tensor,
    _is_vla_tuple,
    _raw_column_to_numpy,
)


def _segment_row_count(value: Any) -> Optional[int]:
    """Row count carried by one column segment, or None for scalar broadcast."""
    if _is_raw_column(value):
        return int(value["shape"][0])
    if _is_torch_tensor(value):
        return int(value.shape[0])
    if isinstance(value, list):
        return len(value)
    if _is_vla_tuple(value):
        return int(value[1].reshape(-1).shape[0]) - 1
    return None


def _read_raw_ranges_as_chunk(
    reader: Any,
    col_list: list[str],
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Assemble native raw ranges into owned NumPy columns for Arrow conversion."""
    import numpy as np

    n_total = sum(length for _, length in ranges)
    if n_total == 0:
        return {}

    expected: Optional[set[str]] = set(col_list) if col_list else None
    fixed_parts: dict[str, list[Any]] = {}
    vla_parts: dict[str, list[tuple[Any, Any]]] = {}

    for start0, length in ranges:
        if length <= 0:
            continue
        seg = reader.read_rows_raw(col_list, start0 + 1, length)
        row_lo, row_hi = start0, start0 + length
        if not seg:
            raise RuntimeError(
                f"empty row segment for rows [{row_lo}, {row_hi}): "
                "refusing to fabricate rows"
            )
        if expected is None:
            expected = set(seg.keys())
        missing = expected - set(seg.keys())
        if missing:
            raise RuntimeError(
                f"row segment for rows [{row_lo}, {row_hi}) is missing "
                f"columns {sorted(missing)}"
            )

        for name, value in seg.items():
            if not _is_raw_column(value):
                raise TypeError(
                    f"raw table reader returned {type(value).__name__} for {name!r}"
                )
            n_got = _segment_row_count(value)
            if n_got != length:
                raise RuntimeError(
                    f"short row segment for rows [{row_lo}, {row_hi}): "
                    f"column {name!r} returned {n_got} of {length} rows"
                )
            materialized = _raw_column_to_numpy(value)
            if value["kind"] == "vla":
                vla_parts.setdefault(name, []).append(materialized)
            else:
                fixed_parts.setdefault(name, []).append(materialized)

    out: dict[str, Any] = {}
    for name, parts in fixed_parts.items():
        # One range is the common case; concatenating a single part would copy
        # the native buffer for no reason.
        out[name] = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    for name, parts in vla_parts.items():
        if len(parts) == 1:
            out[name] = parts[0]
            continue
        values = np.concatenate([part[0] for part in parts], axis=0)
        offsets = [parts[0][1]]
        running = int(parts[0][1][-1])
        for _values, next_offsets in parts[1:]:
            base = running
            running += int(next_offsets[-1])
            offsets.append(np.asarray(next_offsets[1:], dtype=np.int64) + base)
        out[name] = (
            values,
            np.concatenate(offsets).astype(np.int64, copy=False),
        )
    return out


def _read_ranges_as_chunk(
    reader: Any,
    col_list: list[str],
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Read multiple row ranges from a TableReader and assemble into one chunk.

    Each ``(start0, length)`` range triggers one ``reader.read_rows`` round-trip
    to CFITSIO, so callers MUST pass *coalesced* ranges (adjacent/contiguous
    rows merged into a single range) to avoid N small reads for scattered row
    lists. Zero-length ranges are skipped without a round-trip.

    A segment that is empty or short for a non-empty range raises instead of
    fabricating rows: unreadable rows used to surface as zeros (tensors) or
    ``None`` (lists) and short list segments silently spliced the buffer
    (r5a-06, A-14).
    """
    if hasattr(reader, "read_rows_raw"):
        return _read_raw_ranges_as_chunk(reader, col_list, ranges)

    import numpy as np

    out_sorted: dict[str, Any] = {}
    n_total = sum(length for _, length in ranges)
    if n_total == 0:
        return {}

    expected: Optional[set[str]] = set(col_list) if col_list else None
    cursor = 0
    for start0, length in ranges:
        if length <= 0:
            continue
        seg = reader.read_rows(col_list, start0 + 1, length)
        row_lo, row_hi = start0, start0 + length
        if not seg:
            raise RuntimeError(
                f"empty row segment for rows [{row_lo}, {row_hi}): "
                "refusing to fabricate rows"
            )
        if expected is None:
            expected = set(seg.keys())
        missing = expected - set(seg.keys())
        if missing:
            raise RuntimeError(
                f"row segment for rows [{row_lo}, {row_hi}) is missing "
                f"columns {sorted(missing)}"
            )
        for name, value in seg.items():
            buf: Any = out_sorted.get(name)
            if buf is None:
                if _is_torch_tensor(value):
                    import torch

                    # Zero-initialized: an empty reader segment must surface
                    # as zeros, never uninitialized memory. Segments are
                    # length-validated below, so every row slot is filled.
                    buf = torch.zeros(
                        (n_total,) + tuple(value.shape[1:]), dtype=value.dtype
                    )
                else:
                    buf = [None] * n_total
                out_sorted[name] = buf

            n_got = _segment_row_count(value)
            if n_got is not None and n_got != length:
                raise RuntimeError(
                    f"short row segment for rows [{row_lo}, {row_hi}): "
                    f"column {name!r} returned {n_got} of {length} rows"
                )
            if _is_torch_tensor(value):
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


# A CFITSIO ranged read costs roughly as much as streaming ~100 KB of rows
# (measured: ~33 us per range vs ~3 GB/s sequential on a 58-byte-row table).
# Gaps narrower than this many bytes are read through instead of opening a
# new range, so a dense scatter becomes a few long reads while a sparse one
# keeps its per-row ranges. Kept below the break-even point so a sparse
# selection is never slower than one range per row.
_ROW_GAP_READ_THROUGH_BYTES = 32 * 1024
# Upper bound on the rows materialized per batch of merged blocks, so reading
# through gaps never holds much more than the selection itself in memory.
_ROW_BLOCK_BATCH_BYTES = 64 * 1024 * 1024


def _row_blocks(unique_rows: Any, max_gap: int) -> tuple[Any, Any]:
    """``(starts, lengths)`` of blocks covering sorted unique 0-based rows,
    merging neighbours separated by at most ``max_gap`` unrequested rows."""
    import numpy as np

    if unique_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    breaks = np.nonzero(np.diff(unique_rows) > max_gap + 1)[0]
    first = np.insert(breaks + 1, 0, 0)
    last = np.append(breaks, unique_rows.size - 1)
    starts = unique_rows[first]
    lengths = unique_rows[last] - starts + 1
    return starts, lengths


def _take_rows(value: Any, idx: Any) -> Any:
    """Rows ``idx`` (int64 positions) of one assembled column, in that order."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value[idx]
    if isinstance(value, list):
        return [value[i] for i in idx.tolist()]
    if _is_torch_tensor(value):
        import torch

        return value[torch.from_numpy(idx)]
    if _is_vla_tuple(value):
        values, offsets = value
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
        lo = offsets[idx]
        counts = offsets[idx + 1] - lo
        new_offsets = np.zeros(idx.size + 1, dtype=np.int64)
        np.cumsum(counts, out=new_offsets[1:])
        # Element positions: each row's run lo..lo+count, flattened.
        flat = np.repeat(lo - new_offsets[:-1], counts) + np.arange(
            int(new_offsets[-1]), dtype=np.int64
        )
        return values[flat], new_offsets
    raise TypeError(f"cannot select rows from a {type(value).__name__} column")


def _concat_rows(parts: list[Any]) -> Any:
    """Concatenate per-batch selections of one column."""
    import numpy as np

    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(parts, axis=0)
    if isinstance(first, list):
        return [item for part in parts for item in part]
    if _is_torch_tensor(first):
        import torch

        return torch.cat(parts, dim=0)
    values = np.concatenate([p[0] for p in parts], axis=0)
    offsets = [np.asarray(parts[0][1], dtype=np.int64)]
    for _values, part_offsets in parts[1:]:
        offsets.append(np.asarray(part_offsets[1:], dtype=np.int64) + offsets[-1][-1])
    return values, np.concatenate(offsets)


def _read_selected_rows(
    reader: Any,
    col_list: list[str],
    rows: Any,
    *,
    row_bytes: int,
) -> dict[str, Any]:
    """Read 0-based ``rows`` (any order, duplicates allowed) in that order.

    Rows are sorted and grouped into blocks: neighbours closer than
    ``_ROW_GAP_READ_THROUGH_BYTES`` share one ranged read and the rows in
    between are discarded. Blocks are read in batches of at most
    ``_ROW_BLOCK_BATCH_BYTES`` so memory stays bounded by the selection.
    """
    import numpy as np

    unique_rows, inverse = np.unique(rows, return_inverse=True)
    width = max(int(row_bytes), 1)
    max_gap = _ROW_GAP_READ_THROUGH_BYTES // width if row_bytes > 0 else 0
    starts, lengths = _row_blocks(unique_rows, max_gap)
    batch_rows = max(_ROW_BLOCK_BATCH_BYTES // width, 1)

    # Which block each unique row falls in; its offset inside that block.
    block_of = np.searchsorted(starts, unique_rows, side="right") - 1
    within = unique_rows - starts[block_of]

    parts: dict[str, list[Any]] = {}
    b = 0
    n_blocks = int(starts.size)
    while b < n_blocks:
        # Greedy batch of consecutive blocks under the byte budget (always
        # at least one block, however long).
        e = b + 1
        total = int(lengths[b])
        while e < n_blocks and total + int(lengths[e]) <= batch_rows:
            total += int(lengths[e])
            e += 1
        ranges = list(zip(starts[b:e].tolist(), lengths[b:e].tolist()))
        chunk = _read_ranges_as_chunk(reader, col_list, ranges)
        block_base = np.zeros(e - b, dtype=np.int64)
        np.cumsum(lengths[b : e - 1], out=block_base[1:])
        lo = np.searchsorted(block_of, b, side="left")
        hi = np.searchsorted(block_of, e, side="left")
        pos = block_base[block_of[lo:hi] - b] + within[lo:hi]
        # pos is strictly increasing: it spans the whole chunk only when it
        # is 0..total-1, in which case the chunk is already the selection.
        whole = pos.size == total
        for name, value in chunk.items():
            parts.setdefault(name, []).append(
                value if whole else _take_rows(value, pos)
            )
        b = e

    # Back from sorted unique rows to the requested order (and duplicates);
    # an already sorted, duplicate-free request needs no reordering.
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    in_order = unique_rows.size == inverse.size and bool(
        np.all(inverse[1:] > inverse[:-1])
    )
    return {
        name: _concat_rows(col_parts)
        if in_order
        else _take_rows(_concat_rows(col_parts), inverse)
        for name, col_parts in parts.items()
    }
