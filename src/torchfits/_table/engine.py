"""Shared C++ table read dispatch engine.

Provides :func:`_read_ranges_as_chunk` — reads multiple row ranges from a
C++ TableReader and assembles them into a single NumPy-backed dict. Used by
the Arrow table module to avoid duplicate range-assembly logic.
"""

from __future__ import annotations

from typing import Any

from .arrow_convert import _is_vla_tuple


def _read_ranges_as_chunk(
    reader: Any,
    col_list: list[str],
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Read multiple row ranges from a TableReader and assemble into one chunk."""
    import numpy as np

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
