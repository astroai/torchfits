"""Shared C++ table read dispatch engine.

Provides :func:`_read_ranges_as_chunk` — reads multiple row ranges from a
C++ TableReader and assembles them into a single torch-backed dict.
"""

from __future__ import annotations

from typing import Any

import torch

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
        seg = reader.read_rows(col_list, start0 + 1, length)
        if not seg:
            cursor += length
            continue
        for name, value in seg.items():
            buf: Any = out_sorted.get(name)
            if buf is None:
                if isinstance(value, torch.Tensor):
                    buf = torch.empty(
                        (n_total,) + tuple(value.shape[1:]), dtype=value.dtype
                    )
                else:
                    buf = [None] * n_total
                out_sorted[name] = buf

            if isinstance(value, torch.Tensor):
                buf[cursor : cursor + length] = value  # type: ignore[index]
            elif isinstance(value, list):
                buf[cursor : cursor + length] = value  # type: ignore[index]
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
