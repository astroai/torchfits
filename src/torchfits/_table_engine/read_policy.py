"""Backend and where-clause read strategy for FITS table I/O."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .. import fits_schema


class WhereStrategy(str, Enum):
    """How to satisfy a table read that includes a where expression."""

    ARROW_FILTER = "arrow_filter"
    CPP_PUSHDOWN = "cpp_pushdown"


@dataclass(frozen=True)
class WhereReadPlan:
    """Chosen strategy for reading a filtered FITS table."""

    strategy: WhereStrategy
    cpp_pushdown_safe: bool
    unfiltered_backend: str


def should_skip_cpp_for_where(backend: str, where: str | None) -> bool:
    """In auto mode, where= forces the pushdown/scanner path instead of cpp."""
    return backend == "auto" and where is not None


def choose_where_read_plan(
    *,
    header: Mapping[str, Any],
    header_ok: bool,
    columns: Optional[list[str]],
    backend: str,
    n_rows: int,
    mmap: bool = True,
) -> WhereReadPlan:
    """Select Arrow filtering vs C++ pushdown for a where= table read.

    Policy (no size thresholds):

    - ``backend="cpp"`` → fused native pushdown when safe.
    - ``backend="auto"`` + ``mmap=True`` → native mmap-scan pushdown when safe
      (documented WHERE⇒mmap-scan).
    - ``backend="auto"`` + ``mmap=False`` → read then tensor/Arrow filter
      (buffered I/O; comparable to fitsio-style full-read + mask).
    """
    _ = n_rows  # kept for API stability; no N-based strategy forks.
    vla_in_projection = (
        fits_schema.selected_includes_vla(header, columns) if header_ok else True
    )
    cpp_pushdown_safe = header_ok and not vla_in_projection

    if cpp_pushdown_safe and (backend == "cpp" or (backend == "auto" and mmap)):
        strategy = WhereStrategy.CPP_PUSHDOWN
    else:
        strategy = WhereStrategy.ARROW_FILTER
    unfiltered_backend = "cpp" if backend == "auto" else backend

    return WhereReadPlan(
        strategy=strategy,
        cpp_pushdown_safe=cpp_pushdown_safe,
        unfiltered_backend=unfiltered_backend,
    )
