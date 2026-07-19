"""Private per-call C++ FITS handles/readers (CFITSIO §4 Option A).

Sharing a single ``fitsfile*`` across threads corrupts CFITSIO's internal
position state, so every read opens a fresh, privately-owned handle. There is
deliberately no cross-thread handle/reader cache here; the LRUs that used to
live in this module defeated Option A by handing one handle to concurrent
readers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _acquire_cpp_handle(path: str, cpp: Any) -> Any:
    """Open a fresh, privately-owned CFITSIO handle. The caller must close it."""
    return cpp.open_fits_file(path, "r")


def _acquire_cpp_reader(path: str, hdu: int, cpp: Any) -> Any:
    """Open a fresh ``TableReader`` with its own private handle (never shared).

    The filename constructor opens and owns a per-instance ``fitsfile*`` that is
    closed when the reader is garbage-collected, so distinct HDU readers on
    distinct threads never touch the same underlying handle.
    """
    return cpp.TableReader(path, int(hdu))


def _close_all_cached_handles() -> None:
    """No-op: table reads open a private CFITSIO handle per call (Option A)."""


def _invalidate_caches_for_path(path: str) -> None:  # noqa: ARG001
    """No-op: no cross-thread table-handle cache to invalidate."""
