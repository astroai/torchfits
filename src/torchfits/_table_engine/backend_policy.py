"""Backend selection policy for FITS table I/O."""

from __future__ import annotations

import warnings

_LEGACY_BACKEND_ALIASES = {"cpp_numpy": "cpp"}

_TABLE_BACKEND_ORDER = ("auto", "torch", "cpp")
TABLE_BACKENDS = frozenset(_TABLE_BACKEND_ORDER)


def validate_table_backend(backend: str) -> str:
    """Return a validated table backend name or raise with the public error."""
    if backend in _LEGACY_BACKEND_ALIASES:
        warnings.warn(
            "table backend 'cpp_numpy' is deprecated; use 'cpp'",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_BACKEND_ALIASES[backend]
    if backend not in TABLE_BACKENDS:
        allowed = ", ".join(_TABLE_BACKEND_ORDER)
        raise ValueError(f"backend must be one of: {allowed}")
    return backend
