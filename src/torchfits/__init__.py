"""Lean public API for torchfits.

The package root intentionally stays light: importing :mod:`torchfits` must not
load tensor runtimes, NumPy, compiled extensions, or optional integration packages.

Transforms live under :mod:`torchfits.transforms`. Arrow tables under
:mod:`torchfits.table`. HDU types are available as root names and via
:mod:`torchfits.hdu`.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.9.3"

_NAMESPACES: dict[str, str] = {
    "table": "torchfits.table",
    "cache": "torchfits.cache",
    "cpp": "torchfits.cpp",
    "transforms": "torchfits.transforms",
    "data": "torchfits.data",
    "where": "torchfits.where",
    "hdu": "torchfits.hdu",
}

_ROOT_FUNCTIONS: dict[str, tuple[str, str]] = {
    "read": ("torchfits.io", "read"),
    "write": ("torchfits.io", "write"),
    "open": ("torchfits.io", "open"),
    "get_header": ("torchfits.io", "get_header"),
    "read_tensor": ("torchfits.io", "read_tensor"),
    "read_table": ("torchfits.io", "read_table"),
    "read_hdus": ("torchfits.io", "read_hdus"),
    "read_subset": ("torchfits.io", "read_subset"),
    "open_subset_reader": ("torchfits.io", "open_subset_reader"),
    "read_batch": ("torchfits.io", "read_batch"),
    "get_batch_info": ("torchfits.io", "get_batch_info"),
    "get_cache_performance": ("torchfits.io", "get_cache_performance"),
    "read_table_rows": ("torchfits.io", "read_table_rows"),
    "stream_table": ("torchfits.io", "stream_table"),
    "clear_file_cache": ("torchfits.io", "clear_file_cache"),
    "verify_checksums": ("torchfits.io", "verify_checksums"),
    "insert_hdu": ("torchfits.io", "insert_hdu"),
    "replace_hdu": ("torchfits.io", "replace_hdu"),
    "delete_hdu": ("torchfits.io", "delete_hdu"),
    "write_checksums": ("torchfits.io", "write_checksums"),
    "write_tensor": ("torchfits.io", "write_tensor"),
    "to_pandas": ("torchfits.interop", "to_pandas"),
    "to_arrow": ("torchfits.interop", "to_arrow"),
    "to_polars": ("torchfits.interop", "to_polars"),
}

_ROOT_OBJECTS: dict[str, tuple[str, str]] = {
    "Header": ("torchfits.hdu", "Header"),
    "Card": ("torchfits.hdu", "Card"),
    "HDUList": ("torchfits.hdu", "HDUList"),
    "TensorHDU": ("torchfits.hdu", "TensorHDU"),
    "TableHDU": ("torchfits.hdu", "TableHDU"),
    "TableHDURef": ("torchfits.hdu", "TableHDURef"),
}

__all__ = tuple(
    [
        "read",
        "write",
        "open",
        "get_header",
        "read_tensor",
        "read_table",
        "read_hdus",
        "read_subset",
        "open_subset_reader",
        "Header",
        "Card",
        "HDUList",
        "TensorHDU",
        "TableHDU",
        "TableHDURef",
        "read_batch",
        "get_batch_info",
        "get_cache_performance",
        "read_table_rows",
        "stream_table",
        "clear_file_cache",
        "verify_checksums",
        "insert_hdu",
        "replace_hdu",
        "delete_hdu",
        "write_checksums",
        "write_tensor",
        "to_pandas",
        "to_arrow",
        "to_polars",
        *_NAMESPACES,
    ]
)

_RUNTIME_INITIALIZED = False


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _ensure_runtime_init() -> None:
    """Initialize optional runtime caches when an I/O entry point is used."""
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    cache_mb = os.environ.get("TORCHFITS_CFITSIO_CACHE_MB")
    cache_files = os.environ.get("TORCHFITS_CFITSIO_CACHE_FILES")
    cache_limits = None
    if cache_mb is not None or cache_files is not None:
        cache_limits = (
            _positive_env_int("TORCHFITS_CFITSIO_CACHE_FILES", 32),
            _positive_env_int("TORCHFITS_CFITSIO_CACHE_MB", 256),
        )

    cache = import_module("torchfits.cache")
    cache.configure_for_environment()
    # Pre-import torch so its dependency libraries (libcudart.so.12,
    # libtorch_cuda.so, libtorch_python.so) are loaded before torchfits._C.
    import torch  # noqa: F401

    cpp = import_module("torchfits._C")
    if cache_limits is not None:
        cpp.configure_cache(*cache_limits)

    _RUNTIME_INITIALIZED = True


def __getattr__(name: str) -> Any:
    if name in _NAMESPACES:
        if name == "cpp":
            _ensure_runtime_init()
        module = import_module(_NAMESPACES[name])
        globals()[name] = module
        return module

    if name in _ROOT_FUNCTIONS:
        _ensure_runtime_init()
        module_name, attr_name = _ROOT_FUNCTIONS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    if name in _ROOT_OBJECTS:
        module_name, attr_name = _ROOT_OBJECTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from . import (
        cache as cache,
        cpp as cpp,
        data as data,
        hdu as hdu,
        table as table,
        transforms as transforms,
        where as where,
    )
    from .hdu import Card as Card
    from .hdu import HDUList as HDUList
    from .hdu import Header as Header
    from .hdu import TableHDU as TableHDU
    from .hdu import TableHDURef as TableHDURef
    from .hdu import TensorHDU as TensorHDU
    from .io import clear_file_cache as clear_file_cache
    from .io import delete_hdu as delete_hdu
    from .io import get_batch_info as get_batch_info
    from .io import get_cache_performance as get_cache_performance
    from .io import get_header as get_header
    from .io import insert_hdu as insert_hdu
    from .io import open as open
    from .io import open_subset_reader as open_subset_reader
    from .io import read as read
    from .io import read_batch as read_batch
    from .io import read_hdus as read_hdus
    from .io import read_subset as read_subset
    from .io import read_table as read_table
    from .io import read_table_rows as read_table_rows
    from .io import read_tensor as read_tensor
    from .io import replace_hdu as replace_hdu
    from .io import stream_table as stream_table
    from .io import verify_checksums as verify_checksums
    from .io import write as write
    from .io import write_checksums as write_checksums
    from .io import write_tensor as write_tensor
    from .interop import to_arrow as to_arrow
    from .interop import to_pandas as to_pandas
    from .interop import to_polars as to_polars
