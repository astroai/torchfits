"""Explicit public surface for the native FITS extension.

New symbols added to :mod:`torchfits._C` stay private until deliberately added
to ``__all__``. Attribute access for names in ``__all__`` delegates to the
extension.
"""

# ruff: noqa: F822  # nanobind symbols are installed into globals below.

from __future__ import annotations

from typing import Any

import torchfits._C as _C

__all__ = (
    "FITSFile",
    "HDUInfo",
    "SubsetReader",
    "TableReader",
    "append_fits_table_rows",
    "clear_file_cache",
    "clear_shared_read_meta_cache",
    "configure_cache",
    "delete_fits_table_rows",
    "drop_fits_table_columns",
    "get_cache_size",
    "get_hdu_type",
    "get_num_hdus",
    "insert_fits_table_rows",
    "open_and_read_headers",
    "open_fits_file",
    "read_fits_table",
    "read_fits_table_filtered",
    "read_fits_table_from_handle",
    "read_fits_table_rows",
    "read_fits_table_rows_from_handle",
    "read_fits_table_rows_numpy",
    "read_fits_table_rows_numpy_from_handle",
    "read_full",
    "read_full_cached",
    "read_full_nocache",
    "read_full_numpy",
    "read_full_numpy_cached",
    "read_full_raw",
    "read_full_raw_with_scale",
    "read_full_scaled_cpu",
    "read_full_unmapped",
    "read_full_unmapped_raw",
    "read_hdus_batch",
    "read_hdus_sequence_last",
    "read_header",
    "read_header_dict",
    "read_header_string",
    "read_colnames",
    "read_hdu_type",
    "read_keys",
    "read_nrows",
    "read_num_hdus",
    "read_shape",
    "read_table_info",
    "read_images_batch",
    "read_tensor_from_handle",
    "rename_fits_table_columns",
    "resolve_hdu_name_cached",
    "update_fits_table_rows",
    "update_fits_table_rows_mmap",
    "verify_hdu_checksums",
    "write_fits_file",
    "write_fits_file_compressed_images",
    "write_fits_table",
    "write_hdu_checksums",
    "write_hdu_header_cards",
    "delete_hdu_header_key",
)

globals().update({name: getattr(_C, name) for name in __all__})


def __getattr__(name: str) -> Any:
    return getattr(_C, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_C)))
