"""FITS write helpers for the torchfits I/O engine."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Union


from ..hdu import HDUList, Header
from .paths import guard_write_path
from .checksum_api import write_checksums as _write_checksums_impl
from ._tile_compression import CompressionOptions, compression_options
from ._hdu_rewrite import (
    _write_hdus_uncompressed,
    _write_hdus_with_optional_compression,
)
from ._file_ops import (
    _invalidate_path_caches,
    _invalidate_written_target,
)

# Re-exported: _table/write.py imports the atomic publish helper from here.
from ._file_ops import atomic_write_target as atomic_write_target
from ._write_coerce import (
    compressed_hdus_for_write,
    hdulist_hdus_for_write,
    image_hdu_dicts_for_write,
)
from ._write_helpers import (
    _can_use_cpp_table_writer,
    _cpp_header_mapping,
    _normalize_cpp_table_data,
    _normalize_table_input,
    _payload_replay_header,
    _prepare_quantized_table_data_for_write,
    _prepare_unsigned_table_data_for_write,
    _write_boundary_header,
    _write_header_cards_if_supported,
)

from ._hdu_edit import delete_hdu
from ._hdu_rewrite import (
    insert_hdu,
    replace_hdu,
)

__all__ = [
    "write",
    "insert_hdu",
    "replace_hdu",
    "delete_hdu",
]


def _write_all_checksums(path: str) -> None:
    """Write DATASUM/CHECKSUM for every HDU of a freshly written file."""
    import torchfits._C as cpp

    n_hdus = int(cpp.read_num_hdus(str(path)))
    for hdu in range(n_hdus):
        _write_checksums_impl(str(path), hdu=hdu)


def write(
    path: str | os.PathLike[str],
    data: Any,
    header: Optional[Header | Dict[str, Any]] = None,
    overwrite: bool = False,
    compress: Union[bool, str] = False,
    quantize: Any = None,
    checksum: bool = False,
    *,
    tile_shape: Optional[Sequence[int]] = None,
    quantize_level: Optional[float] = None,
    quantize_method: Optional[str] = None,
    dither_seed: Optional[int] = None,
    int_to_float: bool = False,
) -> None:
    """Write data to FITS file.

    Args:
        path: Output file path
        data: Data to write (Tensor, table mapping, or HDUList)
        header: Optional FITS header dictionary
        overwrite: Whether to overwrite existing files
        compress: Whether to use tile compression (Rice algorithm)
        quantize: Opt-in robust int16 packing for float images or table
            columns. ``None`` (default) keeps native float storage.
            For images: ``"robust"`` or ``{"lo_q", "hi_q", "keep_zero"}``.
            For dict tables: ``"robust"`` (all float columns) or
            ``{"col": "robust" | opts}``.
        checksum: When True, compute and write CFITSIO DATASUM/CHECKSUM
            keywords for every HDU after the payload lands (archive-ingest
            friendly). Verify later with :func:`torchfits.verify_checksums`.
        tile_shape: Tile size per image axis for tile-compressed writes, in
            tensor axis order (``(rows, cols)`` for a 2-D image); it covers the
            trailing axes and leading axes get tile size 1. ``-1`` (or a size
            past the axis length) means the whole axis. Default: one image row
            per tile (16 rows for ``HCOMPRESS_1``). Requires ``compress=``.
        quantize_level: CFITSIO quantization of floating-point images in a
            tile-compressed write (``fpack -q``): ``> 0`` sets the step to
            the tile noise sigma divided by the level, ``< 0`` sets an
            absolute step of ``abs(level)``, ``0`` stores floats losslessly
            (``GZIP_1``/``GZIP_2`` only). Default: lossless for ``GZIP_1``,
            4.0 otherwise. Integer images are never quantized.
        quantize_method: ``"subtractive_dither_1"`` (CFITSIO default),
            ``"subtractive_dither_2"`` (also keeps exact zeros) or
            ``"no_dither"`` (default level 16).
        dither_seed: Dither seed in 1..10000 for reproducible quantization
            (default: CFITSIO picks one from the clock).
        int_to_float: Compress integer images as quantized float32, like
            ``fpack -i2f``: physical values (after ``BSCALE``/``BZERO``,
            ``BLANK`` -> NaN) are quantized at ``quantize_level`` (default
            4.0). Lossy; the stored image becomes ``BITPIX = -32``.

    Image tensors on non-CPU devices are detached and copied to CPU before
    the CFITSIO writer runs (in-memory input tensors are not modified).
    """
    path = os.fspath(path)
    guard_write_path(path)
    if str(path).lower().endswith(".bz2"):
        # CFITSIO would create a plain, uncompressed FITS under this name.
        raise ValueError(
            "Writing bzip2-wrapped FITS files ('.bz2') is not supported; "
            "use compress='BZIP2_1' for tile compression inside the file."
        )
    path_exists = os.path.exists(path)
    if not overwrite and path_exists:
        raise FileExistsError(
            f"File '{path}' already exists. Use overwrite=True to overwrite."
        )
    if path_exists and os.path.isdir(path):
        raise IsADirectoryError(path)
    options = compression_options(
        compress,
        tile_shape=tile_shape,
        quantize_level=quantize_level,
        quantize_method=quantize_method,
        dither_seed=dither_seed,
        int_to_float=int_to_float,
    )

    # Always write a fresh temp file and publish it afterwards: a failed write
    # then never creates or deletes anything at ``path`` (the old error path
    # removed whatever was there, including a file another writer created
    # after the existence check), and overwrite=False never clobbers it.
    with atomic_write_target(path, overwrite) as temp_path:
        try:
            _write_new_file(
                temp_path,
                data,
                header=header,
                compress=compress,
                quantize=quantize,
                checksum=checksum,
                options=options,
            )
        except (ValueError, TypeError):
            # Argument errors (non-ASCII text, bad quantize=, a compress= of
            # the wrong type, ...) are documented as ValueError / TypeError;
            # wrapping them hid the contract from callers.
            raise
        except Exception as e:
            # Translation boundary: the documented RuntimeError for a failed
            # write (atomic_write_target has already removed the temp file).
            raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e
    # After the rename, never before: see atomic_write_target.
    _invalidate_written_target(path)


def _write_dict_table(
    path: str,
    data: Dict[str, Any],
    header: Optional[Header | Dict[str, Any]],
    quantize: Any,
    overwrite: bool,
) -> None:
    """Write a dict table through the native CFITSIO table writer."""
    import torchfits._C as cpp

    data, table_schema, _ = _prepare_unsigned_table_data_for_write(data)
    data, q_schema, q_changed = _prepare_quantized_table_data_for_write(
        data, quantize, table_schema
    )
    if q_changed:
        table_schema = q_schema
    if not _can_use_cpp_table_writer(data):
        raise ValueError(
            "Dictionary table writes currently require CFITSIO-native column types "
            "(numeric/bool/complex, strings, or VLA lists). Unsupported object/structure "
            "columns should be converted before writing."
        )
    data = _normalize_cpp_table_data(data)
    header_obj = _write_boundary_header(header)
    cpp.write_fits_table(
        path,
        data,
        _cpp_header_mapping(header_obj),
        overwrite,
        table_schema,
        "binary",
    )
    _write_header_cards_if_supported(path, 1, header_obj)


def _write_new_file(
    path: str,
    data: Any,
    header: Optional[Header | Dict[str, Any]],
    compress: Union[bool, str],
    quantize: Any,
    checksum: bool,
    options: Optional[CompressionOptions] = None,
) -> None:
    """write() body: ``path`` is a temp path that does not exist yet.

    The inputs are coerced into HDUs by ``_write_coerce``; this function only
    picks the writer, runs it, and stamps checksums.
    """
    overwrite = False
    import torchfits._C as cpp

    data = _normalize_table_input(data)

    if compress:
        _write_hdus_with_optional_compression(
            path,
            compressed_hdus_for_write(data, header, quantize),
            compress=compress,
            overwrite=overwrite,
            options=options,
        )
        if checksum:
            _write_all_checksums(path)
        _invalidate_path_caches(path)
        return

    if isinstance(data, HDUList):
        _write_hdus_uncompressed(
            path, hdulist_hdus_for_write(data, header, quantize), overwrite
        )
        if checksum:
            _write_all_checksums(path)
        return

    if isinstance(data, dict) and "data" not in data:
        _write_dict_table(path, data, header, quantize, overwrite)
        if checksum:
            _write_all_checksums(path)
        return

    hdus_to_write = image_hdu_dicts_for_write(data, header, quantize)
    cpp.write_fits_file(path, hdus_to_write, overwrite)
    for idx, item in enumerate(hdus_to_write):
        _write_header_cards_if_supported(
            path, idx, _payload_replay_header(item), invalidate=False
        )
    if checksum:
        _write_all_checksums(path)
    _invalidate_path_caches(path)
