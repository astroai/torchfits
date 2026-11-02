"""HDU rewrite and multi-HDU write paths for FITS I/O."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..hdu import Header, TableHDU, TableHDURef, TensorHDU
from .hdu_api import open_hdulist
from .paths import guard_fits_path
from ._hdu_splice import (
    Piece,
    hdu_byte_spans,
    header_has_card,
    image_extension_as_primary,
    primary_as_image_extension,
    publish_spliced,
    scan_hdus,
    xtension_of,
)
from ._write_helpers import (
    _COMMENTARY_HEADER_KEYS,
    _COMPRESSION_CARD_KEYS,
    _COMPRESSION_CARD_PREFIXES,
    _atomic_replace_target,
    _cpp_header_mapping,
    _filter_header_cards,
    _hdu_with_header,
    _image_hdu_dict_for_fits_write,
    _invalidate_path_caches,
    _invalidate_written_target,
    _is_skippable_empty_primary,
    _merge_fits_write_header,
    _normalize_cpp_table_data,
    _payload_replay_header,
    _prepare_unsigned_table_data_for_write,
    _preserved_table_tform_cards,
    _resolve_compression_algorithm,
    _table_schema_scale_header_cards,
    _write_boundary_header,
    _write_header_cards_if_supported,
)


def _sanitize_header_for_compressed_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mapping of the header cards the compressed dict writer may set.

    Structural/compression keys are dropped so CFITSIO emits canonical
    metadata; HISTORY/COMMENT are excluded (fits_update_key renders them as
    value cards -- the replay step writes them properly); stale CHECKSUM/
    DATASUM never carry over. Byte values must decode as ASCII strictly:
    silently discarding undecodable bytes would write a shortened card.
    """
    import numpy as np

    if not header:
        return {}

    skip_exact = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "EXTEND",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "THEAP",
        "BSCALE",
        "BZERO",
        "DATASUM",
        "CHECKSUM",
        "ZIMAGE",
        "ZCMPTYPE",
        "ZBITPIX",
        "ZNAXIS",
        "ZPCOUNT",
        "ZGCOUNT",
        "ZHECKSUM",
        "ZDATASUM",
    }
    skip_prefix = (
        "NAXIS",
        "ZNAXIS",
        "ZTILE",
        "ZNAME",
        "ZVAL",
        # CFITSIO computes the dither offset for the quantizer it picks;
        # a replayed ZDITHER<n> would contradict the tiles it just wrote.
        "ZDITHER",
        "TTYPE",
        "TFORM",
        "TDIM",
        "TSCAL",
        "TZERO",
        "TNULL",
        "TUNIT",
        "TDISP",
    )

    source = header if isinstance(header, Header) else Header(header)
    out: Dict[str, Any] = {}
    for card in source.cards:
        key_str = str(card.key)
        key_upper = key_str.upper()
        if (
            key_upper in skip_exact
            or key_upper in _COMMENTARY_HEADER_KEYS
            or any(key_upper.startswith(prefix) for prefix in skip_prefix)
        ):
            continue
        if key_str in out:
            continue
        value = card.value
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("ascii")
        out[key_str] = value
    return out


def _sanitize_table_header_for_write(
    header: Optional[Dict[str, Any]],
) -> Header:
    """Write-boundary header for table HDUs.

    Drops the cards the table writer derives from the payload (structural
    block, TTYPE/TFORM/TSCAL/TZERO) plus stale CHECKSUM/DATASUM so a copied
    header can never relabel or rescale the new data. HISTORY/COMMENT are
    kept for the replay step; TUNIT/TDIM/TNULL annotations survive. Rewrites
    re-add the source TFORM spelling per column (see
    ``_preserved_table_tform_cards``).
    """
    return _write_boundary_header(header)


class _TableWriteProxy:
    __slots__ = ("_raw_data", "header", "_schema", "_replay")

    def __init__(
        self,
        raw_data: Any,
        header: Header,
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._raw_data = raw_data
        # The C++ dict writers iterate ``header`` as a plain mapping; the
        # full card list lives in ``_replay`` for the replay step.
        self._replay = header
        self.header = _cpp_header_mapping(header)
        self._schema = schema


def _prepare_table_hdu_for_payload(hdu: Any) -> Optional[_TableWriteProxy]:
    """Normalize a table HDU into a ``_TableWriteProxy`` for the C++ writer.

    Materializes ``TableHDURef`` to CPU; handles ``TableHDU`` and any
    object with ``_raw_data`` + ``header`` attributes. Returns ``None``
    when *hdu* is not a table HDU (the caller must treat it as an image).
    """
    if isinstance(hdu, TableHDURef):
        hdu = hdu.materialize(device="cpu")

    if (
        getattr(hdu, "_replay", None) is not None
        and hasattr(hdu, "_raw_data")
        and hasattr(hdu, "_schema")
    ):
        # Internal write proxy: already normalized with schema-derived cards
        # (TSCAL/TZERO/TNULL) and a card-faithful replay header. Re-preparing
        # its raw data would drop those cards.
        return _TableWriteProxy(hdu._raw_data, hdu._replay, hdu._schema)

    if isinstance(hdu, TableHDU) or (
        hasattr(hdu, "_raw_data") and hasattr(hdu, "header")
    ):
        raw_data = dict(getattr(hdu, "_raw_data", {}))
        raw_data, schema, _ = _prepare_unsigned_table_data_for_write(raw_data)
        scale_cards = _table_schema_scale_header_cards(schema)
        # Source TFORMn spellings (astropy bare 'J'/'I', exotic '8X'/'PJ') are
        # re-added verbatim only for the same column with the same cell
        # layout; decoded scaled columns re-derive instead (r4b-01). Bit and
        # wide string spellings are laid down by the writer via the schema.
        preserved, formats, stale = _preserved_table_tform_cards(hdu.header, raw_data)
        if formats:
            schema = {
                name: {
                    **(schema or {}).get(name, {}),
                    **({"format": formats[name]} if name in formats else {}),
                }
                for name in raw_data
            }
        header = _merge_fits_write_header(
            _merge_fits_write_header(
                _filter_header_cards(
                    _sanitize_table_header_for_write(hdu.header), frozenset(stale)
                ),
                preserved,
            ),
            scale_cards,
        )
        raw_data = _normalize_cpp_table_data(raw_data)
        return _TableWriteProxy(raw_data, header, schema)

    return None


def _write_hdus_uncompressed(path: str, hdus: List[Any], overwrite: bool) -> None:
    """Write an HDU sequence through the uncompressed C++ writer.

    The C++ dict writers receive commentary-free header mappings; the full
    card list (HISTORY/COMMENT included) is replayed once per HDU afterwards
    via fits_write_history/fits_write_comment.
    """
    import torchfits._C as cpp

    guard_fits_path(path)
    if not hdus:
        # A FITS file must contain at least a primary HDU. The C++ writer
        # accepts an empty payload and writes a 2880-byte file whose only card
        # is END -- not a FITS file, which no reader can open. Measured: this
        # is what delete_hdu(path, 0) did to a single-HDU file (5760 valid
        # bytes replaced by 2880 unreadable ones, no error raised), and what
        # write(path, HDUList([])) left behind. The atomic-rename wrapper
        # faithfully replaced a good file with an invalid one, so the refusal
        # belongs here, below the rewrite paths rather than in each caller.
        #
        # write()'s image branch already refused an empty payload with
        # "At least one writable HDU is required"; this is the same contract
        # for the branches that had drifted past it.
        raise ValueError("At least one writable HDU is required")
    if overwrite and os.path.isfile(path):
        with _atomic_replace_target(path) as temp_path:
            _write_hdus_uncompressed(temp_path, hdus, overwrite=False)
        # After the block: that is when the new bytes become visible.
        _invalidate_written_target(path)
        return

    payload: List[Any] = []
    for idx, hdu in enumerate(hdus):  # noqa: B007
        table_proxy = _prepare_table_hdu_for_payload(hdu)
        if table_proxy is not None:
            payload.append(table_proxy)
            continue

        if not isinstance(hdu, TensorHDU):
            raise ValueError(
                f"Unsupported HDU type for write at index {idx}: {type(hdu).__name__}"
            )

        payload.append(
            _image_hdu_dict_for_fits_write(
                hdu.to_tensor("cpu"), getattr(hdu, "header", None)
            )
        )

    _invalidate_path_caches(path)
    cpp.write_fits_file(path, payload, overwrite)
    for idx, item in enumerate(payload):
        _write_header_cards_if_supported(
            path, idx, _payload_replay_header(item), invalidate=False
        )
    _invalidate_path_caches(path)


# Reuse the write-boundary CompImage drop sets instead of restating them here.
# This private copy had already drifted by one card -- it lacked ZCHECKSUM,
# which is exactly the class of stale keyword this module exists to prevent.
_COMPRESSION_HEADER_KEYS = _COMPRESSION_CARD_KEYS
_COMPRESSION_HEADER_PREFIXES = _COMPRESSION_CARD_PREFIXES


def _strip_compression_cards(header: Header) -> Header:
    """Drop CompImage Z* cards so an uncompressed rewrite cannot lie."""
    preserved = Header(header)
    for key in list(preserved.keys()):
        ku = str(key).upper()
        if ku in _COMPRESSION_HEADER_KEYS or any(
            ku.startswith(prefix) for prefix in _COMPRESSION_HEADER_PREFIXES
        ):
            del preserved[key]
    return preserved


def _write_hdus_with_optional_compression(
    path: str,
    hdus: List[Any],
    compress: Union[bool, str] = False,
    *,
    overwrite: bool = True,
) -> None:
    """Rewrite HDUs, optionally using CFITSIO compressed-image writer.

    Like the uncompressed writer, the dict writers get commentary-free header
    mappings and the full card list is replayed once per output HDU (at
    index j+1: the compressed writer always emits its own empty primary).
    The replay also restores BSCALE/BZERO/BLANK that the compressed mapping
    deliberately omits.
    """
    guard_fits_path(path)
    algorithm = _resolve_compression_algorithm(compress)
    if algorithm is None:
        _write_hdus_uncompressed(path, hdus, overwrite=overwrite)
        return

    import torchfits._C as cpp

    payload: list[Any] = []
    for idx, hdu in enumerate(hdus):  # noqa: B007
        table_proxy = _prepare_table_hdu_for_payload(hdu)
        if table_proxy is not None:
            payload.append(table_proxy)
            continue

        if not isinstance(hdu, TensorHDU):
            raise ValueError(
                f"Unsupported HDU type for rewrite at index {idx}: {type(hdu).__name__}"
            )

        # A compressed FITS file uses an empty primary HDU followed by compressed
        # image extensions; skip this placeholder to avoid duplicating it.
        if _is_skippable_empty_primary(idx, hdu):
            continue

        hdu_dict = _image_hdu_dict_for_fits_write(
            hdu.to_tensor("cpu"), getattr(hdu, "header", None)
        )
        replay_header = _payload_replay_header(hdu_dict)
        if replay_header is not None:
            hdu_dict["header"] = _sanitize_header_for_compressed_write(replay_header)
        if algorithm and algorithm.upper() in {"P", "PLIO", "PLIO_1"}:
            d = hdu_dict.get("data")
            if isinstance(d, Tensor) and d.is_floating_point():
                raise RuntimeError(
                    "PLIO_1 compression algorithm does not support floating-point image data"
                )
            scale = replay_header if replay_header is not None else {}
            if (
                isinstance(d, Tensor)
                and d.dtype == torch.int16
                and scale.get("BZERO") == 32768
                and scale.get("BSCALE", 1) == 1
            ):
                # PLIO only stores values in [0, 2**24): hand CFITSIO the
                # physical uint16 pixels instead of the signed -32768 codes.
                hdu_dict["data"] = (d ^ -32768).view(torch.uint16)
        payload.append(hdu_dict)

    _invalidate_path_caches(path)
    cpp.write_fits_file_compressed_images(path, payload, overwrite, algorithm)
    for j, item in enumerate(payload):
        _write_header_cards_if_supported(
            path, j + 1, _payload_replay_header(item), invalidate=False
        )
    _invalidate_path_caches(path)


_CHECKSUM_KEYS = ("CHECKSUM", "DATASUM")


def _new_hdu_from_args(data: Any, header: Optional[Dict[str, Any]]) -> Any:
    if isinstance(data, (TableHDU, TensorHDU)):
        # Never mutate the caller's HDU: build a header-replaced copy.
        return _hdu_with_header(data, Header(header)) if header is not None else data
    if isinstance(data, dict) and "data" not in data:
        return TableHDU(data, header=Header(header or {}))
    if isinstance(data, Tensor):
        return TensorHDU(data=data, header=Header(header or {}))
    raise ValueError(f"Unsupported HDU data type: {type(data)}")


def _file_headers(path: str) -> List[Header]:
    with open_hdulist(path) as hdul:
        return [Header(hdu.header) for hdu in hdul._hdus]


def _file_hdus(path: str, n_headers: int) -> List[Tuple[int, int, int, bytes]]:
    hdus = scan_hdus(path)
    if len(hdus) != n_headers:
        raise ValueError(
            f"Could not locate the byte range of every HDU in '{path}' "
            f"({len(hdus)} found, {n_headers} expected); the file is left unchanged."
        )
    return hdus


def _extname_matches(header: Any, name: str) -> bool:
    """EXTNAME lookup as read() does it: trimmed and case-insensitive."""
    extname = header.get("EXTNAME")
    return extname is not None and str(extname).strip().upper() == name.strip().upper()


def _resolve_target(headers: List[Header], hdu: Union[int, str]) -> int:
    if isinstance(hdu, bool) or not isinstance(hdu, (int, str)):
        raise TypeError("hdu must be an int index or EXTNAME string")
    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(headers):
            raise IndexError(f"hdu index {hdu} out of range for {len(headers)} HDUs")
        return hdu
    for idx, header in enumerate(headers):
        if _extname_matches(header, hdu):
            return idx
    raise KeyError(f"HDU '{hdu}' not found")


def _is_table_hdu(hdu: Any) -> bool:
    return isinstance(hdu, (TableHDU, TableHDURef)) or hasattr(hdu, "_raw_data")


def _require_primary_capable(new_hdu: Any, compress: Union[bool, str]) -> None:
    """Refuse a new HDU that cannot be the primary HDU (index 0)."""
    if _is_table_hdu(new_hdu):
        raise ValueError(
            "A table cannot be the primary HDU (index 0): the file would gain an "
            "empty primary and every HDU would shift by one. Insert or replace "
            "the table at index >= 1, or keep an HDU in front of it."
        )
    if _resolve_compression_algorithm(compress) is not None:
        raise ValueError(
            "compress= cannot apply at index 0: a tile-compressed image is stored "
            "as a binary-table extension and cannot be the primary HDU. Write it "
            "at index >= 1, or without compress=."
        )


@contextlib.contextmanager
def _scratch_hdu(
    new_hdu: Any, *, primary: bool, compress: Union[bool, str]
) -> Iterator[Tuple[str, int, int]]:
    """Write ``new_hdu`` alone to a scratch file; yield ``(path, start, end)``
    of its bytes (the primary HDU, or the extension behind an empty primary).
    """
    scratch_dir = tempfile.mkdtemp(prefix="torchfits-hdu-")
    scratch = os.path.join(scratch_dir, "hdu.fits")
    try:
        if primary:
            hdus = [new_hdu]
        else:
            empty = TensorHDU(data=torch.empty(0), header=Header({"NAXIS": 0}))
            hdus = [empty, new_hdu]
        from . import write_api

        write_api._write_hdus_with_optional_compression(  # type: ignore[attr-defined]
            scratch, hdus, compress=compress, overwrite=False
        )
        start, _data, end = hdu_byte_spans(scratch)[0 if primary else 1]
        yield scratch, start, end
    finally:
        _invalidate_path_caches(scratch)
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _has_checksums(hdus: List[Tuple[int, int, int, bytes]]) -> bool:
    return any(header_has_card(raw, *_CHECKSUM_KEYS) for *_span, raw in hdus)


def _nonempty(pieces: List[Piece]) -> List[Piece]:
    return [p for p in pieces if isinstance(p, bytes) or p[2] is None or p[2] > p[1]]


def insert_hdu(
    path: str,
    data: Any,
    index: int = 1,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Insert a new HDU into an existing FITS file.

    Only the new HDU is written (``compress=`` applies to it alone); every
    other HDU is copied byte-for-byte. Inserting at index 0 turns the old
    primary into an IMAGE extension (header cards converted, data unchanged).
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer HDU position")
    new_hdu = _new_hdu_from_args(data, header)
    hdus = _file_hdus(path, len(_file_headers(path)))
    if index < 0 or index > len(hdus):
        raise IndexError(f"index {index} out of range for {len(hdus)} HDUs")
    stamp = _has_checksums(hdus)
    end = None  # through EOF: keeps bytes after the last HDU

    if index == 0:
        _require_primary_capable(new_hdu, compress)
        _start, old_data, _end, old_raw = hdus[0]
        converted = primary_as_image_extension(old_raw)
        restamp = [0] if stamp else []
        if header_has_card(old_raw, *_CHECKSUM_KEYS):
            restamp.append(1)
        with _scratch_hdu(new_hdu, primary=True, compress=False) as new_piece:
            publish_spliced(
                path,
                _nonempty([new_piece, converted, (path, old_data, end)]),
                restamp=restamp,
            )
        return

    cut = hdus[index - 1][2]
    with _scratch_hdu(new_hdu, primary=False, compress=compress) as new_piece:
        publish_spliced(
            path,
            _nonempty([(path, 0, cut), new_piece, (path, cut, end)]),
            restamp=[index] if stamp else [],
        )


def replace_hdu(
    path: str,
    hdu: Union[int, str],
    data: Any,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Replace an HDU by index or EXTNAME.

    Only the replacement is written (``compress=`` applies to it alone); every
    other HDU is copied byte-for-byte.
    """
    preserve_header = header is None and not isinstance(data, (TableHDU, TensorHDU))
    new_hdu = _new_hdu_from_args(data, header)
    headers = _file_headers(path)
    target = _resolve_target(headers, hdu)

    if preserve_header:
        # Keep the original header (e.g. EXTNAME/WCS) unless the caller
        # overrides it. Scaling and checksum keys describe the REPLACED
        # payload's storage: replaying e.g. an unsigned-convention BZERO=32768
        # over different data makes every reader misinterpret the new bytes,
        # so they are dropped and the writer emits canonical keywords.
        preserved = Header(headers[target])
        for stale_key in (
            "BSCALE",
            "BZERO",
            "BLANK",
            "DATAMIN",
            "DATAMAX",
            "DATASUM",
            "CHECKSUM",
        ):
            if stale_key in preserved:
                del preserved[stale_key]
        preserved = _strip_compression_cards(preserved)
        if isinstance(new_hdu, TensorHDU):
            new_hdu._header = preserved
        else:
            new_hdu.header = preserved

    hdus = _file_hdus(path, len(headers))
    stamp = _has_checksums(hdus)
    primary = target == 0
    if primary:
        _require_primary_capable(new_hdu, compress)
    start, _data, stop, _raw = hdus[target]
    with _scratch_hdu(new_hdu, primary=primary, compress=compress) as new_piece:
        publish_spliced(
            path,
            _nonempty([(path, 0, start), new_piece, (path, stop, None)]),
            restamp=[target] if stamp else [],
        )


def delete_hdu(
    path: str,
    hdu: Union[int, str],
    compress: Union[bool, str] = False,
) -> None:
    """Delete an HDU by index or EXTNAME.

    Every remaining HDU is copied byte-for-byte. Deleting HDU 0 promotes an
    IMAGE extension at index 1 to the primary (header cards converted, data
    unchanged). ``compress=`` is kept for signature compatibility: no HDU is
    written, so a true value raises ``ValueError``.
    """
    if _resolve_compression_algorithm(compress) is not None:
        raise ValueError(
            "delete_hdu writes no HDU, so compress= has nothing to apply to: "
            "every remaining HDU keeps its storage. Call it without compress=."
        )
    headers = _file_headers(path)
    target = _resolve_target(headers, hdu)
    if len(headers) == 1:
        # Refuse before touching anything, so the caller gets an actionable
        # message instead of a file that used to be readable and no longer is.
        raise ValueError(
            "Cannot delete the last HDU: a FITS file must contain at least "
            "one HDU. Write an empty primary instead if that is what you want."
        )
    hdus = _file_hdus(path, len(headers))
    end = None  # through EOF: keeps bytes after the last HDU

    if target == 0:
        _start, next_data, _end, next_raw = hdus[1]
        xtension = xtension_of(next_raw)
        if xtension != "IMAGE":
            kind = (
                "a tile-compressed image"
                if header_has_card(next_raw, "ZIMAGE")
                else f"a table (XTENSION = '{xtension}')"
            )
            raise ValueError(
                f"HDU 1 is {kind} and cannot become the primary HDU; every HDU "
                "would shift by one behind a new empty primary. Delete another "
                "HDU, or replace HDU 0 with an empty image instead."
            )
        restamp = [0] if header_has_card(next_raw, *_CHECKSUM_KEYS) else []
        publish_spliced(
            path,
            _nonempty([image_extension_as_primary(next_raw), (path, next_data, end)]),
            restamp=restamp,
        )
        return

    start, _data, stop, _raw = hdus[target]
    publish_spliced(path, _nonempty([(path, 0, start), (path, stop, end)]))
