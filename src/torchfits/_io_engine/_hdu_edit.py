"""Torch-free HDU edit helpers: locate an HDU by index or EXTNAME, and delete it.

``delete_hdu`` writes no HDU, only copies byte ranges, so it lives apart from
``_hdu_rewrite`` (whose writers import torch) and stays a metadata operation.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union

from ..hdu import Header
from ._file_ops import _resolve_compression_algorithm
from ._hdu_splice import (
    Piece,
    header_has_card,
    image_extension_as_primary,
    publish_spliced,
    scan_hdus,
    xtension_of,
)
from .hdu_api import open_hdulist


_CHECKSUM_KEYS = ("CHECKSUM", "DATASUM")


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


def _has_checksums(hdus: List[Tuple[int, int, int, bytes]]) -> bool:
    return any(header_has_card(raw, *_CHECKSUM_KEYS) for *_span, raw in hdus)


def _nonempty(pieces: List[Piece]) -> List[Piece]:
    return [p for p in pieces if isinstance(p, bytes) or p[2] is None or p[2] > p[1]]


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
