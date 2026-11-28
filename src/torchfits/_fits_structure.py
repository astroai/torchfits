"""Torch-free FITS structure walk: does a byte stream hold every declared HDU?

Shared by the remote download cache (``data/remote.py``), which refuses a
body shorter than its own headers declare, and ``torchfits verify``, which
reports such a file as truncated instead of trusting its checksum keywords
(or their absence). Only the Python standard library is used, so the CLI's
metadata commands stay free of the tensor runtime.
"""

from __future__ import annotations

import bz2
import gzip
import os
from typing import IO

FITS_BLOCK = 2880
FITS_CARD = 80
ByteStream = IO[bytes] | gzip.GzipFile | bz2.BZ2File


class FitsStructureError(ValueError):
    """A body that does not hold every HDU its own headers declare."""


def header_int(cards: dict[str, bytes], key: str, default: int | None = None) -> int:
    raw = cards.get(key)
    if raw is None:
        if default is None:
            raise FitsStructureError(f"header lacks {key}")
        return default
    text = raw.split(b"/", 1)[0].strip()
    try:
        return int(text)
    except ValueError:
        raise FitsStructureError(f"header {key} is not an integer: {text!r}") from None


def hdu_data_size(cards: dict[str, bytes], primary: bool) -> int:
    """Data-unit byte count from BITPIX/NAXISn/PCOUNT/GCOUNT (FITS 4.0 §4.4.1)."""
    bitpix = header_int(cards, "BITPIX")
    naxis = header_int(cards, "NAXIS")
    if bitpix not in (8, 16, 32, 64, -32, -64) or not 0 <= naxis <= 999:
        raise FitsStructureError(f"invalid BITPIX={bitpix} / NAXIS={naxis}")
    if naxis == 0:
        return 0
    axes = [header_int(cards, f"NAXIS{i}") for i in range(1, naxis + 1)]
    if any(n < 0 for n in axes):
        raise FitsStructureError(f"negative axis length in {axes}")
    groups = cards.get("GROUPS", b"").split(b"/", 1)[0].strip() == b"T"
    if primary and groups and axes[0] == 0:
        axes = axes[1:]  # random groups: NAXIS1 = 0 is not a real axis
    count = 1
    for n in axes:
        count *= n
    pcount = header_int(cards, "PCOUNT", 0)
    gcount = header_int(cards, "GCOUNT", 1)
    return abs(bitpix) // 8 * gcount * (pcount + count)


def skip_bytes(handle: ByteStream, n: int, size: int | None) -> int:
    """Advance *n* bytes; return how many were actually there."""
    if n <= 0:
        return 0
    if size is not None:
        pos = handle.tell()
        got = max(0, min(n, size - pos))
        handle.seek(pos + got)
        return got
    got = 0
    while got < n:
        chunk = handle.read(min(n - got, 1 << 20))
        if not chunk:
            break
        got += len(chunk)
    return got


def walk_fits_hdus(handle: ByteStream, size: int | None) -> int:
    """Check that *handle* covers every HDU its headers declare.

    Header blocks are read up to each ``END`` card and each data unit is
    sized from the header; a body that ends inside a header or a data unit
    raises :class:`FitsStructureError`. A missing final 2880-byte pad is
    tolerated (CFITSIO reads such files), and bytes after the last HDU that
    do not start an ``XTENSION`` header are ignored as trailing junk.
    *size* is the byte length for seekable files, ``None`` for streams.
    Returns the number of HDUs walked.
    """
    index = 0
    while True:
        block = handle.read(FITS_BLOCK)
        if index == 0:
            if not block.startswith(b"SIMPLE  ="):
                raise FitsStructureError("no primary header")
        elif not block or not b"XTENSION=".startswith(block[:9]):
            return index
        cards: dict[str, bytes] = {}
        found_end = False
        while not found_end:
            for off in range(0, len(block) - FITS_CARD + 1, FITS_CARD):
                card = block[off : off + FITS_CARD]
                key = card[:8].rstrip().decode("ascii", "replace")
                if key == "END":
                    found_end = True
                    break
                if card[8:10] == b"= " and key not in cards:
                    cards[key] = card[10:]
            if found_end:
                break
            if len(block) < FITS_BLOCK:
                raise FitsStructureError(f"HDU {index}: header ends before END card")
            block = handle.read(FITS_BLOCK)
        if len(block) < FITS_BLOCK:
            # CFITSIO refuses a header whose last 2880-byte record is short,
            # even when the END card itself arrived.
            raise FitsStructureError(f"HDU {index}: header record truncated")
        data_size = hdu_data_size(cards, index == 0)
        got = skip_bytes(handle, data_size, size)
        if got < data_size:
            raise FitsStructureError(
                f"HDU {index}: data truncated ({got} of {data_size} bytes)"
            )
        skip_bytes(handle, -data_size % FITS_BLOCK, size)
        index += 1


def check_plain_fits_extent(path: str | os.PathLike[str]) -> bool:
    """Walk a local, uncompressed FITS file; raise if it is truncated.

    Returns ``True`` when the file was walked and covers every HDU its
    headers declare, ``False`` when it is not a plain FITS file (gzip,
    bzip2, ... or anything without a ``SIMPLE`` card), which this cheap
    seek-only check does not decompress. Raises :class:`FitsStructureError`
    for a plain FITS file that ends inside a header or a data unit.
    """
    with open(path, "rb") as handle:
        if not handle.read(9).startswith(b"SIMPLE  ="):
            return False
        handle.seek(0)
        walk_fits_hdus(handle, os.fstat(handle.fileno()).st_size)
    return True
