"""Header index for gzip-compressed FITS files (``*.fits.gz``).

CFITSIO opens a ``.gz`` file by decompressing all of it into memory, on every
open, and each metadata call opens the file. Listing a 20-HDU, 80 MB file
with ``torchfits.open`` decompressed it ~40 times (8 s), and ``read`` paid
~10 decompressions in metadata probes before touching a pixel.

Here one CFITSIO handle reads every header, HDU type and the HDU count in a
single open (one decompression) and the answers are kept per file. They are
the very strings CFITSIO's per-call probes return. An entry is served only
while the file's ``(size, mtime_ns, inode, device)`` signature is unchanged,
the cache is bounded in entries and header bytes, and any failure leaves the
call to the per-call CFITSIO path (callers get None), so errors are unchanged.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

_MAX_ENTRIES = 64
_MAX_ENTRY_HEADER_BYTES = 8 * 1024 * 1024

_lock = threading.Lock()
# path -> (signature, header strings, HDU type names)
_index: "OrderedDict[str, Tuple[Any, List[str], List[str]]]" = OrderedDict()


def _signature(path: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ino), int(st.st_dev))


def _eligible(path: Any) -> bool:
    """A plain local gzip file named ``.gz`` (no CFITSIO filter / URL syntax)."""
    if not isinstance(path, str) or not path.lower().endswith(".gz"):
        return False
    if "://" in path or path.startswith("!"):
        return False
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _load(path: str) -> Optional[Tuple[List[str], List[str]]]:
    from torchfits._core_api import _core

    try:
        with _core().Metadata(path) as meta:
            n = int(meta.num_hdus())
            headers = [str(meta.header_text(i)) for i in range(n)]
            types = [str(meta.hdu_type(i)) for i in range(n)]
    except (RuntimeError, OSError, ValueError):
        return None
    return headers, types


def _entry(path: Any) -> Optional[Tuple[List[str], List[str]]]:
    if not _eligible(path):
        return None
    sig = _signature(path)
    if sig is None:
        return None
    with _lock:
        cached = _index.get(path)
        if cached is not None:
            if cached[0] == sig:
                _index.move_to_end(path)
                return cached[1], cached[2]
            del _index[path]
    loaded = _load(path)
    if loaded is None:
        return None
    # Never keep answers from a read that raced a rewrite of the file.
    if _signature(path) == sig and sum(map(len, loaded[0])) <= _MAX_ENTRY_HEADER_BYTES:
        with _lock:
            _index[path] = (sig, loaded[0], loaded[1])
            _index.move_to_end(path)
            while len(_index) > _MAX_ENTRIES:
                _index.popitem(last=False)
    return loaded


def _valid_hdu(hdu: Any, count: int) -> bool:
    return isinstance(hdu, int) and not isinstance(hdu, bool) and 0 <= hdu < count


def header_string(path: Any, hdu: Any) -> Optional[str]:
    """CFITSIO's header string for ``hdu`` of a ``.gz`` file, else None."""
    entry = _entry(path)
    if entry is None or not _valid_hdu(hdu, len(entry[0])):
        return None
    return entry[0][int(hdu)]


def num_hdus(path: Any) -> Optional[int]:
    entry = _entry(path)
    return None if entry is None else len(entry[0])


def hdu_type(path: Any, hdu: Any) -> Optional[str]:
    entry = _entry(path)
    if entry is None or not _valid_hdu(hdu, len(entry[1])):
        return None
    return entry[1][int(hdu)]


def invalidate(path: Any) -> None:
    """Drop ``path``'s entry, or every entry when ``path`` is None."""
    with _lock:
        if path is None:
            _index.clear()
        else:
            _index.pop(path, None)
