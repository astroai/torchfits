"""FITS checksum helpers for the torchfits I/O engine."""

from __future__ import annotations

import operator
from typing import Any, Dict, Union


def _cpp() -> Any:
    """Resolve the native extension lazily.

    Importing it maps libtorch but does not import the Python ``torch`` module;
    keep the import lazy so checksum metadata does not initialize tensor state.
    """
    import torchfits._C as cpp

    return cpp


def _validate_hdu(hdu: Union[int, str], path: str) -> int:
    """Index of ``hdu``: a non-negative int, or an EXTNAME resolved like read()."""
    if isinstance(hdu, str):
        from .hdu_api import _resolve_hdu_index, autodetect_hdu

        return _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    if isinstance(hdu, bool):
        raise TypeError("hdu must be a non-negative integer or an EXTNAME string")
    try:
        hdu = operator.index(hdu)
    except TypeError:
        raise TypeError(
            "hdu must be a non-negative integer or an EXTNAME string"
        ) from None
    if hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    return int(hdu)


def write_checksums(path: str, hdu: Union[int, str] = 0) -> None:
    """Compute and write DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    ``hdu`` is an index or an EXTNAME (case-insensitive, as in ``read()``).
    """
    from .paths import coerce_fits_path, guard_fits_path

    from .caches import invalidate_path_caches

    path = coerce_fits_path(path)
    guard_fits_path(path)
    cpp = _cpp()
    cpp.write_hdu_checksums(str(path), _validate_hdu(hdu, str(path)))
    # The header just gained CHECKSUM/DATASUM; cached headers keyed on an
    # unchanged size/mtime (coarse filesystems) would keep serving the old one.
    invalidate_path_caches(str(path))
    cpp.clear_shared_read_meta_cache()


def verify_checksums(path: str, hdu: Union[int, str] = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    ``hdu`` is an index or an EXTNAME (case-insensitive, as in ``read()``).

    CFITSIO ``ffvcks`` status codes (``datastatus`` / ``hdustatus``):
    - ``0`` — checksum keywords absent (nothing to verify)
    - ``1`` — checksum present and correct
    - ``-1`` — checksum present but incorrect (corrupt)

    Returns a dict with ``datastatus``, ``hdustatus``, ``ok``, ``present``,
    and ``status`` (``"ok"``, ``"no_checksums"``, or ``"fail"``).
    ``status`` is ``"fail"`` only when a present checksum is incorrect; a
    correct DATASUM without a CHECKSUM keyword (or vice versa) is ``"ok"``.
    ``present`` is False when CFITSIO reports no checksum keywords at all.
    """
    from .paths import coerce_fits_path, guard_fits_path

    path = coerce_fits_path(path)
    guard_fits_path(path)
    datastatus, hdustatus = _cpp().verify_hdu_checksums(
        str(path), _validate_hdu(hdu, str(path))
    )
    data_i = int(datastatus)
    hdu_i = int(hdustatus)

    if data_i < 0 or hdu_i < 0:
        status_str = "fail"
        ok = False
        present = True
    elif data_i == 0 and hdu_i == 0:
        status_str = "no_checksums"
        ok = True
        present = False
    else:
        status_str = "ok"
        ok = True
        present = True

    return {
        "datastatus": data_i,
        "hdustatus": hdu_i,
        "ok": ok,
        "present": present,
        "status": status_str,
    }
