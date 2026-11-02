"""Typed HDU-lookup and missing-file errors for every read/edit route.

CFITSIO reports a missing file, an unknown EXTNAME and an HDU index past the
end through status codes that reach Python as RuntimeError or OSError with
route-specific text. Rather than match that text, the routes call
:func:`raise_lookup_failure` from their ``except`` block: it *re-checks* the
request (the file exists, the index is below the HDU count, the name resolves)
and raises :class:`FileNotFoundError` or :class:`HDUNotFoundError` only when
that check fails. The successful path pays nothing; an unrelated I/O error
(corrupt data, permissions) re-raises unchanged because the probe passes.

Torch-free: only the metadata core is touched, and only on a failure.
"""

from __future__ import annotations

import os
from typing import Any, Callable, NoReturn, TypeVar

from .._errors import FITSFileNotFoundError, HDUNotFoundError
from .paths import cfitsio_base_path, is_cfitsio_network_url

_T = TypeVar("_T")


def is_local_fits_path(path: Any) -> bool:
    """True for a path CFITSIO opens from the local filesystem."""
    if not isinstance(path, str) or not path:
        return False
    stripped = path.strip()
    if stripped in ("-", "stdin", "stdout"):
        return False
    if is_cfitsio_network_url(stripped) or "://" in stripped:
        return False
    return not stripped.lower().startswith(("vos:", "vault:"))


def missing_file_error(path: Any) -> FITSFileNotFoundError | None:
    """``FileNotFoundError`` for a local path that does not exist, else ``None``."""
    if not is_local_fits_path(path):
        return None
    if os.path.exists(cfitsio_base_path(path)):
        return None
    return FITSFileNotFoundError(f"FITS file not found: {path}")


def require_existing_file(path: Any) -> None:
    """Raise ``FileNotFoundError`` when a local ``path`` does not exist."""
    error = missing_file_error(path)
    if error is not None:
        raise error


def count_hdus(path: str) -> int | None:
    """HDU count of a local file, or ``None`` when it cannot be read."""
    if not is_local_fits_path(path):
        return None
    try:
        from .. import _core_api

        return int(_core_api.read_num_hdus(path))
    except Exception:
        return None


def raise_hdu_index_error(path: str, hdu: int) -> NoReturn:
    """Raise the standard error for an integer ``hdu`` that is not in ``path``.

    A negative index is wrong for every file, so it is reported without
    first checking that ``path`` exists (an argument error comes first).
    """
    if hdu >= 0:
        require_existing_file(path)
    raise HDUNotFoundError.for_request(path, hdu, count_hdus(path))


def raise_lookup_failure(path: Any, hdu: Any, exc: BaseException) -> None:
    """From an ``except`` block: re-raise a lookup failure with its own type.

    Raises ``FileNotFoundError`` when ``path`` is a missing local file, and
    :class:`HDUNotFoundError` when ``hdu`` (index or EXTNAME) is verifiably
    absent. Returns normally when neither holds, so the caller re-raises the
    original exception. Remote paths are never probed.
    """
    if isinstance(exc, (HDUNotFoundError, FileNotFoundError)):
        return
    # Only the native layer's error types can hide a lookup failure; a
    # TypeError/ValueError is an argument error and keeps its type.
    if not isinstance(exc, (RuntimeError, OSError)):
        return
    if not is_local_fits_path(path):
        return
    error = missing_file_error(path)
    if error is not None:
        raise error from exc
    if isinstance(hdu, bool):
        return
    if isinstance(hdu, int):
        n_hdus = count_hdus(path)
        if hdu < 0 or (n_hdus is not None and hdu >= n_hdus):
            raise HDUNotFoundError.for_request(path, hdu, n_hdus) from exc
        return
    if isinstance(hdu, str) and hdu.strip().lower() != "auto":
        from .hdu_api import _resolve_hdu_index, autodetect_hdu

        try:
            _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
        except HDUNotFoundError as lookup:
            raise lookup from exc
        except Exception:
            return


def call_with_lookup(path: Any, hdu: Any, func: Callable[..., _T], *args: Any) -> _T:
    """``func(*args)``, translating a lookup failure via :func:`raise_lookup_failure`."""
    try:
        return func(*args)
    except Exception as exc:
        raise_lookup_failure(path, hdu, exc)
        raise


def resolve_hdu_name(path: str, name: str) -> int:
    """Native EXTNAME/HDUNAME lookup with typed missing-file / missing-HDU errors."""
    import torchfits._C as cpp

    return int(call_with_lookup(path, name, cpp.resolve_hdu_name_cached, path, name))


def raise_lookup_failures(path: Any, hdus: Any, exc: BaseException) -> None:
    """:func:`raise_lookup_failure` for each of several requested HDUs."""
    for hdu in hdus:
        raise_lookup_failure(path, hdu, exc)
