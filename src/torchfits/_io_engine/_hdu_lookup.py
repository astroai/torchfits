"""Typed HDU-lookup errors for every read/edit route.

CFITSIO reports an unknown EXTNAME and an HDU index past the end through status codes that reach Python as RuntimeError or OSError with
route-specific text. Rather than match that text, the routes call
:func:`raise_lookup_failure` from their ``except`` block: it *re-checks* the
request (the index is below the HDU count, the name resolves)
and raises :class:`HDUNotFoundError` only when that check fails. The successful path pays nothing; an unrelated I/O error
(corrupt data, permissions) re-raises unchanged because the probe passes.

Torch-free: only the metadata core is touched, and only on a failure.
"""

from __future__ import annotations

from typing import Any, Callable, NoReturn, TypeVar

from .._errors import HDUNotFoundError
from .paths import is_cfitsio_network_url

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
    """Raise the standard error for an integer ``hdu`` that is not in ``path``."""
    raise HDUNotFoundError.for_request(path, hdu, count_hdus(path))


def raise_lookup_failure(path: Any, hdu: Any, exc: BaseException) -> None:
    """From an ``except`` block: re-raise a lookup failure with its own type.

    Raises :class:`HDUNotFoundError` when ``hdu`` (index or EXTNAME) is
    verifiably absent. Returns normally otherwise, so the caller re-raises the
    original exception. Remote paths are never probed.
    """
    if isinstance(exc, HDUNotFoundError):
        return
    # Only the native layer's error types can hide a lookup failure; a
    # TypeError/ValueError is an argument error and keeps its type.
    if not isinstance(exc, (RuntimeError, OSError)):
        return
    if not is_local_fits_path(path):
        return
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
    """Native EXTNAME/HDUNAME lookup raising HDUNotFoundError when absent."""
    import torchfits._C as cpp

    return int(call_with_lookup(path, name, cpp.resolve_hdu_name_cached, path, name))


def raise_lookup_failures(path: Any, hdus: Any, exc: BaseException) -> None:
    """:func:`raise_lookup_failure` for each of several requested HDUs."""
    for hdu in hdus:
        raise_lookup_failure(path, hdu, exc)
