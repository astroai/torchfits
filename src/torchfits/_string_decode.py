"""Numpy-free string decode for 2-D uint8 FITS byte columns.

FITS stores fixed-width string columns as ``uint8`` tensors of shape
``(N, width)``.  This module provides a single helper, :func:`decode_byte_tensor`,
that decodes such a tensor into a ``list[str]`` using only the Python buffer
protocol — no numpy dependency.
"""

from __future__ import annotations

import codecs
import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch-free at runtime: the Arrow read paths import this
    import torch

DECODE_ERRORS = ("strict", "replace", "ignore")


class DecodeSpec(str):
    """An encoding name that also carries the decode error policy.

    The public table readers take ``encoding=`` and ``decode_errors=``; the
    internal read paths hand one ``encoding`` value down to the few decode
    sites, so the policy travels on that value rather than through every
    internal signature. A plain ``str`` encoding means ``"strict"``.
    """

    errors: str

    def __new__(cls, encoding: str, errors: str = "strict") -> "DecodeSpec":
        if errors not in DECODE_ERRORS:
            raise ValueError(
                f"decode_errors must be one of {', '.join(map(repr, DECODE_ERRORS))}, "
                f"got {errors!r}"
            )
        codecs.lookup(str(encoding))  # LookupError for an unknown encoding
        obj = super().__new__(cls, str(encoding))
        obj.errors = errors
        return obj


def decode_errors_of(encoding: str) -> str:
    """The decode error policy *encoding* carries (``"strict"`` by default)."""
    return getattr(encoding, "errors", "strict")


def _named_decode_error(
    exc: UnicodeDecodeError, column: Any, row: int | None = None
) -> UnicodeDecodeError:
    where = f"column {column!r}" if column is not None else "a string column"
    if row is not None:
        where += f", row {row}"
    return UnicodeDecodeError(
        exc.encoding,
        exc.object,
        exc.start,
        exc.end,
        f"{exc.reason} ({where}); pass encoding='latin-1' (or the encoding the "
        "file uses) to decode it, or decode_errors='replace' / 'ignore' to "
        "accept a lossy decode",
    )


@contextmanager
def column_decode_context(column: Any) -> Iterator[None]:
    """Name *column* in a strict-decode failure raised inside the block."""
    try:
        yield
    except UnicodeDecodeError as exc:
        if "decode_errors=" in str(exc.reason):
            raise
        raise _named_decode_error(exc, column) from exc


def decode_byte_tensor(
    tensor: torch.Tensor,
    encoding: str = "ascii",
    strip: bool = True,
    errors: str | None = None,
    *,
    column: Any = None,
) -> list[str]:
    """Decode a 2-D uint8 tensor ``(N, width)`` into a list of strings.

    Replaces the previous ``np.char.decode`` / ``np.char.rstrip`` path with
    pure-Python byte decoding.  Handles CPU transfer, contiguity, and
    storage offset correctly.

    Args:
        tensor: 2-D ``torch.uint8`` tensor of shape ``(N, width)``.
        encoding: Character encoding for ``bytes.decode``.
        strip: When True, trailing spaces and NULs are stripped.
        errors: ``"strict"`` (raise ``UnicodeDecodeError`` naming *column*
            and the row), ``"replace"`` or ``"ignore"``; ``None`` takes the
            policy *encoding* carries (see :class:`DecodeSpec`), else strict.
            Bytes used to be dropped silently (``errors="ignore"``).
        column: Column name for the error message.

    Returns:
        List of ``N`` decoded strings.
    """
    # detach so we never keep an autograd graph alive for raw byte access
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    n_rows, width = tensor.shape
    if width == 0:
        return [""] * n_rows

    # memcpy the view's window straight from the storage pointer. The obvious
    # ``bytes(tensor.untyped_storage())`` copies the whole backing storage
    # (a small window of a large column would copy the entire column) and
    # ``bytes(UntypedStorage)`` additionally walks elements in Python.
    raw = ctypes.string_at(tensor.data_ptr(), n_rows * width)

    if errors is None:
        errors = decode_errors_of(encoding)
    encoding = str(encoding)
    result: list[str] = []
    for i in range(n_rows):
        try:
            s = raw[i * width : (i + 1) * width].decode(encoding, errors=errors)
        except UnicodeDecodeError as exc:
            raise _named_decode_error(exc, column, i) from exc
        if strip:
            s = s.rstrip(" \x00")
        result.append(s)
    return result
