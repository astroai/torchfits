"""Public exception types shared by every torchfits route.

Torch-free and import-free on purpose: ``import torchfits`` exposes these
without loading the tensor runtime or the native extension.
"""

from __future__ import annotations

import os
from typing import Any


class HDUNotFoundError(KeyError, IndexError, ValueError):
    """The requested HDU does not exist in the file.

    Raised for an EXTNAME (or HDUNAME) that no HDU carries and for an integer
    index that is negative or past the last HDU. It subclasses ``KeyError``,
    ``IndexError`` and ``ValueError`` so code written against any of the types
    earlier releases raised keeps catching it.

    Attributes ``hdu``, ``path`` and ``n_hdus`` (``None`` when unknown) carry
    the request and the file's HDU count.
    """

    def __init__(
        self,
        message: str,
        *,
        hdu: Any = None,
        path: str | None = None,
        n_hdus: int | None = None,
    ) -> None:
        super().__init__(message)
        self.hdu = hdu
        self.path = path
        self.n_hdus = n_hdus

    def __str__(self) -> str:
        # KeyError.__str__ repr-quotes its argument; this is a sentence.
        return str(self.args[0]) if self.args else ""

    def __reduce__(self) -> Any:
        # Keep the keyword attributes across pickling (DataLoader workers).
        return (
            _rebuild_hdu_not_found,
            (str(self), self.hdu, self.path, self.n_hdus),
        )

    @classmethod
    def for_request(
        cls, path: Any, hdu: Any, n_hdus: int | None = None
    ) -> HDUNotFoundError:
        """Build the standard message for ``hdu`` missing from ``path``."""
        where = os.fspath(path) if isinstance(path, (str, os.PathLike)) else path
        count = "" if n_hdus is None else f" ({n_hdus} HDU{'s' if n_hdus != 1 else ''})"
        if isinstance(hdu, str):
            message = f"HDU {hdu!r} not found in {where}{count}"
        else:
            message = f"HDU index {hdu} out of range for {where}{count}"
            if isinstance(hdu, int) and hdu < 0:
                message += ": an HDU index must be non-negative"
        return cls(message, hdu=hdu, path=where, n_hdus=n_hdus)


def _rebuild_hdu_not_found(
    message: str, hdu: Any, path: str | None, n_hdus: int | None
) -> HDUNotFoundError:
    return HDUNotFoundError(message, hdu=hdu, path=path, n_hdus=n_hdus)


__all__ = ["HDUNotFoundError"]
