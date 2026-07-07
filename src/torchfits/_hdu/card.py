"""FITS header card — a single (key, value, comment) record."""

from __future__ import annotations

from typing import Any, NamedTuple


class Card(NamedTuple):
    key: str
    value: Any = None
    comment: str = ""

    @property
    def keyword(self) -> str:
        return self.key
