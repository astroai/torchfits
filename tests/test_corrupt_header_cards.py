"""Corrupt table header cards fail (or read) the same way on every route."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import torchfits


def _table(tmp_path: Path) -> Path:
    path = tmp_path / "t.fits"
    torchfits.table.write(
        str(path),
        {"ID": np.arange(3, dtype=np.int32), "X": np.arange(3, dtype=np.int16)},
    )
    return path


def _patch_card(path: Path, key: str, card: bytes) -> None:
    raw = bytearray(path.read_bytes())
    start = raw.find(key.ljust(8).encode() + b"=", 2880)
    assert start > 0, key
    raw[start : start + 80] = card.ljust(80)
    path.write_bytes(bytes(raw))


def test_stray_byte_in_a_tform_card_is_dropped_everywhere(tmp_path):
    # read_header (and every reader) drops a stray high-bit byte from a card
    # value; read_colnames / read_table_info returned it raw and failed to
    # decode it (UnicodeDecodeError).
    path = _table(tmp_path)
    _patch_card(path, "TFORM2", b"TFORM2  = '1I\xd2     '")
    header = torchfits.read_header(str(path), 1)
    info = torchfits.read_table_info(str(path), 1)
    assert info["colnames"] == torchfits.read_colnames(str(path), 1) == ["ID", "X"]
    assert info["tforms"][1] == str(header["TFORM2"]).strip() == "1I"
    assert torchfits.table.read(str(path))["X"].to_pylist() == [0, 1, 2]

