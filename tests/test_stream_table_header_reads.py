"""``stream_table`` reads the HDU header at most once, and only when needed (r5a-14).

With ``total_rows=`` supplied it used to probe the HDU type again (and, for
``mmap=True``, read the whole header a second time for the scaled/VLA checks)
even when the caller already held that header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import torchfits
import torchfits.table
from torchfits._io_engine.table_streaming import stream_table


@pytest.fixture()
def table_path(tmp_path: Path) -> str:
    path = str(tmp_path / "t.fits")
    torchfits.table.write(
        path,
        {"A": np.arange(10, dtype=np.int32), "B": np.linspace(0, 1, 10)},
        overwrite=True,
    )
    return path


def _spy(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], object]:
    calls: list[str] = []
    real_header = torchfits.read_header
    real_type = torchfits.read_hdu_type

    def get_header(path, hdu):  # type: ignore[no-untyped-def]
        calls.append("header")
        return real_header(path, hdu)

    def read_hdu_type(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("hdu_type")
        return real_type(*args, **kwargs)

    monkeypatch.setattr(torchfits, "read_hdu_type", read_hdu_type)
    monkeypatch.setattr(torchfits, "read_header", get_header)
    return calls, get_header


def _rows(chunks) -> list[int]:  # type: ignore[no-untyped-def]
    return torch.cat([c["A"] for c in chunks]).tolist()


@pytest.mark.parametrize("mmap", [True, False])
def test_header_passed_in_is_not_read_again(table_path, monkeypatch, mmap):
    header = torchfits.read_header(table_path, 1)
    calls, get_header = _spy(monkeypatch)
    chunks = stream_table(
        get_header, table_path, 1, chunk_rows=4, mmap=mmap, total_rows=10, header=header
    )
    assert _rows(chunks) == list(range(10))
    assert calls == []


def test_total_rows_without_mmap_needs_no_header(table_path, monkeypatch):
    calls, get_header = _spy(monkeypatch)
    chunks = stream_table(get_header, table_path, 1, chunk_rows=4, total_rows=10)
    assert _rows(chunks) == list(range(10))
    assert calls == []


def test_mmap_reads_the_header_once(table_path, monkeypatch):
    calls, get_header = _spy(monkeypatch)
    chunks = stream_table(get_header, table_path, 1, chunk_rows=4, mmap=True)
    assert _rows(chunks) == list(range(10))
    assert calls == ["header"]
    calls.clear()
    chunks = stream_table(
        get_header, table_path, 1, chunk_rows=4, mmap=True, total_rows=10
    )
    assert _rows(chunks) == list(range(10))
    assert calls == ["header"]
