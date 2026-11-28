"""Behaviour pinned by the ``except Exception`` audit.

Best-effort fallbacks that used to pass silently now log at DEBUG, and
conversions that caught every exception now catch only the conversion
failures they exist for, so a real error (memory, I/O) propagates.
"""

from __future__ import annotations

import logging

import pytest
import torch

import torchfits
import torchfits._C as cpp
from torchfits._hdu.table_hdu_ref import TableHDURef
from torchfits._io_engine import _write_helpers, caches, hdu_api


def test_reader_eviction_failure_is_logged(monkeypatch, caplog, tmp_path):
    def boom(path):
        raise RuntimeError("evict failed")

    monkeypatch.setattr(cpp, "evict_cached_reader", boom)
    with caplog.at_level(logging.DEBUG, logger=caches.__name__):
        caches.invalidate_path_caches(str(tmp_path / "x.fits"))
    assert "evicting cached table readers" in caplog.text


def test_hdu_type_probe_failure_is_logged_and_skipped(monkeypatch, caplog, tmp_path):
    path = str(tmp_path / "img.fits")
    torchfits.write(path, [torch.zeros(0), torch.ones(3, 3)], overwrite=True)
    torchfits.clear_file_cache()
    real = cpp.get_hdu_type

    def flaky(handle, index):
        if index == 1:
            raise RuntimeError("bad HDU type")
        return real(handle, index)

    monkeypatch.setattr(cpp, "get_hdu_type", flaky)
    with caplog.at_level(logging.DEBUG, logger=hdu_api.__name__):
        assert hdu_api.find_first_hdu(path) is None
    assert "get_hdu_type failed" in caplog.text


class _Column:
    """An Arrow-typed column whose to_numpy fails with *exc*."""

    def __init__(self, exc: BaseException) -> None:
        import pyarrow as pa

        self._exc = exc
        self.type = pa.int32()

    def to_numpy(self, zero_copy_only: bool = True):
        raise self._exc

    def to_pylist(self):
        return [1, 2]


def test_arrow_conversion_failure_falls_back_to_pylist():
    pa = pytest.importorskip("pyarrow")
    column = _Column(pa.ArrowInvalid("cannot convert"))
    assert _write_helpers._arrow_flat_column_values(column) == [1, 2]


def test_arrow_conversion_does_not_swallow_memory_errors():
    pytest.importorskip("pyarrow")
    with pytest.raises(MemoryError):
        _write_helpers._arrow_flat_column_values(_Column(MemoryError()))


def test_table_ref_ascii_probe_propagates_header_read_errors(tmp_path):
    ref = TableHDURef(source_path=str(tmp_path / "missing.fits"), source_hdu=1)
    with pytest.raises((OSError, RuntimeError)):
        ref._is_ascii_table()
    assert TableHDURef(header=object())._is_ascii_table() is False  # type: ignore[arg-type]
