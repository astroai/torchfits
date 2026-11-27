"""Prefetch bookkeeping in ``torchfits.data.remote``.

Prefetch threads, stored errors and download locks are keyed by
``(cache key, destination)``: one URL staged into two cache directories is two
transfers. A staged-cutout iterator retires its look-ahead prefetches when it
closes, so they neither recreate its removed staging directory nor leave
errors behind for the next iterator.
"""

from __future__ import annotations

import threading
import time
import warnings
from pathlib import Path

import pytest
import torch

import torchfits
from torchfits.data import FitsStagedCutoutIterableDataset
from torchfits.data import remote


@pytest.fixture(autouse=True)
def _clean_prefetch_state():
    def reset() -> None:
        for t in threading.enumerate():
            if t.name in ("torchfits-prefetch", "torchfits-stage-reaper"):
                t.join(timeout=10)
        with remote._prefetch_lock:
            remote._prefetch_threads.clear()
            remote._prefetch_errors.clear()
            remote._prefetch_scopes.clear()

    reset()
    yield
    reset()


@pytest.fixture
def fits_payload(tmp_path) -> bytes:
    src = tmp_path / "src.fits"
    torchfits.write(str(src), torch.ones(32, 32), overwrite=True)
    return src.read_bytes()


def _join_background() -> None:
    for t in threading.enumerate():
        if t.name in ("torchfits-prefetch", "torchfits-stage-reaper"):
            t.join(timeout=10)
            assert not t.is_alive()


def test_same_url_into_two_cache_dirs_is_two_transfers(
    tmp_path, monkeypatch, fits_payload
):
    url = "https://example.test/shared.fits"
    cache_a, cache_b = tmp_path / "A", tmp_path / "B"
    calls: list[Path] = []

    def fake_download_http(u, dest, *, scope=None):
        calls.append(dest)
        time.sleep(0.1)
        if dest.parent == cache_a:
            raise OSError("simulated failure in cache A")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fits_payload)
        return dest

    monkeypatch.setattr(remote, "_download_http", fake_download_http)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        remote.prefetch_urls([url], cache_dir=cache_a)
        remote.prefetch_urls([url], cache_dir=cache_b)
        with remote._prefetch_lock:
            assert len(remote._prefetch_threads) == 2
        _join_background()

    # B's transfer succeeded on its own; A's error is not reported for B.
    local_b = remote.resolve_local_path(url, cache_dir=cache_b)
    assert Path(local_b).parent == cache_b
    assert sorted(p.parent.name for p in calls) == ["A", "B"]
    with pytest.raises(OSError, match="cache A"):
        remote.resolve_local_path(url, cache_dir=cache_a)


class _SlowResp:
    """HTTP response whose body trickles in, one small chunk per read."""

    def __init__(self, payload: bytes, delay: float) -> None:
        self.status = 200
        self.headers = {"Content-Length": str(len(payload)), "ETag": '"v1"'}
        self._payload = payload
        self._delay = delay

    def read(self, n: int = -1) -> bytes:
        time.sleep(self._delay)
        chunk, self._payload = self._payload[:512], self._payload[512:]
        return chunk

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_closed_staged_iterator_retires_its_prefetches(
    tmp_path, monkeypatch, fits_payload
):
    urls = [f"https://example.test/f{i}.fits" for i in range(4)]
    fast = urls[0]

    def fake_http_open(u, *, headers=None, timeout=None):
        return _SlowResp(fits_payload, 0.0 if u == fast else 0.05)

    monkeypatch.setattr(remote, "http_open", fake_http_open)
    staging = tmp_path / "staging"
    ds = FitsStagedCutoutIterableDataset(
        urls, cutouts_per_file=1, cutout_size=8, staging_dir=staging, cleanup=True
    )
    it = iter(ds)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        next(it)
        it.close()
        _join_background()

    # Nothing recreated the removed private stage directory, and the
    # retired transfers left neither files nor errors behind.
    assert list(staging.iterdir()) == []
    with remote._prefetch_lock:
        assert remote._prefetch_errors == {}
        assert remote._prefetch_threads == {}
        assert remote._prefetch_scopes == {}

    # The next iterator over the same URLs is unaffected.
    out = list(
        FitsStagedCutoutIterableDataset(
            urls[1:], cutouts_per_file=1, cutout_size=8, staging_dir=staging
        )
    )
    assert len(out) == 3
    _join_background()
    assert list(staging.iterdir()) == []


def test_retired_scope_never_creates_its_directory(tmp_path, monkeypatch):
    """A transfer that starts after retirement does not ``mkdir`` its dest."""
    opened: list[str] = []
    monkeypatch.setattr(
        remote, "http_open", lambda u, **kw: opened.append(u) or _SlowResp(b"", 0)
    )
    scope = remote.PrefetchScope()
    remote.retire_prefetches(scope)
    stage = tmp_path / "gone"
    remote.prefetch_urls(
        ["https://example.test/late.fits"], cache_dir=stage, scope=scope
    )
    with pytest.raises(remote.PrefetchCancelled):
        remote._download_once(
            "https://example.test/x.fits",
            "https://example.test/x.fits",
            stage / "x.fits",
            scope=scope,
        )
    _join_background()
    assert not stage.exists()
    assert opened == []
