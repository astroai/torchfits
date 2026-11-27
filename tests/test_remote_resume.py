"""Download state-machine tests for ``torchfits.data.remote``.

Covers the ``.partial`` -> promote -> cache transition: HTTP 416 recovery on
an over-complete resume offset, empty/short/malformed-length bodies, and
temporary-file hygiene on platforms without ``fcntl``. The SSRF guard and the
pinned fetch itself are covered in ``tests/test_remote_http_range.py`` and
``tests/test_security.py``.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from torchfits.data import remote


@pytest.fixture(autouse=True)
def _clean_prefetch_state():
    with remote._prefetch_lock:
        threads = list(remote._prefetch_threads.values())
    for t in threads:
        t.join(timeout=1.0)
    with remote._prefetch_lock:
        remote._prefetch_threads.clear()
        remote._prefetch_errors.clear()
        remote._prefetch_scopes.clear()
    yield
    with remote._prefetch_lock:
        threads = list(remote._prefetch_threads.values())
    for t in threads:
        t.join(timeout=1.0)
    with remote._prefetch_lock:
        remote._prefetch_threads.clear()
        remote._prefetch_errors.clear()
        remote._prefetch_scopes.clear()


class _FakeResp:
    def __init__(self, status, headers, payload, read_error=None):
        self.status = status
        self.headers = headers
        self._payload = payload
        self._read_error = read_error

    def read(self, n: int = -1) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        data = self._payload
        self._payload = b""
        return data

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _card(text: str) -> bytes:
    return text.ljust(80).encode("ascii")


def _fits_image_bytes(n: int = 600, *, extension_rows: int | None = None) -> bytes:
    """A real, complete FITS file: an int16 vector of *n* pixels (+ optional
    int8 image extension), each HDU padded to 2880 bytes."""

    def hdu(first: str, bitpix: int, axes: list[int], extra: list[str]) -> bytes:
        cards = [first, f"BITPIX  = {bitpix:20d}", f"NAXIS   = {len(axes):20d}"]
        cards += [f"NAXIS{i + 1:<3d}= {a:20d}" for i, a in enumerate(axes)]
        header = b"".join(_card(c) for c in cards + extra + ["END"])
        header = header.ljust(-(-len(header) // 2880) * 2880, b" ")
        count = 1
        for a in axes:
            count *= a
        data = bytes(range(256)) * (count * abs(bitpix) // 8 // 256 + 1)
        data = data[: count * abs(bitpix) // 8]
        return header + data.ljust(-(-len(data) // 2880) * 2880, b"\0")

    body = hdu("SIMPLE  =                    T", 16, [n], [])
    if extension_rows is not None:
        body += hdu(
            "XTENSION= 'IMAGE   '",
            8,
            [100, extension_rows],
            ["PCOUNT  =                    0", "GCOUNT  =                    1"],
        )
    return body


def test_416_on_resume_restarts_instead_of_wedging(tmp_path):
    """A complete-but-unpromoted partial must not wedge every later attempt.

    If a crash lands between the last byte written and ``.partial -> dest``,
    the next resume sends ``Range: bytes=<size>-`` which the server answers
    with 416. That is a dead resume offset: drop the partial and fetch
    cleanly instead of retrying the same doomed Range forever.
    """
    body = _fits_image_bytes()
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/wedged.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(body)  # interrupted after the last byte, before promote
    Path(str(partial) + ".meta").write_text(json.dumps({"etag": '"v1"'}))

    seen: list[dict] = []

    def fake_http_open(u, *, headers=None, timeout=None):
        seen.append(dict(headers or {}))
        if headers and "Range" in headers:
            raise urllib.error.HTTPError(u, 416, "Range Not Satisfiable", None, None)
        return _FakeResp(200, {"Content-Length": str(len(body))}, body)

    with mock.patch("torchfits.data.remote.http_open", side_effect=fake_http_open):
        local = remote.resolve_local_path(url, cache_dir=cache)
    assert Path(local).read_bytes() == body
    # The dead offset was attempted first, then a clean fetch followed.
    assert seen[0]["Range"] == f"bytes={len(body)}-"
    assert "Range" not in seen[1]
    assert not partial.exists()
    assert not Path(str(partial) + ".meta").exists()


def test_empty_body_is_not_promoted_to_cache(tmp_path):
    """A zero-byte response must not become the permanent cached copy.

    With connection-close framing an empty body is indistinguishable from a
    dropped connection; promoting it poisons the cache forever because
    ``resolve_local_path`` treats any existing file as complete.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/empty.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {}, b"")
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="empty download"):
            remote._download_http(url, dest)
    assert not dest.exists()
    assert list(cache.glob("*.partial*")) == []


def test_malformed_content_length_treated_as_unverifiable(tmp_path):
    """A garbage Content-Length must not crash the download.

    ``http.client`` treats a malformed length as unknown framing; the cache
    layer must do the same and verify the body from its FITS structure.
    """
    body = _fits_image_bytes()
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/badlen.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {"Content-Length": "not-a-number"}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        remote._download_http(url, dest)
    assert dest.read_bytes() == body

    resp = _FakeResp(200, {"Content-Length": "not-a-number"}, body[:4000])
    dest.unlink()
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="incomplete FITS download"):
            remote._download_http(url, dest)
    assert not dest.exists()


def test_short_download_is_not_promoted(tmp_path):
    """A body shorter than Content-Length raises and never reaches the cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/short.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {"Content-Length": "100"}, b"only-10b")
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="short download"):
            remote._download_http(url, dest)
    assert not dest.exists()


def test_complete_body_without_content_length_is_promoted(tmp_path):
    """Connection-close framing promotes a body that holds every HDU."""
    body = _fits_image_bytes(extension_rows=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/nolen.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        remote._download_http(url, dest)
    assert dest.read_bytes() == body


def test_complete_body_missing_final_padding_is_promoted(tmp_path):
    """CFITSIO reads a file whose last data unit lacks its 2880 pad."""
    body = _fits_image_bytes(600)
    body = body[: 2880 + 1200]
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/nopad.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    with mock.patch(
        "torchfits.data.remote.http_open", return_value=_FakeResp(200, {}, body)
    ):
        remote._download_http(url, dest)
    assert dest.read_bytes() == body


@pytest.mark.parametrize(
    "cut",
    [
        2000,  # inside the primary header record
        2880 + 1000,  # inside primary pixel data
        2 * 2880 + 4,  # inside the "XTENSION=" keyword of the extension
        2 * 2880 + 2880 + 100,  # inside extension pixel data
    ],
)
def test_truncated_body_without_content_length_is_not_cached(tmp_path, cut):
    """A connection-close body cut short must never reach the permanent cache."""
    body = _fits_image_bytes(extension_rows=40)
    assert len(body) > cut
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/cut.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    with mock.patch(
        "torchfits.data.remote.http_open",
        return_value=_FakeResp(200, {"ETag": '"v1"'}, body[:cut]),
    ):
        with pytest.raises(OSError, match="incomplete FITS download"):
            remote.resolve_local_path(url, cache_dir=cache)
    assert not dest.exists()
    assert list(cache.glob("*.partial*")) == []


@pytest.mark.parametrize("codec", ["gzip", "bz2"])
def test_truncated_compressed_body_without_content_length_is_not_cached(
    tmp_path, codec
):
    import bz2
    import gzip

    compress = gzip.compress if codec == "gzip" else bz2.compress
    packed = compress(_fits_image_bytes(4000, extension_rows=200))
    cache = tmp_path / "cache"
    cache.mkdir()
    url = f"https://example.test/img.fits.{'gz' if codec == 'gzip' else 'bz2'}"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    with mock.patch(
        "torchfits.data.remote.http_open",
        return_value=_FakeResp(200, {}, packed[: len(packed) - 6]),
    ):
        with pytest.raises(OSError, match="incomplete FITS download"):
            remote.resolve_local_path(url, cache_dir=cache)
    assert not dest.exists()

    with mock.patch(
        "torchfits.data.remote.http_open", return_value=_FakeResp(200, {}, packed)
    ):
        local = remote.resolve_local_path(url, cache_dir=cache)
    assert Path(local).read_bytes() == packed


def test_unverifiable_compression_without_content_length_is_not_cached(tmp_path):
    """Unix compress (.Z) has no end-of-stream check: refuse it unframed."""
    body = b"\x1f\x9d\x90" + b"z" * 200
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/img.fits.Z"
    dest = remote.cache_path_for_url(url, cache_dir=cache)
    with mock.patch(
        "torchfits.data.remote.http_open", return_value=_FakeResp(200, {}, body)
    ):
        with pytest.raises(OSError, match="cannot be verified"):
            remote.resolve_local_path(url, cache_dir=cache)
    assert not dest.exists()
    framed = _FakeResp(200, {"Content-Length": str(len(body))}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=framed):
        assert (
            Path(remote.resolve_local_path(url, cache_dir=cache)).read_bytes() == body
        )


def test_truncated_cached_copy_is_dropped_and_refetched(tmp_path):
    """A truncated file cached by an older release heals on the next resolve."""
    body = _fits_image_bytes(extension_rows=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/poisoned-trunc.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)
    dest.write_bytes(body[: 2880 + 1000])
    resp = _FakeResp(200, {"Content-Length": str(len(body))}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp) as fetch:
        local = remote.resolve_local_path(url, cache_dir=cache)
    assert fetch.call_count == 1
    assert Path(local).read_bytes() == body
    # A complete cached copy is served without contacting the server.
    with mock.patch("torchfits.data.remote.http_open") as fetch:
        assert remote.resolve_local_path(url, cache_dir=cache) == local
    assert fetch.call_count == 0


def test_truncated_real_fits_from_loopback_server_is_not_cached(tmp_path, monkeypatch):
    """End to end over a real HTTP/1.0 connection-close response."""
    import http.server

    import torch

    import torchfits
    import torchfits.http_util as hu

    src = tmp_path / "full.fits"
    torchfits.write(str(src), torch.arange(10000.0).reshape(100, 100), overwrite=True)
    full = src.read_bytes()
    state = {"body": full[: 2880 + 10000]}
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):  # noqa: N802 - http.server API
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(hu, "_is_blocked_ip", lambda ip: False)
    url = f"http://127.0.0.1:{server.server_port}/img.fits"
    cache = tmp_path / "cache"
    try:
        with pytest.raises(OSError, match="incomplete FITS download"):
            remote.resolve_local_path(url, cache_dir=cache)
        assert not remote.cache_path_for_url(url, cache_dir=cache).exists()
        state["body"] = full
        local = remote.resolve_local_path(url, cache_dir=cache)
        assert torchfits.read(local).shape == (100, 100)
        assert len(hits) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_failed_download_leaves_no_temporaries_without_fcntl(tmp_path, monkeypatch):
    """Token-named temporaries cannot be resumed; a failed transfer must not
    leave them behind (they would accumulate forever on no-fcntl platforms).
    """
    monkeypatch.setattr(remote, "_fcntl", None)
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/reset.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(
        200,
        {"Content-Length": "4096", "ETag": '"v1"'},
        b"",
        read_error=OSError("connection reset by peer"),
    )
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="connection reset"):
            remote._download_http(url, dest)
    assert list(cache.glob("*.partial*")) == []
    assert list(cache.glob("*.meta*")) == []
    assert not dest.exists()


def test_failed_download_keeps_resume_state_with_fcntl(tmp_path):
    """Canonical (fcntl) partials survive a failed transfer for later resume."""
    if remote._fcntl is None:  # pragma: no cover - non-POSIX
        pytest.skip("resume state requires the canonical fcntl partial name")
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/resume-later.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(
        200,
        {"Content-Length": "4096", "ETag": '"v1"'},
        b"",
        read_error=OSError("connection reset by peer"),
    )
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="connection reset"):
            remote._download_http(url, dest)
    partial = dest.with_suffix(dest.suffix + ".partial")
    assert partial.exists()
    assert Path(str(partial) + ".meta").read_text() == json.dumps(
        {"etag": '"v1"', "last_modified": None}
    )
    assert not dest.exists()


class _CopyState:
    barrier = threading.Barrier(2)
    content = b"VOS-PAYLOAD" * 64


class _FakeVosClient:
    def copy(self, uri, dest):
        # Force both writers to overlap: pass the entry barrier, write, and
        # stay inside copy() until the peer has written too.
        _CopyState.barrier.wait(timeout=10)
        Path(dest).write_bytes(_CopyState.content)
        time.sleep(0.2)


def test_concurrent_vos_downloads_use_private_temporaries(tmp_path, monkeypatch):
    """Two unsynchronized writers must never share one ``.partial``.

    On platforms without ``fcntl`` the cross-process lock is a no-op and
    writer isolation comes from unique per-attempt temp names; ``_download_vos``
    must honor that or concurrent downloaders corrupt/promote each other's
    in-flight file. Two threads calling ``_download_vos`` directly model two
    processes (neither the per-key thread lock nor flock applies there).
    """
    monkeypatch.setattr(remote, "_fcntl", None)
    fake_vos = types.ModuleType("vos")
    fake_vos.Client = _FakeVosClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vos", fake_vos)

    cache = tmp_path / "cache"
    cache.mkdir()
    dest = cache / "mosaic.fits"
    _CopyState.barrier = threading.Barrier(2)

    errors: list[BaseException] = []

    def worker():
        try:
            remote._download_vos("vos:alice/data/mosaic.fits", dest)
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert dest.read_bytes() == _CopyState.content
    assert list(cache.glob("*.partial*")) == []


_HTML = b"<html><body>Service temporarily unavailable</body></html>"


def test_error_page_body_is_not_promoted_to_cache(tmp_path):
    """A 200 HTML error page must not become the permanent cached copy."""
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/outage.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {"Content-Length": str(len(_HTML))}, _HTML)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="not a FITS file") as excinfo:
            remote.resolve_local_path(url, cache_dir=cache)
    assert url in str(excinfo.value)
    assert not dest.exists()
    assert list(cache.glob("*.partial*")) == []


@pytest.mark.parametrize("magic", [None, b"\x1f\x8b\x08", b"BZh9"])
def test_fits_and_compressed_bodies_are_cached(tmp_path, magic):
    # Length-framed compressed bodies are trusted on their Content-Length;
    # plain FITS is always walked, so it must be a real file.
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/ok.fits"
    body = _fits_image_bytes() if magic is None else magic + b"x" * 64
    resp = _FakeResp(200, {"Content-Length": str(len(body))}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        local = remote.resolve_local_path(url, cache_dir=cache)
    assert Path(local).read_bytes() == body


def test_previously_poisoned_cache_entry_is_refetched(tmp_path):
    """A non-FITS file left in the cache by an older release is replaced."""
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/poisoned.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_HTML)
    body = _fits_image_bytes()
    resp = _FakeResp(200, {"Content-Length": str(len(body))}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp) as fetch:
        local = remote.resolve_local_path(url, cache_dir=cache)
    assert fetch.call_count == 1
    assert Path(local).read_bytes() == body


def test_cached_copy_read_failure_names_the_url(tmp_path, monkeypatch):
    """A broken cached copy reports the URL it came from, not just a hash path."""
    import torchfits
    from torchfits._io_engine import subset

    cached = tmp_path / "cached.fits"
    cached.write_bytes(b"SIMPLE  =" + b" " * 2871)  # FITS magic, broken header
    url = "https://example.test/broken.fits"

    def no_range(*args, **kwargs):
        raise subset.HttpRangeNotSatisfied("no ranges")

    monkeypatch.setattr(subset, "read_subset_http", no_range)
    monkeypatch.setattr(
        "torchfits._io_engine.http_subset.locate_uncompressed_2d", no_range
    )
    monkeypatch.setattr(remote, "resolve_local_path", lambda p, **kw: str(cached))
    monkeypatch.setattr(
        torchfits.http_util, "_resolve_public_addrs", lambda u: ("8.8.8.8",)
    )
    with pytest.raises((OSError, RuntimeError)) as excinfo:
        torchfits.read_subset(url, 0, 0, 0, 2, 2)
    text = str(excinfo.value) + " ".join(getattr(excinfo.value, "__notes__", []))
    assert url in text


def test_python_fetched_https_works_without_libcurl(tmp_path, monkeypatch):
    """read_subset fetches http(s) in Python, so a build without libcurl must
    still serve it; only CFITSIO-opened routes need the curl driver."""
    import torch

    import torchfits
    from torchfits._io_engine import paths, subset

    cached = tmp_path / "cached.fits"
    torchfits.write(str(cached), torch.arange(16.0).reshape(4, 4), overwrite=True)
    url = "https://example.test/image.fits"

    def no_range(*args, **kwargs):
        raise subset.HttpRangeNotSatisfied("no ranges")

    monkeypatch.setattr(paths, "has_curl_support", lambda: False)
    monkeypatch.setattr(subset, "read_subset_http", no_range)
    monkeypatch.setattr(
        "torchfits._io_engine.http_subset.locate_uncompressed_2d", no_range
    )
    monkeypatch.setattr(remote, "resolve_local_path", lambda p, **kw: str(cached))
    monkeypatch.setattr(
        torchfits.http_util, "_resolve_public_addrs", lambda u: ("8.8.8.8",)
    )
    cut = torchfits.read_subset(url, 0, 0, 0, 2, 2)
    assert cut.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    with pytest.raises(ValueError, match="libcurl"):
        torchfits.read(url)
