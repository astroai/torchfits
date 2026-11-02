"""Open failures carry CFITSIO's own reason, and TLS URLs on builds without
libcurl fail with an actionable message instead of a generic open error."""

from __future__ import annotations

import pytest

import torchfits
import torchfits._C as _C
from torchfits import http_util

import _loopback_http_server as fixture_server


@pytest.fixture()
def loopback(tmp_path, monkeypatch):
    proc, port, log = fixture_server.start(tmp_path)
    monkeypatch.setattr(http_util, "_resolve_public_addrs", lambda url: ("127.0.0.1",))
    try:
        yield port
    finally:
        proc.kill()
        proc.wait()


def test_has_curl_flag_is_exposed():
    assert isinstance(_C.HAS_CURL, bool)


@pytest.mark.parametrize(
    "url", ["https://archive.example.org/x.fits", "FTPS://archive.example.org/x.fits"]
)
def test_tls_url_without_libcurl_names_the_cause(monkeypatch, url):
    monkeypatch.setattr(http_util, "_resolve_public_addrs", lambda url: ("8.8.8.8",))
    monkeypatch.setattr(_C, "HAS_CURL", False, raising=False)
    for fn in (torchfits.read, torchfits.read_header, torchfits.open):
        with pytest.raises(ValueError, match="without libcurl"):
            fn(url)


def test_open_failure_includes_cfitsio_reason(loopback):
    url = f"http://localhost:{loopback}/missing.fits"
    with pytest.raises((OSError, RuntimeError)) as excinfo:
        torchfits.read_header(url)
    message = str(excinfo.value)
    assert f"Could not open FITS file: {url} (" in message
    assert "http_open" in message or "ffopen" in message


def test_open_does_not_repeat_its_prefix(loopback):
    url = f"http://localhost:{loopback}/missing.fits"
    with pytest.raises(RuntimeError) as excinfo:
        torchfits.open(url)
    assert str(excinfo.value).count("Failed to open FITS file") == 1


def test_cross_host_redirect_error_says_why(loopback):
    with pytest.raises((OSError, RuntimeError), match="redirect .* refused"):
        torchfits.read_header(f"http://localhost:{loopback}/redirect-other")
