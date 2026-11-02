"""CFITSIO's http driver must not follow a redirect to a host the guard never saw.

The SSRF guard validates only the URL handed to CFITSIO. A public server
answering 302 to an internal host used to be followed by the driver without
any re-check; the vendored patch refuses redirects that leave the original
host. The fixture server runs out of process (CFITSIO holds the GIL).
"""

from __future__ import annotations

import pytest

import torchfits
from torchfits import http_util

import _loopback_http_server as fixture_server


@pytest.fixture()
def loopback(tmp_path, monkeypatch):
    proc, port, log = fixture_server.start(tmp_path)

    # Treat the loopback fixture as a public host so the guard lets CFITSIO in.
    monkeypatch.setattr(http_util, "_resolve_public_addrs", lambda url: ("127.0.0.1",))
    try:
        yield port, log
    finally:
        proc.kill()
        proc.wait()


def test_cross_host_redirect_is_not_followed(loopback):
    port, log = loopback
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read(f"http://localhost:{port}/redirect-other")
    hits = log.read_text()
    assert "GET /redirect-other" in hits
    assert "/secret.fits" not in hits


def test_same_host_redirect_still_followed(loopback):
    port, log = loopback
    data = torchfits.read(f"http://localhost:{port}/redirect-same")
    assert data.flatten().tolist() == list(range(12))
    assert "GET /data.fits" in log.read_text()
