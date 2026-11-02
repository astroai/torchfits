"""An '@' after the authority must not pick the host CFITSIO connects to.

CFITSIO's http driver used to take the host after the first '@' anywhere in
the URL, path and query included, while the SSRF guard parses the URL like
every other client (``urlparse``). ``http://public/x@127.0.0.1:port/a.fits``
therefore passed the guard and reached loopback. The guard now refuses '@' in
CFITSIO network URLs, and the vendored driver only honours an '@' inside the
authority. The fixture servers run out of process (CFITSIO holds the GIL).
"""

from __future__ import annotations

import pytest

import torchfits
from torchfits import http_util
from torchfits._io_engine import paths

import _loopback_http_server as fixture_server


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/x@127.0.0.1:8080/a.fits",
        "http://8.8.8.8:80/p?q=@127.0.0.1:8080/a.fits",
        "http://user:pw@8.8.8.8/a.fits",
        "httpmem://8.8.8.8/x@127.0.0.1/a.fits",
        "ftp://8.8.8.8/x@127.0.0.1/a.fits",
        "!http://8.8.8.8/x@127.0.0.1/a.fits",
    ],
)
def test_guard_refuses_at_sign_in_cfitsio_urls(monkeypatch, url):
    # The host before '@' resolves as public: only the '@' rule can refuse.
    monkeypatch.setattr(http_util, "_resolve_public_addrs", lambda u: ("8.8.8.8",))
    with pytest.raises(http_util.HttpBlockedError, match="'@'"):
        http_util.guard_cfitsio_remote_path(url)
    with pytest.raises(http_util.HttpBlockedError, match="'@'"):
        torchfits.read_header(url)


@pytest.fixture()
def two_servers(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    proc_a, port_a, log_a = fixture_server.start(tmp_path / "a")
    proc_b, port_b, log_b = fixture_server.start(tmp_path / "b")
    # Bypass the guard entirely so only the patched CFITSIO parser decides.
    monkeypatch.setattr(paths, "guard_cfitsio_remote_path", lambda path: None)
    try:
        yield (port_a, log_a), (port_b, log_b)
    finally:
        for proc in (proc_a, proc_b):
            proc.kill()
            proc.wait()


@pytest.mark.parametrize(
    "suffix",
    ["/x@127.0.0.1:{port_b}/data.fits", "/p?q=@127.0.0.1:{port_b}/data.fits"],
)
def test_cfitsio_ignores_at_sign_outside_authority(two_servers, suffix):
    (port_a, log_a), (port_b, log_b) = two_servers
    url = f"http://127.0.0.1:{port_a}" + suffix.format(port_b=port_b)
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read_header(url)
    assert log_b.read_text() == ""
    hits = log_a.read_text()
    assert "GET /" in hits
    # No Authorization header derived from the path prefix either: the
    # fixture logs only the request line, so check the host it reached.
    assert f"Host=127.0.0.1:{port_a}" in hits
