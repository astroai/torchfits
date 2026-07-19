"""HTTP ``torchfits probe`` path — SSRF policy for loopback fixtures.

Range-replay against a local ThreadingHTTPServer used to be the smoke path;
``probe`` now rejects private/loopback/link-local URLs before fetch. Happy-path
HTTP Range coverage lives behind real non-internal hosts; here we lock the
safety bar.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from torchfits.cli.cmds_probe import (
    _ValidatingRedirectHandler,
    _is_internal_url,
    _probe_http,
)
from torchfits.cli.common import IoError
from torchfits.http_util import HttpBlockedError


def test_http_range_probe_blocks_loopback() -> None:
    with pytest.raises(IoError, match="internal or private networks is blocked"):
        _probe_http("http://127.0.0.1:9/probe.fits", header_bytes=5760, timeout=1.0)


def test_is_internal_url_blocks_private_ipv6_and_unresolvable() -> None:
    assert _is_internal_url("http://127.0.0.1/x.fits")
    assert _is_internal_url("http://169.254.169.254/x.fits")
    assert _is_internal_url("http://[::1]/x.fits")
    assert _is_internal_url("http://10.0.0.5/x.fits")
    # getaddrinfo failure -> treat as internal (block).
    assert _is_internal_url("http://no-such-host.invalid/x.fits")


def test_redirect_to_internal_is_blocked() -> None:
    handler = _ValidatingRedirectHandler()
    with pytest.raises((IoError, HttpBlockedError), match="redirect to internal"):
        handler.redirect_request(
            None, None, 302, "Found", {}, "http://127.0.0.1/evil.fits"
        )


def test_cli_probe_http_blocks_loopback() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchfits.cli",
            "probe",
            "http://127.0.0.1:9/probe.fits",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "access to internal or private networks is blocked" in result.stderr
