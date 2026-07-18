"""Local HTTP Range replay fixture for ``torchfits probe`` (network path)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

import torchfits
from torchfits.cli.cmds_probe import _probe_http


class _RangeHandler(BaseHTTPRequestHandler):
    """Minimal Range-capable FITS server for probe smoke tests."""

    file_bytes: bytes = b""

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        data = self.file_bytes
        range_hdr = self.headers.get("Range")
        if range_hdr and range_hdr.startswith("bytes="):
            spec = range_hdr.removeprefix("bytes=")
            start_s, _, end_s = spec.partition("-")
            start = int(start_s or 0)
            end = int(end_s) if end_s else len(data) - 1
            end = min(end, len(data) - 1)
            chunk = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_http_range_probe_replay(tmp_path: Path) -> None:
    path = tmp_path / "probe.fits"
    torchfits.write(
        str(path),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        header={"OBJECT": "HTTP_PROBE"},
        overwrite=True,
    )
    payload = path.read_bytes()
    handler = type(
        "FitsRangeHandler",
        (_RangeHandler,),
        {"file_bytes": payload},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        url = f"http://{host}:{port}/probe.fits"
        record = _probe_http(url)
        assert record["source"] == "http"
        assert int(record["bitpix"]) == -32
        assert int(record["naxis"]) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_probe_http_json(tmp_path: Path) -> None:
    path = tmp_path / "probe.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8), dtype=torch.float32),
        overwrite=True,
    )
    payload = path.read_bytes()
    handler = type(
        "FitsRangeHandler",
        (_RangeHandler,),
        {"file_bytes": payload},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        url = f"http://{host}:{port}/probe.fits"
        result = subprocess.run(
            [sys.executable, "-m", "torchfits.cli", "probe", url, "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        rows = json.loads(result.stdout)
        assert rows[0]["source"] == "http"
        assert int(rows[0]["bitpix"]) == -32
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
