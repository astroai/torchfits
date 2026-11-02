"""Loopback HTTP fixture server, run as a separate process.

CFITSIO's network drivers hold the GIL while they connect and read, so an
in-process server thread would deadlock against them. Tests start this module
with ``sys.executable`` via :func:`start` and get back the port; every request
line is appended to a log file the test can inspect.

Routes:

* ``/data.fits`` / ``/secret.fits`` -- a 3x4 int16 primary image (0..11)
* ``/data.fits.gz`` -- the same image, gzip-compressed
* ``/redirect-same`` -- 302 to ``http://localhost:<port>/data.fits``
* ``/redirect-other`` -- 302 to ``http://127.0.0.1:<port>/secret.fits``
* ``/error-page.fits`` -- 200 with an HTML body (a CDN-style soft error)
* ``/no-reply.fits`` -- the connection is closed without any response
* anything else -- 404
"""

from __future__ import annotations

import gzip
import http.server
import os
import subprocess
import sys
import time
from pathlib import Path


def fits_image_bytes() -> bytes:
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  =                    4",
        "NAXIS2  =                    3",
        "END",
    ]
    header = "".join(card.ljust(80) for card in cards).ljust(2880).encode("ascii")
    data = b"".join(i.to_bytes(2, "big") for i in range(12)).ljust(2880, b"\0")
    return header + data


class _Handler(http.server.BaseHTTPRequestHandler):
    log_path: str = os.devnull

    def _send(self, status: int, body: bytes, **headers: str) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        with open(type(self).log_path, "a", encoding="utf-8") as log:
            log.write(f"{self.command} {self.path} Host={self.headers.get('Host')}\n")
        port = self.server.server_address[1]
        path = self.path.split("?", 1)[0]
        if path in ("/data.fits", "/secret.fits"):
            self._send(200, fits_image_bytes())
        elif path == "/data.fits.gz":
            self._send(200, gzip.compress(fits_image_bytes()))
        elif path == "/redirect-same":
            self._send(302, b"", Location=f"http://localhost:{port}/data.fits")
        elif path == "/redirect-other":
            self._send(302, b"", Location=f"http://127.0.0.1:{port}/secret.fits")
        elif path == "/error-page.fits":
            self._send(200, b"<html><body>quota exceeded</body></html>")
        elif path == "/no-reply.fits":
            self.close_connection = True
        else:
            self._send(404, b"not found")

    do_HEAD = do_GET

    def log_message(self, *args: object) -> None:
        pass


def start(tmp_path: Path) -> tuple[subprocess.Popen[bytes], int, Path]:
    """Start the server out of process; return ``(proc, port, log_path)``."""
    port_file = tmp_path / "port.txt"
    log_path = tmp_path / "hits.log"
    log_path.write_text("")
    err_path = tmp_path / "server-stderr.txt"
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(
            [sys.executable, __file__, str(port_file), str(log_path)],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            stderr=err,
        )
    # CI macOS runners can take a while to start an interpreter under load.
    deadline = time.monotonic() + 60
    while not (port_file.exists() and port_file.read_text().strip()):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            detail = err_path.read_text(errors="replace").strip()
            raise RuntimeError(
                "loopback fixture server failed to start "
                f"(returncode={proc.poll()}): {detail or 'no stderr'}"
            )
        time.sleep(0.05)
    return proc, int(port_file.read_text()), log_path


def main() -> None:
    port_file, log_path = sys.argv[1], sys.argv[2]
    _Handler.log_path = log_path
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    tmp = port_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(str(server.server_address[1]))
    os.replace(tmp, port_file)
    server.serve_forever()


if __name__ == "__main__":
    main()
