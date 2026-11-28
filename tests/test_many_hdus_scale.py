"""Many-HDU files: reaching HDU n must not cost a walk over the n-1 before it.

CFITSIO finds HDU n by reading every header in front of it unless the handle
already knows where it starts. Every torchfits read opens a private handle, so
a loop over the HDUs of a 10,000-extension file was quadratic: ``open()`` took
minutes, as did ``torchfits verify`` / ``stats`` / ``info``.

The cost is measured as bytes read through the kernel (``rchar`` in
``/proc/self/io``) rather than wall time, so the assertions are exact on a
loaded host. Linux only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import torchfits

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.path.exists("/proc/self/io"),
    reason="needs Linux /proc/self/io read accounting",
)

BLOCK = 2880


def _card(key: str, value: object) -> str:
    if isinstance(value, bool):
        text = f"{'T' if value else 'F':>20}"
    elif isinstance(value, int):
        text = f"{value:>20}"
    else:
        text = f"{repr(value).replace(chr(34), chr(39)):<20}"
    return f"{key:<8}= {text}".ljust(80)


def _header(cards: list[tuple[str, object]]) -> bytes:
    text = "".join(_card(k, v) for k, v in cards) + "END".ljust(80)
    return (text + " " * ((-len(text)) % BLOCK)).encode("ascii")


def _write_mef(path, n_ext: int, first_rows: int = 2) -> None:
    """Primary + ``n_ext`` int16 (first_rows x 3) IMAGE extensions, EXTNAME E<i>.

    Pixel [0, 0] of extension i is i, so a wrong HDU offset reads a wrong value.
    """
    with open(path, "wb") as fh:
        fh.write(
            _header([("SIMPLE", True), ("BITPIX", 16), ("NAXIS", 0), ("EXTEND", True)])
        )
        for i in range(1, n_ext + 1):
            rows = first_rows if i == 1 else 2
            fh.write(
                _header(
                    [
                        ("XTENSION", "IMAGE"),
                        ("BITPIX", 16),
                        ("NAXIS", 2),
                        ("NAXIS1", 3),
                        ("NAXIS2", rows),
                        ("PCOUNT", 0),
                        ("GCOUNT", 1),
                        ("EXTNAME", f"E{i}"),
                    ]
                )
            )
            data = np.full(rows * 3, i, dtype=">i2").tobytes()
            fh.write(data + b"\0" * ((-len(data)) % BLOCK))


def _rchar() -> int:
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    raise AssertionError("no rchar in /proc/self/io")


def _bytes_read(fn) -> int:
    before = _rchar()
    fn()
    return _rchar() - before


N_EXT = 400


def test_open_lists_a_many_hdu_file_in_one_pass(tmp_path):
    path = str(tmp_path / "mef.fits")
    _write_mef(path, N_EXT)
    torchfits.clear_all_caches()

    def _open():
        with torchfits.open(path) as hdul:
            assert len(hdul) == N_EXT + 1
            assert hdul[N_EXT].header["EXTNAME"] == f"E{N_EXT}"

    # One pass reads each 2880-byte header once (~N blocks). Opening the file
    # per HDU re-read the headers in front of it: ~N^2 / 2 blocks.
    assert _bytes_read(_open) < 8 * N_EXT * BLOCK
