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


def test_path_reads_of_a_late_hdu_do_not_rescan_the_headers(tmp_path):
    path = str(tmp_path / "mef.fits")
    _write_mef(path, N_EXT)
    torchfits.clear_all_caches()
    assert int(torchfits.read(path, hdu=N_EXT)[0, 0]) == N_EXT  # learns offsets

    reads = {
        "read": lambda: torchfits.read(path, hdu=N_EXT),
        "read_tensor": lambda: torchfits.read_tensor(path, hdu=N_EXT),
        "verify_checksums": lambda: torchfits.verify_checksums(path, N_EXT),
        "read_keys": lambda: torchfits.read_keys(path, ["NAXIS"], N_EXT),
    }
    for name, fn in reads.items():
        # A handle that knows the offsets reads the primary header and the
        # target HDU: a handful of blocks, not the N_EXT headers before it.
        assert _bytes_read(fn) < 16 * BLOCK, name
    assert int(torchfits.read(path, hdu=N_EXT - 1)[0, 0]) == N_EXT - 1
    assert int(torchfits.read(path, hdu=7)[0, 0]) == 7


def test_learned_hdu_offsets_do_not_survive_a_rewrite(tmp_path):
    path = str(tmp_path / "mef.fits")
    _write_mef(path, 50)
    torchfits.clear_all_caches()
    assert int(torchfits.read(path, hdu=50)[0, 0]) == 50
    # Out-of-band rewrite (not through torchfits): extension 1 grows by
    # several blocks, so every later HDU moves.
    _write_mef(path, 60, first_rows=3000)
    for hdu in (60, 50, 2):
        assert int(torchfits.read(path, hdu=hdu)[0, 0]) == hdu
    assert torchfits.read_num_hdus(path) == 61
    # A rewrite through torchfits invalidates the offsets too.
    torchfits.delete_hdu(path, 1)
    assert int(torchfits.read(path, hdu=59)[0, 0]) == 60
    assert int(torchfits.read(path, hdu=1)[0, 0]) == 2


def test_concurrent_reads_of_a_many_hdu_file_stay_correct(tmp_path):
    """Threads opening the same file share CFITSIO's FITSfile; seeding the
    learned offsets must not race another thread's open or HDU move."""
    from concurrent.futures import ThreadPoolExecutor

    path = str(tmp_path / "mef.fits")
    _write_mef(path, 120)
    torchfits.clear_all_caches()

    def _worker(seed: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng(seed)
        out = []
        for hdu in rng.integers(1, 121, size=150):
            out.append((int(hdu), int(torchfits.read(path, hdu=int(hdu))[0, 0])))
            if hdu % 7 == 0:
                torchfits.verify_checksums(path, int(hdu))
        return out

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = [pair for chunk in pool.map(_worker, range(12)) for pair in chunk]
    assert len(results) == 12 * 150
    assert all(hdu == value for hdu, value in results)
