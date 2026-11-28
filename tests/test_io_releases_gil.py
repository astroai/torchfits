"""Long native file passes run without holding the GIL.

``ffpcks`` / ``ffvcks`` read every byte of the HDU (~2 s on a 4.5 GB image),
and opening a ``.gz`` file inflates all of it into memory; every other Python
thread used to stall meanwhile. None of these touches a Python object, so the
bindings release the GIL for the call.
"""

from __future__ import annotations

import sys
import threading
import time

import torch

import torchfits
import torchfits._C as cpp


def _other_thread_ran_during(call) -> tuple[bool, float]:
    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    t = threading.Thread(target=ticker)
    t.start()
    try:
        time.sleep(0.01)
        start = time.perf_counter()
        call()
        end = time.perf_counter()
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old_interval)
    duration = end - start
    # Ignore the edges, where the ticker may run before the native call takes
    # (or after it gives back) the GIL even when the call holds it throughout.
    margin = min(0.002, duration / 4)
    return any(start + margin < x < end - margin for x in ticks), duration


def test_checksum_write_and_verify_let_other_threads_run(tmp_path) -> None:
    path = str(tmp_path / "big.fits")
    torchfits.write(path, torch.ones(4096, 4096, dtype=torch.float32), overwrite=True)

    ran, duration = _other_thread_ran_during(lambda: cpp.write_hdu_checksums(path, 0))
    assert ran, f"no other thread ran during a {duration * 1e3:.1f} ms checksum write"

    result: list[tuple[int, int]] = []
    ran, duration = _other_thread_ran_during(
        lambda: result.append(cpp.verify_hdu_checksums(path, 0))
    )
    assert ran, f"no other thread ran during a {duration * 1e3:.1f} ms checksum verify"
    assert tuple(result[0]) == (1, 1)
    assert torchfits.verify_checksums(path, 0)["ok"] is True


def test_opening_a_gzip_file_lets_other_threads_run(tmp_path) -> None:
    """CFITSIO inflates a whole .gz file into memory when it opens it."""
    import gzip
    import shutil

    plain = tmp_path / "big.fits"
    torchfits.write(
        str(plain), torch.arange(4096 * 4096, dtype=torch.float32).reshape(4096, 4096)
    )
    packed = tmp_path / "big.fits.gz"
    with open(plain, "rb") as src, gzip.open(packed, "wb", compresslevel=1) as dst:
        shutil.copyfileobj(src, dst)

    handles: list[object] = []
    ran, duration = _other_thread_ran_during(
        lambda: handles.append(cpp.open_fits_file(str(packed), "r"))
    )
    assert ran, f"no other thread ran during a {duration * 1e3:.1f} ms .gz open"
    handles[0].close()
