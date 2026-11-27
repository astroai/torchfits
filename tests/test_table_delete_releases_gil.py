"""``delete_fits_table_rows`` runs its file rewrite without holding the GIL.

Deleting rows near the start of a table makes CFITSIO shift every later row,
which is pure file I/O. The binding takes no Python input, so it now releases
the GIL for the whole call; before, every other Python thread stalled until
the rewrite finished. It also invalidates the path's native caches once the
rewrite is done, so a read that ran meanwhile cannot leave a stale row count.
(Such a read can still make the delete's READWRITE open fail -- CFITSIO
refuses it while another thread holds the file open -- as it always could.)
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

import torchfits
import torchfits._C as cpp

N_ROWS = 2_000_000


def test_delete_rows_lets_other_threads_run(tmp_path) -> None:
    path = str(tmp_path / "big.fits")
    torchfits.write(path, {"a": np.arange(N_ROWS, dtype=np.int64)}, overwrite=True)

    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())

    # A short switch interval bounds how long the ticker can keep the GIL
    # around the edges of the call, so the window checked below is the call.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    t = threading.Thread(target=ticker)
    t.start()
    try:
        time.sleep(0.01)
        start = time.perf_counter()
        cpp.delete_fits_table_rows(path, 1, 1, 1)
        end = time.perf_counter()
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old_interval)

    duration = end - start
    # Ignore the edges, where the ticker may run before the native call takes
    # (or after it gives back) the GIL even when the call holds it throughout.
    margin = min(0.002, duration / 4)
    during = [x for x in ticks if start + margin < x < end - margin]
    assert during, f"no other thread ran during a {duration * 1e3:.1f} ms delete"

    assert torchfits.read_nrows(path, 1) == N_ROWS - 1
    got = cpp.read_fits_table_rows(path, 1, ["a"], 1, 3, False)["a"]
    assert got.tolist() == [1, 2, 3]
