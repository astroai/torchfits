"""Row/column mutations wait out a read in flight on another thread.

A cached TableReader holds a CFITSIO handle open READONLY, and CFITSIO refuses
to reopen a file READWRITE while one is registered. The writers evicted the
caches and retried exactly once, so a read running on another thread at that
moment (its reader is checked out of the cache, out of reach of the eviction)
made ``delete_rows`` fail. The writers now keep evicting and retrying with a
short backoff while the path is a writable file.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import torchfits


def test_delete_rows_succeeds_while_another_thread_reads(tmp_path) -> None:
    path = str(tmp_path / "t.fits")
    n = 200_000
    torchfits.table.write(
        path,
        {"a": np.arange(n, dtype=np.float64), "b": np.arange(n, dtype=np.int32)},
        overwrite=True,
    )
    stop = threading.Event()
    reader_errors: list[BaseException] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                torchfits.table.read_torch(path, mmap=False)
            except BaseException as exc:  # pragma: no cover - reported below
                reader_errors.append(exc)
            time.sleep(0.002)

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        for _ in range(15):
            torchfits.table.delete_rows(path, (0, 1))
            torchfits.table.update_rows(
                path, {"b": np.array([7], dtype=np.int32)}, row_slice=(0, 1), mmap=False
            )
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert not reader_errors
    data = torchfits.table.read_torch(path)
    assert data["a"].shape[0] == n - 15
    assert float(data["a"][0]) == 15.0
    assert int(data["b"][0]) == 7
