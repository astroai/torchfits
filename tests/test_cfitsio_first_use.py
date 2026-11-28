"""A process's first CFITSIO calls may start on many threads at once.

CFITSIO made its global lock recursive lazily, inside the first open, by
re-running ``pthread_mutex_init`` on a mutex other entry points (the error
stack, the open-file table) were already locking. When the first reads of a
process started concurrently, one thread could re-initialise the lock while
another held it, leaving it locked for good: every thread then hung in
``ffxmsg`` / ``fits_clear_Fptr``. libtorchfits_core now initialises CFITSIO
while it loads.

Each attempt is a fresh interpreter (the race only exists before the first
open) that never imports torch, so a run costs a fraction of a second.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np

import torchfits

SCRIPT = textwrap.dedent(
    """
    import sys, threading, time
    from torchfits import _core

    path = sys.argv[1]
    n = 32
    barrier = threading.Barrier(n)
    errors = []

    def work(i):
        try:
            barrier.wait()
            for _ in range(3):
                assert _core.read_num_hdus(path) == 2
                assert _core.read_nrows(path, 1) == 5
        except BaseException as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 20
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):
        print("HANG")
        sys.stdout.flush()
        import os
        os._exit(3)
    assert not errors, errors
    print("OK")
    """
)


def test_concurrent_first_cfitsio_use_does_not_deadlock(tmp_path) -> None:
    path = str(tmp_path / "t.fits")
    torchfits.write(path, {"a": np.arange(5, dtype=np.int32)}, overwrite=True)
    for attempt in range(8):
        proc = subprocess.run(
            [sys.executable, "-c", SCRIPT, path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0 and "OK" in proc.stdout, (
            f"attempt {attempt}: rc={proc.returncode}\n{proc.stdout}{proc.stderr}"
        )
