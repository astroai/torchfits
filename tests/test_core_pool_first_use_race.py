"""Concurrent first use of the core thread pool builds exactly one pool.

``pool()`` used to do an unsynchronised ``if (!g_pool) g_pool = new Pool``, so
threads making their first ``parallel_for`` call together could each build a
pool, leaking every extra worker. A forked child starts with no pool, so each
fork below is a fresh first use; the child counts its OS threads afterwards.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent(
    """
    import os, subprocess, sys, threading
    import torch, torchfits
    from torchfits import _core

    torch.set_num_threads(1)
    path = sys.argv[1]
    # int8 is stored with the signed-byte offset, so a read runs the core
    # pool's sign-bit pass.
    torchfits.write(path, torch.randint(-128, 127, (1024, 1024), dtype=torch.int8),
                    overwrite=True)

    def nthreads():
        if os.path.isdir("/proc/self/task"):
            return len(os.listdir("/proc/self/task"))
        out = subprocess.run(["ps", "-M", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout
        return len(out.strip().splitlines()) - 1

    workers = _core.thread_count() - 1
    for attempt in range(20):
        pid = os.fork()
        if pid == 0:
            base = nthreads()
            barrier = threading.Barrier(16)
            def work():
                barrier.wait()
                torchfits.read(path)
            ts = [threading.Thread(target=work) for _ in range(16)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            os._exit(0 if nthreads() - base <= workers else 1)
        _, status = os.waitpid(pid, 0)
        if os.waitstatus_to_exitcode(status) != 0:
            print(f"EXTRA POOL attempt={attempt}")
            sys.exit(1)
    print("OK")
    """
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
def test_concurrent_first_use_builds_one_pool(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "i8.fits")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
