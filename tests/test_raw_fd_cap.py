"""The shared read cache's raw-descriptor memo is bounded by TORCHFITS_MAX_CACHED_FDS.

This is the Python half of ``tests/cpp/test_raw_fd_retry.cpp``: SharedReadMeta
memoises each path's raw descriptor so a repeat read skips an ``open()``, and the
memo had no bound -- a loop over N files left N descriptors open for the life of
the process (measured on macOS with ``RLIMIT_NOFILE=64``: 90 reads held 64/64
descriptors and 199 of 200 further opens failed with EMFILE).

The cap is read once at native static init, so it can only be set in a child,
and the descriptors are per-process state, so they are counted there too. The
unbounded run (``TORCHFITS_MAX_CACHED_FDS=0``) is the control: it proves the
probe counts the memo at all, so a green bounded run is not vacuous.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import torchfits

_FD_PROBE = textwrap.dedent(
    """
    import os, sys
    import torchfits

    paths = sys.argv[1:]

    def open_fds():
        d = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
        return len([e for e in os.listdir(d) if e.isdigit()])

    before = open_fds()
    for p in paths:
        torchfits.read(p, hdu=0)
    print(open_fds() - before)
    """
)

N_FILES = 40
CAP = 8


def _retained(paths: list[str], cap: int) -> int:
    env = {**os.environ, "TORCHFITS_MAX_CACHED_FDS": str(cap)}
    result = subprocess.run(
        [sys.executable, "-c", _FD_PROBE, *paths],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return int(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(
    not (os.path.isdir("/proc/self/fd") or os.path.isdir("/dev/fd")),
    reason="no per-process descriptor listing on this platform",
)
def test_shared_read_cache_does_not_retain_one_descriptor_per_path(tmp_path):
    data = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    paths = []
    for i in range(N_FILES):
        path = str(tmp_path / f"f{i:03d}.fits")
        torchfits.write(path, data)
        paths.append(path)

    unbounded = _retained(paths, 0)
    assert unbounded >= N_FILES, (
        f"control: with the cap disabled {N_FILES} reads retained only "
        f"{unbounded} descriptors, so the probe is not measuring the memo"
    )
    bounded = _retained(paths, CAP)
    assert bounded <= CAP, (
        f"the shared read cache kept {bounded} descriptors open after "
        f"{N_FILES} reads with TORCHFITS_MAX_CACHED_FDS={CAP}"
    )
