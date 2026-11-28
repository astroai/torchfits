"""A failing ``table.read`` releases its descriptors without a gc pass.

The "Invalid VLA geometry" re-raise helper kept the exception in a local of
its own frame, which is on the exception's traceback: a reference cycle
(exception -> traceback -> frame -> exception) that kept every frame on the
traceback alive -- including the caller's native reader and its two file
descriptors -- until the cyclic garbage collector happened to run.
"""

from __future__ import annotations

import gc
import os
import struct
import sys

import numpy as np
import pytest

import torchfits

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd"
)

BLOCK = 2880


def _hdu1_data_start(raw: bytes) -> int:
    pos = 0
    for _ in range(2):
        while True:
            block = raw[pos : pos + BLOCK]
            pos += BLOCK
            if any(block[i : i + 8] == b"END     " for i in range(0, BLOCK, 80)):
                break
    return pos


def _corrupt_vla_table(path: str) -> None:
    torchfits.table.write(
        path, {"V": [np.arange(n, dtype=np.float64) for n in (1, 2, 3)]}
    )
    raw = bytearray(open(path, "rb").read())
    start = _hdu1_data_start(bytes(raw))
    raw[start + 4 : start + 8] = struct.pack(">i", 10_000_000)  # heap offset
    open(path, "wb").write(bytes(raw))


def _nfd() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.parametrize(
    "call",
    [
        lambda p: torchfits.table.read(p, hdu=1),
        lambda p: torchfits.table.read(p, hdu=1, mmap=False),
        lambda p: torchfits.table.read(p, hdu=1, where="V > 0"),
    ],
    ids=["read", "read-nommap", "read-where"],
)
def test_failed_table_read_closes_descriptors_without_gc(tmp_path, call):
    path = str(tmp_path / "vla.fits")
    _corrupt_vla_table(path)
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(3):
            torchfits.clear_all_caches()
            before = _nfd()
            with pytest.raises(RuntimeError, match="Invalid VLA geometry"):
                call(path)
            sys.last_exc = sys.last_value = sys.last_traceback = None
            torchfits.clear_all_caches()
            assert _nfd() == before
    finally:
        if was_enabled:
            gc.enable()
