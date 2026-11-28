"""A corrupt VLA table read through ``torchfits.read`` reports the corruption.

``torchfits.read`` tries its readers in order (mmap first) and, when all of
them fail, reports the first one's error. For a VLA column the mmap reader
cannot read the table at all, so a corrupt VLA heap was reported as "VLA
columns not supported for mmap" instead of the buffered reader's diagnosis.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import torchfits

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


@pytest.mark.parametrize("mmap", [True, False, "auto"])
def test_corrupt_vla_heap_reports_the_corruption(tmp_path, mmap):
    path = str(tmp_path / "vla.fits")
    rows = [np.arange(n, dtype=np.float64) for n in (1, 2, 3)]
    torchfits.table.write(path, {"V": rows})
    tform = torchfits.read_header(path, 1)["TFORM1"]
    assert tform.startswith("1P"), tform
    raw = bytearray(open(path, "rb").read())
    start = _hdu1_data_start(bytes(raw))
    # Row 0's descriptor (int32 count, int32 heap offset) points far past
    # the heap.
    raw[start + 4 : start + 8] = struct.pack(">i", 10_000_000)
    open(path, "wb").write(bytes(raw))
    torchfits.clear_file_cache()
    with pytest.raises(RuntimeError) as info:
        torchfits.read(path, hdu=1, mmap=mmap)
    message = str(info.value)
    assert "not supported for mmap" not in message
    assert "VLA" in message


def test_valid_vla_table_still_reads(tmp_path):
    path = str(tmp_path / "vla_ok.fits")
    rows = [np.arange(n, dtype=np.float64) for n in (1, 2, 3)]
    torchfits.table.write(path, {"V": rows})
    out = torchfits.read(path, hdu=1)
    assert [len(v) for v in out["V"]] == [1, 2, 3]
