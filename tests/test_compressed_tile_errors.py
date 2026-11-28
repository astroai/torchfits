"""A corrupt tile-compressed image names the problem.

CFITSIO has no message for NO_COMPRESSED_TILE (status 415), which a tile
whose descriptor says it holds no bytes raises: reads failed with
"status=415 msg=unknown error status". DATA_DECOMPRESSION_ERR (414), which
the vendored bounds patches return for a truncated tile, said only "error
uncompressing image", and the Rice decoder's status 1 "non-CFITSIO program
error".
"""

from __future__ import annotations

import struct

import pytest
import torch

import torchfits

BLOCK = 2880


def _hdu1_data_start(raw: bytes) -> int:
    """Byte offset of HDU 1's data (primary has no data)."""
    pos = 0
    for _ in range(2):  # primary header, then HDU 1's header
        while True:
            block = raw[pos : pos + BLOCK]
            pos += BLOCK
            if any(block[i : i + 8] == b"END     " for i in range(0, BLOCK, 80)):
                break
    return pos


def _rice_file(tmp_path) -> tuple[str, bytearray, int]:
    path = str(tmp_path / "rice.fits")
    data = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    torchfits.write(path, data, compress="RICE_1")
    assert torchfits.read_header(path, 1)["TFORM1"].startswith("1P")
    raw = bytearray(open(path, "rb").read())
    return path, raw, _hdu1_data_start(bytes(raw))


READS = {
    "read": lambda p: torchfits.read(p, hdu=1),
    "read_tensor": lambda p: torchfits.read_tensor(p, hdu=1),
    "subset": lambda p: torchfits.read_subset(p, 1, 0, 0, 4, 4),
}


@pytest.mark.parametrize("reader", sorted(READS))
def test_missing_tile_reports_truncated_or_corrupt(tmp_path, reader):
    path, raw, start = _rice_file(tmp_path)
    # Row 0's descriptor (int32 count, int32 heap offset): no bytes at all.
    raw[start : start + 4] = struct.pack(">i", 0)
    open(path, "wb").write(bytes(raw))
    torchfits.clear_file_cache()
    with pytest.raises(RuntimeError) as info:
        READS[reader](path)
    message = str(info.value)
    assert "status=415" in message
    assert "unknown error status" not in message
    assert "compressed tile data is truncated or corrupt" in message


def test_short_tile_reports_truncated_or_corrupt(tmp_path):
    path, raw, start = _rice_file(tmp_path)
    # Row 0 now claims a single heap byte: the Rice stream ends early.
    raw[start : start + 4] = struct.pack(">i", 1)
    open(path, "wb").write(bytes(raw))
    torchfits.clear_file_cache()
    with pytest.raises(RuntimeError) as info:
        torchfits.read(path, hdu=1)
    message = str(info.value)
    # The Rice decoder reports a short stream as status 1, other decoders
    # (and the vendored bounds checks) as 414.
    assert "status=1 " in message or "status=414" in message
    assert "compressed tile data is truncated or corrupt" in message
