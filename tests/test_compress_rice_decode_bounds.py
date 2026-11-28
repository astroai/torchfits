"""A corrupt RICE_1 tile is refused without reading past its buffer.

The RICE decoders checked for the end of the compressed stream only after each
block of pixels, so a tile whose descriptor claims fewer bytes than its pixels
need was decoded past the end of the heap buffer before the error surfaced
(cfitsio-4.7.0-ricecomp-input-bounds). Out-of-bounds reads only show under ASan
(the sanitizer workflow runs this file); everywhere the read must fail with a
RuntimeError, never crash or return pixels.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import torchfits


def _hdu_data_offsets(raw: bytes) -> list[int]:
    """Byte offset of each HDU's data unit (headers only; no data sizes needed
    beyond the first HDU, which is an empty primary here)."""
    offsets = []
    pos = 0
    while pos < len(raw):
        block = pos
        while True:
            chunk = raw[block : block + 2880]
            cards = [chunk[i : i + 80] for i in range(0, 2880, 80)]
            block += 2880
            if any(
                card.startswith(b"END     ") or card.rstrip() == b"END"
                for card in cards
            ):
                break
        offsets.append(block)
        if len(offsets) == 2:
            break
        pos = block  # the primary HDU has no data
    return offsets


@pytest.mark.parametrize(
    ("dtype", "keep"),
    [(np.int16, 3), (np.int32, 5), (np.uint8, 2), (np.int16, 1), (np.int32, 0)],
)
def test_truncated_rice_tile_is_an_error(tmp_path: Path, dtype, keep) -> None:
    path = tmp_path / "rice.fits"
    info = np.iinfo(dtype)
    x = np.random.default_rng(0).integers(
        int(info.min), int(info.max), size=(8, 64), dtype=dtype, endpoint=True
    )
    torchfits.write(str(path), x, compress="RICE_1", overwrite=True)
    header = torchfits.read_header(str(path), 1)
    tforms = [
        str(header[f"TFORM{i}"]).strip() for i in range(1, int(header["TFIELDS"]) + 1)
    ]
    ttypes = [
        str(header[f"TTYPE{i}"]).strip() for i in range(1, int(header["TFIELDS"]) + 1)
    ]
    col = ttypes.index("COMPRESSED_DATA")
    assert all(tf.startswith(("1P", "P")) for tf in tforms[: col + 1]), tforms
    raw = bytearray(path.read_bytes())
    data = _hdu_data_offsets(bytes(raw))[1]
    # Shrink the first tile's descriptor to `keep` bytes of its stream.
    desc = data + 8 * col
    _, heap_offset = struct.unpack(">ii", raw[desc : desc + 8])
    raw[desc : desc + 8] = struct.pack(">ii", keep, heap_offset)
    path.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError):
        torchfits.read(str(path), hdu=1)
