"""TensorHDU.data.dtype must equal the dtype the readers actually return.

The DataView used a 1e-5 tolerance on BSCALE/BZERO while the native reader
compares exactly, so near-identity scales reported the raw integer dtype for
data that reads back as float.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits


def _image(path, bitpix: int, cards: list[str]) -> None:
    dtype = {16: ">i2", 32: ">i4", 8: "u1"}[bitpix]
    header = [
        "SIMPLE  =                    T",
        f"BITPIX  = {bitpix:>20}",
        "NAXIS   =                    2",
        "NAXIS1  =                    2",
        "NAXIS2  =                    2",
        *cards,
        "END",
    ]
    raw = "".join(c.ljust(80) for c in header).ljust(2880).encode("ascii")
    data = np.arange(4, dtype=dtype).tobytes().ljust(2880, b"\0")
    path.write_bytes(raw + data)


@pytest.mark.parametrize(
    "bitpix, cards",
    [
        (32, ["BZERO   =             0.000001"]),
        (16, ["BSCALE  =             1.000001"]),
        (16, ["BZERO   =                32768", "BSCALE  =             1.000001"]),
        (32, ["BZERO   =           2147483648", "BSCALE  =           1.00000001"]),
        (16, ["BZERO   =                32768"]),
        (8, ["BZERO   =                 -128"]),
        (16, ["BSCALE  =                  2.0"]),
        (16, ["BLANK   =                    3"]),
    ],
)
def test_dataview_dtype_matches_read(tmp_path, bitpix, cards):
    path = tmp_path / "img.fits"
    _image(path, bitpix, cards)
    expected = torchfits.read(str(path)).dtype
    with torchfits.open(str(path)) as hdul:
        view = hdul[0].data
        assert view.dtype == expected
        assert view[0:2, 0:2].dtype == expected
