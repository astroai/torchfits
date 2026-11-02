"""Column edits on a gzip-compressed table keep scaled columns intact.

The stored (pre-TSCAL/TZERO) cells were read with np.fromfile at the
decompressed data offset, i.e. from the *compressed* bytes, so a rewrite of
x.fits.gz replaced scaled columns with garbage (or failed).
"""

from __future__ import annotations

import gzip

import numpy as np
import torch

import torchfits


def _scaled_table_bytes() -> bytes:
    primary = [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ]
    table = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    2",
        "NAXIS2  =                    3",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TTYPE1  = 'S'",
        "TFORM1  = 'I'",
        "TSCAL1  =                  2.0",
        "TZERO1  =                 10.0",
        "END",
    ]

    def block(cards):
        return "".join(c.ljust(80) for c in cards).ljust(2880).encode("ascii")

    data = np.array([1, 2, 3], dtype=">i2").tobytes().ljust(2880, b"\0")
    return block(primary) + block(table) + data


def test_insert_column_on_gzip_table_keeps_scaled_values(tmp_path):
    path = tmp_path / "t.fits.gz"
    path.write_bytes(gzip.compress(_scaled_table_bytes()))
    before = torchfits.read(str(path), hdu=1)["S"].tolist()
    assert before == [12.0, 14.0, 16.0]

    torchfits.table.insert_column(str(path), "N", torch.arange(3), hdu=1)

    after = torchfits.read(str(path), hdu=1)
    assert after["S"].tolist() == before
    assert after["N"].tolist() == [0, 1, 2]
