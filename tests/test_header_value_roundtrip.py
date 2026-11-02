"""Every value read_header produces can be written back.

read_header yields None (undefined), complex and integers beyond 64 bits for
legal cards; the native writers used to refuse them, so
write(path, data, header=read_header(src)) failed on valid files.
"""

from __future__ import annotations

import pytest
import torch

import torchfits

VALUES = {
    "UNDEF": None,
    "CPLX": complex(1.5, -2.0),
    "ICPLX": complex(1, 2),
    "BIGINT": 2**70,
    "NEGBIG": -(2**80) - 7,
    "PLAIN": 42,
}


def _write(kind, path, header):
    if kind == "image":
        torchfits.write(path, torch.zeros(2, 2), header=header, overwrite=True)
    elif kind == "compressed":
        torchfits.write(
            path, torch.zeros(4, 4), header=header, overwrite=True, compress="GZIP_1"
        )
    elif kind == "mef":
        torchfits.write(
            path,
            [torch.zeros(1, 1), {"data": torch.ones(2, 2), "header": header}],
            overwrite=True,
        )
    else:
        torchfits.write(path, {"A": torch.arange(3)}, header=header, overwrite=True)


@pytest.mark.parametrize("kind", ["image", "compressed", "mef", "table"])
def test_read_header_values_write_back(tmp_path, kind):
    path = str(tmp_path / f"{kind}.fits")
    hdr = torchfits.Header([(k, v, f"c-{k}") for k, v in VALUES.items()])
    _write(kind, path, hdr)
    hdu = 0 if kind == "image" else 1
    got = torchfits.read_header(path, hdu)
    for key, value in VALUES.items():
        assert key in got, key
        assert got[key] == value, key
        assert type(got[key]) is type(value), key

    again = str(tmp_path / "again.fits")
    _write(kind, again, got)
    back = torchfits.read_header(again, hdu)
    for key, value in VALUES.items():
        assert back[key] == value, key
