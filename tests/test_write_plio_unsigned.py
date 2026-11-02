"""uint16 images compress with PLIO_1.

The writer converted uint16 to signed int16 codes (value - 32768) before the
compressed writer, and PLIO only stores values in [0, 2**24): status 413.
"""

import pytest
import torch

import torchfits


@pytest.mark.parametrize("wrap", [lambda x: x, lambda x: [x], lambda x: {"data": x}])
def test_uint16_plio_roundtrip(tmp_path, wrap):
    path = str(tmp_path / "plio.fits")
    img = torch.tensor([[65535, 1, 2, 3], [0, 40000, 7, 8]], dtype=torch.uint16)
    torchfits.write(path, wrap(img), compress="PLIO_1")
    header = torchfits.read_header(path, 1)
    assert header["ZCMPTYPE"] == "PLIO_1" and float(header["BZERO"]) == 32768.0
    got = torchfits.read(path, hdu=1)
    assert got.dtype == torch.uint16 and torch.equal(got, img)
