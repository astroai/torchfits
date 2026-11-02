"""HDU edits write untouched scaled images back with their stored codes.

insert/replace/delete decoded every image, so an untouched BITPIX=16 HDU with
BSCALE/BZERO/BLANK came back as BITPIX=-32 floats without its scale cards.
"""

import torch

import torchfits

SCALE_KEYS = ("BITPIX", "BSCALE", "BZERO", "BLANK")


def _scale_cards(path, hdu):
    header = torchfits.read_header(path, hdu)
    return {k: header.get(k) for k in SCALE_KEYS}


def test_untouched_scaled_images_keep_storage(tmp_path):
    path = str(tmp_path / "mef.fits")
    codes = torch.arange(400, dtype=torch.int16).reshape(20, 20)
    codes[0, 0] = -32768
    torchfits.write(
        path,
        [
            torch.zeros(1, 1),
            {
                "data": codes,
                "header": {"BSCALE": 0.5, "BZERO": 10.0, "BLANK": -32768},
            },
            torch.tensor([[0, 1000, 65535]], dtype=torch.uint16),
            torch.tensor([[-5, 0, 127]], dtype=torch.int8),
        ],
    )
    before = [_scale_cards(path, i) for i in range(4)]
    raw = [torchfits.read(path, hdu=i, raw_scale=True) for i in range(1, 4)]
    phys = [torchfits.read(path, hdu=i) for i in range(1, 4)]

    torchfits.insert_hdu(path, torch.ones(2, 2), index=4)
    torchfits.replace_hdu(path, 4, torch.zeros(2, 2))
    torchfits.delete_hdu(path, 4)

    assert [_scale_cards(path, i) for i in range(4)] == before
    for i in range(1, 4):
        assert torch.equal(torchfits.read(path, hdu=i, raw_scale=True), raw[i - 1])
        assert torch.allclose(
            torchfits.read(path, hdu=i).double(), phys[i - 1].double(), equal_nan=True
        )
