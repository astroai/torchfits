"""replace_hdu/delete_hdu find EXTNAMEs the way read() does.

They compared EXTNAME case-sensitively, so replace_hdu(p, "sci", ...) raised
KeyError on a file that read(p, hdu="sci") reads.
"""

import torch

import torchfits


def _mef(path):
    torchfits.write(
        path,
        [
            torch.zeros(1, 1),
            {"data": torch.ones(2, 2), "header": {"EXTNAME": "SCI"}},
            {"data": torch.full((2, 2), 2.0), "header": {"EXTNAME": "ERR"}},
        ],
    )


def test_extname_lookup_is_case_insensitive(tmp_path):
    path = str(tmp_path / "en.fits")
    _mef(path)
    torchfits.replace_hdu(path, "sci", torch.full((2, 2), 9.0))
    assert torchfits.read(path, hdu="SCI")[0, 0].item() == 9.0
    torchfits.delete_hdu(path, " err ")
    with torchfits.open(path) as hdul:
        assert len(hdul) == 2
