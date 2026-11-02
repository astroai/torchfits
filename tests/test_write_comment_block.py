"""Rewrites do not accumulate copies of CFITSIO's standard COMMENT block.

CFITSIO stamps a 2-line COMMENT block on every primary it creates; replaying
the source header appended another copy per insert/delete/overwrite.
"""

import torch

import torchfits

STANDARD = "FITS (Flexible Image Transport System)"


def _comments(path, hdu=0):
    cards = torchfits.read_header(path, hdu).cards
    return [str(c.value).strip() for c in cards if c.key == "COMMENT"]


def test_rewrites_keep_one_standard_block(tmp_path):
    path = str(tmp_path / "acc.fits")
    header = torchfits.Header()
    header.add_comment("user note")
    torchfits.write(path, torch.ones(2, 2), header=header)
    first = _comments(path)
    assert len(first) == 3 and first.count("user note") == 1

    for _ in range(3):
        torchfits.insert_hdu(path, torch.zeros(2, 2), index=1)
        torchfits.delete_hdu(path, 1)
    with torchfits.open(path) as hdul:
        torchfits.write(path, hdul, overwrite=True)
    torchfits.write(
        path, torch.ones(2, 2), header=torchfits.read_header(path), overwrite=True
    )
    assert _comments(path) == first

    torchfits.write(
        path,
        torch.ones(2, 2),
        header=torchfits.read_header(path),
        compress=True,
        overwrite=True,
    )
    assert sum(STANDARD in c for c in _comments(path)) == 1
    assert _comments(path, 1) == ["user note"]
