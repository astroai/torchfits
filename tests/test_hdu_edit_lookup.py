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


def _kinds(path):
    with torchfits.open(path) as hdul:
        return [type(h).__name__ for h in hdul]


def test_edits_that_would_renumber_hdus_raise(tmp_path):
    import pytest

    path = str(tmp_path / "idx.fits")
    torchfits.write(path, [torch.zeros(2, 2), torch.ones(2, 2)])
    before = open(path, "rb").read()
    table = {"x": torch.arange(3)}
    with pytest.raises(ValueError, match="primary"):
        torchfits.insert_hdu(path, table, index=0)
    with pytest.raises(ValueError, match="primary"):
        torchfits.replace_hdu(path, 0, table)
    with pytest.raises(ValueError, match="primary"):
        torchfits.insert_hdu(path, torch.zeros(2, 2), index=0, compress=True)
    with pytest.raises(ValueError, match="primary"):
        torchfits.replace_hdu(path, 0, torch.zeros(2, 2), compress=True)
    assert open(path, "rb").read() == before

    tab = str(tmp_path / "tab.fits")
    torchfits.write(tab, table)
    with pytest.raises(ValueError, match="primary"):
        torchfits.delete_hdu(tab, 0)
    assert _kinds(tab) == ["TensorHDU", "TableHDURef"]

    # Edits that land where asked still work.
    torchfits.insert_hdu(path, table, index=1)
    assert torchfits.read(path, hdu=1)["x"].tolist() == [0, 1, 2]
    comp = str(tmp_path / "comp.fits")
    torchfits.write(comp, [torch.ones(4, 4)], compress=True)
    torchfits.insert_hdu(comp, torch.zeros(4, 4), index=2, compress=True)
    assert torchfits.read(comp, hdu=2).shape == (4, 4)
