"""read_extname on an HDU without EXTNAME must not take the error path.

Most primary HDUs have no EXTNAME. The native keyword read used to raise for
a missing key, which sent every such call through the lookup re-check (an
extra file open). It now asks for the key with missing_ok.
"""

import pytest
import torch

import torchfits
import torchfits._io_engine._hdu_lookup as lookup
from torchfits import HDUNotFoundError


def test_missing_extname_is_none_without_error_path(tmp_path, monkeypatch):
    path = str(tmp_path / "x.fits")
    torchfits.write(path, torch.zeros(2, 2), overwrite=True)

    def boom(*_args, **_kwargs):
        raise AssertionError("lookup re-check ran for a normal call")

    monkeypatch.setattr(lookup, "count_hdus", boom)
    monkeypatch.setattr(lookup, "raise_lookup_failure", boom)
    assert torchfits.read_extname(path, 0) is None


def test_present_extname_and_lookup_errors(tmp_path):
    path = str(tmp_path / "y.fits")
    hdus = [torch.zeros(2, 2), torch.ones(3, 3)]
    torchfits.write(path, hdus, overwrite=True)
    torchfits.write_checksums(path)  # any edit keeps EXTNAME handling simple
    with torchfits.open(path) as hdul:
        names = [h.header.get("EXTNAME") for h in hdul]
    assert torchfits.read_extname(path, 1) == names[1]
    with pytest.raises(HDUNotFoundError):
        torchfits.read_extname(path, 9)
    with pytest.raises(FileNotFoundError):
        torchfits.read_extname(str(tmp_path / "missing.fits"), 0)
