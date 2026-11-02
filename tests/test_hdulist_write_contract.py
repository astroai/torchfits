"""``HDUList.write`` follows the same path contract as ``torchfits.write``.

It used to call the raw HDU writer directly: an existing file with
``overwrite=False`` surfaced CFITSIO's RuntimeError instead of
FileExistsError, and a ``!``-prefixed path let CFITSIO clobber the file.
"""

from __future__ import annotations

import os

import pytest
import torch

import torchfits


def _hdul():
    return torchfits.HDUList([torchfits.TensorHDU(torch.arange(6.0).reshape(2, 3))])


def test_existing_file_without_overwrite_raises_file_exists(tmp_path):
    target = tmp_path / "out.fits"
    target.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        _hdul().write(str(target))
    assert target.read_bytes() == b"keep me"


def test_bang_prefix_does_not_clobber(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "victim.fits"
    victim.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="clobber"):
        _hdul().write("!victim.fits")
    assert victim.read_bytes() == b"keep me"


def test_overwrite_replaces_and_roundtrips(tmp_path):
    target = tmp_path / "out.fits"
    target.write_bytes(b"old")
    _hdul().write(os.fspath(target), overwrite=True)
    assert torch.equal(torchfits.read(str(target)), torch.arange(6.0).reshape(2, 3))


def test_rewrite_open_file_onto_itself(tmp_path):
    target = tmp_path / "self.fits"
    torchfits.write(str(target), torch.arange(4.0), overwrite=True)
    with torchfits.open(str(target)) as hdul:
        hdul.write(str(target), overwrite=True)
    assert torch.equal(torchfits.read(str(target)), torch.arange(4.0))
