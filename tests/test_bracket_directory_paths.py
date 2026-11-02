"""A '[' in a directory name is part of the path, not a CFITSIO filter.

CFITSIO's extended-filename parser reads the first '[' anywhere as the start of
a filter, so every write (and read-write open) under ``/data/[br]/`` failed.
Plain local paths that contain '[' without a trailing filter section are now
opened and created literally.
"""

from __future__ import annotations

import os

import pytest
import torch

import torchfits


@pytest.fixture()
def bracket_dir(tmp_path):
    d = tmp_path / "[br]" / "x[1]y"
    d.mkdir(parents=True)
    return d


def test_image_write_read_roundtrip(bracket_dir):
    path = str(bracket_dir / "img.fits")
    data = torch.arange(12.0).reshape(3, 4)
    torchfits.write(path, data)
    assert os.listdir(bracket_dir) == ["img.fits"]
    assert torch.equal(torchfits.read(path), data)
    assert torchfits.read_header(path)["NAXIS1"] == 4
    with torchfits.open(path) as hdul:
        assert len(hdul) == 1
    torchfits.write(path, data * 2, overwrite=True)
    assert torch.equal(torchfits.read(path), data * 2)


def test_compressed_and_mef_edits(bracket_dir):
    path = str(bracket_dir / "mef.fits")
    torchfits.write(path, torch.ones(16, 16), compress="RICE")
    torchfits.insert_hdu(path, torch.zeros(2), 2, header={"EXTNAME": "NEW"})
    torchfits.write_checksums(path, hdu="NEW")
    assert torchfits.verify_checksums(path, hdu="NEW")["status"] == "ok"
    assert torch.equal(torchfits.read(path, hdu="NEW"), torch.zeros(2))


def test_table_write_append_read(bracket_dir):
    path = str(bracket_dir / "tab.fits")
    torchfits.table.write(path, {"A": [1, 2]})
    torchfits.table.append_rows(path, {"A": [3]})
    assert torchfits.table.read(path).column("A").to_pylist() == [1, 2, 3]


def test_relative_path_under_bracket_directory(bracket_dir, monkeypatch):
    monkeypatch.chdir(bracket_dir.parent.parent)
    rel = os.path.join("[br]", "x[1]y", "rel.fits")
    torchfits.write(rel, torch.ones(3))
    assert torch.equal(torchfits.read(rel), torch.ones(3))


def test_filter_on_file_in_plain_directory_still_works(tmp_path):
    path = str(tmp_path / "f.fits")
    torchfits.write(path, torch.arange(10.0))
    assert torchfits.read_header(path + "[0]")["NAXIS1"] == 10
