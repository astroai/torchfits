"""Writes and HDU edits of x.fits.gz keep the file gzip-compressed.

The atomic temp file was named '.tmp.fits', so CFITSIO wrote plain FITS and
the overwrite/edit published it under the '.gz' name.
"""

import gzip

import torch

import torchfits

GZIP_MAGIC = b"\x1f\x8b"


def _magic(path):
    with open(path, "rb") as fh:
        return fh.read(2)


def test_overwrite_and_edits_keep_gzip(tmp_path):
    path = str(tmp_path / "a.fits.gz")
    img = torch.arange(10000, dtype=torch.float32).reshape(100, 100)
    torchfits.write(path, img)
    assert _magic(path) == GZIP_MAGIC
    torchfits.write(path, img * 2, overwrite=True)
    assert _magic(path) == GZIP_MAGIC
    assert torch.equal(torchfits.read(path), img * 2)

    torchfits.insert_hdu(path, {"a": torch.arange(3)}, index=1)
    assert _magic(path) == GZIP_MAGIC
    torchfits.replace_hdu(path, 1, {"a": torch.arange(4)})
    assert _magic(path) == GZIP_MAGIC
    assert torchfits.read(path, hdu=1)["a"].tolist() == [0, 1, 2, 3]
    torchfits.delete_hdu(path, 1)
    assert _magic(path) == GZIP_MAGIC
    assert gzip.decompress(open(path, "rb").read())[:6] == b"SIMPLE"
    assert [p.name for p in tmp_path.iterdir()] == ["a.fits.gz"]


def test_header_and_checksum_on_gzip_target(tmp_path):
    path = str(tmp_path / "h.fits.gz")
    torchfits.write(path, torch.ones(4, 4), header={"OBJECT": "M31"}, checksum=True)
    assert _magic(path) == GZIP_MAGIC
    assert torchfits.read_header(path)["OBJECT"] == "M31"
