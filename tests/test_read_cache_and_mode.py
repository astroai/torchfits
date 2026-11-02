"""Read-path regressions: cache identity, explicit mode, HDU lists, EXTNAME scan.

- The Python read/header caches stat'ed ``mef.fits[1]`` / ``file://`` names
  literally, failed, and kept the entries forever (stale after a rewrite).
- The read-cache key has no ``mode``, so ``mode='image'`` returned a cached
  table dict; a cached table HDU type also overrode ``mode='image'``.
- ``read(p, hdu=[...])`` dropped ``mode``, sent EXTNAMEs to the native batch
  with ``mmap=True`` (TypeError), and had no fallback when one HDU was a table.
- The EXTNAME scan fallback called ``.get`` on a list of cards.
- An HTTP cutout of a BINTABLE returned raw row bytes as a uint8 image.
"""

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as C
import torchfits._io_engine.http_subset as http_subset


class _Table:
    def __init__(self, data, extname):
        self._raw_data = data
        self.header = {"EXTNAME": extname}
        self._schema = None


@pytest.fixture
def mef(tmp_path):
    """Primary 3x4 image, HDU 1 table 'CAT', HDU 2 image 'SCI'."""
    path = str(tmp_path / "mef.fits")
    image = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    C.write_fits_file(
        path,
        [
            {"data": image},
            _Table({"A": torch.arange(5, dtype=torch.int32)}, "CAT"),
            {"data": image + 100, "header": {"EXTNAME": "SCI"}},
        ],
        True,
    )
    return path, image


def test_extended_syntax_cache_sees_rewrite(mef):
    path, image = mef
    data, _hdr = torchfits.read(path + "[2]", return_header=True)
    assert torch.equal(data, image + 100)
    torchfits.replace_hdu(path, 2, torch.full((2, 2), 7.0), header={"NEWKEY": 5})
    data, hdr = torchfits.read(path + "[2]", return_header=True)
    assert torch.equal(data, torch.full((2, 2), 7.0))
    assert torchfits.read_header(path + "[2]").get("NEWKEY") == 5


def test_file_url_cache_sees_rewrite(tmp_path):
    path = tmp_path / "u.fits"
    url = "file://" + str(path)
    torchfits.write(str(path), torch.ones(2, 2), overwrite=True)
    assert torchfits.read(url, return_header=True)[0].shape == (2, 2)
    torchfits.write(str(path), torch.full((3, 3), 2.0), overwrite=True)
    assert torch.equal(
        torchfits.read(url, return_header=True)[0], torch.full((3, 3), 2.0)
    )


def test_explicit_mode_is_not_answered_from_cache(mef):
    path, _ = mef
    torchfits.read(path, 1, return_header=True)  # caches the table read
    torchfits.read(path, 1)
    for kwargs in ({"return_header": True}, {}):
        with pytest.raises(Exception):
            torchfits.read(path, 1, mode="image", **kwargs)
    torchfits.read(path, 2, return_header=True)
    with pytest.raises(Exception):
        torchfits.read(path, 2, mode="table", return_header=True)


def test_hdu_list_honors_mode_names_and_mixed_types(mef):
    path, image = mef
    with pytest.raises(Exception):
        torchfits.read(path, hdu=[1], mode="image")
    with pytest.raises(Exception):
        torchfits.read(path, hdu=[2], mode="table")
    got = torchfits.read(path, hdu=["SCI"], mmap=True)
    assert torch.equal(got[0], image + 100)
    mixed = torchfits.read(path, hdu=[1, 2], mmap=True)
    assert mixed[0]["A"].tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(mixed[1], image + 100)


def test_missing_extname_reports_not_found(mef):
    path, _ = mef
    with pytest.raises(Exception, match="not found"):
        torchfits.read(path, hdu="NOPE")


def test_http_cutout_refuses_table_hdu(mef, monkeypatch):
    path, _ = mef
    blob = open(path, "rb").read()

    def fake_range(url, start, end, timeout=None):
        return blob[start : end + 1]

    monkeypatch.setattr(http_subset, "http_read_range", fake_range)
    with pytest.raises(ValueError, match="not an image"):
        http_subset.locate_uncompressed_2d("https://example.org/mef.fits", 1)
    info = http_subset.locate_uncompressed_2d("https://example.org/mef.fits", 2)
    assert (info["naxis1"], info["naxis2"]) == (4, 3)
    assert np.dtype(str(info["dtype"]).replace("torch.", "")) == np.float32


def test_http_cutout_of_compressed_hdu_falls_back(tmp_path, monkeypatch):
    """A tile-compressed image is a BINTABLE: it must ask for a full download
    (HttpRangeUnsupported), not be rejected as a table."""
    path = str(tmp_path / "c.fits")
    torchfits.write(
        path,
        torch.arange(64, dtype=torch.int16).reshape(8, 8),
        compress="RICE",
        overwrite=True,
    )
    blob = open(path, "rb").read()

    def fake_range(url, start, end, timeout=None):
        return blob[start : end + 1]

    monkeypatch.setattr(http_subset, "http_read_range", fake_range)
    with pytest.raises(http_subset.HttpRangeUnsupported):
        http_subset.locate_uncompressed_2d("https://example.org/c.fits", 1)


def test_table_write_accepts_pathlib_path(tmp_path):
    import torchfits.table as table

    path = tmp_path / "t.fits"
    table.write(path, {"A": np.array([1, 2], dtype=np.uint16)}, overwrite=True)
    assert torchfits.read(str(path), hdu=1)["A"].tolist() == [1, 2]
