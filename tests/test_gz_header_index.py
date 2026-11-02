"""Metadata of ``.fits.gz`` files comes from one decompression, not one per call.

CFITSIO inflates a whole ``.gz`` file on every open; ``torchfits.open`` on a
20-HDU file used to open it ~40 times. The header index answers header
strings, HDU types and the HDU count from a single CFITSIO handle, must agree
exactly with the per-call probes, and must never serve a file that changed.
"""

from __future__ import annotations

import gzip
import os
import shutil

import numpy as np
import pytest
import torch

import torchfits
from torchfits import HDUList, TensorHDU, _core_api
from torchfits._io_engine import gz_index


def _gzip(src, dst):
    with open(src, "rb") as s, gzip.open(dst, "wb", compresslevel=1) as o:
        shutil.copyfileobj(s, o)


@pytest.fixture()
def gz_mef(tmp_path):
    plain = tmp_path / "mef.fits"
    hdus = HDUList(
        [
            TensorHDU(torch.zeros(2, 2), header={"OBJECT": "prim"}),
            TensorHDU(torch.arange(12.0).reshape(3, 4), header={"EXTNAME": "SCI"}),
            TensorHDU(torch.ones(5, dtype=torch.int16), header={"EXTNAME": "AUX"}),
        ]
    )
    torchfits.write(str(plain), hdus)
    torchfits.table.write(str(tmp_path / "t.fits"), {"x": np.arange(3.0)})
    torchfits.insert_hdu(str(plain), {"x": np.arange(3.0)}, index=3)
    torchfits.insert_hdu(
        str(plain), torch.ones(8, 8, dtype=torch.int32), index=4, compress="GZIP_1"
    )
    gz = tmp_path / "mef.fits.gz"
    _gzip(plain, gz)
    gz_index.invalidate(None)
    yield str(gz)
    gz_index.invalidate(None)


def test_index_matches_per_call_cfitsio(gz_mef):
    core = _core_api._core()
    n = int(core.read_num_hdus(gz_mef))
    assert n == 5
    assert gz_index.num_hdus(gz_mef) == n
    for i in range(n):
        assert gz_index.header_string(gz_mef, i) == core.read_header_string(gz_mef, i)
        assert gz_index.hdu_type(gz_mef, i) == core.read_hdu_type(gz_mef, i)
    assert [gz_index.hdu_type(gz_mef, i) for i in range(n)] == [
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "BINARY_TABLE",
        "IMAGE",
    ]


def test_metadata_calls_share_one_decompression(gz_mef, monkeypatch):
    calls = []
    real = gz_index._load
    monkeypatch.setattr(gz_index, "_load", lambda p: calls.append(p) or real(p))
    hdul = torchfits.open(gz_mef)
    assert len(hdul) == 5
    assert hdul[1].header["EXTNAME"] == "SCI"
    assert torchfits.read_header(gz_mef, 2)["EXTNAME"] == "AUX"
    x = torchfits.read(gz_mef, hdu=1)
    assert torch.equal(x, torch.arange(12.0).reshape(3, 4))
    assert len(calls) == 1


def test_changed_file_is_never_served_stale(gz_mef, tmp_path):
    assert torchfits.read_header(gz_mef, 1)["EXTNAME"] == "SCI"
    # Rewritten through torchfits (atomic replace) ...
    torchfits.write(gz_mef, torch.ones(2, 2), header={"OBJECT": "new"}, overwrite=True)
    assert torchfits.read_num_hdus(gz_mef) == 1
    assert torchfits.read_header(gz_mef, 0)["OBJECT"] == "new"
    # ... and in place by another program.
    other = tmp_path / "o.fits"
    torchfits.write(
        str(other), HDUList([TensorHDU(torch.zeros(1)), TensorHDU(torch.ones(3))])
    )
    with open(other, "rb") as s, gzip.open(gz_mef, "wb") as o:
        shutil.copyfileobj(s, o)
    assert torchfits.read_num_hdus(gz_mef) == 2
    assert len(torchfits.open(gz_mef)) == 2


def test_insert_hdu_on_gz_sees_new_layout(gz_mef):
    assert len(torchfits.open(gz_mef)) == 5
    torchfits.insert_hdu(gz_mef, torch.zeros(4, 4), index=1, header={"EXTNAME": "NEW"})
    hdul = torchfits.open(gz_mef)
    assert len(hdul) == 6
    assert hdul[1].header["EXTNAME"] == "NEW"
    assert hdul[2].header["EXTNAME"] == "SCI"


def test_index_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(gz_index, "_MAX_ENTRIES", 2)
    gz_index.invalidate(None)
    for i in range(4):
        p = tmp_path / f"f{i}.fits"
        torchfits.write(str(p), torch.full((2,), float(i)))
        _gzip(p, tmp_path / f"f{i}.fits.gz")
        assert torchfits.read_num_hdus(str(tmp_path / f"f{i}.fits.gz")) == 1
    assert len(gz_index._index) == 2
    gz_index.invalidate(None)


def test_errors_still_come_from_cfitsio(gz_mef, tmp_path):
    plain = gz_mef[: -len(".gz")]
    for probe in (torchfits.read_header, torchfits.read_hdu_type):
        errors = []
        for path in (gz_mef, plain):
            with pytest.raises(Exception) as info:
                probe(path, 9)
            errors.append((type(info.value), str(info.value).replace(path, "P")))
        assert errors[0] == errors[1]
    raw = gzip.decompress(open(gz_mef, "rb").read())
    # Whatever CFITSIO answers for a truncated file -- a count or an error --
    # the indexed path answers the same. (A fresh path per cut: an in-place
    # rewrite of equal size within one clock tick defeats any stat check.)
    core = _core_api._core()
    for cut in (1, 2, 5):
        truncated = str(tmp_path / f"trunc{cut}.fits.gz")
        with gzip.open(truncated, "wb") as o:
            o.write(raw[: len(raw) - 2880 * cut])
        try:
            expected = int(core.read_num_hdus(truncated))
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                torchfits.read_num_hdus(truncated)
            assert gz_index.num_hdus(truncated) is None
        else:
            assert torchfits.read_num_hdus(truncated) == expected


def test_plain_fits_is_not_indexed(tmp_path):
    p = tmp_path / "plain.fits"
    torchfits.write(str(p), torch.ones(2))
    assert gz_index.num_hdus(str(p)) is None
    assert gz_index.header_string(str(p), 0) is None
    os.rename(p, tmp_path / "fake.fits.gz")  # not gzip data despite the name
    assert gz_index.num_hdus(str(tmp_path / "fake.fits.gz")) is None
