"""write_checksums / verify_checksums accept an EXTNAME like the other HDU APIs."""

from __future__ import annotations

import pytest
import torch

import torchfits


@pytest.fixture()
def mef(tmp_path):
    path = tmp_path / "mef.fits"
    torchfits.write(
        str(path),
        torchfits.HDUList(
            [
                torchfits.TensorHDU(torch.empty(0)),
                torchfits.TensorHDU(torch.ones(3, 3), {"EXTNAME": "SCI"}),
                torchfits.TensorHDU(torch.zeros(2), {"EXTNAME": "ERR"}),
            ]
        ),
    )
    return str(path)


@pytest.mark.parametrize("name", ["SCI", "sci"])
def test_checksums_by_extname(mef, name):
    torchfits.write_checksums(mef, hdu=name)
    assert "CHECKSUM" in torchfits.read_header(mef, hdu=1)
    assert "CHECKSUM" not in torchfits.read_header(mef, hdu=2)
    assert torchfits.verify_checksums(mef, hdu=name)["status"] == "ok"
    assert torchfits.verify_checksums(mef, hdu=1)["status"] == "ok"
    assert torchfits.verify_checksums(mef, hdu="ERR")["status"] == "no_checksums"


def test_unknown_extname_is_refused(mef):
    with pytest.raises((KeyError, ValueError)):
        torchfits.write_checksums(mef, hdu="NOPE")
    with pytest.raises((KeyError, ValueError)):
        torchfits.verify_checksums(mef, hdu="NOPE")


@pytest.mark.parametrize("bad", [True, 1.0, None])
def test_non_index_types_still_rejected(mef, bad):
    with pytest.raises(TypeError):
        torchfits.verify_checksums(mef, hdu=bad)
