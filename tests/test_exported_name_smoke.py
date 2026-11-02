"""Behavioural smoke for public names that no other test referenced."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits import data, table, transforms


@pytest.fixture()
def catalog(tmp_path):
    path = tmp_path / "cat.fits"
    table.write(
        str(path),
        {"ID": np.arange(4, dtype=np.int32), "MAG": np.linspace(18.0, 19.5, 4)},
        extname="CAT",
    )
    return str(path)


def test_fits_polars_frame_wraps_read_polars(catalog):
    pytest.importorskip("polars")
    frame = table.read_polars(catalog)
    assert isinstance(frame, table.FITSPolarsFrame)
    assert len(frame) == 4 and frame.height == 4
    assert frame.columns == ["ID", "MAG"]
    assert frame["ID"].to_list() == [0, 1, 2, 3]
    assert "MAG" in frame.field_meta
    assert "FITSPolarsFrame(frame=" in repr(frame)


def test_read_astropy_returns_table(catalog):
    pytest.importorskip("astropy")
    tbl = table.read_astropy(catalog, hdu="CAT", columns=["MAG"])
    assert tbl.colnames == ["MAG"]
    assert len(tbl) == 4
    assert float(tbl["MAG"][0]) == pytest.approx(18.0)


def test_discover_bands_and_band_info(tmp_path):
    path = tmp_path / "bands.fits"
    torchfits.write(
        str(path),
        torchfits.HDUList(
            [
                torchfits.TensorHDU(torch.empty(0)),
                torchfits.TensorHDU(
                    torch.ones(4, 5), {"EXTNAME": "G", "MAGZP": 22.5, "EXPTIME": 2.0}
                ),
                torchfits.TensorHDU(torch.ones(4, 5), {"EXTNAME": "G_IVAR"}),
                torchfits.TensorHDU(torch.ones(3), {"EXTNAME": "SPEC1D"}),
            ]
        ),
    )
    bands = data.discover_bands(path)
    assert [b.name for b in bands] == ["G", "G_IVAR"]
    g, ivar = bands
    assert isinstance(g, data.BandInfo)
    assert (g.index, g.shape, g.role) == (1, (4, 5), "flux")
    assert ivar.role == "ivar"
    assert g.zeropoint == pytest.approx(22.5)
    assert g.flux_scale() == pytest.approx(10 ** (-0.4 * 22.5))
    assert g.flux_scale(exptime_normalized=False) == pytest.approx(
        10 ** (-0.4 * 22.5) / 2.0
    )
    assert ivar.flux_scale() is None
    assert g.to_dict()["name"] == "G"


def test_calibration_state():
    state = transforms.calibration_state({"BSCALE": 2.0})
    assert state is transforms.DataState.PHYSICAL
    assert (
        transforms.calibration_state({}, raw_scale=True) is transforms.DataState.STORED
    )


def test_affine_transform_round_trip():
    x = torch.tensor([1.0, 2.0, 4.0])
    affine = transforms.AffineTransform(scale=2.0, offset=1.0)
    y = affine(x)
    assert torch.allclose(y, torch.tensor([3.0, 5.0, 9.0]))
    assert torch.allclose(affine.inverse(y), x)
    with pytest.raises(ValueError, match="non-zero"):
        transforms.AffineTransform(scale=0.0)
