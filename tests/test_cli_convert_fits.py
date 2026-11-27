"""``torchfits convert --to fits``: a FITS table rewrite keeps its metadata.

The output must keep the stored column layout (TFORM, TSCAL/TZERO, TNULL
sentinels, TDIM, unsigned convention), TUNIT and other per-column cards
(renumbered under ``--columns``), EXTNAME and other user keywords -- not
the float64/NaN, ``EXTNAME='Table'`` re-encoding of a round trip through
Arrow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits.cli.main import main


@pytest.fixture
def catalog(tmp_path):
    src = tmp_path / "cat.fits"
    data = {
        "flux": np.array([1, -999, 3], dtype=np.int32),
        "mag": np.array([10.0, 11.0, 12.0], dtype=np.float32),
        "sc": np.array([1, 2, 4], dtype=np.int16),
        "vec": np.arange(18.0).reshape(3, 6),
        "u": np.array([1, 2, 2**32 - 1], dtype=np.uint32),
        "name": ["a", "bb", "ccc"],
    }
    schema = {
        "flux": {"format": "J", "unit": "pix", "tnull": -999},
        "mag": {"unit": "mag"},
        "sc": {"format": "I", "bscale": 0.5, "bzero": 10.0},
        "vec": {"format": "6D", "dim": "(3,2)"},
        "name": {"format": "5A"},
    }
    torchfits.table.write(
        str(src),
        data,
        schema=schema,
        header={"EXTNAME": "CAT", "OBSERVER": "me", "TDISP2": "F6.2"},
        overwrite=True,
    )
    return src


def _header(path):
    return torchfits.read_header(str(path), 1)


def test_convert_fits_keeps_layout_and_metadata(catalog, tmp_path):
    out = tmp_path / "out.fits"
    assert main(["convert", str(catalog), str(out), "--to", "fits"]) == 0
    hdr = _header(out)
    assert hdr["EXTNAME"] == "CAT"
    assert hdr["OBSERVER"] == "me"
    assert hdr["TTYPE1"] == "flux"
    assert hdr["TFORM1"].strip().lstrip("1") == "J"
    assert hdr["TUNIT1"] == "pix"
    assert hdr["TNULL1"] == -999
    assert hdr["TUNIT2"] == "mag"
    assert hdr["TDISP2"] == "F6.2"
    assert hdr["TFORM3"].strip().lstrip("1") == "I"
    assert hdr["TSCAL3"] == 0.5 and hdr["TZERO3"] == 10
    assert hdr["TDIM4"] == "(3,2)"
    assert hdr["TZERO5"] == 2**31
    assert hdr["TFORM6"] == "5A"
    got = torchfits.table.read(str(out), hdu=1).to_pydict()
    assert got == torchfits.table.read(str(catalog), hdu=1).to_pydict()
    assert got["flux"] == [1, None, 3]
    raw = torchfits.table.read_torch(str(out), hdu=1, columns=["flux"])["flux"]
    assert raw.dtype == torch.int32


def test_convert_fits_columns_renumbers_per_column_cards(catalog, tmp_path):
    out = tmp_path / "out.fits"
    code = main(["convert", str(catalog), str(out), "--to", "fits", "-c", "mag,flux,u"])
    assert code == 0
    hdr = _header(out)
    assert hdr["TFIELDS"] == 3
    assert [hdr["TTYPE1"], hdr["TTYPE2"], hdr["TTYPE3"]] == ["mag", "flux", "u"]
    assert hdr["TUNIT1"] == "mag" and hdr["TDISP1"] == "F6.2"
    assert hdr["TUNIT2"] == "pix" and hdr["TNULL2"] == -999
    assert hdr["TZERO3"] == 2**31
    assert "TNULL1" not in hdr and "TDISP2" not in hdr and "TSCAL4" not in hdr
    assert hdr["EXTNAME"] == "CAT"
    got = torchfits.table.read(str(out), hdu=1).to_pydict()
    assert got == {
        "mag": [10.0, 11.0, 12.0],
        "flux": [1, None, 3],
        "u": [1, 2, 2**32 - 1],
    }


def test_convert_fits_where_filters_rows_keeping_metadata(catalog, tmp_path):
    out = tmp_path / "out.fits"
    code = main(["convert", str(catalog), str(out), "--to", "fits", "-w", "mag > 10"])
    assert code == 0
    hdr = _header(out)
    assert hdr["NAXIS2"] == 2
    assert hdr["TNULL1"] == -999 and hdr["TSCAL3"] == 0.5
    got = torchfits.table.read(str(out), hdu=1).to_pydict()
    assert got["flux"] == [None, 3]
    assert got["sc"] == [11.0, 12.0]
    assert got["name"] == ["bb", "ccc"]
    assert got["u"] == [2, 2**32 - 1]
    assert got["vec"][0] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_convert_fits_where_and_columns(catalog, tmp_path):
    out = tmp_path / "out.fits"
    code = main(
        [
            "convert",
            str(catalog),
            str(out),
            "--to",
            "fits",
            "-c",
            "name,sc",
            "-w",
            "flux > 0",
        ]
    )
    assert code == 0
    hdr = _header(out)
    assert hdr["TFIELDS"] == 2 and hdr["TSCAL2"] == 0.5
    got = torchfits.table.read(str(out), hdu=1).to_pydict()
    assert got == {"name": ["a", "ccc"], "sc": [10.5, 12.0]}


def test_convert_fits_where_selecting_no_rows(catalog, tmp_path):
    out = tmp_path / "out.fits"
    code = main(["convert", str(catalog), str(out), "--to", "fits", "-w", "mag > 100"])
    assert code == 0
    assert _header(out)["NAXIS2"] == 0
    assert _header(out)["TFORM6"] == "5A"


@pytest.mark.parametrize(
    "extra",
    [["-c", "nope"], ["-w", "nope > 1"], ["-w", "mag >"], ["-w", "name > 3"]],
)
def test_convert_fits_bad_selection_is_usage_error(catalog, tmp_path, extra):
    out = tmp_path / "out.fits"
    out.write_bytes(b"keep")
    assert main(["convert", str(catalog), str(out), "--to", "fits", *extra]) == 2
    assert out.read_bytes() == b"keep"


def test_convert_fits_ascii_table(tmp_path):
    src = tmp_path / "asc.fits"
    torchfits.table.write(
        str(src),
        {"a": np.array([1, 2, 3], dtype=np.int32), "b": np.array([1.5, 2.5, 3.5])},
        table_type="ascii",
        header={"EXTNAME": "ASC"},
        overwrite=True,
    )
    out = tmp_path / "out.fits"
    assert main(["convert", str(src), str(out), "--to", "fits", "-w", "a >= 2"]) == 0
    hdr = _header(out)
    assert hdr["XTENSION"].strip() == "TABLE"
    assert hdr["EXTNAME"] == "ASC"
    assert torchfits.table.read(str(out), hdu=1).to_pydict() == {
        "a": [2, 3],
        "b": [2.5, 3.5],
    }
