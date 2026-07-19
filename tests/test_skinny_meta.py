"""Skinny CFITSIO metadata: read_* helpers (no full header dump)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

import torchfits


def _write_sample(tmp_path):
    path = tmp_path / "skinny_meta.fits"
    data = np.zeros((8, 4), dtype=np.float32)
    primary = fits.PrimaryHDU(data)
    primary.header["EXPTIME"] = 12.5
    primary.header["OBJECT"] = "demo"
    for i in range(50):
        primary.header.add_history(f"pad history line {i}")

    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.arange(100, dtype=np.int32)),
            fits.Column(name="X", format="E", array=np.arange(100, dtype=np.float32)),
        ],
        name="CAT",
    )
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return path


def test_skinny_meta_suite(tmp_path):
    path = _write_sample(tmp_path)
    p = str(path)

    assert torchfits.read_num_hdus(p) == 2
    assert torchfits.read_hdu_type(p, hdu=0) == "IMAGE"
    assert torchfits.read_hdu_type(p, hdu=1) == "BINARY_TABLE"

    bitpix, shape = torchfits.read_shape(p, hdu=0)
    assert bitpix == -32
    assert shape == (8, 4)

    assert torchfits.read_nrows(p, hdu=1) == 100
    assert torchfits.read_nrows(p, hdu="CAT") == 100
    assert torchfits.read_colnames(p, hdu=1) == ["ID", "X"]
    assert torchfits.read_extname(p, hdu=1) == "CAT"
    assert torchfits.read_extname(p, hdu=0) in (None, "PRIMARY", "")

    info = torchfits.read_table_info(p, hdu=1)
    assert info["nrows"] == 100
    assert info["colnames"] == ["ID", "X"]
    assert len(info["tforms"]) == 2

    keys = torchfits.read_keys(p, ["BITPIX", "NAXIS1", "NAXIS2", "EXPTIME"], hdu=0)
    assert keys["BITPIX"] == -32
    assert keys["NAXIS1"] == 4
    assert keys["NAXIS2"] == 8
    assert keys["EXPTIME"] == 12.5

    full = torchfits.read_header(p, hdu=0)
    assert keys["EXPTIME"] == full["EXPTIME"]
    assert torchfits.read_nrows(p, hdu=1) == int(
        torchfits.read_header(p, hdu=1)["NAXIS2"]
    )


def test_read_keys_missing_raises(tmp_path):
    path = tmp_path / "missing_key.fits"
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32)).writeto(path, overwrite=True)
    try:
        torchfits.read_keys(str(path), ["NOTAKEY"], hdu=0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_read_nrows_on_image_raises(tmp_path):
    path = tmp_path / "image_only.fits"
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32)).writeto(path, overwrite=True)
    try:
        torchfits.read_nrows(str(path), hdu=0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_resolve_extname_skips_primary_without_extname(tmp_path):
    """Primary often lacks EXTNAME; named resolve must still find later HDUs."""
    path = _write_sample(tmp_path)
    # Force the skinny EXTNAME scan by using a name (resolve_hdu_name_cached
    # usually succeeds; this still asserts the public path works).
    assert torchfits.read_nrows(str(path), hdu="CAT") == 100
    assert torchfits.read_extname(str(path), hdu="CAT") == "CAT"
