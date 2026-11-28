"""HDUList header typing (r6a-06) and table row-count probes (r3a-17)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits.hdu import Header, TableHDU, TableHDURef


def test_hdulist_headers_are_typed_like_read_header(tmp_path):
    path = str(tmp_path / "mef.fits")
    torchfits.write(
        path,
        [
            {"data": torch.zeros(2, 2), "header": {"OBJECT": "M31", "EXPTIME": 1.5}},
            {"data": torch.ones(3, 4), "header": {"EXTNAME": "SCI", "FLAG": True}},
        ],
        overwrite=True,
    )
    torchfits.table.write(
        str(tmp_path / "t.fits"), {"A": np.arange(3, dtype=np.int32)}, overwrite=True
    )
    for p in (path, str(tmp_path / "t.fits")):
        with torchfits.open(p) as hdul:
            for index in range(len(hdul)):
                assert dict(hdul[index].header) == dict(torchfits.read_header(p, index))
    with torchfits.open(path) as hdul:
        assert hdul[0].header["NAXIS"] == 2
        assert isinstance(hdul[1].header["NAXIS1"], int)
        assert hdul[0].header["EXPTIME"] == 1.5
        assert hdul[1].header["FLAG"] is True


def test_table_ref_num_rows_reads_naxis2(tmp_path):
    path = str(tmp_path / "t.fits")
    torchfits.table.write(path, {"A": np.arange(5, dtype=np.int32)}, overwrite=True)
    with torchfits.open(path) as hdul:
        assert hdul[1].num_rows == len(hdul[1]) == 5
    assert TableHDURef(header=Header({"XTENSION": "BINTABLE"})).num_rows == 0


@pytest.mark.parametrize("bad", ["abc", 2.5, True])
def test_malformed_naxis2_is_not_zero_rows(bad):
    ref = TableHDURef(header=Header({"XTENSION": "BINTABLE", "NAXIS2": bad}))
    with pytest.raises(ValueError, match="NAXIS2"):
        _ = ref.num_rows
    table = TableHDU({}, header=Header({"NAXIS2": bad}))
    with pytest.raises(ValueError, match="NAXIS2"):
        _ = table.num_rows


def test_table_ref_num_rows_propagates_header_errors(tmp_path):
    path = tmp_path / "t.fits"
    torchfits.table.write(str(path), {"A": np.arange(5)}, overwrite=True)
    ref = TableHDURef(source_path=str(path), source_hdu=1)
    path.write_bytes(b"not a FITS file at all" * 200)
    with pytest.raises((OSError, RuntimeError)):
        _ = ref.num_rows
