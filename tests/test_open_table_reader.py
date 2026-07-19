"""open_table_reader: reuse one CFITSIO handle for multiple column reads."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits


def test_open_table_reader_matches_cold_read_torch(tmp_path):
    path = tmp_path / "open_reader.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="A", format="J", array=np.arange(50, dtype=np.int32)),
            fits.Column(name="B", format="E", array=np.arange(50, dtype=np.float32)),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)
    p = str(path)

    cold_a = torchfits.table.read_torch(p, hdu=1, columns=["A"])
    cold_b = torchfits.table.read_torch(p, hdu=1, columns=["B"])

    with torchfits.open_table_reader(p, hdu=1) as reader:
        assert reader.num_rows() == 50
        a = reader.read_torch(columns=["A"])
        b = reader.read_torch(columns=["B"])

    assert torch.equal(a["A"].cpu(), cold_a["A"].cpu())
    assert torch.allclose(b["B"].cpu(), cold_b["B"].cpu())

    with pytest.raises(RuntimeError):
        reader.read_torch(columns=["A"])
