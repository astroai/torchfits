"""A 0-row numpy string column writes with its dtype width.

With no cell to infer the type from, the writer raised std::bad_cast.
"""

import numpy as np
import pytest

import torchfits


@pytest.mark.parametrize(
    ("dtype", "table_type", "tform"),
    [
        ("S3", "binary", "3A"),
        ("U5", "binary", "5A"),
        ("S3", "ascii", "A3"),
    ],
)
def test_empty_string_column_keeps_width(tmp_path, dtype, table_type, tform):
    path = str(tmp_path / "s.fits")
    torchfits.table.write(
        path,
        {"s": np.array([], dtype=dtype), "x": np.zeros(0)},
        overwrite=True,
        table_type=table_type,
    )
    header = torchfits.read_header(path, 1)
    assert header["TFORM1"] == tform
    assert header["NAXIS2"] == 0
    table = torchfits.table.read(path, hdu=1)
    assert table.num_rows == 0 and str(table.schema.field("s").type) == "string"


def test_root_write_empty_string_column(tmp_path):
    path = str(tmp_path / "w.fits")
    torchfits.write(path, {"s": np.array([], dtype="S3")})
    assert torchfits.read_header(path, 1)["TFORM1"] == "3A"
