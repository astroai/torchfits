"""BIT ('X') columns whose repeat is not a multiple of 8 must read back per row.

CFITSIO's TBIT reader includes each cell's pad bits in a multi-row read, so a
single fits_read_col over all rows misaligns every row after the first.
"""

import numpy as np
import pytest

import torchfits
from torchfits._table.write import write as write_table


@pytest.mark.parametrize(
    ("tform", "values"),
    [
        ("1X", np.array([True, False, True, True, False, True])),
        ("3X", np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=bool)),
        (
            "8X",
            np.array([[1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1]], dtype=bool),
        ),
        ("10X", np.array([[1] * 9 + [0], [0] * 9 + [1]], dtype=bool)),
    ],
)
def test_bit_column_roundtrip(tmp_path, tform, values):
    path = tmp_path / "bits.fits"
    write_table(
        str(path), {"F": values}, schema={"F": {"format": tform}}, overwrite=True
    )

    expected = (
        values.reshape(values.shape[0], -1).squeeze(-1) if tform == "1X" else values
    )
    got = torchfits.read(str(path), hdu=1)["F"]
    assert got.numpy().reshape(expected.shape).tolist() == expected.tolist()

    # The mmap path decodes raw bytes and must agree.
    via_mmap = torchfits._C.read_fits_table_rows(str(path), 1, ["F"], 1, -1, True)["F"]
    assert via_mmap.numpy().reshape(expected.shape).tolist() == expected.tolist()
