"""insert_column / replace_column infer the TZERO convention like table.write.

uint16/uint32/uint64 used to raise TypeError ("FITS has no native unsigned
... format") and int8 had no inferable TFORM at all, although table.write,
append_rows and update_rows accept those dtypes.
"""

import numpy as np
import pytest
import torch

import torchfits
from torchfits import table

CASES = [
    (np.uint16, "1I", 32768, torch.uint16),
    (np.uint32, "1J", 2147483648, torch.uint32),
    (np.uint64, "1K", 2**63, torch.uint64),
    (np.int8, "1B", -128, torch.int8),
]


def _values(dt):
    info = np.iinfo(dt)
    return np.array([info.min, 1, info.max], dtype=dt)


@pytest.mark.parametrize(("dt", "tform", "tzero", "tdtype"), CASES)
@pytest.mark.parametrize("as_torch", [False, True])
def test_insert_column_infers_offset_convention(
    tmp_path, dt, tform, tzero, tdtype, as_torch
):
    path = str(tmp_path / "t.fits")
    table.write(path, {"X": np.arange(3)})
    values = _values(dt)
    table.insert_column(path, "U", torch.from_numpy(values) if as_torch else values)

    reference = str(tmp_path / "ref.fits")
    table.write(reference, {"X": np.arange(3), "U": values})
    for p in (path, reference):
        header = torchfits.read_header(p, 1)
        assert header["TFORM2"].lstrip("1") == tform[-1]
        assert int(header["TZERO2"]) == tzero
    got = table.read_torch(path, hdu=1)["U"]
    assert got.dtype == tdtype
    assert got.tolist() == values.tolist()


@pytest.mark.parametrize(("dt", "tform", "tzero", "tdtype"), CASES)
def test_replace_column_infers_offset_convention(tmp_path, dt, tform, tzero, tdtype):
    path = str(tmp_path / "t.fits")
    table.write(path, {"X": np.arange(3), "Y": np.arange(3.0)})
    # Dropping the format lets replace_column infer it from the new values.
    table.drop_columns(path, ["Y"])
    table.insert_column(path, "Y", _values(dt))
    table.replace_column(path, "Y", _values(dt)[::-1].copy())
    header = torchfits.read_header(path, 1)
    assert header["TFORM2"].lstrip("1") == tform[-1]
    assert int(header["TZERO2"]) == tzero
    got = table.read_torch(path, hdu=1)["Y"]
    assert got.dtype == tdtype
    assert got.tolist() == _values(dt)[::-1].tolist()


@pytest.mark.parametrize("dt", [np.uint16, np.int8])
def test_ascii_table_still_refuses_offset_dtypes(tmp_path, dt):
    path = str(tmp_path / "a.fits")
    table.write(path, {"X": np.arange(3)}, table_type="ascii")
    with pytest.raises(ValueError, match="ASCII table"):
        table.insert_column(path, "U", _values(dt))
