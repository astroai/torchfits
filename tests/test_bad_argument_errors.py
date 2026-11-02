"""Bad arguments raise the argument error, not a RuntimeError wrapper.

``open(p, mode="update")`` wrapped HDUList's ValueError in RuntimeError, and
``table.read(p, hdu=-1)`` reached CFITSIO ("illegal HDU number") instead of
the ValueError ``read`` / ``read_header`` raise for a negative index.
"""

from __future__ import annotations

import pytest

import torchfits


@pytest.fixture()
def table_path(tmp_path):
    path = tmp_path / "t.fits"
    torchfits.table.write(str(path), {"A": [1, 2, 3]})
    return str(path)


@pytest.mark.parametrize("mode", ["update", "w", "rw"])
def test_open_unsupported_mode_raises_value_error(table_path, mode):
    with pytest.raises(ValueError, match="mode='r' only") as excinfo:
        torchfits.open(table_path, mode=mode)
    assert not isinstance(excinfo.value, RuntimeError)


def test_open_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        torchfits.open(str(tmp_path / "missing.fits"))


@pytest.mark.parametrize("hdu", [-1, -2])
def test_table_read_negative_hdu_raises_value_error(table_path, hdu):
    # HDUNotFoundError is a ValueError, so this contract still holds.
    with pytest.raises(torchfits.HDUNotFoundError, match="non-negative"):
        torchfits.table.read(table_path, hdu=hdu)
    with pytest.raises(ValueError, match="non-negative"):
        torchfits.read(table_path, hdu=hdu)
