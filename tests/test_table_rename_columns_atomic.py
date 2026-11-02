"""rename_columns must validate the whole mapping before writing any TTYPEn.

CFITSIO caches column names, so resolving renames one at a time saw stale
names: a valid rotation failed after half-applying, two sources could take the
same target through the native call, and a case-only rename was refused.
"""

import numpy as np
import pytest

import torchfits
import torchfits._C as C
from torchfits._table.mutation import rename_columns
from torchfits._table.write import write as write_table


@pytest.fixture
def table_path(tmp_path):
    path = tmp_path / "ren.fits"
    write_table(
        str(path),
        {"A": np.arange(3, dtype=np.int32), "B": np.arange(3, 6, dtype=np.int32)},
        overwrite=True,
    )
    return str(path)


def _columns(path):
    return list(torchfits.read(path, hdu=1).keys())


def test_rotation_rename(table_path):
    rename_columns(table_path, {"B": "C", "A": "B"})
    data = torchfits.read(table_path, hdu=1)
    assert list(data.keys()) == ["B", "C"]
    assert data["B"].tolist() == [0, 1, 2]
    assert data["C"].tolist() == [3, 4, 5]


def test_case_only_rename(table_path):
    rename_columns(table_path, {"A": "a"})
    assert _columns(table_path) == ["a", "B"]


def test_duplicate_target_is_rejected_without_writing(table_path):
    with pytest.raises(RuntimeError, match="already exists"):
        C.rename_fits_table_columns(table_path, 1, {"A": "X", "B": "X"})
    assert _columns(table_path) == ["A", "B"]


def test_collision_is_rejected_without_writing(table_path):
    with pytest.raises(RuntimeError, match="already exists"):
        C.rename_fits_table_columns(table_path, 1, {"A": "Z", "B": "z"})
    assert _columns(table_path) == ["A", "B"]
