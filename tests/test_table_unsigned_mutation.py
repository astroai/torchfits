"""update_rows/append_rows on unsigned (TZERO-convention) columns.

Every unsigned width failed: the native writers had no uint16/uint32/uint64
payload type ("Unsupported dtype for update_rows"), the mmap update accepted
only signed payloads, and a range check ignoring TZERO warned that valid
values "will wrap on disk".
"""

import warnings

import numpy as np
import pytest

import torchfits
from torchfits._table.mutation import append_rows, update_rows


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("use_mmap", [True, False])
def test_unsigned_update_and_append(tmp_path, dtype, use_mmap):
    top = int(np.iinfo(dtype).max)
    mid = 1 << (np.dtype(dtype).itemsize * 8 - 1)
    path = str(tmp_path / "u.fits")
    torchfits.write(path, {"U": np.array([1, 2, 3], dtype=dtype)}, overwrite=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        update_rows(
            path, {"U": np.array([top, mid], dtype=dtype)}, slice(1, 3), mmap=use_mmap
        )
    append_rows(path, {"U": np.array([7, mid + 1], dtype=dtype)})

    got = torchfits.read(path, hdu=1)["U"]
    assert got.numpy().tolist() == [1, top, mid, 7, mid + 1]


def test_byte_column_range_warning_uses_unsigned_bounds(tmp_path):
    path = str(tmp_path / "b.fits")
    torchfits.write(path, {"B": np.array([1, 2], dtype=np.uint8)}, overwrite=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        update_rows(path, {"B": np.array([200, 255], dtype=np.int16)}, slice(0, 2))
    assert torchfits.read(path, hdu=1)["B"].tolist() == [200, 255]
