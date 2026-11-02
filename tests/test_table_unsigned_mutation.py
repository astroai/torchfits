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


@pytest.mark.parametrize("mmap", ["auto", True, False])
@pytest.mark.parametrize(
    "payload", [[5, 6], np.array([5, 6], dtype=np.int64)], ids=["list", "int64"]
)
def test_uint64_update_with_signed_payload_is_physical(tmp_path, mmap, payload):
    path = str(tmp_path / "u64.fits")
    torchfits.write(path, {"U": np.zeros(3, dtype=np.uint64)}, overwrite=True)
    update_rows(path, {"U": payload}, slice(0, 2), mmap=mmap)
    assert torchfits.read(path, hdu=1)["U"].tolist() == [5, 6, 0]


def test_uint64_update_rejects_negative_values(tmp_path):
    path = str(tmp_path / "u64neg.fits")
    torchfits.write(path, {"U": np.zeros(2, dtype=np.uint64)}, overwrite=True)
    before = open(path, "rb").read()
    with pytest.raises(ValueError, match="negative"):
        update_rows(path, {"U": [-1, 2]}, slice(0, 2))
    assert open(path, "rb").read() == before


def test_native_mmap_update_int64_payload_on_uint64_column(tmp_path):
    import torchfits._C as cpp

    path = str(tmp_path / "u64native.fits")
    torchfits.write(path, {"U": np.zeros(2, dtype=np.uint64)}, overwrite=True)
    cpp.update_fits_table_rows_mmap(path, 1, {"U": np.array([5, 6])}, 1, 2)
    torchfits.clear_all_caches()
    assert torchfits.read(path, hdu=1)["U"].tolist() == [5, 6]
