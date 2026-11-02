"""Row and column mutations keep a stamped HDU's CHECKSUM/DATASUM valid."""

import numpy as np
import pytest

import torchfits

OPS = {
    "append_rows": lambda p: torchfits.table.append_rows(p, {"a": [9], "s": ["q"]}),
    "insert_rows": lambda p: torchfits.table.insert_rows(p, {"a": [9]}, row=0),
    "update_rows_mmap": lambda p: torchfits.table.update_rows(
        p, {"a": np.array([7], dtype=np.int32)}, slice(0, 1), mmap=True
    ),
    "update_rows_cfitsio": lambda p: torchfits.table.update_rows(
        p, {"a": np.array([7], dtype=np.int32)}, slice(0, 1), mmap=False
    ),
    "update_rows_str": lambda p: torchfits.table.update_rows(
        p, {"s": ["w"]}, slice(0, 1)
    ),
    "delete_rows": lambda p: torchfits.table.delete_rows(p, 0),
    "rename_columns": lambda p: torchfits.table.rename_columns(p, {"a": "b"}),
    "drop_columns": lambda p: torchfits.table.drop_columns(p, ["s"]),
    "insert_column": lambda p: torchfits.table.insert_column(
        p, "z", np.arange(3, dtype=np.int16)
    ),
    "replace_column": lambda p: torchfits.table.replace_column(
        p, "a", np.arange(3, dtype=np.int32)
    ),
}


@pytest.mark.parametrize("op", sorted(OPS))
def test_mutation_restamps_checksums(tmp_path, op):
    path = str(tmp_path / "sum.fits")
    torchfits.table.write(
        path,
        {"a": np.arange(3, dtype=np.int32), "s": ["x", "y", "z"]},
        overwrite=True,
    )
    torchfits.write_checksums(path, hdu=1)
    assert torchfits.verify_checksums(path, hdu=1)["status"] == "ok"
    OPS[op](path)
    result = torchfits.verify_checksums(path, hdu=1)
    assert result["present"] and result["status"] == "ok"


def test_unstamped_hdu_gains_no_checksum(tmp_path):
    path = str(tmp_path / "nosum.fits")
    torchfits.table.write(path, {"a": np.arange(3, dtype=np.int32)}, overwrite=True)
    torchfits.table.append_rows(path, {"a": [9]})
    assert not torchfits.verify_checksums(path, hdu=1)["present"]
