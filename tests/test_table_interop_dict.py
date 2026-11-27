"""``write_parquet`` / ``write_csv`` / ``write_ipc`` accept a column dict (r7c-23)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits

pq = pytest.importorskip("pyarrow.parquet")


@pytest.mark.parametrize("writer", ["write_parquet", "write_csv", "write_ipc"])
def test_interop_writers_accept_column_dict(tmp_path, writer):
    dest = tmp_path / "out.bin"
    data = {
        "a": torch.tensor([1, 2, 3], dtype=torch.int32),
        "b": np.array([1.5, 2.5, 3.5]),
        "c": ["x", "y", "z"],
    }
    getattr(torchfits.table, writer)(str(dest), data)
    if writer == "write_parquet":
        got = pq.read_table(dest)
    elif writer == "write_ipc":
        import pyarrow.ipc as ipc

        got = ipc.open_file(str(dest)).read_all()
    else:
        import pyarrow.csv as pacsv

        got = pacsv.read_csv(str(dest))
    assert got.column("a").to_pylist() == [1, 2, 3]
    assert got.column("b").to_pylist() == [1.5, 2.5, 3.5]
    assert got.column("c").to_pylist() == ["x", "y", "z"]


def test_interop_writer_dict_rejects_read_kwargs(tmp_path):
    with pytest.raises(TypeError, match="require a file path"):
        torchfits.table.write_parquet(str(tmp_path / "x.parquet"), {"a": [1]}, hdu=1)
