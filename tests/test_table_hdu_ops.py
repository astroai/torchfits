"""TableHDU (in memory) and TableHDURef (file-backed) row/column operations.

Covers the in-memory editors (add_column, append_rows, head, iter_rows, VLA
lengths), the lazy reference's file mutators (*_file), and the VLA / complex
coercion they route through.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits import TableHDU, table

pytest.importorskip("pyarrow")

VLA_LENGTHS = [1, 0, 3, 2, 4]


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "r.fits")
    table.write(
        p,
        {
            "ID": np.arange(5, dtype=np.int32),
            "X": np.arange(5, dtype=np.float32) * 2,
            "V": [torch.arange(n, dtype=torch.int32) for n in VLA_LENGTHS],
            "Z": np.arange(5).astype(np.complex64) * (1 + 1j),
        },
        overwrite=True,
    )
    return p


def _ref(path):
    with torchfits.open(path) as hdul:
        return hdul[1]


def _ids(path):
    return table.read(path, columns=["ID"]).column("ID").to_pylist()


# -- in-memory TableHDU ---------------------------------------------------------


def test_table_hdu_head_iter_rows_and_vla_lengths(path):
    t = _ref(path).materialize()
    assert isinstance(t, TableHDU)
    assert t.vla_lengths == {"V": VLA_LENGTHS}
    assert t.get_vla_lengths("V") == VLA_LENGTHS
    with pytest.raises(KeyError, match="not a VLA"):
        t.get_vla_lengths("ID")

    assert t.head(2)["ID"].tolist() == [0, 1]
    assert t.head(2).head(9).num_rows == 2
    with pytest.raises(ValueError, match="n >= 0"):
        t.head(-1)

    batches = list(t.iter_rows(batch_size=3))
    assert [b["ID"].tolist() for b in batches] == [[0, 1, 2], [3, 4]]
    assert [len(v) for v in batches[1]["V"]] == [2, 4]
    with pytest.raises(ValueError, match="batch_size"):
        list(t.iter_rows(batch_size=0))


def test_table_hdu_add_column_and_append_rows(path):
    t = _ref(path).materialize()
    added = t.add_column("Y", torch.ones(5))
    assert added.columns[-1] == "Y" and "Y" not in t.columns
    with pytest.raises(KeyError, match="already exists"):
        t.add_column("ID", torch.zeros(5))
    assert t.add_column("ID", torch.zeros(5), overwrite=True)["ID"].sum() == 0
    with pytest.raises(ValueError, match="has 2 rows, expected 5"):
        t.add_column("W", torch.ones(2))

    rows = {
        "ID": torch.tensor([9], dtype=torch.int32),
        "X": torch.tensor([1.0]),
        "V": [torch.tensor([7, 7], dtype=torch.int32)],
        "Z": torch.tensor([2 - 1j], dtype=torch.complex64),
    }
    grown = t.append_rows(rows)
    assert grown.num_rows == 6
    assert grown["ID"].tolist() == [0, 1, 2, 3, 4, 9]
    assert grown.get_vla_lengths("V") == VLA_LENGTHS + [2]
    with pytest.raises(ValueError, match="exactly matching columns"):
        t.append_rows({"ID": torch.tensor([1], dtype=torch.int32)})
    with pytest.raises(ValueError, match="non-empty"):
        t.append_rows({})


# -- file-backed TableHDURef ----------------------------------------------------


def test_ref_reads_rows_and_arrow(path):
    ref = _ref(path)
    assert ref.num_rows == len(ref) == 5
    assert ref.head(2).num_rows == 2
    assert ref.vla_lengths == {"V": VLA_LENGTHS}
    batches = list(ref.select(["ID"]).iter_rows(batch_size=2))
    assert [b["ID"].tolist() for b in batches] == [[0, 1], [2, 3], [4]]
    arrow = ref.select(["ID", "V"]).head(3).to_arrow()
    assert arrow.column_names == ["ID", "V"]
    assert arrow.column("V").to_pylist() == [[0], [], [0, 1, 2]]
    with pytest.raises(TypeError, match="columns"):
        ref.to_arrow(columns=["ID"])


def test_ref_row_mutators(path):
    ref = _ref(path)
    ref = ref.insert_rows_file(
        {
            "ID": np.array([10, 11], dtype=np.int32),
            "X": np.array([1.5, 2.5], dtype=np.float32),
            "V": [[5], [6, 6, 6]],
            "Z": np.array([1j, -1j], dtype=np.complex64),
        },
        row=1,
    )
    assert ref.num_rows == 7
    assert _ids(path) == [0, 10, 11, 1, 2, 3, 4]
    assert ref.vla_lengths["V"] == [1, 1, 3, 0, 3, 2, 4]

    ref = ref.delete_rows_file(slice(1, 3))
    assert _ids(path) == [0, 1, 2, 3, 4]
    assert ref.num_rows == 5

    ref = ref.update_rows_file(
        {
            "X": np.array([-1.0, -2.0], dtype=np.float32),
            "V": [[8, 8], []],
            "Z": np.array([3 + 4j, 0], dtype=np.complex64),
        },
        (0, 2),
    )
    got = table.read(path, columns=["X", "V"]).to_pydict()
    assert got["X"][:3] == [-1.0, -2.0, 4.0]
    assert got["V"][:3] == [[8, 8], [], [0, 1, 2]]
    # Arrow has no complex type; the torch reader returns the column as-is.
    z = torchfits.read(path, hdu=1, mode="table")["Z"]
    assert z.dtype == torch.complex64
    assert z.reshape(-1)[:3].tolist() == [3 + 4j, 0j, 2 + 2j]


def test_ref_column_mutators(path):
    ref = _ref(path).rename_columns_file({"X": "XX"})
    assert ref.columns == ["ID", "XX", "V", "Z"]
    ref = ref.drop_columns_file(["Z", "V"])
    assert ref.columns == ["ID", "XX"]
    assert table.read(path).column_names == ["ID", "XX"]


def test_ref_append_rejects_complex_for_real_column(path):
    with pytest.raises(TypeError, match="complex dtype is not supported"):
        _ref(path).append_rows_file(
            {
                "ID": np.array([1j]),
                "X": np.array([1.0], dtype=np.float32),
                "V": [[1]],
                "Z": np.array([1j], dtype=np.complex64),
            }
        )


def test_ref_append_rejects_real_for_complex_column(path):
    with pytest.raises(TypeError, match="must be complex"):
        _ref(path).append_rows_file(
            {
                "ID": np.array([1], dtype=np.int32),
                "X": np.array([1.0], dtype=np.float32),
                "V": [[1]],
                "Z": torch.tensor([1.0]),
            }
        )
