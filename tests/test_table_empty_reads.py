"""Empty reads keep every column, typed as a non-empty read types it.

A 0-row table used to read back as ``{}`` and ``num_rows=0`` raised
``ValueError`` on most routes; the column set and dtypes must not depend on
how many rows the window selects.
"""

import numpy as np
import pytest
import torch

import torchfits
from torchfits import table
from torchfits.data import FitsTableDataset, FitsTableIterableDataset

SCHEMA = {"SC": {"bscale": 0.5, "bzero": 1.0}}


def _columns(n):
    return {
        "F": np.arange(n, dtype=np.float32),
        "I8": np.arange(n, dtype=np.int8),
        "U16": np.arange(n, dtype=np.uint16),
        "U32": np.arange(n, dtype=np.uint32),
        "U64": np.arange(n, dtype=np.uint64),
        "B": np.ones(n, dtype=bool),
        "S": np.array(["ab"] * n, dtype="S5"),
        "V": np.arange(3 * n, dtype=np.int32).reshape(n, 3),
        "SC": np.arange(n, dtype=np.int16),
    }


def _layout(data):
    return {
        k: (v.dtype, tuple(v.shape[1:])) if isinstance(v, torch.Tensor) else type(v)
        for k, v in data.items()
    }


@pytest.fixture
def fixed(tmp_path):
    full = str(tmp_path / "full.fits")
    empty = str(tmp_path / "empty.fits")
    table.write(full, _columns(4), schema=SCHEMA)
    table.write(empty, _columns(0), schema=SCHEMA)
    return full, empty


@pytest.fixture
def vla(tmp_path):
    full = str(tmp_path / "vla.fits")
    empty = str(tmp_path / "vla0.fits")
    data = {
        "X": np.arange(4.0),
        "VL": [np.arange(i, dtype=np.float32) for i in range(4)],
    }
    table.write(full, data)
    table.write(empty, data)
    table.delete_rows(empty, (0, 4))
    assert torchfits.read_nrows(empty, 1) == 0
    return full, empty


def test_fixed_width_empty_reads_are_typed(fixed):
    full, empty = fixed
    expected = _layout(table.read_torch(full, hdu=1))
    assert expected["SC"][0] == torch.float64
    assert expected["U64"][0] == torch.uint64
    assert expected["S"] == (torch.uint8, (5,))
    routes = [
        ("read_torch", table.read_torch(empty, hdu=1)),
        ("read_torch nommap", table.read_torch(empty, hdu=1, mmap=False)),
        ("read", torchfits.read(empty, hdu=1)),
        ("read nommap", torchfits.read(empty, hdu=1, mmap=False)),
        ("read table", torchfits.read(empty, hdu=1, mode="table")),
        ("n0", table.read_torch(full, hdu=1, start_row=2, num_rows=0)),
        ("n0 nommap", table.read_torch(full, hdu=1, num_rows=0, mmap=False)),
        ("n0 read", torchfits.read(full, hdu=1, num_rows=0)),
        ("n0 where", table.read_torch(full, hdu=1, num_rows=0, where="F > 1")),
        ("where", table.read_torch(empty, hdu=1, where="F > 1")),
    ]
    for name, got in routes:
        assert _layout(got) == expected, name
        assert all(v.shape[0] == 0 for v in got.values()), name

    sub = table.read_torch(full, hdu=1, columns=["SC", "U16"], start_row=5, num_rows=0)
    assert _layout(sub) == {k: expected[k] for k in ("SC", "U16")}

    with torchfits.open(empty) as hdul:
        assert {k: hdul[1][k].dtype for k in expected} == {
            k: v[0] for k, v in expected.items()
        }


def test_vla_empty_reads_keep_the_list_column(vla):
    full, empty = vla
    for name, got in [
        ("read_torch", table.read_torch(empty, hdu=1)),
        ("read", torchfits.read(empty, hdu=1)),
        ("n0", table.read_torch(full, hdu=1, start_row=2, num_rows=0)),
        ("n0 read", torchfits.read(full, hdu=1, num_rows=0)),
        ("n0 read table", torchfits.read(full, hdu=1, mode="table", num_rows=0)),
    ]:
        assert list(got) == ["X", "VL"], name
        assert got["X"].dtype == torch.float64 and got["X"].shape == (0,), name
        assert got["VL"] == [], name


@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_arrow_empty_window_keeps_typed_columns(fixed, vla, backend):
    for full, empty in (fixed, vla):
        expected = table.read(full, hdu=1, backend=backend).schema
        for got in (
            table.read(empty, hdu=1, backend=backend),
            table.read(full, hdu=1, row_slice=(2, 2), backend=backend),
        ):
            assert got.num_rows == 0
            assert got.schema.names == expected.names
            assert got.schema.types == expected.types


def test_datasets_over_an_empty_table(fixed):
    _full, empty = fixed
    assert len(FitsTableDataset(empty, hdu=1)) == 0
    assert list(FitsTableIterableDataset(empty, hdu=1)) == []


def test_negative_num_rows_other_than_minus_one_still_raises(fixed):
    full, _ = fixed
    with pytest.raises(ValueError, match="num_rows"):
        table.read_torch(full, hdu=1, num_rows=-2)
    with pytest.raises(ValueError, match="num_rows"):
        torchfits.read(full, hdu=1, num_rows=-2)
