"""N-D table cells: TDIM on read (``shape_cells=True``) and on write.

``TDIM = '(a,b)'`` makes a ``TFORM = '<a*b>X'`` cell an ``(b, a)`` array in C
order. Writers store ``(N, d1, d2)`` values flat with that TDIM; readers
return them flat by default and shaped with ``shape_cells=True``.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import torch

import torchfits

T = torchfits.table
CUBE = np.arange(24, dtype=np.float32).reshape(4, 2, 3)


def _cards(path):
    hdr = torchfits.read_header(str(path), 1)
    return {
        k: v for k, v in hdr.items() if k[:5] in ("TFORM", "TDIM1", "TDIM2", "TDIM3")
    }


@pytest.fixture
def cells(tmp_path):
    path = tmp_path / "cells.fits"
    T.write(
        str(path),
        {
            "img": CUBE,
            "id": np.arange(4, dtype=np.int64),
            "k": torch.arange(16, dtype=torch.int32).reshape(4, 2, 2),
        },
        overwrite=True,
    )
    return str(path)


def test_write_nd_cells_emits_tform_and_tdim(cells):
    cards = _cards(cells)
    assert cards["TFORM1"] == "6E" and cards["TDIM1"] == "(3,2)"
    assert cards["TFORM3"] == "4J" and cards["TDIM3"] == "(2,2)"
    assert "TDIM2" not in cards


def test_default_reads_stay_flat(cells):
    assert T.read_torch(cells)["img"].shape == (4, 6)
    assert T.read(cells).schema.field("img").type == pa.list_(pa.float32(), 6)


def test_read_torch_shape_cells(cells):
    got = T.read_torch(cells, shape_cells=True)
    assert torch.equal(got["img"], torch.from_numpy(CUBE))
    assert got["k"].shape == (4, 2, 2)
    assert got["id"].shape == (4,)
    data, _hdr = T.read_torch(cells, shape_cells=True, return_header=True)
    assert data["img"].shape == (4, 2, 3)


@pytest.mark.parametrize("kwargs", [{}, {"mmap": False}, {"where": "id > 1"}])
def test_read_shape_cells_nests_fixed_size_lists(cells, kwargs):
    got = T.read(cells, shape_cells=True, **kwargs)
    assert got.schema.field("img").type == pa.list_(pa.list_(pa.float32(), 3), 2)
    expected = CUBE[2:] if "where" in kwargs else CUBE
    assert np.array(got["img"].to_pylist()).tolist() == expected.tolist()


def test_streaming_and_schema_readers_agree(cells):
    nested = pa.list_(pa.list_(pa.float32(), 3), 2)
    assert {
        b.schema.field("img").type
        for b in T.scan(cells, shape_cells=True, batch_size=3)
    } == {nested}
    assert [
        b["img"].shape for b in T.scan_torch(cells, shape_cells=True, batch_size=3)
    ] == [
        (3, 2, 3),
        (1, 2, 3),
    ]
    assert T.schema(cells, shape_cells=True).field("img").type == nested
    empty = T.reader(cells, shape_cells=True, where="id > 100")
    assert empty.schema.field("img").type == nested
    assert T.to_polars(cells, shape_cells=True).schema["img"].shape == (2, 3)


@pytest.mark.parametrize("metadata", [False, True])
def test_fits_arrow_fits_round_trip_keeps_tdim(cells, tmp_path, metadata):
    # Shaped cells carry TDIM in their nesting; flat cells read with
    # include_fits_metadata=True carry it as fits_tdim field metadata.
    table = (
        T.read(cells, include_fits_metadata=True)
        if metadata
        else T.read(cells, shape_cells=True)
    )
    out = tmp_path / "rt.fits"
    T.write(str(out), table, overwrite=True)
    cards = _cards(out)
    assert cards["TDIM1"] == "(3,2)" and cards["TDIM3"] == "(2,2)"
    assert torch.equal(
        T.read_torch(str(out), shape_cells=True)["img"], torch.from_numpy(CUBE)
    )


def test_write_fixed_shape_tensor_column(tmp_path):
    storage = pa.FixedSizeListArray.from_arrays(pa.array(CUBE.ravel()), 6)
    tensor = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(pa.float32(), (2, 3)), storage
    )
    out = tmp_path / "t.fits"
    T.write(str(out), pa.table({"img": tensor}), overwrite=True)
    assert _cards(out)["TDIM1"] == "(3,2)"
    assert torch.equal(
        T.read_torch(str(out), shape_cells=True)["img"], torch.from_numpy(CUBE)
    )


def test_mutations_take_nd_cells(tmp_path):
    path = str(tmp_path / "m.fits")
    T.write(path, {"id": np.arange(4)}, overwrite=True)
    T.insert_column(path, "img", CUBE)
    assert _cards(path)["TDIM2"] == "(3,2)"
    T.update_rows(path, {"img": -CUBE[:2]}, row_slice=(0, 2))
    T.append_rows(path, {"img": CUBE[:1], "id": np.array([9])})
    got = T.read_torch(path, shape_cells=True)["img"]
    assert got.shape == (5, 2, 3)
    assert got[0].tolist() == (-CUBE[0]).tolist()
    assert got[4].tolist() == CUBE[0].tolist()
    T.replace_column(path, "img", np.zeros((5, 3, 2), np.float32))
    assert _cards(path)["TDIM2"] == "(2,3)"
    with pytest.raises(ValueError, match="does not describe"):
        T.insert_column(path, "bad", np.zeros((5, 2, 3)), dim="(2,3)")


def test_streaming_writer_takes_nd_cells(tmp_path):
    path = str(tmp_path / "s.fits")
    with T.writer(path, overwrite=True) as w:
        w.write({"img": CUBE[:2]})
        w.write({"img": CUBE[2:]})
    assert _cards(path)["TDIM1"] == "(3,2)"
    assert torch.equal(
        T.read_torch(path, shape_cells=True)["img"], torch.from_numpy(CUBE)
    )
    with pytest.raises(ValueError, match="cell shape"):
        with T.writer(str(tmp_path / "s2.fits"), overwrite=True) as w:
            w.write({"img": CUBE[:2]})
            w.write({"img": np.zeros((1, 3, 2), np.float32)})


def test_tnull_and_string_tdim_columns(tmp_path):
    path = str(tmp_path / "n.fits")
    v = np.array([[1, -99, 3, 4, 5, 6], [7, 8, 9, 10, 11, -99]], dtype=np.int32)
    T.write(
        path,
        {"s": ["ab   cd", "efghijklmn"], "v": v},
        schema={
            "s": {"format": "10A", "dim": "(5,2)"},
            "v": {"format": "6J", "dim": "(3,2)", "tnull": -99},
        },
        overwrite=True,
    )
    got = T.read(path, shape_cells=True)
    # A string column's TDIM is kept as metadata; the column stays one string.
    assert got["s"].to_pylist() == ["ab   cd", "efghijklmn"]
    assert got["v"].to_pylist() == [
        [[1, None, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, None]],
    ]
    assert T.read_torch(path, shape_cells=True)["v"].shape == (2, 2, 3)
    meta = (
        T.read(path, include_fits_metadata=True, shape_cells=True)
        .schema.field("s")
        .metadata
    )
    assert meta[b"fits_tdim"] == b"(5,2)"


def test_tdim_not_describing_the_whole_cell_stays_flat(tmp_path):
    path = str(tmp_path / "p.fits")
    T.write(
        path,
        {"v": np.zeros((2, 6), np.float32)},
        schema={"v": {"format": "6E", "dim": "(2,2)"}},
        overwrite=True,
    )
    assert T.read_torch(path, shape_cells=True)["v"].shape == (2, 6)


def test_to_astropy_shape_cells(cells):
    pytest.importorskip("astropy")
    assert T.to_astropy(cells, shape_cells=True)["img"].shape == (4, 2, 3)
