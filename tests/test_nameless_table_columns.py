"""A nameless table column must look the same to schema and to the readers.

FITS permits a column with no ``TTYPE`` card.  The readers synthesize
``COL<i>``; the Python schema used to drop the column entirely (shifting every
later index) and, after a first fix, named it ``""``.  Either way ``schema``
disagreed with ``read``/``read_torch``, so nothing could address the column.
"""

import pytest

import torchfits
from torchfits._io_engine._write_helpers import _delete_header_key_if_supported


@pytest.fixture()
def nameless_second_column(tmp_path):
    path = str(tmp_path / "nameless.fits")
    torchfits.table.write(path, {"x": [0, 1, 2], "y": [10, 20, 30]})
    # Same route as `torchfits setkey --delete`: CFITSIO card delete, so the
    # file really has no TTYPE2 (writing a mutated Header re-adds the card).
    _delete_header_key_if_supported(path, 1, "TTYPE2")
    return path


def test_schema_and_readers_agree_on_nameless_column(nameless_second_column):
    path = nameless_second_column

    with torchfits.open(path) as hdul:
        names = [c["name"] for c in hdul[1].schema["columns"]]
    assert names == ["x", "COL2"]

    arrow = torchfits.table.read(path, hdu=1)
    assert arrow.column_names == names
    assert arrow.to_pydict() == {"x": [0, 1, 2], "COL2": [10, 20, 30]}

    tensors = torchfits.table.read_torch(path, hdu=1)
    assert list(tensors.keys()) == names
    assert tensors["COL2"].tolist() == [10, 20, 30]


def test_nameless_column_is_not_duplicated_on_sliced_or_filtered_reads(
    nameless_second_column,
):
    """One nameless column must stay one column once slices or filters are involved.

    The filtered and sliced paths build their field list from the header walk
    plus (historically) a separate unnamed-column pass. Once the walk stopped
    dropping nameless columns, the two sources listed it twice and an empty or
    filtered read grew a duplicate ``COL2``.
    """
    path = nameless_second_column

    empty = torchfits.table.read(path, hdu=1, row_slice=(0, 0))
    assert empty.column_names == ["x", "COL2"]
    assert empty.num_rows == 0

    filtered = torchfits.table.read(path, hdu=1, where="x > 0")
    assert filtered.column_names == ["x", "COL2"]
    assert filtered.to_pydict() == {"x": [1, 2], "COL2": [20, 30]}

    filtered_torch = torchfits.table.read_torch(path, hdu=1, where="x > 0")
    assert list(filtered_torch.keys()) == ["x", "COL2"]
    assert filtered_torch["COL2"].tolist() == [20, 30]


def _ttypes(path):
    header = torchfits.read_header(path, 1)
    return {k: v for k, v in header.items() if str(k).startswith("TTYPE")}


def _columns(path):
    return torchfits.table.read(path, hdu=1).to_pydict()


def test_insert_column_keeps_nameless_column(nameless_second_column):
    """insert_column used to rewrite the table without the TTYPE-less column."""
    path = nameless_second_column
    torchfits.table.insert_column(path, "z", [7, 8, 9])
    assert _columns(path) == {"x": [0, 1, 2], "COL2": [10, 20, 30], "z": [7, 8, 9]}
    # Still nameless: the rewrite does not invent a TTYPE2 card.
    assert _ttypes(path) == {"TTYPE1": "x", "TTYPE3": "z"}


def test_insert_column_before_nameless_column_keeps_its_name(
    nameless_second_column,
):
    """A nameless column that moves keeps the name reads gave it."""
    path = nameless_second_column
    torchfits.table.insert_column(path, "z", [7, 8, 9], index=0)
    assert _columns(path) == {"z": [7, 8, 9], "x": [0, 1, 2], "COL2": [10, 20, 30]}
    assert _ttypes(path) == {"TTYPE1": "z", "TTYPE2": "x", "TTYPE3": "COL2"}


def test_replace_column_keeps_nameless_column(nameless_second_column):
    path = nameless_second_column
    torchfits.table.replace_column(path, "x", [4, 5, 6])
    assert _columns(path) == {"x": [4, 5, 6], "COL2": [10, 20, 30]}
    assert _ttypes(path) == {"TTYPE1": "x"}

    torchfits.table.replace_column(path, "COL2", [11, 21, 31])
    assert _columns(path) == {"x": [4, 5, 6], "COL2": [11, 21, 31]}
    assert _ttypes(path) == {"TTYPE1": "x"}


def test_row_mutations_accept_nameless_column_name(nameless_second_column):
    """append/insert/update_rows took 'x' but refused the read-side 'COL2'."""
    path = nameless_second_column
    torchfits.table.append_rows(path, {"x": [3], "COL2": [40]})
    assert _columns(path) == {"x": [0, 1, 2, 3], "COL2": [10, 20, 30, 40]}

    torchfits.table.insert_rows(path, {"x": [-1], "COL2": [-10]}, row=0)
    assert _columns(path) == {"x": [-1, 0, 1, 2, 3], "COL2": [-10, 10, 20, 30, 40]}

    for i, mmap in enumerate((True, False, "auto")):
        torchfits.table.update_rows(path, {"COL2": [100 + i]}, (0, 1), mmap=mmap)
        assert _columns(path)["COL2"][0] == 100 + i
    assert _ttypes(path) == {"TTYPE1": "x"}


def test_rename_and_drop_accept_nameless_column_name(nameless_second_column):
    path = nameless_second_column
    torchfits.table.rename_columns(path, {"COL2": "y"})
    assert _columns(path) == {"x": [0, 1, 2], "y": [10, 20, 30]}

    _delete_header_key_if_supported(path, 1, "TTYPE2")
    torchfits.table.drop_columns(path, ["COL2"])
    assert _columns(path) == {"x": [0, 1, 2]}


def test_nameless_fallback_does_not_shadow_a_named_column(tmp_path):
    """'COL2' only means column 2 when column 2 has no name of its own."""
    path = str(tmp_path / "named.fits")
    torchfits.table.write(path, {"x": [0, 1], "y": [10, 20]})
    with pytest.raises(KeyError):
        torchfits.table.append_rows(path, {"x": [3], "COL2": [40]})
    with pytest.raises(KeyError):
        torchfits.table.drop_columns(path, ["COL2"])
    assert _columns(path) == {"x": [0, 1], "y": [10, 20]}


def test_native_fallback_only_for_nameless_columns(tmp_path):
    """The native COL<i> fallback must not resolve a column that has a name."""
    import torchfits._C as cpp

    path = str(tmp_path / "named.fits")
    torchfits.table.write(path, {"x": [0, 1], "y": [10, 20]})
    with pytest.raises(RuntimeError, match="Column not found"):
        cpp.drop_fits_table_columns(path, 1, ["COL2"])
    assert _columns(path) == {"x": [0, 1], "y": [10, 20]}
