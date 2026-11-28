"""A repeated TTYPE must not make a column disappear from reads.

With TTYPE1 = TTYPE2 = 'AA', the header-only schema() listed three columns
while read() / schema(where=...) returned two (the second 'AA' was dropped by
the name-keyed readers), and schema(columns=['AA']) described the second
column while read(columns=['AA']) returned the first. Every reader now names
a column whose TTYPE repeats an earlier one COL<i>, like a column with no
TTYPE, so all columns are readable and addressable, and 'AA' means the first.
"""

import numpy as np
import pytest

import torchfits
from torchfits._io_engine._write_helpers import _write_header_cards_if_supported

T = torchfits.table


@pytest.fixture()
def dup_table(tmp_path):
    path = str(tmp_path / "dup.fits")
    T.write(
        path,
        {
            "AA": np.array([1, 2, 3], dtype=np.int32),
            "BB": np.array([1.5, 2.5, 3.5]),
            "S": np.array(["a", "b", "c"]),
        },
    )
    _write_header_cards_if_supported(path, 1, {"TTYPE2": "AA"})
    assert torchfits.read_header(path, 1)["TTYPE2"] == "AA"
    return path


def _fields(schema):
    return [(f.name, str(f.type)) for f in schema]


EXPECTED = [("AA", "int32"), ("COL2", "double"), ("S", "string")]


def test_every_read_path_keeps_the_repeated_column(dup_table):
    p = dup_table
    assert _fields(T.schema(p)) == EXPECTED
    assert _fields(T.schema(p, where="S == 'a'")) == EXPECTED
    assert _fields(T.schema(p, where="S == 'zz'")) == EXPECTED
    assert _fields(T.read(p).schema) == EXPECTED
    assert _fields(T.read(p, where="S == 'zz'").schema) == EXPECTED
    assert T.read(p).to_pydict() == {
        "AA": [1, 2, 3],
        "COL2": [1.5, 2.5, 3.5],
        "S": ["a", "b", "c"],
    }
    assert T.read(p, where="COL2 > 2").to_pydict()["AA"] == [2, 3]
    assert [b.to_pydict() for b in T.scan(p)][0]["COL2"] == [1.5, 2.5, 3.5]
    assert list(T.read_torch(p)) == ["AA", "COL2", "S"]
    assert torchfits.read_colnames(p, 1) == ["AA", "COL2", "S"]
    with torchfits.open(p) as hdul:
        assert hdul[1].columns == ["AA", "COL2", "S"]


def test_name_selection_agrees_across_paths(dup_table):
    p = dup_table
    assert _fields(T.schema(p, columns=["AA"])) == [("AA", "int32")]
    assert _fields(T.schema(p, columns=["AA"], where="S == 'a'")) == [("AA", "int32")]
    assert _fields(T.read(p, columns=["AA"]).schema) == [("AA", "int32")]
    assert T.read(p, columns=["COL2"]).to_pydict() == {"COL2": [1.5, 2.5, 3.5]}


def test_mutations_address_the_repeated_column(dup_table):
    p = dup_table
    T.update_rows(p, {"COL2": [9.5]}, (0, 1), mmap=False)
    T.update_rows(p, {"AA": [7]}, (0, 1))
    T.append_rows(p, {"AA": [4], "COL2": [4.5], "S": ["d"]})
    assert T.read(p).to_pydict() == {
        "AA": [7, 2, 3, 4],
        "COL2": [9.5, 2.5, 3.5, 4.5],
        "S": ["a", "b", "c", "d"],
    }
    # A rewrite keeps the file's TTYPE2 card.
    T.insert_column(p, "z", [0, 0, 0, 0])
    header = torchfits.read_header(p, 1)
    assert (header["TTYPE1"], header["TTYPE2"]) == ("AA", "AA")
    assert T.read(p).column_names == ["AA", "COL2", "S", "z"]
