"""String columns with sub-strings (TFORM rAw, e.g. 20A5) must keep one value per row.

CFITSIO counts TSTRING elements in w-character sub-strings for rAw columns.
Writing one element per row put each row's first w characters into the first
row's sub-strings and left the other rows empty, on write, update and append.
"""

import numpy as np

import torchfits._C as C
from torchfits._table.mutation import append_rows, update_rows
from torchfits._table.write import write as write_table

ROWS = ["aaaaabbbbbcccccddddd", "eeeeefffffggggghhhhh", "iiiiijjjjjkkkkklllll"]


def _strings(path):
    return [
        bytes(r).decode().rstrip("\x00 ")
        for r in C.read_fits_table(path, 1)["S"].tolist()
    ]


def test_substring_column_write_update_append(tmp_path):
    path = str(tmp_path / "sub.fits")
    write_table(
        path,
        {"S": list(ROWS), "N": np.arange(3, dtype=np.int32)},
        schema={"S": {"format": "20A5"}},
        overwrite=True,
    )
    assert _strings(path) == ROWS

    update_rows(
        path, {"S": ["XXXXXYYYYYZZZZZWWWWW", "PPPPPQQQQQ"]}, slice(1, 3), mmap=False
    )
    assert _strings(path) == [ROWS[0], "XXXXXYYYYYZZZZZWWWWW", "PPPPPQQQQQ"]

    append_rows(
        path, {"S": ["QQQQQRRRRR", "short"], "N": np.array([3, 4], dtype=np.int32)}
    )
    assert _strings(path)[3:] == ["QQQQQRRRRR", "short"]
    assert C.read_fits_table(path, 1)["N"].tolist() == [0, 1, 2, 3, 4]
