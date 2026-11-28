"""``torchfits convert`` to CSV/TSV refuses array columns as a usage error.

Arrow's CSV writer cannot express a list cell: a table with a vector or VLA
column failed mid-write with "Unsupported Type" and exit 3. It is now a
usage error (exit 2) naming the columns, raised before the output exists.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import torchfits
from torchfits.cli.main import main


@pytest.fixture
def vector_table(tmp_path):
    path = str(tmp_path / "vec.fits")
    torchfits.table.write(
        path,
        {
            "A": np.arange(3, dtype=np.int32),
            "V": np.ones((3, 4), np.float32),
            "L": [np.arange(n, dtype=np.int16) for n in (1, 2, 3)],
        },
    )
    return path


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_vector_columns_are_a_usage_error(vector_table, tmp_path, capsys, suffix):
    out = str(tmp_path / f"out{suffix}")
    assert main(["convert", vector_table, out]) == 2
    err = capsys.readouterr().err
    assert "V, L" in err
    assert "--columns" in err and "parquet" in err
    assert not os.path.exists(out)


def test_scalar_selection_still_converts(vector_table, tmp_path):
    out = str(tmp_path / "out.csv")
    assert main(["convert", vector_table, out, "--columns", "A"]) == 0
    assert open(out).read().split() == ['"A"', "0", "1", "2"]


def test_parquet_keeps_vector_columns(vector_table, tmp_path):
    pytest.importorskip("pyarrow.parquet")
    out = str(tmp_path / "out.parquet")
    assert main(["convert", vector_table, out]) == 0
