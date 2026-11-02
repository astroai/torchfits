"""Round-trip a small FITS table through every torchfits.table interop exporter.

One table with int (with a TNULL row), float (with NaN), string, bool and
unsigned 16-bit columns; each exporter must keep values, nulls and dtypes.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from torchfits import table

pa = pytest.importorskip("pyarrow")

EXPECTED = {
    "I": [1, None, 3],
    "F": [0.5, math.nan, 2.5],
    "S": ["ab", "c", ""],
    "B": [True, False, True],
    "U": [0, 40000, 65535],
}


def _same(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        if isinstance(w, float) and math.isnan(w):
            assert g is not None and math.isnan(g)
        else:
            assert g == w


def _check_columns(columns: dict) -> None:
    for name, want in EXPECTED.items():
        _same(list(columns[name]), want)


@pytest.fixture
def fits_table(tmp_path):
    path = str(tmp_path / "t.fits")
    table.write(
        path,
        {
            "I": np.array([1, -999, 3], dtype=np.int32),
            "F": np.array([0.5, np.nan, 2.5]),
            "S": ["ab", "c", ""],
            "B": np.array([True, False, True]),
            "U": np.array([0, 40000, 65535], dtype=np.uint16),
        },
        schema={"I": {"tnull": -999}},
        overwrite=True,
    )
    return path


def _arrow_types(schema) -> dict:
    return {f.name: str(f.type) for f in schema}


ARROW_TYPES = {"I": "int32", "F": "double", "S": "string", "B": "bool", "U": "uint16"}


@pytest.mark.parametrize("stream", [False, True])
def test_write_parquet_round_trips(fits_table, tmp_path, stream):
    pq = pytest.importorskip("pyarrow.parquet")
    dest = str(tmp_path / "t.parquet")
    table.write_parquet(dest, fits_table, stream=stream)
    got = pq.read_table(dest)
    assert _arrow_types(got.schema) == ARROW_TYPES
    _check_columns(got.to_pydict())


@pytest.mark.parametrize("stream", [False, True])
def test_write_ipc_round_trips(fits_table, tmp_path, stream):
    ipc = pytest.importorskip("pyarrow.ipc")
    dest = str(tmp_path / "t.arrow")
    with warnings.catch_warnings():
        # pyarrow >= 24 deprecates write_feather, which the non-stream path uses.
        warnings.simplefilter("ignore", FutureWarning)
        table.write_ipc(dest, fits_table, stream=stream)
    got = ipc.open_file(dest).read_all()
    assert _arrow_types(got.schema) == ARROW_TYPES
    _check_columns(got.to_pydict())


@pytest.mark.parametrize("stream", [False, True])
def test_write_csv_round_trips(fits_table, tmp_path, stream):
    dest = str(tmp_path / "t.csv")
    table.write_csv(dest, fits_table, stream=stream)
    lines = open(dest).read().splitlines()
    assert lines == [
        '"I","F","S","B","U"',
        '1,0.5,"ab",true,0',
        ',nan,"c",false,40000',
        '3,2.5,"",true,65535',
    ]


def test_write_csv_tsv_from_arrow_table(fits_table, tmp_path):
    dest = str(tmp_path / "t.tsv")
    table.write_csv(dest, table.read(fits_table, columns=["I", "U"]), delimiter="\t")
    assert open(dest).read().splitlines()[1] == "1\t0"


def test_detached_input_rejects_io_kwargs(fits_table, tmp_path):
    arrow = table.read(fits_table)
    with pytest.raises(TypeError, match="require a file path"):
        table.write_parquet(str(tmp_path / "x.parquet"), arrow, hdu=1)
    with pytest.raises(TypeError, match="require a file path"):
        table.to_pandas(arrow, hdu=1)


def test_to_pandas_keeps_values_and_dtypes(fits_table):
    df = table.to_pandas(fits_table)
    assert {k: str(v) for k, v in df.dtypes.items() if k != "S"} == {
        "I": "float64",  # pandas has no nullable int by default: NULL -> NaN
        "F": "float64",
        "B": "bool",
        "U": "uint16",
    }
    assert df["I"].isna().tolist() == [False, True, False]
    assert df["S"].tolist() == EXPECTED["S"]
    assert df["U"].tolist() == EXPECTED["U"]

    frames = list(table.to_pandas(fits_table, stream=True, batch_size=2))
    assert [len(f) for f in frames] == [2, 1]
    from_arrow = table.to_pandas(table.read(fits_table))
    assert from_arrow["U"].tolist() == EXPECTED["U"]


def test_to_polars_read_polars_and_scan_polars(fits_table):
    pytest.importorskip("polars")
    df = table.to_polars(fits_table)
    assert {k: str(v) for k, v in df.schema.items()} == {
        "I": "Int32",
        "F": "Float64",
        "S": "String",
        "B": "Boolean",
        "U": "UInt16",
    }
    _check_columns(df.to_dict(as_series=False))

    wrapped = table.read_polars(fits_table)
    assert wrapped.height == 3 and len(wrapped) == 3
    assert wrapped.field_meta["I"]["fits_tnull"] == "-999"
    assert wrapped.field_meta["U"]["fits_tzero"] == "32768.0"
    assert wrapped.table_meta["fits_hdu"] == "1"
    _check_columns(wrapped.frame.to_dict(as_series=False))

    batches = list(table.scan_polars(fits_table, batch_size=2))
    assert [b.height for b in batches] == [2, 1]
    assert batches[1]["U"].to_list() == [65535]

    streamed = list(table.to_polars(table.read(fits_table), stream=True))
    assert sum(b.height for b in streamed) == 3


def test_to_duckdb_and_duckdb_query(fits_table):
    duckdb = pytest.importorskip("duckdb")
    rel = table.to_duckdb(fits_table, relation_name="cat")
    rows = rel.order("U").fetchall()
    assert [r[4] for r in rows] == EXPECTED["U"]
    assert rows[1][0] is None and math.isnan(rows[1][1])

    con = duckdb.connect()
    result = table.duckdb_query(
        fits_table,
        "SELECT SUM(U) AS s, COUNT(I) AS n FROM cat",
        relation_name="cat",
        connection=con,
    )
    if hasattr(result, "read_all"):
        result = result.read_all()
    assert result.to_pydict() == {"s": [105535], "n": [2]}

    with pytest.raises(ValueError, match="exactly one SQL statement"):
        table.duckdb_query(fits_table, "SELECT 1; SELECT 2")
    with pytest.raises(ValueError, match="SELECT or EXPLAIN"):
        table.duckdb_query(fits_table, "DROP TABLE fits_table")
    with pytest.raises(ValueError, match="relation_name"):
        table.to_duckdb(fits_table, relation_name="")
