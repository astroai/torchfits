"""``convert --columns/--where`` streams non-FITS output."""

from __future__ import annotations

import numpy as np
import pytest

import torchfits
from torchfits.cli.main import main

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture
def catalog(tmp_path):
    src = tmp_path / "cat.fits"
    torchfits.table.write(
        str(src),
        {
            "RA": np.arange(10, dtype=np.float64),
            "MAG": np.arange(10, dtype=np.float32) + 15,
            "NAME": [f"s{i}" for i in range(10)],
        },
        overwrite=True,
    )
    return src


@pytest.fixture
def no_materialize(monkeypatch):
    """Fail any full-table read: the streaming path must use table.reader."""
    from torchfits._table import interop

    def boom(*args, **kwargs):
        raise AssertionError("convert materialized the table with table.read")

    monkeypatch.setattr(interop, "read", boom)
    monkeypatch.setattr(torchfits.table, "read", boom)


@pytest.mark.parametrize("fmt", ["parquet", "csv", "tsv", "arrow"])
def test_convert_columns_where_streams(catalog, tmp_path, fmt, no_materialize):
    out = tmp_path / f"out.{fmt}"
    code = main(
        [
            "convert",
            str(catalog),
            str(out),
            "--to",
            fmt,
            "-c",
            "MAG,RA",
            "-w",
            "MAG >= 22",
        ]
    )
    assert code == 0
    if fmt == "parquet":
        got = pq.read_table(out)
    elif fmt == "arrow":
        import pyarrow.ipc as ipc

        got = ipc.open_file(str(out)).read_all()
    else:
        import pyarrow.csv as pacsv

        got = pacsv.read_csv(
            str(out),
            parse_options=pacsv.ParseOptions(delimiter="\t" if fmt == "tsv" else ","),
        )
    assert got.column_names == ["MAG", "RA"]
    assert got.column("RA").to_pylist() == [7.0, 8.0, 9.0]


@pytest.mark.parametrize(
    "extra",
    [
        ["-c", "NOPE"],
        ["-w", "NOPE > 1"],
        ["-w", "RA >"],
        ["-w", "NAME > 3"],
    ],
)
@pytest.mark.parametrize("fmt", ["parquet", "csv", "arrow"])
def test_convert_bad_selection_is_usage_error_before_output(
    catalog, tmp_path, fmt, extra
):
    out = tmp_path / f"out.{fmt}"
    out.write_bytes(b"keep")
    assert main(["convert", str(catalog), str(out), "--to", fmt, *extra]) == 2
    assert out.read_bytes() == b"keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cat.fits", out.name]


def test_convert_missing_hdu_with_where_is_usage_error(catalog, tmp_path):
    # An -e index the file does not have is a usage error in every
    # subcommand (docs/cli.md exit codes); a corrupt HDU stays exit 3.
    out = tmp_path / "out.parquet"
    assert main(["convert", str(catalog), str(out), "-e", "7", "-w", "RA > 1"]) == 2
