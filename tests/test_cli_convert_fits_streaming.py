"""``torchfits convert --to fits`` streams fixed-width tables chunk by chunk.

The FITS -> FITS convert used to read the whole table (and its predicate
columns) before writing. It now copies the stored row bytes of the selected
columns through ``table.writer`` one chunk at a time, so memory is bounded by
the chunk size and every stored cell -- TNULL sentinels, scaled and unsigned
integers, bit and complex columns -- is byte-identical. A selection with a
variable-length column falls back to the materializing rewrite.
"""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest

import torchfits
from torchfits.cli import cmds_convert
from torchfits.cli.main import main

N = 257


@pytest.fixture
def catalog(tmp_path):
    src = tmp_path / "cat.fits"
    rng = np.random.default_rng(0)
    flux = rng.integers(-5, 100, N).astype(np.int32)
    flux[::7] = -999
    data = {
        "ID": np.arange(N, dtype=np.int64),
        "flux": flux,
        "mag": rng.uniform(10, 20, N).astype(np.float32),
        "sc": rng.integers(0, 100, N).astype(np.int16),
        "u": rng.integers(0, 2**32 - 1, N, dtype=np.uint64).astype(np.uint32),
        "bits": rng.integers(0, 2, (N, 11)).astype(bool),
        "name": [f"n{i}" for i in range(N)],
    }
    schema = {
        "flux": {"format": "J", "unit": "pix", "tnull": -999},
        "sc": {"format": "I", "bscale": 0.5, "bzero": 10.0},
        "bits": {"format": "11X"},
        "name": {"format": "6A"},
    }
    torchfits.table.write(
        str(src),
        data,
        schema=schema,
        header={"EXTNAME": "CAT", "OBSERVER": "me", "HISTORY": "made"},
        overwrite=True,
    )
    return src


@pytest.fixture
def streamed_only(monkeypatch):
    """Fail the materializing rewrite and force many small chunks."""

    def boom(*args, **kwargs):
        raise AssertionError("convert --to fits materialized the table")

    monkeypatch.setattr(cmds_convert, "_convert_table_to_fits_materialized", boom)
    monkeypatch.setattr(cmds_convert, "_RAW_CHUNK_BYTES", 500)


def _stored(path, columns=None, where=None):
    return torchfits.table.read(
        str(path), hdu=1, columns=columns, where=where, apply_fits_nulls=False
    ).to_pydict()


def test_full_copy_streams_byte_identical(catalog, tmp_path, streamed_only):
    out = tmp_path / "out.fits"
    assert main(["convert", str(catalog), str(out), "--to", "fits"]) == 0
    src_hdr = torchfits.read_header(str(catalog), 1)
    out_hdr = torchfits.read_header(str(out), 1)
    for key in src_hdr.keys():
        if key not in {"CHECKSUM", "DATASUM"}:
            assert out_hdr[key] == src_hdr[key], key
    spans_src = torchfits._io_engine._hdu_splice.hdu_byte_spans(str(catalog))[1]
    spans_out = torchfits._io_engine._hdu_splice.hdu_byte_spans(str(out))[1]
    with open(catalog, "rb") as a, open(out, "rb") as b:
        a.seek(spans_src[1])
        b.seek(spans_out[1])
        nbytes = N * int(src_hdr["NAXIS1"])
        assert a.read(nbytes) == b.read(nbytes)


def test_complex_columns_stream(tmp_path, streamed_only):
    src = tmp_path / "z.fits"
    z = (np.arange(50) + 1j * np.arange(50)[::-1]).astype(np.complex64)
    torchfits.table.write(str(src), {"z": z, "w": z.astype(np.complex128)})
    out = tmp_path / "out.fits"
    assert main(["convert", str(src), str(out), "--to", "fits"]) == 0
    got = torchfits.table.read_torch(str(out))
    assert np.array_equal(got["z"].numpy(), z)
    assert np.array_equal(got["w"].numpy(), z.astype(np.complex128))


def test_columns_and_where_stream_across_chunks(catalog, tmp_path, streamed_only):
    out = tmp_path / "out.fits"
    cols = "name,u,flux,bits,sc"
    where = "mag > 15"
    code = main(
        ["convert", str(catalog), str(out), "--to", "fits", "-c", cols, "-w", where]
    )
    assert code == 0
    want = _stored(catalog, cols.split(","), where)
    assert 0 < len(want["u"]) < N
    assert _stored(out) == want
    assert (
        torchfits.table.read(str(out)).to_pydict()
        == torchfits.table.read(
            str(catalog), columns=cols.split(","), where=where
        ).to_pydict()
    )
    hdr = torchfits.read_header(str(out), 1)
    assert hdr["TTYPE3"] == "flux" and hdr["TNULL3"] == -999 and hdr["TUNIT3"] == "pix"
    assert hdr["TFORM4"] == "11X"
    assert hdr["TSCAL5"] == 0.5 and hdr["TZERO5"] == 10
    assert hdr["TZERO2"] == 2**31
    assert hdr["EXTNAME"] == "CAT" and hdr["OBSERVER"] == "me"
    assert hdr["NAXIS2"] == len(want["u"])


def test_ascii_table_streams(tmp_path, streamed_only):
    src = tmp_path / "asc.fits"
    torchfits.table.write(
        str(src),
        {
            "a": np.arange(40, dtype=np.int32),
            "s": [f"x{i}" for i in range(40)],
            "b": np.arange(40) / 4.0,
        },
        table_type="ascii",
        header={"EXTNAME": "ASC"},
        overwrite=True,
    )
    out = tmp_path / "out.fits"
    assert (
        main(
            [
                "convert",
                str(src),
                str(out),
                "--to",
                "fits",
                "-c",
                "b,a",
                "-w",
                "a >= 10",
            ]
        )
        == 0
    )
    assert torchfits.read_header(str(out), 1)["XTENSION"].strip() == "TABLE"
    assert torchfits.table.read(str(out)).to_pydict() == {
        "b": [i / 4.0 for i in range(10, 40)],
        "a": list(range(10, 40)),
    }


def test_gzip_source_streams(catalog, tmp_path, streamed_only):
    gz = tmp_path / "cat.fits.gz"
    with open(catalog, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    out = tmp_path / "out.fits"
    assert main(["convert", str(gz), str(out), "--to", "fits", "-w", "ID < 100"]) == 0
    assert _stored(out) == _stored(catalog, where="ID < 100")


def test_failure_mid_stream_keeps_output(catalog, tmp_path, monkeypatch):
    out = tmp_path / "out.fits"
    out.write_bytes(b"keep")
    monkeypatch.setattr(cmds_convert, "_RAW_CHUNK_BYTES", 500)
    real = cmds_convert._chunk_where_mask
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise cmds_convert.UsageError("--where: boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(cmds_convert, "_chunk_where_mask", flaky)
    assert (
        main(["convert", str(catalog), str(out), "--to", "fits", "-w", "ID >= 0"]) == 2
    )
    assert out.read_bytes() == b"keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cat.fits", "out.fits"]


def test_variable_length_column_falls_back(tmp_path, monkeypatch):
    src = tmp_path / "vla.fits"
    torchfits.table.write(
        str(src),
        {
            "a": np.arange(3, dtype=np.int32),
            "v": [np.arange(k, dtype=np.float64) for k in (1, 2, 3)],
        },
        overwrite=True,
    )
    out = tmp_path / "out.fits"
    assert main(["convert", str(src), str(out), "--to", "fits", "-w", "a > 0"]) == 0
    got = torchfits.table.read(str(out)).to_pydict()
    assert got == {"a": [1, 2], "v": [[0.0, 1.0], [0.0, 1.0, 2.0]]}

    # Without the variable-length column the copy streams.
    def boom(*args, **kwargs):
        raise AssertionError("materialized")

    monkeypatch.setattr(cmds_convert, "_convert_table_to_fits_materialized", boom)
    assert main(["convert", str(src), str(out), "--to", "fits", "-c", "a"]) == 0
    assert torchfits.table.read(str(out)).to_pydict() == {"a": [0, 1, 2]}
