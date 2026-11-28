"""Incremental FITS table writes: ``table.write`` of batch streams and ``table.writer``.

``table.write`` used to need complete columnar input; a RecordBatchReader or a
generator of batches failed with "data must be a non-empty dictionary or
table/dataframe object". Streams are now written batch by batch into a temp
file that is published atomically on success.
"""

from __future__ import annotations

import gzip
import os
import tracemalloc

import numpy as np
import pyarrow as pa
import pytest

import torchfits
from torchfits import table


def _batches(n_batches: int, rows: int, *, start: int = 0):
    for k in range(start, start + n_batches):
        base = k * rows
        yield {
            "ID": np.arange(base, base + rows, dtype=np.int64),
            "FLUX": np.full((rows, 3), k, dtype=np.float32),
            "U16": (np.arange(rows) % 60000).astype(np.uint16),
            "I8": (np.arange(rows) % 200 - 100).astype(np.int8),
            "NAME": np.array([f"s{k}_{j % 10}" for j in range(rows)], dtype="S8"),
            "OK": np.arange(rows) % 3 == 0,
        }


def _concat(batches):
    batches = list(batches)
    return {name: np.concatenate([b[name] for b in batches]) for name in batches[0]}


def test_generator_of_dicts_matches_one_shot_write(tmp_path) -> None:
    streamed = str(tmp_path / "streamed.fits")
    oneshot = str(tmp_path / "oneshot.fits")
    table.write(streamed, _batches(4, 250), extname="CAT", header={"ORIGIN": "x"})
    table.write(
        oneshot, _concat(_batches(4, 250)), extname="CAT", header={"ORIGIN": "x"}
    )

    assert table.read(streamed).equals(table.read(oneshot))
    h_stream = torchfits.read_header(streamed, 1)
    h_once = torchfits.read_header(oneshot, 1)
    assert h_stream["NAXIS2"] == 1000
    for key in (
        "NAXIS1",
        "TFORM1",
        "TFORM2",
        "TFORM3",
        "TZERO3",
        "TFORM4",
        "TZERO4",
        "TFORM5",
        "TFORM6",
        "EXTNAME",
        "ORIGIN",
    ):
        assert h_stream[key] == h_once[key], key


def test_record_batch_reader_round_trip(tmp_path) -> None:
    src = str(tmp_path / "src.fits")
    dst = str(tmp_path / "dst.fits")
    table.write(src, _concat(_batches(3, 400)))
    reader = table.reader(src, batch_size=170)
    table.write(dst, reader)
    assert table.read(dst).equals(table.read(src))
    # Vector columns come back as fixed-width vectors, not variable-length.
    assert (
        torchfits.read_header(dst, 1)["TFORM2"]
        == torchfits.read_header(src, 1)["TFORM2"]
    )


def test_list_of_record_batches_and_tables(tmp_path) -> None:
    path = str(tmp_path / "t.fits")
    t = pa.table({"A": pa.array([1, 2, 3], pa.int32()), "B": [1.5, None, 2.5]})
    batches = [t.to_batches()[0], t, t.slice(1)]
    table.write(path, batches)
    out = table.read(path)
    assert out["A"].to_pylist() == [1, 2, 3, 1, 2, 3, 2, 3]
    got = out["B"].to_numpy(zero_copy_only=False)
    assert np.isnan(got[[1, 4, 6]]).all()


def test_writer_context_manager_and_num_rows(tmp_path) -> None:
    path = str(tmp_path / "w.fits")
    with table.writer(path, checksum=True) as w:
        for batch in _batches(3, 100):
            w.write(batch)
        assert w.num_rows == 300
        assert not os.path.exists(path)  # published only on close
    assert w.closed
    assert torchfits.read_nrows(path, 1) == 300
    report = torchfits.verify_checksums(path)
    assert report["ok"], report


def test_many_batches_bounded_memory(tmp_path) -> None:
    path = str(tmp_path / "big.fits")
    rows = 4096
    n_batches = 200  # 200 * 4096 * 40 B ~= 33 MB of table data

    def gen():
        for k in range(n_batches):
            yield {
                "ID": np.arange(k * rows, (k + 1) * rows, dtype=np.int64),
                "V": np.full((rows, 4), k, dtype=np.float64),
            }

    tracemalloc.start()
    try:
        table.write(path, gen())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    total = n_batches * rows * 40
    assert peak < total / 8, f"peak {peak} B for {total} B of data"

    assert torchfits.read_nrows(path, 1) == n_batches * rows
    ids = table.read(path, columns=["ID"], row_slice=(rows * 150, rows * 150 + 3))
    assert ids["ID"].to_pylist() == [rows * 150 + i for i in range(3)]
    tail = table.read(
        path, columns=["V"], row_slice=(n_batches * rows - 1, n_batches * rows)
    )
    assert tail["V"].to_pylist() == [[n_batches - 1.0] * 4]


def test_string_overflow_is_refused_and_destination_kept(tmp_path) -> None:
    path = str(tmp_path / "s.fits")
    with open(path, "wb") as fh:
        fh.write(b"precious")
    batches = [{"S": ["ab", "c"]}, {"S": ["toolong"]}]
    with pytest.raises(
        ValueError, match=r"does not fit.*schema=\{'S': \{'format': '7A'\}\}"
    ):
        table.write(path, iter(batches), overwrite=True)
    with open(path, "rb") as fh:
        assert fh.read() == b"precious"
    assert sorted(os.listdir(tmp_path)) == ["s.fits"]

    table.write(path, iter(batches), overwrite=True, schema={"S": {"format": "8A"}})
    assert table.read(path)["S"].to_pylist() == ["ab", "c", "toolong"]


def test_schema_mismatches_raise_clearly(tmp_path) -> None:
    path = str(tmp_path / "m.fits")
    with pytest.raises(ValueError, match="do not match the table's columns"):
        table.write(path, [{"A": np.arange(2)}, {"B": np.arange(2)}])
    with pytest.raises(ValueError, match="do not match the table's columns"):
        table.write(
            path,
            [
                {"A": np.arange(2), "B": np.arange(2)},
                {"B": np.arange(2), "A": np.arange(2)},
            ],
        )
    with pytest.raises(TypeError, match="float64 cannot be stored"):
        table.write(path, [{"A": np.arange(2, dtype=np.int32)}, {"A": np.ones(2)}])
    with pytest.raises(ValueError, match="cell shape"):
        table.write(path, [{"A": np.zeros((2, 3))}, {"A": np.zeros((2, 4))}])
    with pytest.raises(TypeError, match="str values"):
        table.write(path, [{"A": np.arange(2)}, {"A": ["x", "y"]}])
    assert not os.path.exists(path)
    assert os.listdir(tmp_path) == []

    # Safe widening is accepted and stored in the first batch's type.
    table.write(
        path, [{"A": np.arange(2, dtype=np.int64)}, {"A": np.arange(2, dtype=np.int16)}]
    )
    assert table.read(path)["A"].to_pylist() == [0, 1, 0, 1]


def test_producer_failure_leaves_no_file(tmp_path) -> None:
    path = str(tmp_path / "p.fits")

    def gen():
        yield {"A": np.arange(3)}
        raise RuntimeError("producer died")

    with pytest.raises(RuntimeError, match="producer died"):
        table.write(path, gen())
    assert os.listdir(tmp_path) == []

    with pytest.raises(KeyError):
        with table.writer(path) as w:
            w.write({"A": np.arange(3)})
            raise KeyError("boom")
    assert os.listdir(tmp_path) == []


def test_overwrite_false_refuses_existing_before_consuming(tmp_path) -> None:
    path = str(tmp_path / "x.fits")
    table.write(path, {"A": np.arange(2)})
    consumed = []

    def gen():
        consumed.append(1)
        yield {"A": np.arange(2)}

    with pytest.raises(FileExistsError):
        table.write(path, gen())
    assert consumed == []


def test_empty_streams(tmp_path) -> None:
    path = str(tmp_path / "e.fits")
    with pytest.raises(ValueError, match="no batches were written"):
        table.write(path, iter([]))
    assert not os.path.exists(path)

    schema = pa.schema([("A", pa.int64()), ("S", pa.string())])
    table.write(path, pa.RecordBatchReader.from_batches(schema, []))
    out = table.read(path)
    assert out.num_rows == 0
    assert out.column_names == ["A", "S"]

    # Leading empty batches (a filtered scan yields them) do not fix the
    # string width.
    empty = pa.record_batch([pa.array([], pa.string())], names=["S"])
    full = pa.record_batch([pa.array(["abcdef"])], names=["S"])
    table.write(path, [empty, full, empty], overwrite=True)
    assert table.read(path)["S"].to_pylist() == ["abcdef"]


def test_integer_nulls_need_tnull(tmp_path) -> None:
    path = str(tmp_path / "n.fits")
    first = pa.record_batch([pa.array([1, 2], pa.int32())], names=["A"])
    nulls = pa.record_batch([pa.array([None, 4], pa.int32())], names=["A"])
    with pytest.raises(ValueError, match="null values but no TNULL"):
        table.write(path, [first, nulls])
    assert not os.path.exists(path)

    table.write(path, [first, nulls], schema={"A": {"tnull": -99}})
    assert torchfits.read_header(path, 1)["TNULL1"] == -99
    assert table.read(path)["A"].to_pylist() == [1, 2, None, 4]

    with pytest.raises(ValueError, match="logical columns"):
        table.write(
            str(tmp_path / "b.fits"),
            [pa.record_batch([pa.array([True, None])], names=["B"])],
        )


def test_schema_scaling_is_applied_once(tmp_path) -> None:
    path = str(tmp_path / "scaled.fits")
    schema = {"A": {"format": "J", "bscale": 0.5, "bzero": 10}}
    stored = [np.array([0, 2], np.int32), np.array([4, 6], np.int32)]
    table.write(path, [{"A": s} for s in stored], schema=schema)
    assert table.read(path)["A"].to_pylist() == [10.0, 11.0, 12.0, 13.0]


def test_vla_and_ascii_streams(tmp_path) -> None:
    vla = str(tmp_path / "vla.fits")
    rows = [np.arange(k, dtype=np.int32) for k in range(1, 4)]
    table.write(vla, [{"V": rows}, {"V": rows[::-1]}])
    got = table.read(vla)["V"].to_pylist()
    assert got == [list(range(k)) for k in (1, 2, 3, 3, 2, 1)]

    asc = str(tmp_path / "ascii.fits")
    table.write(
        asc,
        [
            {"I": np.arange(3, dtype=np.int32), "S": ["a", "bb", "c"]},
            {"I": np.arange(3, 5, dtype=np.int32), "S": ["dd", "e"]},
        ],
        table_type="ascii",
    )
    out = table.read(asc)
    assert out["I"].to_pylist() == [0, 1, 2, 3, 4]
    assert out["S"].to_pylist() == ["a", "bb", "c", "dd", "e"]


def test_equal_length_vla_first_batch_stays_vla(tmp_path) -> None:
    # A first batch whose VLA cells all have the same length was laid out as
    # a fixed-width vector (table.write makes it a VLA), so a later batch with
    # other lengths was refused.
    path = str(tmp_path / "vla_eq.fits")
    one_shot = str(tmp_path / "one_shot.fits")
    first = [np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)]
    table.write(one_shot, {"V": first})
    table.write(path, [{"V": first}, {"V": [np.arange(3, dtype=np.int32)]}])
    assert torchfits.read_table_info(path)["tforms"][0].startswith("1PJ")
    assert torchfits.read_table_info(one_shot)["tforms"][0].startswith("1PJ")
    assert table.read(path)["V"].to_pylist() == [[0, 1], [0, 1], [0, 1, 2]]


def test_gzip_target_is_gzip(tmp_path) -> None:
    path = str(tmp_path / "t.fits.gz")
    table.write(path, _batches(2, 50))
    with gzip.open(path, "rb") as fh:
        assert fh.read(6) == b"SIMPLE"
    assert table.read(path).num_rows == 100


def test_quantize_is_refused_for_streams(tmp_path) -> None:
    with pytest.raises(ValueError, match="quantize"):
        table.write(
            str(tmp_path / "q.fits"), iter([{"A": np.ones(2)}]), quantize="robust"
        )


def test_checksum_option_on_one_shot_write(tmp_path) -> None:
    path = str(tmp_path / "c.fits")
    table.write(path, {"A": np.arange(4)}, checksum=True)
    assert torchfits.verify_checksums(path)["ok"]
    path2 = str(tmp_path / "c2.fits")
    table.write(path2, {"A": np.arange(4, dtype=np.uint16)}, checksum=True)
    assert torchfits.verify_checksums(path2)["ok"]
