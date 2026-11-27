"""Every non-FITS output and multi-step CLI output is published atomically.

A failure part-way through a write must leave a pre-existing destination
byte-identical and must not leave a partial (or temp) file behind. Each test
injects a failure *after* some bytes reached the file the writer was given.
"""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits.cli.main import main

pa = pytest.importorskip("pyarrow")

OLD = b"previous destination contents\n"


class _Boom(RuntimeError):
    pass


def _seed(path) -> None:
    path.write_bytes(OLD)


def _assert_intact(path) -> None:
    assert path.read_bytes() == OLD
    leftovers = sorted(p.name for p in path.parent.iterdir() if ".tmp" in p.name)
    assert leftovers == []


def _table() -> object:
    return pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def _partial_then_fail(dest, *args, **kwargs):
    with open(dest, "wb") as handle:
        handle.write(b"PARTIAL")
    raise _Boom("injected failure")


# -- interop writers ---------------------------------------------------------


def test_write_parquet_failure_keeps_destination(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    dest = tmp_path / "out.parquet"
    _seed(dest)
    monkeypatch.setattr(
        pq, "write_table", lambda table, where, **kw: _partial_then_fail(where)
    )
    with pytest.raises(_Boom):
        torchfits.table.write_parquet(str(dest), _table())
    _assert_intact(dest)


def test_write_parquet_stream_failure_keeps_destination(tmp_path):
    dest = tmp_path / "out.parquet"
    _seed(dest)

    def batches():
        yield pa.record_batch({"a": [1, 2]})
        raise _Boom("injected failure")

    with pytest.raises(_Boom):
        torchfits.table.write_parquet(str(dest), batches(), stream=True)
    _assert_intact(dest)


def test_write_csv_failure_keeps_destination(tmp_path, monkeypatch):
    import pyarrow.csv as pacsv

    dest = tmp_path / "out.csv"
    _seed(dest)
    monkeypatch.setattr(
        pacsv, "write_csv", lambda table, where, **kw: _partial_then_fail(where)
    )
    with pytest.raises(_Boom):
        torchfits.table.write_csv(str(dest), _table())
    _assert_intact(dest)


def test_write_csv_stream_failure_keeps_destination(tmp_path):
    dest = tmp_path / "out.tsv"
    _seed(dest)

    def batches():
        yield pa.record_batch({"a": [1, 2]})
        raise _Boom("injected failure")

    with pytest.raises(_Boom):
        torchfits.table.write_csv(str(dest), batches(), delimiter="\t", stream=True)
    _assert_intact(dest)


def test_write_ipc_failure_keeps_destination(tmp_path, monkeypatch):
    import pyarrow.feather as feather

    dest = tmp_path / "out.arrow"
    _seed(dest)
    monkeypatch.setattr(
        feather, "write_feather", lambda table, where, **kw: _partial_then_fail(where)
    )
    with pytest.raises(_Boom):
        torchfits.table.write_ipc(str(dest), _table())
    _assert_intact(dest)


def test_write_ipc_stream_failure_keeps_destination(tmp_path):
    dest = tmp_path / "out.arrow"
    _seed(dest)

    def batches():
        yield pa.record_batch({"a": [1, 2]})
        raise _Boom("injected failure")

    with pytest.raises(_Boom):
        torchfits.table.write_ipc(str(dest), batches(), stream=True)
    _assert_intact(dest)


def test_interop_success_replaces_existing_destination(tmp_path):
    import pyarrow.parquet as pq

    dest = tmp_path / "out.parquet"
    _seed(dest)
    torchfits.table.write_parquet(str(dest), _table())
    assert pq.read_table(dest).column("a").to_pylist() == [1, 2, 3]
    assert [p.name for p in tmp_path.iterdir()] == ["out.parquet"]


def test_interop_gz_name_is_not_gzip_wrapped(tmp_path):
    dest = tmp_path / "out.csv.gz"
    torchfits.table.write_csv(str(dest), _table())
    assert dest.read_bytes().startswith(b'"a","b"')


def test_convert_cli_parquet_failure_keeps_destination(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    src = tmp_path / "cat.fits"
    torchfits.table.write(str(src), {"a": [1, 2, 3]}, overwrite=True)
    dest = tmp_path / "out.parquet"
    _seed(dest)

    real_writer = pq.ParquetWriter

    class FailingWriter(real_writer):  # type: ignore[misc,valid-type]
        def write_batch(self, *args, **kwargs):
            super().write_batch(*args, **kwargs)
            raise _Boom("injected failure")

    monkeypatch.setattr(pq, "ParquetWriter", FailingWriter)
    assert main(["convert", str(src), str(dest)]) != 0
    _assert_intact(dest)


# -- PNG ---------------------------------------------------------------------


def test_png_failure_keeps_destination(tmp_path, monkeypatch):
    import builtins

    import importlib

    rgb = importlib.import_module("torchfits.transforms.rgb")

    dest = tmp_path / "out.png"
    _seed(dest)
    real_open = builtins.open

    class _FailingHandle:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            return False

        def write(self, data):
            self._handle.write(data[:8])
            raise _Boom("injected failure")

    def failing_open(path, mode="r", *args, **kwargs):
        return _FailingHandle(real_open(path, mode, *args, **kwargs))

    monkeypatch.setattr(rgb, "open", failing_open, raising=False)
    with pytest.raises(_Boom):
        rgb.write_rgb_image(str(dest), torch.rand(4, 4, 3))
    _assert_intact(dest)


# -- copy / compress / decompress / setkey -----------------------------------


def _image(path, value: float = 1.0) -> None:
    torchfits.write(str(path), torch.full((4, 4), value), overwrite=True)


def test_copy_failure_keeps_destination(tmp_path, monkeypatch):
    from torchfits.cli import cmds_copy

    src = tmp_path / "src.fits"
    _image(src)
    dest = tmp_path / "dest.fits"
    _seed(dest)

    def failing_copy2(source, target, *args, **kwargs):
        with open(source, "rb") as fin, open(target, "wb") as fout:
            fout.write(fin.read(100))
        raise _Boom("injected failure")

    monkeypatch.setattr(cmds_copy.shutil, "copy2", failing_copy2)
    assert main(["copy", str(src), str(dest)]) == 3
    _assert_intact(dest)


def test_copy_success_is_byte_identical(tmp_path):
    src = tmp_path / "src.fits"
    _image(src)
    dest = tmp_path / "dest.fits"
    _seed(dest)
    assert main(["copy", str(src), str(dest)]) == 0
    assert dest.read_bytes() == src.read_bytes()


def test_compress_post_edit_failure_keeps_destination(tmp_path, monkeypatch):
    src = tmp_path / "src.fits"
    _image(src)  # primary image: compress moves it and stamps ZSIMPLE
    dest = tmp_path / "dest.fits"
    _seed(dest)

    def failing_cards(*args, **kwargs):
        raise _Boom("injected failure")

    monkeypatch.setattr(torchfits.io, "_write_header_cards_if_supported", failing_cards)
    assert main(["compress", str(src), str(dest), "-J", "1"]) == 3
    _assert_intact(dest)


def test_compress_publishes_zsimple(tmp_path):
    src = tmp_path / "src.fits"
    _image(src, 3.0)
    dest = tmp_path / "dest.fits"
    _seed(dest)
    assert main(["compress", str(src), str(dest), "-J", "1"]) == 0
    assert torchfits.read_header(str(dest), 1).get("ZSIMPLE") is True
    assert torch.equal(torchfits.read_tensor(str(dest), hdu=1), torch.full((4, 4), 3.0))


def test_decompress_post_edit_failure_keeps_destination(tmp_path, monkeypatch):
    src = tmp_path / "src.fits"
    _image(src)
    packed = tmp_path / "packed.fits"
    assert main(["compress", str(src), str(packed), "-J", "1"]) == 0
    dest = tmp_path / "dest.fits"
    _seed(dest)

    def failing_read_header(*args, **kwargs):
        raise _Boom("injected failure")

    monkeypatch.setattr(torchfits, "read_header", failing_read_header)
    assert main(["decompress", str(packed), str(dest), "-J", "1"]) == 3
    _assert_intact(dest)


def test_setkey_out_failure_keeps_existing_destination(tmp_path, monkeypatch):
    from torchfits.cli import cmds_setkey

    src = tmp_path / "src.fits"
    _image(src)
    dest = tmp_path / "dest.fits"
    _seed(dest)
    real_write_plan = cmds_setkey._write_plan

    def failing_write_plan(path, plan):
        real_write_plan(path, plan)
        raise _Boom("injected failure")

    monkeypatch.setattr(cmds_setkey, "_write_plan", failing_write_plan)
    code = main(["setkey", str(src), "-k", "OBSERVER", "--value", "x", "-o", str(dest)])
    assert code == 3
    _assert_intact(dest)


def test_setkey_out_success_replaces_destination(tmp_path):
    src = tmp_path / "src.fits"
    _image(src)
    before = src.read_bytes()
    dest = tmp_path / "dest.fits"
    _seed(dest)
    code = main(["setkey", str(src), "-k", "OBSERVER", "--value", "x", "-o", str(dest)])
    assert code == 0
    assert torchfits.read_header(str(dest), 0)["OBSERVER"] == "x"
    assert src.read_bytes() == before
