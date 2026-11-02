"""A write that fails part-way leaves the original file byte-identical.

Every whole-file writer and HDU edit writes to a hidden sibling temp file and
publishes it only on success. These tests inject a failure after some bytes
already reached the temp file (a native writer that dies mid-write, or a
splice copy that dies mid-stream) and check that the target is unchanged and
no temp file is left behind.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as C
from torchfits import table
from torchfits._io_engine import _hdu_splice


class InjectedFault(RuntimeError):
    pass


def _dying_writer(monkeypatch, name):
    """Replace native writer *name* with one that writes junk, then raises."""
    calls = []

    def fake(path, *args, **kwargs):
        calls.append(path)
        with open(os.fspath(path).lstrip("!"), "wb") as fh:
            fh.write(b"SIMPLE  =                    T" + b" " * 2000)
        raise InjectedFault(f"injected failure in {name}")

    monkeypatch.setattr(C, name, fake)
    return calls


def _dying_splice(monkeypatch):
    real = _hdu_splice._copy_range
    calls = []

    def fake(src, dst, start, end):
        calls.append((start, end))
        real(src, dst, start, end)
        raise InjectedFault("injected failure while splicing")

    monkeypatch.setattr(_hdu_splice, "_copy_range", fake)
    return calls


def _scratch_dirs():
    return {
        d for d in os.listdir(tempfile.gettempdir()) if d.startswith("torchfits-hdu-")
    }


@pytest.fixture
def original(tmp_path):
    path = tmp_path / "orig.fits"
    torchfits.write(
        str(path),
        torchfits.HDUList(
            [
                torchfits.TensorHDU(torch.arange(12.0).reshape(3, 4)),
                torchfits.TableHDU({"A": torch.arange(5)}),
                torchfits.TensorHDU(torch.ones(2, 2)),
            ]
        ),
    )
    return path


def _assert_untouched(path, before):
    assert path.read_bytes() == before
    assert sorted(os.listdir(path.parent)) == [path.name], "temp file left behind"
    torchfits.clear_file_cache()
    assert torch.equal(
        torchfits.read(str(path), hdu=0), torch.arange(12.0).reshape(3, 4)
    )


@pytest.mark.parametrize(
    ("native", "data"),
    [
        ("write_fits_file", torch.zeros(4, 4)),
        ("write_fits_file", torchfits.HDUList([torchfits.TensorHDU(torch.zeros(2))])),
        ("write_fits_table", {"B": torch.arange(3)}),
    ],
)
def test_write_overwrite_failure_keeps_original(monkeypatch, original, native, data):
    before = original.read_bytes()
    calls = _dying_writer(monkeypatch, native)
    with pytest.raises(RuntimeError, match="injected failure"):
        torchfits.write(str(original), data, overwrite=True)
    assert calls and os.fspath(calls[0]).lstrip("!") != str(original)
    _assert_untouched(original, before)


def test_table_write_overwrite_failure_keeps_original(monkeypatch, original):
    before = original.read_bytes()
    calls = _dying_writer(monkeypatch, "write_fits_table")
    with pytest.raises(RuntimeError, match="injected failure"):
        table.write(str(original), {"B": np.arange(3)}, overwrite=True)
    assert calls
    _assert_untouched(original, before)


def test_write_new_file_failure_creates_nothing(monkeypatch, tmp_path):
    _dying_writer(monkeypatch, "write_fits_file")
    with pytest.raises(RuntimeError, match="injected failure"):
        torchfits.write(str(tmp_path / "new.fits"), torch.zeros(2))
    assert os.listdir(tmp_path) == []


EDITS = {
    "insert": lambda p: torchfits.insert_hdu(p, torch.zeros(2, 2), index=1),
    "insert_primary": lambda p: torchfits.insert_hdu(p, torch.zeros(2, 2), index=0),
    "replace": lambda p: torchfits.replace_hdu(p, 2, torch.zeros(3)),
    "replace_table": lambda p: torchfits.replace_hdu(p, 1, {"A": torch.arange(2)}),
}


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_hdu_edit_native_failure_keeps_original(monkeypatch, original, edit):
    before = original.read_bytes()
    scratch_before = _scratch_dirs()
    calls = _dying_writer(monkeypatch, "write_fits_file")
    with pytest.raises(RuntimeError, match="injected failure"):
        EDITS[edit](str(original))
    assert calls
    _assert_untouched(original, before)
    assert _scratch_dirs() == scratch_before


@pytest.mark.parametrize(
    "edit",
    [
        *sorted(EDITS),
        "delete",
        "delete_primary",
    ],
)
def test_hdu_edit_splice_failure_keeps_original(monkeypatch, tmp_path, original, edit):
    if edit == "delete_primary":
        # Deleting HDU 0 needs an IMAGE extension at index 1.
        torchfits.delete_hdu(str(original), 1)
    before = original.read_bytes()
    scratch_before = _scratch_dirs()
    calls = _dying_splice(monkeypatch)
    action = {
        "delete": lambda p: torchfits.delete_hdu(p, 1),
        "delete_primary": lambda p: torchfits.delete_hdu(p, 0),
        **EDITS,
    }[edit]
    with pytest.raises(InjectedFault):
        action(str(original))
    assert calls
    assert original.read_bytes() == before
    assert sorted(os.listdir(tmp_path)) == [original.name], "temp file left behind"
    assert _scratch_dirs() == scratch_before
