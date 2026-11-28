"""Every spelling of a path shares one native metadata cache entry.

The shared per-path metadata cache (``SharedReadMeta``) was keyed by the raw
string the caller passed, while writers invalidate by whichever spelling *they*
were given. Reading ``./x.fits`` and then appending through ``/abs/x.fits``
left the ``./x.fits`` entry alive, so ``read_nrows('./x.fits')`` kept reporting
the old row count until the one-second validation window ran out. Entries are
now keyed by the canonical (realpath) base file plus any extended-syntax
suffix, and an invalidation drops every entry that reads the same file.

The validation window is left at its default (1 s) on purpose: these tests
must pass without the stat re-check, so they cover the invalidation path only.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import torchfits


def _write_table(path: str, nrows: int) -> None:
    torchfits.write(path, {"a": np.arange(nrows, dtype=np.int32)}, overwrite=True)


def _append_one(path: str) -> None:
    torchfits.table.append_rows(path, {"a": np.array([99], dtype=np.int32)})


@pytest.fixture()
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_relative_reader_sees_append_through_absolute_path(in_tmp) -> None:
    _write_table("x.fits", 3)
    assert torchfits.read_nrows("./x.fits") == 3
    _append_one(os.path.abspath("x.fits"))
    assert torchfits.read_nrows("./x.fits") == 4
    assert torchfits.read_nrows(os.path.abspath("x.fits")) == 4


def test_absolute_reader_sees_append_through_relative_path(in_tmp) -> None:
    _write_table("x.fits", 3)
    absolute = os.path.abspath("x.fits")
    assert torchfits.read_nrows(absolute) == 3
    _append_one("./x.fits")
    assert torchfits.read_nrows(absolute) == 4
    assert torchfits.read_nrows("x.fits") == 4


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_symlink_alias_shares_invalidation(in_tmp) -> None:
    _write_table("real.fits", 3)
    try:
        os.symlink("real.fits", "link.fits")
    except OSError:
        pytest.skip("symlinks not permitted here")
    assert torchfits.read_nrows("link.fits") == 3
    assert torchfits.read_nrows("real.fits") == 3

    _append_one("real.fits")
    assert torchfits.read_nrows("link.fits") == 4

    _append_one("link.fits")
    assert torchfits.read_nrows("real.fits") == 5
    assert torchfits.read_nrows("link.fits") == 5


def test_extended_syntax_entry_invalidated_by_base_path(in_tmp) -> None:
    _write_table("x.fits", 3)
    extended = "x.fits[1]"
    # Extended syntax positions on HDU 1, so the HDU argument is relative (0).
    assert torchfits.read_nrows(extended, 0) == 3
    assert torchfits.read_nrows(os.path.abspath(extended), 0) == 3

    _append_one(os.path.abspath("x.fits"))
    assert torchfits.read_nrows(extended, 0) == 4
    assert torchfits.read_nrows("./" + extended, 0) == 4


def test_aliases_share_one_registry_entry(in_tmp) -> None:
    import torchfits._core as core

    _write_table("x.fits", 3)
    spellings = [
        "x.fits",
        "./x.fits",
        os.path.abspath("x.fits"),
        os.path.join("..", in_tmp.name, "x.fits"),
    ]
    core.clear_shared_read_meta_cache()
    for spelling in spellings:
        assert torchfits.read_nrows(spelling) == 3
    assert core.shared_meta_entry_count() == 1


def test_relative_name_follows_chdir(tmp_path, monkeypatch) -> None:
    # The name -> file resolution is remembered, but a relative name is
    # remembered together with the directory it was resolved in: after a chdir
    # the same spelling must reach the other directory's file at once.
    for name, nrows in (("a", 3), ("b", 5)):
        (tmp_path / name).mkdir()
        _write_table(str(tmp_path / name / "x.fits"), nrows)
    monkeypatch.chdir(tmp_path / "a")
    assert torchfits.read_nrows("x.fits") == 3
    monkeypatch.chdir(tmp_path / "b")
    assert torchfits.read_nrows("x.fits") == 5
    monkeypatch.chdir(tmp_path / "a")
    assert torchfits.read_nrows("./x.fits") == 3


_RESOLUTION_WINDOW_SCRIPT = r"""
import os, sys
import numpy as np
import torchfits

root = sys.argv[1]
os.chdir(root)
torchfits.write("a.fits", {"a": np.arange(3, dtype=np.int32)}, overwrite=True)
torchfits.write("b.fits", {"a": np.arange(5, dtype=np.int32)}, overwrite=True)
os.symlink("a.fits", "link.fits")
assert torchfits.read_nrows("link.fits") == 3

# Retarget the link behind torchfits' back. Like a rewrite of the file itself,
# that is noticed when the validation window ends, not on every call: the
# probe must not pay a realpath() (a readlink per path component) each time.
os.symlink("b.fits", "link.tmp")
os.replace("link.tmp", "link.fits")
assert torchfits.read_nrows("link.fits") == 3, "resolution was not remembered"
# The remembered resolution must not leak b.fits' identity into a.fits' entry.
assert torchfits.read_nrows("a.fits") == 3
assert torchfits.read_nrows("b.fits") == 5

# An in-process write invalidates, which drops every remembered resolution.
torchfits.table.append_rows("b.fits", {"a": np.array([9], dtype=np.int32)})
assert torchfits.read_nrows("link.fits") == 6
assert torchfits.read_nrows("a.fits") == 3
print("OK")
"""


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_name_resolution_is_remembered_for_the_validation_window(tmp_path) -> None:
    import subprocess
    import sys

    probe = tmp_path / "probe"
    try:
        os.symlink("nowhere", probe)
    except OSError:
        pytest.skip("symlinks not permitted here")
    env = dict(os.environ, TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS="600000")
    proc = subprocess.run(
        [sys.executable, "-c", _RESOLUTION_WINDOW_SCRIPT, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
