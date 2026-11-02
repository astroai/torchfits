"""insert/replace/delete_hdu refuse a read-only file like other in-place edits.

The splice publishes through an atomic rename, which only needs a writable
directory, so a 0444 file used to be edited silently. Whole-file
``write(..., overwrite=True)`` keeps replace semantics.
"""

from __future__ import annotations

import os
import stat

import pytest
import torch

import torchfits

pytestmark = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes"
)


@pytest.fixture()
def readonly_mef(tmp_path):
    path = tmp_path / "ro.fits"
    hdus = [
        torchfits.TensorHDU(torch.empty(0)),
        torchfits.TensorHDU(torch.ones(3), {"EXTNAME": "A"}),
        torchfits.TensorHDU(torch.zeros(2), {"EXTNAME": "B"}),
    ]
    torchfits.write(str(path), torchfits.HDUList(hdus))
    os.chmod(path, 0o444)
    try:
        yield path
    finally:
        os.chmod(path, 0o644)


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: torchfits.insert_hdu(p, torch.ones(2), 1),
        lambda p: torchfits.replace_hdu(p, "A", torch.ones(5)),
        lambda p: torchfits.delete_hdu(p, "B"),
    ],
    ids=["insert", "replace", "delete"],
)
def test_hdu_edit_refuses_readonly_file(readonly_mef, edit):
    before = readonly_mef.read_bytes()
    with pytest.raises(PermissionError):
        edit(str(readonly_mef))
    assert readonly_mef.read_bytes() == before
    assert stat.S_IMODE(readonly_mef.stat().st_mode) == 0o444


def test_write_overwrite_still_replaces_readonly_file(readonly_mef):
    torchfits.write(str(readonly_mef), torch.arange(3.0), overwrite=True)
    assert torch.equal(torchfits.read(str(readonly_mef)), torch.arange(3.0))
