"""A rewrite that lands during a read must not be pinned in the read cache.

The fallback read path took the file signature *after* reading, so data read
from the old file was cached under the new file's signature and then served
as current.
"""

import torch

import torchfits
import torchfits._C as cpp


def test_rewrite_during_read_is_not_pinned(tmp_path, monkeypatch):
    path = str(tmp_path / "r6.fits")
    torchfits.write(path, torch.ones(4, 4))
    original = cpp.read_full
    fired = []

    def read_then_rewrite(*args, **kwargs):
        data = original(*args, **kwargs)
        if not fired:
            fired.append(True)
            torchfits.write(path, torch.zeros(5, 5), overwrite=True)
        return data

    monkeypatch.setattr(cpp, "read_full", read_then_rewrite)
    old, _ = torchfits.read(path, return_header=True)
    assert fired and old.shape == (4, 4)
    monkeypatch.setattr(cpp, "read_full", original)
    new, header = torchfits.read(path, return_header=True)
    assert new.shape == (5, 5) and int(header["NAXIS1"]) == 5
