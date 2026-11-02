"""A compressed MEF whose primary is ``TensorHDU(torch.empty(0))`` writes.

The compressed writer emits its own empty primary, so the caller's empty
primary must be skipped. Only a ``NAXIS=0`` header card was recognised; the
examples' idiom (an empty tensor with no header) failed with CFITSIO status
414 for every ``compress=`` value.
"""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits import HDUList, TensorHDU


@pytest.mark.parametrize("compress", ["RICE", True])
def test_empty_tensor_primary_with_compression(tmp_path, compress):
    sci = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    err = torch.ones(20, 20, dtype=torch.float32)
    hdul = HDUList(
        [
            TensorHDU(torch.empty(0)),
            TensorHDU(sci, {"EXTNAME": "SCI"}),
            TensorHDU(err, {"EXTNAME": "ERR"}),
        ]
    )
    path = str(tmp_path / f"mef_{compress}.fits")
    torchfits.write(path, hdul, compress=compress, overwrite=True)
    assert torch.allclose(torchfits.read(path, hdu="SCI"), sci)
    assert torch.allclose(torchfits.read(path, hdu="ERR"), err)
    with torchfits.open(path) as opened:
        assert len(opened) == 3
