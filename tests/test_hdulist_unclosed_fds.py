"""An HDUList dropped without close() releases its file descriptor.

The open-HDUList registry (which lets a mutation close a list's handle) held
strong references to every list ``torchfits.open`` returned, so a list the
caller never closed -- ``[h.data for h in torchfits.open(path)]`` -- kept its
file open for the life of the process, one descriptor per call.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest
import torch

import torchfits

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="counts /proc/self/fd"
)


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_unclosed_hdulists_do_not_leak_descriptors(tmp_path):
    path = str(tmp_path / "mef.fits")
    torchfits.write(path, [torch.zeros(3, 3), torch.ones(4, 4)])
    [h.data for h in torchfits.open(path)]  # warm up imports and caches
    gc.collect()
    before = _open_fds()
    for _ in range(20):
        [h.data for h in torchfits.open(path)]
    gc.collect()
    assert _open_fds() - before <= 1


def test_hdu_outlives_its_unclosed_list(tmp_path):
    path = str(tmp_path / "mef.fits")
    torchfits.write(path, [torch.zeros(3, 3), torch.ones(4, 4)])
    hdu = torchfits.open(path)[1]
    gc.collect()
    assert torch.equal(hdu.to_tensor("cpu"), torch.ones(4, 4))
    view = torchfits.open(path)[1].data
    gc.collect()
    assert tuple(view.shape) == (4, 4)
