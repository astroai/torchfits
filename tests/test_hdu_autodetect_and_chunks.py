"""Payload-HDU autodetection and TensorHDU.chunks()."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits import Header, HDUList, TableHDU, TensorHDU
from torchfits._io_engine.hdu_api import autodetect_hdu, find_first_hdu


def _empty_primary() -> TensorHDU:
    return TensorHDU(torch.zeros(0, dtype=torch.int16))


def test_image_after_table_is_preferred(tmp_path):
    path = str(tmp_path / "mixed.fits")
    HDUList(
        [
            _empty_primary(),
            TableHDU({"A": np.arange(3, dtype=np.int32)}),
            TensorHDU(torch.ones(4, 5), header=Header({"EXTNAME": "SCI"})),
        ]
    ).write(path)
    assert find_first_hdu(path) == 2
    assert autodetect_hdu(path) == 2
    assert torch.equal(torchfits.read(path, hdu="auto"), torch.ones(4, 5))


def test_table_only_falls_back_to_first_table(tmp_path):
    path = str(tmp_path / "tab.fits")
    torchfits.table.write(path, {"A": np.arange(3, dtype=np.int32)})
    assert find_first_hdu(path) == 1
    assert autodetect_hdu(path) == 1


def test_compressed_image_is_detected(tmp_path):
    path = str(tmp_path / "rice.fits")
    data = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    torchfits.write(path, data, compress="RICE_1")
    assert find_first_hdu(path) == 1
    assert torch.equal(torchfits.read(path, hdu="auto"), data)


def test_primary_image_wins_and_payloadless_file_is_zero(tmp_path):
    path = str(tmp_path / "img.fits")
    torchfits.write(path, torch.ones(2, 2))
    assert find_first_hdu(path) == 0
    assert autodetect_hdu(path) == 0

    empty = str(tmp_path / "empty.fits")
    HDUList([_empty_primary()]).write(empty)
    assert find_first_hdu(empty) is None
    assert autodetect_hdu(empty) == 0
    assert autodetect_hdu(empty) == 0  # cached negative answer


def test_autodetect_follows_a_rewrite(tmp_path):
    path = str(tmp_path / "re.fits")
    torchfits.write(path, torch.ones(2, 2))
    assert autodetect_hdu(path) == 0
    HDUList([_empty_primary(), TensorHDU(torch.ones(3, 3))]).write(path, overwrite=True)
    assert autodetect_hdu(path) == 1


@pytest.mark.parametrize("step", [1, 3, 10])
def test_chunks_tile_the_image(tmp_path, step):
    data = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
    in_memory = list(TensorHDU(data).chunks((step, 4)))
    assert torch.equal(torch.cat(in_memory), data)
    assert all(c.shape[0] <= step for c in in_memory)

    path = str(tmp_path / "c.fits")
    torchfits.write(path, data)
    with torchfits.open(path) as hdul:
        from_file = list(hdul[0].chunks((step,)))
    assert [c.shape for c in from_file] == [c.shape for c in in_memory]
    assert torch.equal(torch.cat(from_file), data)


def test_chunks_of_a_cube_follow_the_first_axis(tmp_path):
    data = torch.arange(3 * 4 * 5, dtype=torch.int32).reshape(3, 4, 5)
    path = str(tmp_path / "cube.fits")
    torchfits.write(path, data)
    with torchfits.open(path) as hdul:
        chunks = list(hdul[0].chunks((2, 4, 5)))
    assert [tuple(c.shape) for c in chunks] == [(2, 4, 5), (1, 4, 5)]
    assert torch.equal(torch.cat(chunks), data)
