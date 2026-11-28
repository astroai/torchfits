"""One table device move: tensors and VLA row lists, batched per column.

The read cache, the fallback reader, ``read_torch``, ``scan_torch`` and the
datasets each carried their own copy of the column-dict device move (P12);
they now share ``device.move_table_columns``. VLA columns (lists of row
tensors) used to cost one host-to-device copy per row; runs of same-dtype CPU
rows now go over in one transfer and are split back on the device.

The "meta" device stands in for an accelerator: it exercises the real
``Tensor.to`` path without needing CUDA or MPS.
"""

from __future__ import annotations

import torch

from torchfits._io_engine import device as device_mod
from torchfits._io_engine.device import move_table_columns


def _count_transfers(monkeypatch) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    real = device_mod.to_device

    def spy(tensor, device, *, non_blocking=False):
        shapes.append(tuple(tensor.shape))
        return real(tensor, device, non_blocking=non_blocking)

    monkeypatch.setattr(device_mod, "to_device", spy)
    return shapes


def test_cpu_device_returns_input_unchanged():
    data = {"A": torch.arange(3), "V": [torch.arange(2)]}
    assert move_table_columns(data, "cpu") is data


def test_vla_rows_move_in_one_transfer_and_keep_their_shapes(monkeypatch):
    shapes = _count_transfers(monkeypatch)
    rows = [torch.arange(n, dtype=torch.float32) for n in (3, 0, 5, 1)]
    rows.append(torch.ones(2, 2))
    data = {
        "A": torch.arange(5),
        "V": rows,
        "S": ["a", "b", "c", "d", "e"],
        "K": 7,
    }
    out = move_table_columns(data, "meta")
    assert out["A"].device.type == "meta"
    assert [t.device.type for t in out["V"]] == ["meta"] * 5
    assert [tuple(t.shape) for t in out["V"]] == [(3,), (0,), (5,), (1,), (2, 2)]
    assert out["S"] == data["S"] and out["K"] == 7
    # One transfer for "A", one for the whole VLA column (13 elements).
    assert shapes == [(5,), (13,)]


def test_mixed_rows_fall_back_per_run(monkeypatch):
    shapes = _count_transfers(monkeypatch)
    rows = [
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
        None,
        torch.zeros(4, dtype=torch.float64),
    ]
    out = move_table_columns({"V": rows}, "meta")
    assert out["V"][2] is None
    assert [t.dtype for t in (out["V"][0], out["V"][1], out["V"][3])] == [
        torch.int32,
        torch.int32,
        torch.float64,
    ]
    assert shapes == [(5,), (4,)]


def test_vla_transfer_is_bounded_by_the_batch_budget(monkeypatch):
    monkeypatch.setattr(device_mod, "_ROW_LIST_TRANSFER_BYTES", 16)
    shapes = _count_transfers(monkeypatch)
    rows = [torch.zeros(2, dtype=torch.float32) for _ in range(5)]
    out = move_table_columns({"V": rows}, "meta")
    assert len(out["V"]) == 5
    assert shapes == [(4,), (4,), (2,)]


def test_moved_rows_match_values_on_a_real_copy(monkeypatch):
    # Split views are rebuilt from the flat buffer: check values survive by
    # routing the transfer through a dtype-preserving CPU clone.
    real = device_mod.to_device
    monkeypatch.setattr(
        device_mod,
        "to_device",
        lambda t, device, *, non_blocking=False: real(t, "cpu").clone(),
    )
    rows = [torch.arange(n) * 10 for n in (1, 4, 2)]
    out = move_table_columns({"V": rows}, "cuda")
    for got, want in zip(out["V"], rows):
        assert torch.equal(got, want)


def test_dataset_table_chunks_move_vla_rows_with_their_siblings():
    """The map-style dataset's chunk move left VLA rows on the CPU."""
    from torchfits import data as data_mod

    chunk = data_mod._move_table_chunk(
        {"A": torch.arange(3), "V": [torch.arange(n) for n in (1, 2, 3)]}, "meta"
    )
    assert chunk["A"].device.type == "meta"
    assert [t.device.type for t in chunk["V"]] == ["meta"] * 3
