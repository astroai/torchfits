"""Signed-byte device path applies FITS BZERO=-128 on host before copy."""

from __future__ import annotations

import torch

from torchfits._io_engine._read_pipeline import _apply_scale_on_device


def test_signed_byte_converts_on_host_then_copies() -> None:
    raw = torch.tensor([0, 128, 255], dtype=torch.uint8)
    out = _apply_scale_on_device(
        raw, scaled=True, bscale=1.0, bzero=-128.0, device="cpu"
    )
    assert out.dtype == torch.int8
    assert out.tolist() == [-128, 0, 127]
