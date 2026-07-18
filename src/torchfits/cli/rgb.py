"""Minimal Lupton+ (2004) asinh RGB using torch only."""

from __future__ import annotations

import binascii
import struct
import zlib
from typing import Any

import torch


def lupton_rgb(
    r: Any,
    g: Any,
    b: Any,
    *,
    Q: float = 8.0,
    stretch: float = 0.5,
    minimum: float = 0.0,
) -> torch.Tensor:
    """Return float RGB tensor with shape (H, W, 3) in [0, 1]."""
    red = torch.as_tensor(r, dtype=torch.float64)
    green = torch.as_tensor(g, dtype=torch.float64)
    blue = torch.as_tensor(b, dtype=torch.float64)
    intensity = (red + green + blue) / 3.0
    floor = torch.clamp(intensity, min=minimum)
    f_intensity = torch.asinh(Q * floor)
    f_intensity = torch.where(
        f_intensity > 0, f_intensity, torch.ones_like(f_intensity)
    )
    ir = torch.pow(floor, stretch)
    channels = torch.stack(
        (
            torch.asinh(Q * red) / f_intensity * ir,
            torch.asinh(Q * green) / f_intensity * ir,
            torch.asinh(Q * blue) / f_intensity * ir,
        ),
        dim=-1,
    )
    if channels.numel() > 0:
        peak = float(channels.max().item())
        if peak > 0:
            channels = channels / peak
    return torch.clamp(channels, 0.0, 1.0)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_rgb_image(path: str, rgb: torch.Tensor) -> None:
    """Write RGB float image as PNG (stdlib only; no Pillow / NumPy import)."""
    if rgb.dim() != 3 or int(rgb.shape[-1]) != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    height, width, _ = map(int, rgb.shape)
    flat = (
        torch.clamp(rgb, 0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .contiguous()
        .reshape(-1)
    )
    # ponytail: bytes(flat.tolist()) is fine for CLI preview sizes; ctypes from data_ptr if huge
    raw = bytes(flat.tolist())
    row_bytes = width * 3
    scanlines = bytearray((row_bytes + 1) * height)
    for row in range(height):
        offset = row * (row_bytes + 1)
        scanlines[offset] = 0
        start = row * row_bytes
        scanlines[offset + 1 : offset + 1 + row_bytes] = raw[start : start + row_bytes]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)
