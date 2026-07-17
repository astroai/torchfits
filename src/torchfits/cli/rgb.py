"""Minimal Lupton+ (2004) asinh RGB using torch only."""

from __future__ import annotations

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
    peak = float(channels.max().item())
    if peak > 0:
        channels = channels / peak
    return torch.clamp(channels, 0.0, 1.0)


def write_rgb_image(path: str, rgb: torch.Tensor) -> None:
    """Write RGB float image as binary PPM (no Pillow / NumPy import)."""
    flat = (
        torch.clamp(rgb, 0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .contiguous()
        .reshape(-1)
    )
    height, width, _ = rgb.shape
    with open(path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(bytes(flat.tolist()))
