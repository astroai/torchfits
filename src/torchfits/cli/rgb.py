"""Minimal Lupton+ (2004) asinh RGB without astropy."""

from __future__ import annotations

from typing import Any

import numpy as np


def lupton_rgb(
    r: Any,
    g: Any,
    b: Any,
    *,
    Q: float = 8.0,
    stretch: float = 0.5,
    minimum: float = 0.0,
) -> np.ndarray:
    """Return float RGB array with shape (H, W, 3) in [0, 1]."""
    red = np.asarray(r, dtype=np.float64)
    green = np.asarray(g, dtype=np.float64)
    blue = np.asarray(b, dtype=np.float64)
    intensity = (red + green + blue) / 3.0
    floor = np.maximum(intensity, minimum)
    f_intensity = np.arcsinh(Q * floor)
    f_intensity = np.where(f_intensity > 0, f_intensity, 1.0)
    ir = np.power(floor, stretch)
    channels = (
        np.arcsinh(Q * red) / f_intensity * ir,
        np.arcsinh(Q * green) / f_intensity * ir,
        np.arcsinh(Q * blue) / f_intensity * ir,
    )
    rgb = np.dstack(channels)
    peak = float(rgb.max())
    if peak > 0:
        rgb = rgb / peak
    return np.clip(rgb, 0.0, 1.0)


def write_rgb_image(path: str, rgb: np.ndarray) -> None:
    """Write RGB float image to PNG (Pillow) or PPM (stdlib fallback)."""
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError:
        _write_ppm(path, arr)
        return
    Image.fromarray(arr, mode="RGB").save(path)


def _write_ppm(path: str, arr: np.ndarray) -> None:
    height, width, _ = arr.shape
    with open(path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(arr.tobytes())
