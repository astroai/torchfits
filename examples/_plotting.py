"""Before/after plot helpers for the transform gallery."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path


def _note_skip(name: str) -> None:
    print(f"skipping figures: matplotlib not installed ({name})")


def save_image_before_after(
    raw: torch.Tensor | np.ndarray,
    transformed: torch.Tensor | np.ndarray,
    name: str,
    *,
    titles: Sequence[str] = ("input", "transformed"),
    cmap: str = "gray",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _note_skip(name)
        return None

    a = _to_numpy(raw)
    b = _to_numpy(transformed)
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-2], a.shape[-1])[0]
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-2], b.shape[-1])[0]

    path = output_path(name)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, data, title in zip(axes, (a, b), titles, strict=True):
        finite = data[np.isfinite(data)]
        vmin = float(np.percentile(finite, 1)) if finite.size else None
        vmax = float(np.percentile(finite, 99)) if finite.size else None
        ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_1d_before_after(
    wave: torch.Tensor | np.ndarray | None,
    flux: torch.Tensor | np.ndarray,
    transformed: torch.Tensor | np.ndarray,
    name: str,
    *,
    titles: Sequence[str] = ("flux", "transformed"),
) -> Path | None:
    """1-D before/after plot (table columns or image slices)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _note_skip(name)
        return None

    y0 = _to_numpy(flux).reshape(-1)
    y1 = _to_numpy(transformed).reshape(-1)
    x0 = np.arange(y0.size) if wave is None else _to_numpy(wave).reshape(-1)[: y0.size]
    x1 = np.arange(y1.size) if wave is None or y1.size != y0.size else x0[: y1.size]
    path = output_path(name)

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 4.4), sharex=False, constrained_layout=True
    )
    axes[0].plot(x0, y0, color="#1f4e79", lw=0.8)
    axes[0].set_ylabel(titles[0])
    axes[1].plot(x1, y1, color="#1f4e79", lw=0.8)
    axes[1].set_ylabel(titles[1])
    axes[1].set_xlabel("index / pixel")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_lightcurve_before_after(
    time: torch.Tensor | np.ndarray,
    flux: torch.Tensor | np.ndarray,
    transformed_x: torch.Tensor | np.ndarray,
    transformed_y: torch.Tensor | np.ndarray,
    name: str,
    *,
    titles: Sequence[str] = ("light curve", "transformed"),
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _note_skip(name)
        return None

    t0 = _to_numpy(time).reshape(-1)
    y0 = _to_numpy(flux).reshape(-1)
    t1 = _to_numpy(transformed_x).reshape(-1)
    y1 = _to_numpy(transformed_y).reshape(-1)
    path = output_path(name)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)
    axes[0].plot(t0, y0, ".", ms=2, color="#1f4e79")
    axes[0].set_title(titles[0])
    axes[0].set_xlabel("time")
    axes[1].plot(t1, y1, ".", ms=2, color="#c45c26")
    axes[1].set_title(titles[1])
    axes[1].set_xlabel("phase" if t1.max() <= 1.5 else "time")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_image_triplet(
    raw: torch.Tensor | np.ndarray,
    mid: torch.Tensor | np.ndarray,
    final: torch.Tensor | np.ndarray,
    name: str,
    *,
    titles: Sequence[str] = ("raw", "mid", "final"),
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _note_skip(name)
        return None

    panels = [_to_numpy(x) for x in (raw, mid, final)]
    for i, p in enumerate(panels):
        if p.ndim > 2:
            panels[i] = p.reshape(-1, p.shape[-2], p.shape[-1])[0]
    path = output_path(name)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, data, title in zip(axes, panels, titles, strict=True):
        finite = data[np.isfinite(data)]
        vmin = float(np.percentile(finite, 1)) if finite.size else None
        vmax = float(np.percentile(finite, 99)) if finite.size else None
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
