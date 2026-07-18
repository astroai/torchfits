"""Example: write a custom reversible ``FITSTransform`` and compose it.

Uses the HorseHead sample when cached, else a synthetic image (always
succeeds — no network dependency for this one).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402
from torchfits.data import FitsImageDataset  # noqa: E402
from torchfits.transforms import BackgroundSubtract, Compose, FITSTransform  # noqa: E402


class ScaleOffset(FITSTransform):
    """Affine transform: ``forward(x) = x * scale + offset``."""

    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x * self.scale + self.offset

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (x - self.offset) / self.scale


def _load_image() -> torch.Tensor:
    path = try_ensure_sample("horsehead")
    if path is not None:
        return torchfits.read_tensor(str(path), hdu=0).float()
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
    )
    return torch.exp(-(xx**2 + yy**2) * 3) * 500 + 20


def main() -> int:
    image = _load_image()
    print(f"image: shape={tuple(image.shape)} mean={image.mean():.2f}")

    xf = ScaleOffset(scale=2.0, offset=-10.0)
    forward = xf(image)
    back = xf.inverse(forward)
    print(f"ScaleOffset: max_err={(image - back).abs().max().item():.2e}")

    pipeline = Compose([BackgroundSubtract(), xf])
    piped = pipeline(image.clone())
    decoded = pipeline.inverse(piped)
    print(
        f"Compose(BackgroundSubtract, ScaleOffset): out mean={piped.mean():.2f} "
        f"round-trip max_err={(image - decoded).abs().max().item():.2e}"
    )

    tmp_dir = tempfile.mkdtemp(prefix="torchfits_custom_transform_")
    try:
        path = os.path.join(tmp_dir, "image_000.fits")
        torchfits.write(path, image, overwrite=True)
        dataset = FitsImageDataset(path, hdu=0, transform=xf)
        transformed = dataset[0][0]
        print(
            f"FitsImageDataset(transform=ScaleOffset): shape={tuple(transformed.shape)}"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
