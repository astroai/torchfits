#!/usr/bin/env python
"""Image transform gallery — before/after PNGs (HorseHead or synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._plotting import (  # noqa: E402
    save_image_before_after,
    save_image_triplet,
)
from examples._sample_data import SampleUnavailable, try_ensure_sample  # noqa: E402

from torchfits.transforms import (  # noqa: E402
    ArcsinhStretch,
    AsymmetricSigmaClip,
    BackgroundSubtract,
    Compose,
    FITSHeaderNormalize,
    FITSHeaderScale,
    GlobalScalarNorm,
    LogStretch,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaClip,
    SqrtStretch,
    ZScaleNormalize,
)


def _load_image() -> torch.Tensor:
    path = try_ensure_sample("horsehead")
    if path is not None:
        import torchfits

        return torchfits.read_tensor(str(path), hdu=0).float()
    # Synthetic fallback (CI / offline).
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="ij"
    )
    return torch.exp(-((xx * 2.2) ** 2 + (yy * 1.4) ** 2) * 4) * 800 + 40


def main() -> int:
    img = _load_image()
    print(f"image shape={tuple(img.shape)} min={img.min():.3g} max={img.max():.3g}")

    stretches = [
        ("arcsinh", ArcsinhStretch(a=0.1)),
        ("log", LogStretch(a=1000.0)),
        ("sqrt", SqrtStretch()),
    ]
    for tag, xf in stretches:
        out = xf(img)
        p = save_image_before_after(img, out, f"image_{tag}", titles=("raw", tag))
        print("wrote", p)

    norms = [
        ("zscale", ZScaleNormalize()),
        ("robust", RobustNormalize()),
        ("minmax", MinMaxNormalize()),
        ("percentile", PercentileClipNormalize(lower_pct=1, upper_pct=99)),
        ("background", BackgroundSubtract()),
        ("global_median", GlobalScalarNorm(stat="median")),
    ]
    for tag, xf in norms:
        out = xf(img.clone())
        p = save_image_before_after(img, out, f"image_{tag}", titles=("raw", tag))
        print("wrote", p)

    clipped = SigmaClip(n_sigma=3.0)(img.clone())
    print("wrote", save_image_before_after(img, clipped, "image_sigma_clip"))
    aclip = AsymmetricSigmaClip(n_low=3.0, n_high=5.0)(img.clone())
    print("wrote", save_image_before_after(img, aclip, "image_asymmetric_sigma_clip"))

    scale = FITSHeaderScale(bscale=0.5, bzero=100.0)
    physical = scale(img.clone())
    print(
        "wrote",
        save_image_before_after(
            img, physical, "image_fits_header_scale", titles=("stored", "BSCALE/BZERO")
        ),
    )
    hdr_norm = FITSHeaderNormalize({"BITPIX": 16, "BSCALE": 1.0, "BZERO": 0.0})
    # Integer-like counts for header normalize path.
    counts = (img / img.max() * 20000).to(torch.float32)
    print(
        "wrote",
        save_image_before_after(
            counts,
            hdr_norm(counts.clone()),
            "image_fits_header_normalize",
            titles=("counts", "normalized"),
        ),
    )

    pipe = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1), ZScaleNormalize()])
    mid = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1)])(img.clone())
    final = pipe(img.clone())
    print(
        "wrote",
        save_image_triplet(
            img,
            mid,
            final,
            "image_compose_pipeline",
            titles=("raw", "bg+arcsinh", "+zscale"),
        ),
    )
    print("gallery_images OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SampleUnavailable as exc:
        print(f"SKIP: {exc}")
        raise SystemExit(0) from exc
