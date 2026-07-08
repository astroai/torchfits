"""Example: ML-friendly FITS image transforms with forward/inverse pipelines.

Demonstrates the :mod:`torchfits.transforms` module with:
- High dynamic range arcsinh stretching (LSST standard)
- Background subtraction
- IRAF zscale normalization
- Percentile clipping
- Pipeline composition with reversible inverse
"""

import os
import tempfile

import numpy as np
from astropy.io import fits

import torchfits
from torchfits.transforms import (
    ArcsinhStretch,
    BackgroundSubtract,
    Compose,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    LogStretch,
    SqrtStretch,
    ZScaleNormalize,
)


def _create_test_file(path: str) -> None:
    """Create a FITS image with high dynamic range (simulated galaxy)."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[-128:128, -128:128]
    # Exponential disk + Gaussian bulge
    disk = 5000 * np.exp(-np.sqrt(x**2 + y**2) / 50)
    bulge = 20000 * np.exp(-(x**2 + y**2) / (2 * 20**2))
    background = rng.normal(100, 5, (256, 256))
    data = (disk + bulge + background).astype(np.float32)
    # Add a few bright sources
    for _ in range(10):
        sx, sy = rng.integers(32, 224, 2)
        data[sy, sx] = rng.uniform(500, 5000)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = "ELECTRONS"
    hdu.writeto(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # Read the image as a torch tensor
        image = torchfits.read_tensor(path, hdu=0)
        print(f"Raw image:  shape={image.shape},  dtype={image.dtype}")
        print(
            f"  min={image.min().item():.4g},  max={image.max().item():.4g},  "
            f"median={image.median().item():.4g}"
        )

        # ----- 1. Individual transforms -----

        # Arcsinh: the gold standard for high-DR astronomy (LSST, SDSS)
        arcsinh = ArcsinhStretch(a=0.1)
        stretched = arcsinh(image)
        print(
            f"\nArcsinhStretch(a=0.1): min={stretched.min().item():.4f}, "
            f"max={stretched.max().item():.4f}"
        )
        restored = arcsinh.inverse(stretched)
        print(f"  Inverse error (max): {(image - restored).abs().max().item():.2e}")

        # Background subtraction
        bgsub = BackgroundSubtract()
        bg_free = bgsub(image)
        print(
            f"\nBackgroundSubtract:  median_before={image.median().item():.2f}, "
            f"median_after={bg_free.median().item():.4f}"
        )

        # ZScale: IRAF auto-contrast
        zscale = ZScaleNormalize(contrast=0.25)
        z_norm = zscale(image)
        print(
            f"\nZScaleNormalize:  min={z_norm.min().item():.4f}, "
            f"max={z_norm.max().item():.4f}"
        )
        z_restored = zscale.inverse(z_norm)
        print(f"  Inverse error (max): {(image - z_restored).abs().max().item():.2e}")

        # Percentile clip
        pclip = PercentileClipNormalize(lower_pct=5, upper_pct=95)
        clipped = pclip(image)
        print(
            f"\nPercentileClipNormalize(5, 95):  min={clipped.min().item():.4f}, "
            f"max={clipped.max().item():.4f}"
        )

        # Robust normalization (median + MAD)
        robust = RobustNormalize()
        robust_norm = robust(image)
        print(
            f"\nRobustNormalize:  median={robust_norm.median().item():.4f}, "
            f"std={robust_norm.std().item():.4f}"
        )

        # Log stretch
        log_stretch = LogStretch(a=100)
        log_norm = log_stretch(image)
        print(
            f"\nLogStretch(a=100):  min={log_norm.min().item():.4f}, "
            f"max={log_norm.max().item():.4f}"
        )

        # Sqrt (Poisson variance stabilizing)
        sqrt_stretch = SqrtStretch()
        sqrt_norm = sqrt_stretch(image)
        print(
            f"\nSqrtStretch:  min={sqrt_norm.min().item():.4f}, "
            f"max={sqrt_norm.max().item():.4f}"
        )

        # MinMax
        mm = MinMaxNormalize()
        mm_norm = mm(image)
        print(
            f"\nMinMaxNormalize:  min={mm_norm.min().item():.4f}, "
            f"max={mm_norm.max().item():.4f}"
        )

        # ----- 2. Composed pipeline with inverse -----
        pipeline = Compose(
            [
                BackgroundSubtract(),
                ArcsinhStretch(a=0.1),
                ZScaleNormalize(contrast=0.3),
            ]
        )
        print(f"\nComposed pipeline:\n{pipeline}")

        preprocessed = pipeline(image)
        print(
            f"  Forward:  min={preprocessed.min().item():.4f}, "
            f"max={preprocessed.max().item():.4f}"
        )

        decoded = pipeline.inverse(preprocessed)
        print(f"  Inverse error (max): {(image - decoded).abs().max().item():.2e}")

        # ----- 3. Use with torchfits.FITSDataset -----
        from torch.utils.data import DataLoader

        dataset = torchfits.FITSDataset(
            path,
            hdu=0,
            transform=pipeline,  # auto-applied in __getitem__
        )
        for sample in DataLoader(dataset, batch_size=1):
            print(
                f"\nFITSDataset + transform pipeline: shape={sample.shape}, "
                f"device={sample.device}"
            )

    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
