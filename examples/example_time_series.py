"""Example: Time-domain workflows (e.g., exoplanet light curves).

Demonstrates:
- Creating a synthetic exoplanet transit light curve.
- Writing the light curve to a FITS table using torchfits.table.write.
- Reading the table columns back as PyTorch Tensors.
- Applying AsymmetricSigmaClip to filter out random outlier spikes.
"""

import os
import tempfile

import numpy as np
import torch

import torchfits
from torchfits.transforms import AsymmetricSigmaClip


def _generate_synthetic_lightcurve(
    n_points: int = 2400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic exoplanet transit light curve."""
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 100.0, n_points)

    flux = np.ones_like(time)
    period = 5.0
    phase = (time / period) % 1.0
    transit_mask = (phase < 0.02) | (phase > 0.98)
    flux[transit_mask] -= 0.02

    flux_err = np.full_like(time, 0.001)
    flux += rng.normal(0.0, 0.001, size=n_points)

    spike_indices = rng.choice(n_points, size=10, replace=False)
    flux[spike_indices] += rng.uniform(0.015, 0.03, size=10)

    return time, flux, flux_err


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        time_np, flux_np, err_np = _generate_synthetic_lightcurve()
        print(f"Generated synthetic time series with {len(time_np)} observations.")

        data = {
            "TIME": torch.from_numpy(time_np),
            "FLUX": torch.from_numpy(flux_np),
            "FLUX_ERR": torch.from_numpy(err_np),
        }
        torchfits.table.write(
            path,
            data,
            header={"EXTNAME": "LIGHTCURVE", "PERIOD": 5.0, "OBJECT": "WASP-12b"},
            overwrite=True,
        )

        tensors = torchfits.table.read_torch(path, hdu=1)
        flux = tensors["FLUX"]
        print(
            f"\nRead FITS table: OBJECT={torchfits.read_header(path, hdu=1).get('OBJECT')}"
        )
        print(f"  FLUX column shape: {flux.shape}, dtype: {flux.dtype}")

        clipper = AsymmetricSigmaClip(n_low=3.0, n_high=3.0, dim=(-1,))
        clean_flux = clipper(flux)

        clipped_mask = clean_flux != flux
        n_clipped = clipped_mask.sum().item()
        print("\nOutlier Rejection (AsymmetricSigmaClip):")
        print(f"  Clipped {n_clipped} outlier spike(s).")
        print(f"  Min flux after clip: {clean_flux.min().item():.4f}")
        print(f"  Max flux after clip: {clean_flux.max().item():.4f}")

    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
