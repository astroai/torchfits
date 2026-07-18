"""Example: Time-domain and time-series workflows (e.g., exoplanet light curves).

Demonstrates:
- Creating a synthetic exoplanet transit light curve (periodic time series).
- Writing the light curve to a FITS table using torchfits.table.write.
- Reading the table columns back as PyTorch Tensors.
- Applying AsymmetricSigmaClip to filter out random outlier spikes.
- Applying PhaseFold to fold the time series onto a uniform phase grid.
- Applying SavitzkyGolayFilter to smooth the folded transit profile.
"""

import os
import tempfile

import numpy as np
import torch

import torchfits
from torchfits.transforms import (
    AsymmetricSigmaClip,
    Compose,
    PhaseFold,
    SavitzkyGolayFilter,
)


def _generate_synthetic_lightcurve(
    n_points: int = 2400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic exoplanet transit light curve.

    Time is in days (hourly samples for 100 days).
    Transit period is 5.0 days, duration is 0.2 days (4.8 hours), depth is 2%.
    Add Gaussian noise and a few outlier spikes (flares/instrumental glitches).
    """
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 100.0, n_points)

    # Out-of-transit flux normalized to 1.0
    flux = np.ones_like(time)

    # Compute phase relative to period=5.0
    period = 5.0
    phase = (time / period) % 1.0

    # Apply box-shaped transit dip
    transit_mask = (phase < 0.02) | (phase > 0.98)
    flux[transit_mask] -= 0.02

    # Add Gaussian noise
    flux_err = np.full_like(time, 0.001)
    flux += rng.normal(0.0, 0.001, size=n_points)

    # Inject random outlier spikes (e.g., stellar flares or cosmic rays)
    spike_indices = rng.choice(n_points, size=10, replace=False)
    flux[spike_indices] += rng.uniform(0.015, 0.03, size=10)

    return time, flux, flux_err


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        # 1. Generate and write the light curve to a FITS table
        time_np, flux_np, err_np = _generate_synthetic_lightcurve()
        print(f"Generated synthetic time series with {len(time_np)} observations.")

        data = {
            "TIME": torch.from_numpy(time_np),
            "FLUX": torch.from_numpy(flux_np),
            "FLUX_ERR": torch.from_numpy(err_np),
        }
        # Save as a binary table
        torchfits.table.write(
            path,
            data,
            header={"EXTNAME": "LIGHTCURVE", "PERIOD": 5.0, "OBJECT": "WASP-12b"},
            overwrite=True,
        )

        # 2. Read the table back as PyTorch tensors
        # read_table returns a dict mapping column names to PyTorch Tensors
        tensors = torchfits.read_table(path, hdu=1)
        flux = tensors["FLUX"]
        print(
            f"\nRead FITS table: OBJECT={torchfits.read_header(path, hdu=1).get('OBJECT')}"
        )
        print(f"  FLUX column shape: {flux.shape}, dtype: {flux.dtype}")

        # 3. Clean up outlier spikes using AsymmetricSigmaClip
        # Spike outliers (flares) are positive, so we use asymmetric clipping
        # n_low=3.0, n_high=3.0. Replaces clipped values with the median.
        clipper = AsymmetricSigmaClip(n_low=3.0, n_high=3.0, dim=(-1,))
        clean_flux = clipper(flux)

        # Let's count how many spikes were clipped
        clipped_mask = clean_flux != flux
        n_clipped = clipped_mask.sum().item()
        print("\nOutlier Rejection (AsymmetricSigmaClip):")
        print(f"  Clipped {n_clipped} outlier spike(s).")

        # 4. Fold the light curve into phase bins
        # Period = 5.0 days, fold into 100 phase bins
        folder = PhaseFold(period=5.0, n_bins=100)
        folded = folder(clean_flux)
        print("\nPhase Folding (PhaseFold):")
        print(f"  Folded flux shape: {folded.shape} (100 phase bins)")

        # 5. Smooth the folded transit profile using Savitzky-Golay filter
        # Window length 11, polynomial order 3
        smoother = SavitzkyGolayFilter(window_length=11, polyorder=3, dim=-1)
        smoothed = smoother(folded)
        print("\nSmoothing (SavitzkyGolayFilter):")
        print(f"  Smoothed folded flux shape: {smoothed.shape}")

        # Composed pipeline: outlier clip -> phase fold -> smooth
        pipeline = Compose(
            [
                AsymmetricSigmaClip(n_low=3.0, n_high=3.0, dim=(-1,)),
                PhaseFold(period=5.0, n_bins=100),
                SavitzkyGolayFilter(window_length=11, polyorder=3, dim=-1),
            ]
        )
        direct_pipeline_flux = pipeline(flux)
        assert torch.allclose(direct_pipeline_flux, smoothed)
        print("\nSuccessfully ran composed time-series pipeline.")
        print(f"  Min folded flux (transit dip center): {smoothed.min().item():.4f}")
        print(f"  Max folded flux (out-of-transit level): {smoothed.max().item():.4f}")

    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
