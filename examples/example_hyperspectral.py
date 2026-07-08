"""Example: Hyperspectral transforms for spectral data and multi-band images.

Demonstrates the :mod:`torchfits.transforms` hyperspectral-specific tools:

- **SpectralBinning**: reduce spectral resolution by binning adjacent channels
- **ContinuumRemoval**: subtract a polynomial or B-spline baseline
- **BandMath**: compute band ratios (NDVI, WBI, etc.) on multi-spectral data
- **WaveletDecompose**: split spectrum into frequency bands (Haar DWT)
- **SavitzkyGolayFilter**: polynomial smoothing with invertible decomposition

All transforms operate on arbitrary tensor layouts and are compatible with
:class:`~torch.utils.data.DataLoader` pipelines.
"""

import math

import torch

from torchfits.transforms import (
    BandMath,
    ContinuumRemoval,
    SavitzkyGolayFilter,
    SpectralBinning,
    WaveletDecompose,
)


def _synthetic_hyperspectral_cube(
    n_bands: int = 256,
    height: int = 64,
    width: int = 64,
) -> torch.Tensor:
    """Create a synthetic hyperspectral cube with mineral absorption features.

    Returns a tensor of shape ``[n_bands, height, width]`` simulating
    a push-broom hyperspectral image:
    - Each pixel has a reflectance-like spectrum with a continuum shape
    - Absorption bands at specific wavelengths (e.g., iron oxides, clay)
    - Spatial variation in mineral abundance
    """
    # Wavelength grid (e.g. VIS–NIR, 400–2500 nm)
    wavelength = torch.linspace(0.4, 2.5, n_bands)

    # Continuum: quadratic baseline typical of reflectance spectra
    continuum = 0.5 - 0.15 * (wavelength - 1.0) + 0.05 * (wavelength - 1.0) ** 2

    # Absorption features: Gaussian dips at specific mineral bands
    features = [
        {"center": 0.65, "width": 0.03, "depth": 0.12},  # Fe³⁺ (hematite)
        {"center": 0.90, "width": 0.04, "depth": 0.08},  # Fe²⁺ (pyroxene)
        {"center": 1.40, "width": 0.05, "depth": 0.15},  # OH⁻ / H₂O
        {"center": 1.90, "width": 0.06, "depth": 0.18},  # H₂O (clay)
        {"center": 2.20, "width": 0.04, "depth": 0.20},  # Al-OH (kaolinite)
        {"center": 2.35, "width": 0.03, "depth": 0.10},  # CO₃²⁻ (carbonate)
    ]

    # Build 1D spectrum
    spectrum = continuum.clone()
    for f in features:
        spectrum -= f["depth"] * torch.exp(
            -((wavelength - f["center"]) ** 2) / (2 * f["width"] ** 2)
        )

    # Create spatial variation: each pixel has a different mineral mixture
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing="ij",
    )

    # Abundance maps (smooth spatial gradients)
    abundance_iron = 0.3 + 0.7 * torch.sigmoid(3 * x_grid)  # left → right
    abundance_clay = 0.2 + 0.8 * torch.sigmoid(-3 * y_grid)  # top → bottom
    abundance_water = 0.1 + 0.4 * (1 + torch.cos(math.pi * x_grid))  # centre peak

    # Assemble cube: each pixel's spectrum = continuum + abundance-weighted absorptions
    cube = continuum.unsqueeze(-1).unsqueeze(-1).expand(-1, height, width).clone()

    # Fe³⁺ absorption: stronger where iron abundance is high
    iron_dip = features[0]["depth"] * torch.exp(
        -((wavelength - features[0]["center"]) ** 2) / (2 * features[0]["width"] ** 2)
    ).unsqueeze(-1).unsqueeze(-1)
    cube -= iron_dip * abundance_iron.unsqueeze(0)

    # H₂O absorption: stronger where water abundance is high
    water_dip = features[3]["depth"] * torch.exp(
        -((wavelength - features[3]["center"]) ** 2) / (2 * features[3]["width"] ** 2)
    ).unsqueeze(-1).unsqueeze(-1)
    cube -= water_dip * abundance_water.unsqueeze(0)

    # Al-OH absorption: stronger where clay abundance is high
    clay_dip = features[4]["depth"] * torch.exp(
        -((wavelength - features[4]["center"]) ** 2) / (2 * features[4]["width"] ** 2)
    ).unsqueeze(-1).unsqueeze(-1)
    cube -= clay_dip * abundance_clay.unsqueeze(0)

    # Add a small amount of noise
    cube += torch.randn_like(cube) * 0.005

    return cube.float()


def main() -> None:
    # ------------------------------------------------------------------
    # Create a synthetic hyperspectral cube
    # ------------------------------------------------------------------
    cube = _synthetic_hyperspectral_cube(n_bands=256, height=64, width=64)
    print(f"Hyperspectral cube: shape={cube.shape}, dtype={cube.dtype}")
    print(
        f"  min={cube.min().item():.4f}, max={cube.max().item():.4f}, "
        f"mean={cube.mean().item():.4f}"
    )

    # ------------------------------------------------------------------
    # 1. SpectralBinning — reduce spectral resolution
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. SpectralBinning")
    print("=" * 60)

    binner = SpectralBinning(factor=4, mode="mean", dim=0)
    binned = binner(cube)
    print("  Binned 256→{} bands (factor=4, mean)".format(binned.shape[0]))
    print(
        "    Binned:  min={:.4f}, max={:.4f}".format(
            binned.min().item(), binned.max().item()
        )
    )

    # Demonstrate invertibility: nearest-neighbour upsample
    restored = binner.inverse(binned)
    print(f"  Inverse restored shape: {restored.shape}")
    # The restored cube is block-constant; verify per-bin mean conservation
    bin_check = restored[:252].reshape(63, 4, 64, 64).mean(dim=1)
    err = (bin_check - binned[:63]).abs().max().item()
    print(f"  Bin-mean conservation error: {err:.2e}")

    # ------------------------------------------------------------------
    # 2. ContinuumRemoval — polynomial and spline baseline subtraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. ContinuumRemoval")
    print("=" * 60)

    # Extract a single-pixel spectrum for demonstration
    pixel_00 = cube[:, 0, 0]  # [256]
    pixel_32 = cube[:, 32, 32]  # [256] — different mineral mix

    # Polynomial continuum removal
    poly_remover = ContinuumRemoval(method="polynomial", order=2, n_sigma=2.0)
    residual_00 = poly_remover(pixel_00.unsqueeze(0))  # [1, 256]
    print(
        f"  Polynomial (order=2): continuum-subtracted pixel (0,0) — "
        f"range [{residual_00.min().item():.4f}, {residual_00.max().item():.4f}]"
    )

    # Verify roundtrip
    roundtrip = poly_remover.inverse(residual_00)
    err_poly = (roundtrip - pixel_00.unsqueeze(0)).abs().max().item()
    print(f"  Roundtrip error (polynomial): {err_poly:.2e}")

    # Spline continuum removal on the other pixel
    spline_remover = ContinuumRemoval(method="spline", n_knots=12, n_sigma=2.0)
    residual_32 = spline_remover(pixel_32.unsqueeze(0))
    print(
        "  Spline (n_knots=12): continuum-subtracted pixel (32,32) — "
        "range [{:.4f}, {:.4f}]".format(
            residual_32.min().item(), residual_32.max().item()
        )
    )

    roundtrip_s = spline_remover.inverse(residual_32)
    err_spline = (roundtrip_s - pixel_32.unsqueeze(0)).abs().max().item()
    print(f"  Roundtrip error (spline): {err_spline:.2e}")

    # Demonstrate the difference: ContinuumRemoval (subtracts) vs.
    # ContinuumNormalize (divides)
    from torchfits.transforms import ContinuumNormalize

    norm = ContinuumNormalize(order=2)
    normalized_00 = norm(pixel_00.unsqueeze(0))
    restored_norm = norm.inverse(normalized_00)
    print(
        "\n  ContinuumNormalize (divides by continuum, compare to Removal which"
        " subtracts):"
    )
    print(
        f"    Normalized range: [{normalized_00.min().item():.4f}, "
        f"{normalized_00.max().item():.4f}]"
    )
    print(
        "    Roundtrip error: "
        f"{(restored_norm - pixel_00.unsqueeze(0)).abs().max().item():.2e}"
    )

    # ------------------------------------------------------------------
    # 3. BandMath — NDVI-style band ratios
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. BandMath")
    print("=" * 60)

    # Create a simple 3-band "satellite" image (simulating R, G, NIR)
    # Bands along dim=0: [Red, Green, NIR] at spatial resolution 128x128
    h, w = 128, 128
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )

    # Red: vegetation has low reflectance
    red = 0.1 + 0.05 * torch.cos(math.pi * x) * torch.cos(math.pi * y)
    # NIR: vegetation has high reflectance
    nir = 0.5 + 0.20 * torch.cos(math.pi * x) * torch.cos(math.pi * y)
    # Green: intermediate
    green = 0.3 + 0.10 * torch.cos(math.pi * x) * torch.cos(math.pi * y)

    rgb_nir = torch.stack([red, green, nir], dim=0)  # [3, 128, 128]
    print(f"  3-band image: shape={rgb_nir.shape}  (bands at dim=0: [R, G, NIR])")

    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = BandMath(
        lambda b: (b[2] - b[0]) / (b[2] + b[0] + 1e-8),
        band_dim=0,
    )
    ndvi_result = ndvi(rgb_nir)
    print(
        "  NDVI: shape={}, range=[{:.4f}, {:.4f}]".format(
            ndvi_result.shape, ndvi_result.min().item(), ndvi_result.max().item()
        )
    )
    # NDVI for "vegetation" (centre) should be positive
    print(f"    centre pixel NDVI: {ndvi_result[64, 64].item():.4f} (vegetation)")
    # NDVI for "bare soil" (corner) should be near 0
    print(f"    corner pixel NDVI: {ndvi_result[0, 0].item():.4f} (bare soil)")

    # GNDVI: (NIR - Green) / (NIR + Green) — chlorophyll-sensitive variant
    gndvi = BandMath(
        lambda b: (b[2] - b[1]) / (b[2] + b[1] + 1e-8),
        band_dim=0,
    )
    gndvi_result = gndvi(rgb_nir)
    print(
        "  GNDVI: shape={}, range=[{:.4f}, {:.4f}]".format(
            gndvi_result.shape, gndvi_result.min().item(), gndvi_result.max().item()
        )
    )

    # ------------------------------------------------------------------
    # 4. Composed pipeline: bin → remove continuum → band ratio
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. Composed pipeline: bin → continuum removal on extracted spectra")
    print("=" * 60)

    # Note: ContinuumRemoval expects spectra along dim=-1.
    # For a hyperspectral cube [bands, H, W] (bands at dim=0), first bin
    # spectrally, then apply per-pixel continuum removal by moving the
    # spectral axis to the last position.
    binner8 = SpectralBinning(factor=8, mode="mean", dim=0)
    binned_cube = binner8(cube)  # [32, 64, 64], bands still at dim=0

    # Permute so bands become the last dim for ContinuumRemoval
    panel = binned_cube.movedim(0, -1)  # [64, 64, 32]
    remover = ContinuumRemoval(method="polynomial", order=2)
    residual_panel = remover(panel)  # continuum-subtracted per pixel
    print("  Pipeline: bin(256→32) → permute → continuum removal per pixel")
    print(f"    Result shape: {residual_panel.shape}  ([H, W, bands])")
    print(
        f"    Range: [{residual_panel.min().item():.4f}, "
        f"{residual_panel.max().item():.4f}]"
    )
    print(
        f"    Mean (should be ~0 after subtraction): {residual_panel.mean().item():.4f}"
    )

    # Inverse: add continuum back, permute back, upsample
    panel_restored = remover.inverse(residual_panel)
    binned_restored = panel_restored.movedim(-1, 0)  # back to [32, 64, 64]
    cube_restored = binner8.inverse(binned_restored)  # [256, 64, 64]
    trimmed = 256 - (256 % 8)
    err_cube = (cube[:trimmed] - cube_restored[:trimmed]).abs().max().item()
    print(f"    Inverse error (binned region): {err_cube:.2e}")

    # ------------------------------------------------------------------
    # 5. WaveletDecompose — frequency split into approx + detail bands
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. WaveletDecompose")
    print("=" * 60)

    # Decompose a single-pixel spectrum into 3 levels
    spec_00 = cube[:, 0, 0]  # [256]
    wavelet = WaveletDecompose(levels=3, dim=-1)

    # Forward: stacked [approx_3, detail_3, detail_2, detail_1]
    coeffs = wavelet(spec_00.unsqueeze(0))  # [1, 256]
    # The first N/8 elements are the level-3 approximation (broadband continuum),
    # the remaining are detail coefficients at three scales.
    approx_len = spec_00.shape[-1] >> 3  # 32
    approx_energy = (coeffs[0, :approx_len] ** 2).sum().item()
    detail_energy = (coeffs[0, approx_len:] ** 2).sum().item()
    print("  WaveletDecompose (levels=3, Haar):")
    print(
        "    Approximation (low-freq / continuum) energy: {:.4f}".format(approx_energy)
    )
    print(
        "    Detail (high-freq / features) energy:       {:.4f}".format(detail_energy)
    )
    # The continuum (quadratic baseline) dominates the approximation
    assert approx_energy > detail_energy, (
        "Broadband continuum should dominate approximation"
    )

    # Verify perfect reconstruction
    restored_spec = wavelet.inverse(coeffs)
    err_wavelet = (restored_spec - spec_00.unsqueeze(0)).abs().max().item()
    print("    Roundtrip error: {:.2e}".format(err_wavelet))

    # Demonstrate inverse: reconstructing from approx-only (discarding details)
    # fills in the absorption features with broadband continuum
    coeffs_zero_detail = coeffs.clone()
    coeffs_zero_detail[0, approx_len:] = 0.0
    continuum_only = wavelet.inverse(coeffs_zero_detail)
    print("    Continuum-only reconstruction (details zeroed):")
    print(
        "      range [{:.4f}, {:.4f}]".format(
            continuum_only.min().item(), continuum_only.max().item()
        )
    )

    # ------------------------------------------------------------------
    # 6. SavitzkyGolayFilter — polynomial smoothing with residual channel
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. SavitzkyGolayFilter")
    print("=" * 60)

    # Add noise to the spectrum to demonstrate smoothing
    spec_noisy = spec_00 + torch.randn_like(spec_00) * 0.01
    sg = SavitzkyGolayFilter(window_length=11, polyorder=3, dim=-1)
    spec_smooth = sg(spec_noisy.unsqueeze(0))  # [1, 256]
    print("  SG filter (window=11, polyorder=3):")
    # Smoothing should reduce std (noise suppression)
    noisy_std = spec_noisy.std().item()
    smooth_std = spec_smooth.std().item()
    print("    Noisy spectrum std:    {:.5f}".format(noisy_std))
    print("    Smoothed spectrum std: {:.5f}".format(smooth_std))
    assert smooth_std < noisy_std, "SG filter should reduce noise"

    # Verify invertibility via additive decomposition
    spec_restored = sg.inverse(spec_smooth)
    err_sg = (spec_restored - spec_noisy.unsqueeze(0)).abs().max().item()
    print("    Roundtrip error: {:.2e}".format(err_sg))

    print("\nDone — all transforms compatible with torch.utils.data.DataLoader.")


if __name__ == "__main__":
    main()
