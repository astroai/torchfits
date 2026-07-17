# Transform gallery

Before/after figures for `torchfits.transforms`. Run the galleries locally
(writes PNGs under `examples/output/`):

```bash
pixi run python examples/gallery_images.py
pixi run python examples/gallery_spectra.py
pixi run python examples/gallery_tables_lc.py
```

Public samples (HorseHead, SDSS spectrum, Chandra events) download once into
`~/.cache/torchfits/samples/`. With `TORCHFITS_EXAMPLE_FAST=1`, galleries use
synthetic fallbacks when the cache is empty (CI).

API formulas: [Transforms reference](api-transforms.md).

---

## Images (HorseHead)

Astropy [FITS images](https://learn.astropy.org/tutorials/FITS-images.html)
pattern: open → stretch / normalize → inspect.

![Arcsinh stretch](assets/gallery/image_arcsinh.png)

![ZScale normalize](assets/gallery/image_zscale.png)

![Compose: background → arcsinh → zscale](assets/gallery/image_compose_pipeline.png)

Also covered by `gallery_images.py`: `LogStretch`, `SqrtStretch`,
`RobustNormalize`, `MinMaxNormalize`, `PercentileClipNormalize`,
`BackgroundSubtract`, `GlobalScalarNorm`, `SigmaClip`,
`AsymmetricSigmaClip`, `FITSHeaderScale`, `FITSHeaderNormalize`.

---

## Spectra and continuum (SDSS / synthetic)

Specutils-style continuum workflow: flux → continuum estimate → normalize or
subtract → plot.

![Continuum normalize](assets/gallery/spectrum_continuum_normalize.png)

![Continuum removal](assets/gallery/spectrum_continuum_removal.png)

![Doppler shift](assets/gallery/spectrum_doppler_shift.png)

`gallery_spectra.py` also demos continuum estimators
(`SavitzkyGolayFilter`, `RunningPercentile`, `UpperEnvelopeContinuum`,
`AsymmetricLeastSquares`, `AlphaShapeContinuum`, `WaveletDecompose`),
`SpectralBinning`, and `BandMath` (NDVI-like on a small cube).

!!! note "SpectralBinning limits"
    Binning is adjacent-channel mean/sum — not full flux-conserving resample
    onto an arbitrary wavelength grid (specutils `FluxConservingResampler`).

---

## Light curves

![Phase fold](assets/gallery/lightcurve_phase_fold.png)

`gallery_tables_lc.py` also covers `SigmaClip` / `AsymmetricSigmaClip`,
Savitzky–Golay on the folded curve, plus table `FITSScaleColumns` /
`TNullToNan`.

---

## Cubes

See [`examples/example_hyperspectral.py`](../examples/example_hyperspectral.py)
for multi-band cubes (binning, continuum, band math) with optional figure
output under `examples/output/`.
