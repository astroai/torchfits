# Transforms

Header-aware preprocessing for FITS images, spectra, and tables.

## When to use

- High-dynamic-range **visualization** (arcsinh, zscale, log/sqrt stretches)
- **Model input** scaling that you want reusable across a Dataset
- FITS **BSCALE / null** hygiene (`FITSHeaderScale`, `TNullToNan`, column scale)

## When not to

- You need raw ADU / physical values as stored — use `read_tensor` / `table.read`
- One-off arithmetic on a single tensor — plain PyTorch is enough
- Catalog filtering — use `table.read(..., where=)` (C++ pushdown), not transforms

Wire a pipeline into training with `FitsTensorDataset(..., transform=pipeline)`
or call it on a tensor from `read_tensor`. See [Data module](api-data.md) for
when to introduce a Dataset / `make_loader`.

!!! note "1.0 transform boundary"
    **Core (kept):** stretches, zscale / robust norms, FITS BSCALE / null
    hygiene, basic continuum divide/subtract.
    **Advanced (frozen for 1.0):** ALS / alpha-shape / BandMath / PhaseFold /
    wavelet and specialty continuum estimators stay in-tree but are not expanded
    this release — candidates to move to torchsky later. Do not treat the
    advanced set as a growing public surface mid-rc.

!!! note "RGB"
    1.0 ships Lupton asinh RGB via `torchfits.transforms.lupton_rgb` (same as
    `torchfits convert … --to png`). Richer multi-band variants → **1.1**.

All transforms implement the `FITSTransform` callable protocol
(`forward` / `inverse` / `__call__`). They are **not**
`torch.nn.Module` subclasses — use them as callables with
`torch.utils.data.Dataset` / `DataLoader`. Inverse state is
**instance-local** (`_last_*` fields); construct one pipeline per worker
when `num_workers > 0`.

```python
from torchfits.transforms import ArcsinhStretch, BackgroundSubtract, Compose, ZScaleNormalize

pipeline = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1), ZScaleNormalize()])
normalized = pipeline(image)
restored = pipeline.inverse(normalized)
```
### Masks

Most transforms accept an optional boolean `mask` (`True` = valid).
`Compose` forwards the same mask to every child:

```python
pipeline(image, mask=finite_mask)
pipeline.inverse(normalized, mask=finite_mask)
```
### Invertibility

| Kind | `inverse()` |
|---|---|
| Stretches, normalizers, FITS scale, continuum divide/subtract, baseline estimators, wavelet | Yes (state cached on the instance) |
| `BandMath`, `PhaseFold`, `SigmaClip`, `AsymmetricSigmaClip`, `TNullToNan` | No — lossy / many-to-one |

Stateless stretches are the most likely to work under `torch.compile`;
data-dependent normalizers cache Python-side state and may graph-break.
There is no certified compile matrix yet.

The implementation lives under `torchfits.transforms` as a small package
(`stretch`, `normalize`, `fits_meta`, `spectral`, `continuum`,
`clip`) re-exported from `torchfits.transforms`.

---

## Stretches

Stateless, always invertible. Apply non-linear flux scaling for visualization.

### `ArcsinhStretch(a=1.0)`

Lupton+ (2004) arcsinh stretch — LSST/SDSS standard.

$$\text{output} = \frac{\operatorname{arcsinh}(a \cdot x)}{\operatorname{arcsinh}(a)}$$

**Inverse:** $x = \frac{\sinh(\text{output} \cdot \operatorname{arcsinh}(a))}{a}$

| Param | Default | Description |
|---|---|---|
| `a` | `1.0` | Softening parameter — smaller values are more linear near zero |

!!! info "When to use"
    The default choice for astronomical image display. Handles the huge dynamic
    range of sky images gracefully — stars and galaxies both remain visible.

### `LogStretch(a=1000.0, eps=1e-9)`

Logarithmic stretch for heavy-tailed flux distributions.

$$\text{output} = \frac{\log_{10}(1 + a \cdot \max(x, 0))}{\log_{10}(1 + a)}$$

**Inverse:** $x = \frac{10^{\text{output} \cdot \log_{10}(1 + a)} - 1}{a}$

| Param | Default | Description |
|---|---|---|
| `a` | `1000.0` | Scale factor before log — larger compresses low flux more gently |
| `eps` | `1e-9` | Floor to prevent log(0) |

!!! info "When to use"
    When you need stronger compression than arcsinh, e.g., for very wide-field
    images with bright stars and faint diffuse emission.

### `SqrtStretch()`

Square-root stretch — stabilizes Poisson variance.

$$\text{output} = \sqrt{\max(x, 0)}$$

**Inverse:** $x = \text{output}^2$

!!! info "When to use"
    Quick and simple. Good default for photon-counting data where Poisson
    statistics apply.

---

## Normalizers

Data-dependent, invertible. Compute statistics from the image and cache them
for inverse transforms.

### `ZScaleNormalize(contrast=0.25, dim=(-2, -1))`

IRAF zscale auto-contrast algorithm. Maps display range to [0, 1].

$$z_1 = \text{median} - \frac{\text{MAD} \times 1.4826}{\max(\text{contrast}, 10^{-5})}$$

$$z_2 = \text{median} + \frac{\text{MAD} \times 1.4826}{\max(\text{contrast}, 10^{-5})}$$

Both clamped to $[x_{\min}, x_{\max}]$.

$$\text{output} = \frac{x - z_1}{z_2 - z_1}$$

**Inverse:** $x = \text{output} \cdot (z_2 - z_1) + z_1$

| Param | Default | Description |
|---|---|---|
| `contrast` | `0.25` | Controls how aggressively to trim outliers — smaller = tighter range |
| `dim` | `(-2, -1)` | Dimensions for statistics |

!!! info "When to use"
    The standard choice for astronomical image display. Automatically adapts to
    the dynamic range of the data. Use for quick visualization or when you want
    a [0, 1] normalized image that preserves relative contrast.

### `RobustNormalize(dim=(-2, -1))`

Subtract median, divide by MAD-derived standard deviation.

$$\text{output} = \frac{x - \text{median}(x)}{\max(\text{MAD} \times 1.4826,\ 10^{-9})}$$

where $\text{MAD} = \text{median}(|x - \text{median}(x)|)$.

**Inverse:** $x = \text{output} \cdot \text{std\_approx} + \text{median}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for statistics |

!!! info "When to use"
    Universal ML preprocessing. Produces zero-mean, unit-variance-like data
    robust to outliers. Use as the last step before feeding to a neural
    network.

### `BackgroundSubtract(dim=(-2, -1))`

Subtract median background level.

$$\text{output} = x - \text{median}(x)$$

**Inverse:** $x = \text{output} + \text{median}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for background estimation |

!!! info "When to use"
    First step in most image pipelines. Removes the constant sky background
    before stretching or normalization.

### `PercentileClipNormalize(lower_pct=1, upper_pct=99, dim=(-2, -1))`

Clip to percentile range, scale to [0, 1].

$$\text{lower} = Q_{\text{lower\_pct}/100}(x), \quad \text{upper} = Q_{\text{upper\_pct}/100}(x)$$

$$\text{output} = \frac{\text{clamp}(x,\ \text{lower},\ \text{upper}) - \text{lower}}{\text{upper} - \text{lower}}$$

**Inverse:** $x = \text{output} \cdot (\text{upper} - \text{lower}) + \text{lower}$

| Param | Default | Description |
|---|---|---|
| `lower_pct` | `1.0` | Lower percentile |
| `upper_pct` | `99.0` | Upper percentile |
| `dim` | `(-2, -1)` | Dimensions for quantile computation |

!!! info "When to use"
    More aggressive than zscale — hard-clips outliers. Good for display when
    you know the percentile range of "interesting" data.

### `MinMaxNormalize(dim=(-2, -1))`

Min-max normalization to [0, 1] with ULP-safe epsilon.

$$\text{output} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

**Inverse:** $x = \text{output} \cdot (v_{\max} - v_{\min}) + v_{\min}$

| Param | Default | Description |
|---|---|---|
| `dim` | `(-2, -1)` | Dimensions for min/max |

!!! info "When to use"
    Simple normalization when you know the data has no outliers. Avoid for
    astronomical images — a single bright star dominates the range.

### `GlobalScalarNorm(stat="median", dim=None)`

Divide by a single scalar statistic. Minimal linear prep.

$$\text{output} = \frac{x}{\max(\text{scalar},\ 10^{-30})}$$

where scalar is one of: `median(x)`, `max(x)`, `mean(x)`, or
$\sqrt{\text{mean}(x^2)}$ (RMS).

**Inverse:** $x = \text{output} \times \text{scalar}$

| Param | Default | Description |
|---|---|---|
| `stat` | `"median"` | `"median"`, `"max"`, `"mean"`, or `"rms"` |
| `dim` | `None` | Dimensions (None = all) |

!!! info "When to use"
    When you want the simplest possible normalization — just scale by the
    typical value. Good for quick comparisons between spectra.

---

## Spectral Transforms

1D astronomy-specific transforms for spectra and hyperspectral cubes.

### `ContinuumNormalize(order=3, n_sigma=2.0, max_iter=3)`

Fit polynomial continuum with sigma-clipping, then **divide** spectrum by it.

1. Build Vandermonde matrix $A$ of size $[L, \text{order}+1]$ on $t \in [-1, 1]$.
2. Iteratively solve $(A^T W A + \lambda I) c = A^T W y$ with sigma-clipped weights.
3. Continuum $= Ac$

$$\text{output} = \frac{x}{\max(|\text{continuum}|,\ 10^{-30})}$$

**Inverse:** $x = \text{output} \times \text{continuum}$

| Param | Default | Description |
|---|---|---|
| `order` | `3` | Polynomial order |
| `n_sigma` | `2.0` | Sigma-clipping threshold |
| `max_iter` | `3` | Max clipping iterations |

!!! info "When to use"
    Standard step in stellar/galaxy spectroscopy. Removes the broad continuum
    shape so absorption/emission features can be analyzed. Use before
    measuring equivalent widths or feeding spectra to ML models.

### `ContinuumRemoval(method="polynomial", order=3, n_knots=10)`

Fit continuum and **subtract** it (additive decomposition).

- `method="polynomial"`: same polynomial fit as `ContinuumNormalize`.
- `method="spline"`: cubic B-spline via $(B^T W B + \lambda I)c = B^T W y$.

$$\text{output} = x - \text{baseline}$$

**Inverse:** $x = \text{output} + \text{baseline}$

| Param | Default | Description |
|---|---|---|
| `method` | `"polynomial"` | `"polynomial"` or `"spline"` |
| `order` | `3` | Polynomial order (if polynomial) |
| `n_knots` | `10` | B-spline knots (if spline) |
| `n_sigma` | `2.0` | Sigma-clipping threshold |
| `max_iter` | `3` | Max iterations |

!!! info "When to use"
    Use when you want to isolate spectral features by removing the baseline
    additively (unlike `ContinuumNormalize` which divides). Better for
    emission-line spectra where you want to measure line fluxes.

### `DopplerShift(z=0.0)`

Redshift/blueshift via linear interpolation resampling.

$$L_{\text{new}} = \max(2,\ \lfloor L \cdot (1 + z) \rfloor)$$

Resamples the last dimension by factor $(1 + z)$ using `F.interpolate`.

**Inverse:** Applies opposite shift $1/(1+z)$.

| Param | Default | Description |
|---|---|---|
| `z` | `0.0` | Redshift (+ = redshift, − = blueshift) |

!!! info "When to use"
    Data augmentation for spectral ML — randomly shift spectra to improve
    redshift robustness. Also used to correct known redshifts.

### `SpectralBinning(factor=2, mode="mean", dim=-1)`

Bin adjacent channels. Trailing partial bins are dropped.

$$\text{output} = \text{reduce}(x_{\text{reshaped}},\ \text{along factor dim})$$

- `mode="mean"`: $\text{mean}$ — flux-conserving.
- `mode="sum"`: $\text{sum}$ — total flux per bin.

**Inverse:** Nearest-neighbour repeat upsample (pads with zeros if trailing bins were dropped).

| Param | Default | Description |
|---|---|---|
| `factor` | `2` | Channels per bin |
| `mode` | `"mean"` | `"mean"` or `"sum"` |
| `dim` | `-1` | Spectral dimension |

!!! info "When to use"
    Increase signal-to-noise by trading spectral resolution. Use factor=2 for
    a quick SNR boost, or larger factors for coarse binning.

### `BandMath(func, band_dim=0)`

!!! note "Advanced"
    Generic NDVI-style band arithmetic — not FITS-specific. Prefer survey
    code or a one-liner for most pipelines.

Dimension-agnostic band arithmetic via `torch.unbind`.

$$\text{output} = \text{func}(\text{bands})$$

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `func` | *(required)* | Callable receiving tuple of band tensors |
| `band_dim` | `0` | Dimension containing spectral bands |

```python
# NDVI for remote sensing
ndvi = BandMath(lambda b: (b[3] - b[2]) / (b[3] + b[2] + 1e-8))
```
!!! info "When to use"
    Compute spectral indices (NDVI, color ratios, etc.) from multi-band data.
    The function receives unbound band tensors and can do any arithmetic.

---

## Continuum / Baseline Estimators

Additive decomposition: $\text{Original} = \text{Estimate} + \text{Residuals}$.
`inverse()` re-adds stored residuals for perfect recovery.

### `SavitzkyGolayFilter(window_length=7, polyorder=3, dim=-1)`

Polynomial smoothing via conv1d with pre-computed SG coefficients.

Solves least-squares polynomial fit at each window position:

$$c = \text{lstsq}(A,\ y_{\text{impulse}})$$

where $A$ is the Vandermonde matrix for positions $[-h, \ldots, h]$.

**Inverse:** $x = \text{output} + \text{residuals}$

| Param | Default | Description |
|---|---|---|
| `window_length` | `7` | Odd window length (≥ 3) |
| `polyorder` | `3` | Polynomial order (< `window_length`) |
| `dim` | `-1` | Dimension to filter along |

!!! info "When to use"
    Fast, non-parametric smoothing that preserves peaks better than a moving
    average. Good for cleaning spectra before continuum fitting.

### `RunningPercentile(percentile=90, window_size=21, dim=-1)`

Sliding-window percentile via `unfold` + `torch.quantile`.

$$\text{continuum}[i] = Q_{\text{percentile}/100}(\text{window centered at } i)$$

**Inverse:** $x = \text{output} + \text{residuals}$

| Param | Default | Description |
|---|---|---|
| `percentile` | `90.0` | Percentile for each window |
| `window_size` | `21` | Odd sliding window size |
| `dim` | `-1` | Dimension to filter along |

!!! info "When to use"
    Robust continuum estimate that ignores absorption features (use high
    percentile like 90-95) or emission features (use low percentile like 5-10).

### `UpperEnvelopeContinuum(window=11, smooth=0.0, dim=-1)`

Local-max detection + linear interpolation between maxima.

1. Detect local maxima within window.
2. Linearly interpolate between nearest left/right maxima.
3. Optional Gaussian smoothing.

**Inverse:** $x = \text{output} + \text{residuals}$

| Param | Default | Description |
|---|---|---|
| `window` | `11` | Half-width for local-max detection |
| `smooth` | `0.0` | Gaussian smoothing sigma (0 = none) |
| `dim` | `-1` | Dimension to operate along |

!!! info "When to use"
    Fast upper-envelope estimate. Works well for emission-dominated spectra
    where the continuum connects the peaks.

### `AsymmetricLeastSquares(lam=1e5, p=0.01, max_iter=10, dim=-1, envelope="lower")`

!!! note "Advanced"
    Raman/NIR specialty baseline. Prefer `UpperEnvelopeContinuum` or
    `SavitzkyGolayFilter` for generic astronomy spectra unless you need ALS.

Eilers (2003) penalized baseline with asymmetric weights. Standard in Raman
and NIR spectroscopy.

Iteratively solves $(W + \lambda D^T D)z = Wy$ where:
- $D$ is the second-difference operator.
- $W$ is diagonal with $w_i = p$ if $y_i > z_i$, else $w_i = 1-p$ (for `envelope="lower"`).

Solved via O(n) banded Cholesky factorization (pentadiagonal matrix).

**Inverse:** $x = \text{output} + \text{residuals}$

| Param | Default | Description |
|---|---|---|
| `lam` | `1e5` | Smoothness (larger = stiffer baseline) |
| `p` | `0.01` | Asymmetry weight in (0, 1) |
| `max_iter` | `10` | Max reweighting iterations |
| `dim` | `-1` | Dimension to operate along |
| `envelope` | `"lower"` | `"lower"` (hug absorption) or `"upper"` (hug emission) |

!!! info "When to use"
    The gold standard for baseline estimation in vibrational spectroscopy
    (Raman, NIR, IR). Produces smooth baselines that closely follow the lower
    envelope of broad fluorescence backgrounds.

### `AlphaShapeContinuum(half_window=15, iterations=1, dim=-1)`

!!! note "Advanced"
    Morphological upper-envelope variant. Prefer `UpperEnvelopeContinuum`
    unless you specifically want closing (dilate+erode).

Morphological closing (dilation then erosion) via `unfold`.

$$\text{dilated} = \text{max-pool}(x,\ \text{window}=2h+1)$$

$$\text{continuum} = \text{min-pool}(\text{dilated},\ \text{window}=2h+1)$$

**Inverse:** $x = \text{output} + \text{residuals}$

| Param | Default | Description |
|---|---|---|
| `half_window` | `15` | Half-width of structuring element |
| `iterations` | `1` | Number of closing operations |
| `dim` | `-1` | Dimension to operate along |

!!! info "When to use"
    Guaranteed upper envelope. Fast and non-parametric. Good for emission
    spectra where the continuum connects peaks.

### `WaveletDecompose(levels=3, dim=-1)`

Multi-level Haar discrete wavelet transform. Fully invertible frequency split.

At each level:

$$\text{approx}[i] = \frac{x[2i] + x[2i+1]}{2}, \qquad \text{detail}[i] = \frac{x[2i] - x[2i+1]}{2}$$

Output: $[\text{approx}_L, \text{detail}_L, \ldots, \text{detail}_1]$

**Inverse:** Reconstructs via inverse Haar DWT:
$x[2i] = \text{approx}[i] + \text{detail}[i]$, $x[2i+1] = \text{approx}[i] - \text{detail}[i]$.

| Param | Default | Description |
|---|---|---|
| `levels` | `3` | Decomposition levels (1–8) |
| `dim` | `-1` | Dimension to decompose along |

!!! info "When to use"
    Multi-scale analysis of spectra or images. The approximation coefficients
    capture the broad continuum; details capture noise and features at
    different scales. Fully invertible — no information is lost.

---

## Outlier Rejection

### `SigmaClip(n_sigma=3.0, max_iter=5, dim=(-2,-1), fill="mean")`

Iterative sigma-clipping with mean or median fill.

1. Compute mean $\mu$ and std $\sigma$ over dims.
2. Mask pixels where $|x - \mu| > n_\sigma \cdot \sigma$.
3. Repeat until convergence or `max_iter`.
4. Replace clipped values with surviving mean or median.

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `n_sigma` | `3.0` | Clipping threshold |
| `max_iter` | `5` | Max iterations |
| `dim` | `(-2, -1)` | Dimensions for statistics |
| `fill` | `"mean"` | `"mean"` or `"median"` replacement |

!!! info "When to use"
    Standard for cleaning cosmic rays and hot pixels from images. Use
    `fill="median"` for more robust replacement in the presence of many
    outliers.

### `AsymmetricSigmaClip(n_low=3.0, n_high=3.0, dim=(-2,-1))`

One-pass asymmetric sigma-clip via `estimate_background` (median + MAD).

$$\text{lower} = \text{med} - n_{\text{low}} \cdot \text{std}, \qquad \text{upper} = \text{med} + n_{\text{high}} \cdot \text{std}$$

Outliers replaced with median.

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `n_low` | `3.0` | Std below median to clip |
| `n_high` | `3.0` | Std above median to clip |
| `dim` | `(-2, -1)` | Dimensions for statistics |

!!! info "When to use"
    Faster than iterative sigma-clip. Use different `n_low`/`n_high` when
    the outlier distribution is asymmetric (e.g., bright stars are more
    common than dark holes).

---

## Time-Domain

### `PhaseFold(period=1.0, n_bins=64, t0=0.0)`

!!! note "Advanced"
    Time-series / variable-star tool — lossy. Not part of the core FITS image
    ML path.

Fold periodic time series into phase bins.

$$\text{phase}[i] = \left(\frac{t_i - t_0}{\text{period}}\right) \bmod 1$$

Bin each sample by phase into `n_bins` uniform bins in $[0, 1)$.

**Inverse:** None (many-to-one, lossy).

| Param | Default | Description |
|---|---|---|
| `period` | `1.0` | Folding period |
| `n_bins` | `64` | Number of phase bins (≥ 2) |
| `t0` | `0.0` | Phase zero-point |

!!! info "When to use"
    Essential for variable star and exoplanet transit analysis. Folds noisy
    time series into a clean phase curve.

---

## FITS Metadata

### `FITSHeaderScale(bscale=1.0, bzero=0.0)`

Apply/remove FITS BSCALE/BZERO linear scaling.

$$\text{output} = \text{BSCALE} \cdot x + \text{BZERO}$$

**Inverse:** $x = \frac{\text{output} - \text{BZERO}}{\text{BSCALE}}$

| Param | Default | Description |
|---|---|---|
| `bscale` | `1.0` | FITS BSCALE keyword |
| `bzero` | `0.0` | FITS BZERO keyword |

Factory: `FITSHeaderScale.from_header(header)` — extracts BSCALE/BZERO from
a FITS header dict.

### `FITSScaleColumns(scales)`

Per-column BSCALE/BZERO for table tensors.

$$\text{output}[c] = \text{TSCAL}_c \cdot x[c] + \text{TZERO}_c$$

**Inverse:** $x[c] = \frac{\text{output}[c] - \text{TZERO}_c}{\text{TSCAL}_c}$

| Param | Default | Description |
|---|---|---|
| `scales` | *(required)* | `dict[str, (TSCAL, TZERO)]` |

Factory: `FITSScaleColumns.from_header(header)`.

### `TNullToNan(nulls)`

Map FITS TNULL sentinels to NaN in table columns.

$$\text{output}[i] = \begin{cases} \text{NaN} & \text{if } x[i] = \text{TNULL} \\ x[i] & \text{otherwise} \end{cases}$$

**Inverse:** None (lossy).

| Param | Default | Description |
|---|---|---|
| `nulls` | *(required)* | `dict[str, TNULL_value]` |

Factory: `TNullToNan.from_header(header)`.

### `FITSHeaderNormalize(header, scale_floats=False)`

Auto-normalize from BITPIX/BSCALE/BZERO. Integer types mapped to [0, 1].

- **Integer types** (BITPIX 8/16/32/64): min-max normalize using the
  native range scaled by BSCALE/BZERO.
- **Float types** (BITPIX -32/-64): identity unless `scale_floats=True`.

**Inverse:** Yes (for normalized types).

| Param | Default | Description |
|---|---|---|
| `header` | *(required)* | FITS header dict |
| `scale_floats` | `False` | Also normalize float data |

---

## Utility

### `Compose(transforms)`

Chain transforms; `inverse()` unwinds in reverse order.

```python
pipeline = Compose([
    BackgroundSubtract(),
    ArcsinhStretch(a=0.1),
    ZScaleNormalize(),
])
normalized = pipeline(image)
original = pipeline.inverse(normalized)
```
### `FITSTransform`

Base class for custom transforms. Override `forward()` and optionally
`inverse()`. `__call__` delegates to `forward()`. Not an `nn.Module`.

### Helpers

| Function | Role |
|---|---|
| `safe_arcsinh`, `safe_log` | Numerically stable stretch primitives |
| `estimate_background` | Shared background estimator for normalizers / clip |
| `zscale_limits` | IRAF-style zscale limit finder used by `ZScaleNormalize` |

---

## Importing

Import transform classes from `torchfits.transforms` (namespace-only since
0.9.2 — they are not re-exported at the package root):

```python
from torchfits.transforms import (
    ArcsinhStretch, BackgroundSubtract, Compose, ZScaleNormalize,
    RobustNormalize, MinMaxNormalize, PercentileClipNormalize,
    LogStretch, SqrtStretch,
    SpectralBinning, ContinuumRemoval, BandMath, ContinuumNormalize,
    DopplerShift, PhaseFold, GlobalScalarNorm,
    SavitzkyGolayFilter, RunningPercentile, UpperEnvelopeContinuum,
    WaveletDecompose, AsymmetricLeastSquares, AlphaShapeContinuum,
    AsymmetricSigmaClip, SigmaClip,
    FITSScaleColumns, TNullToNan, FITSHeaderNormalize,
)
```
See `examples/example_transforms.py` (image pipeline) and
`examples/example_hyperspectral.py` (spectral cube) for runnable demos.
