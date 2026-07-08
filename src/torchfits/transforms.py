"""Machine-learning friendly transformations for FITS images and high-DR data.

All transforms are compatible with :class:`torch.utils.data.Dataset`,
:class:`~torch.utils.data.IterableDataset`, and multi-worker :class:`~torch.utils.data.DataLoader`.
Every transform provides a matching ``.inverse()`` for decoding model outputs back
to physical flux units.

Design principles
-----------------
* **Gradient-safe** — float64 intermediates for arcsinh / log, stable clamping near zero.
* **High dynamic range** — :class:`ArcsinhStretch` (Lupton+ 1999 / LSST) preserves both
  bright and faint structure without saturating.
* **Astronomy-native** — IRAF-style zscale, robust background subtraction, percentile clipping.
* **Streaming-friendly** — data-dependent state (``.inverse()``) is thread-local;
  each worker holds its own copy (safe under ``num_workers > 0``).

Basic usage::

    import torchfits
    from torchfits.transforms import ArcsinhStretch, ZScaleNormalize, BackgroundSubtract, Compose

    pipeline = Compose([
        BackgroundSubtract(),
        ArcsinhStretch(a=0.1),
        ZScaleNormalize(),
    ])

    image = torchfits.read_tensor("galaxy.fits")
    normalized = pipeline(image)          # forward → model input
    restored  = pipeline.inverse(normalized)  # inverse  → physical flux (approx.)

See ``examples/example_transforms.py`` for a full walkthrough.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.linalg


# ---------------------------------------------------------------------------
# Multi-dim reduction helpers (compatible with PyTorch builds that lack
# tuple dim support in median / quantile / amin / amax).
# ---------------------------------------------------------------------------


def _normalize_dims(ndim: int, dim: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert negative dims to positive and return sorted unique dims."""
    return tuple(sorted({d if d >= 0 else ndim + d for d in dim}))


def _flatten_dims(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    """Collapse *dims* (sorted, positive) into a single trailing dim."""
    ndim = x.ndim
    keep = [d for d in range(ndim) if d not in dims]
    x_moved = x.permute(*keep, *dims)
    return x_moved.reshape(*x_moved.shape[: len(keep)], -1)


def _unflatten_result(
    reduced: torch.Tensor, shape: tuple[int, ...], dims: tuple[int, ...]
) -> torch.Tensor:
    """Reshape a reduced tensor back to *shape* with *dims* set to 1."""
    shape_out = list(shape)
    for d in dims:
        shape_out[d] = 1
    return reduced.reshape(shape_out)


def _reduce_keepdim(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    func,
) -> torch.Tensor:
    """Reduce *x* over *dim* using *func* (single-dim reducer), keepdim."""
    ndim = x.ndim
    dims = _normalize_dims(ndim, dim)
    if len(dims) == 1:
        return func(x, dims[0], True)
    x_flat = _flatten_dims(x, dims)
    result = func(x_flat, -1, True)
    # Reshape back to original ndim with reduced dims set to 1
    shape_out = list(x.shape)
    for d in dims:
        shape_out[d] = 1
    return result.reshape(shape_out)


def _median(x: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """torch.median over tuple dim (compatibility wrapper)."""
    return _reduce_keepdim(
        x, dim, lambda t, d, k: torch.median(t, dim=d, keepdim=k).values
    )


def _amin(x: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """torch.amin over tuple dim (compatibility wrapper)."""
    return _reduce_keepdim(x, dim, lambda t, d, k: torch.amin(t, dim=d, keepdim=k))


def _amax(x: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """torch.amax over tuple dim (compatibility wrapper)."""
    return _reduce_keepdim(x, dim, lambda t, d, k: torch.amax(t, dim=d, keepdim=k))


def _quantile(x: torch.Tensor, q: float, dim: Tuple[int, ...]) -> torch.Tensor:
    """torch.quantile over tuple dim (compatibility wrapper)."""
    return _reduce_keepdim(
        x, dim, lambda t, d, k: torch.quantile(t, q, dim=d, keepdim=k)
    )


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------


def safe_arcsinh(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``arcsinh(scale * x)`` using float64 internally.

    This preserves precision across the large dynamic range typical of
    astronomical images (LSST / SDSS convention).
    """
    orig_dtype = x.dtype
    out = torch.arcsinh(x.double() * scale)
    return out.to(orig_dtype)


def safe_log(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute ``log(x)`` with a floor at *eps* to avoid -inf.

    Uses float64 internally for precision.
    """
    orig_dtype = x.dtype
    out = torch.log(torch.clamp_min(x.double(), eps))
    return out.to(orig_dtype)


def estimate_background(
    x: torch.Tensor, dim: Tuple[int, ...] = (-2, -1)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust background estimator: median and MAD-based dispersion.

    Returns
    -------
    med : Tensor
        Per-pixel-group median (keepdim=True).
    std_approx : Tensor
        MAD × 1.4826 ≈ standard deviation of the background.
    """
    with torch.no_grad():
        med = _median(x, dim)
        mad = _median(torch.abs(x - med), dim)
        std_approx = mad.mul_(1.4826)
    return med, std_approx


def zscale_limits(
    x: torch.Tensor, contrast: float = 0.25, dim: Tuple[int, ...] = (-2, -1)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IRAF-style zscale auto-contrast limits (fast proxy).

    Returns (z1, z2) clipped to [vmin, vmax] with a fallback when the image
    is constant (z1 == z2).
    """
    with torch.no_grad():
        med, std = estimate_background(x, dim=dim)
        z1 = med - (std / max(contrast, 1e-5))
        z2 = med + (std / max(contrast, 1e-5))

        vmin = _amin(x, dim)
        vmax = _amax(x, dim)
        z1 = torch.where(std == 0, vmin, torch.maximum(z1, vmin))
        z2 = torch.where(
            std == 0, vmax, torch.minimum(z2, vmax)
        )  # Use a data-relative epsilon to avoid float32 underflow for large values.
        # 1e-6 relative guarantees >1 ULP margin even at float32 extremes.
        _eps = torch.maximum(
            torch.tensor(1e-6, device=x.device, dtype=z1.dtype), z1.abs() * 1e-6
        )
        z2 = torch.where(z1 == z2, z1 + _eps, z2)
    return z1, z2


# ---------------------------------------------------------------------------
# Base transform
# ---------------------------------------------------------------------------


class FITSTransform:
    """Protocol for astronomy transforms with forward and inverse passes.

    Subclasses should override ``forward`` and ``inverse``.
    Calling an instance directly delegates to :meth:`forward`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class Compose(FITSTransform):
    """Chain transforms; ``.inverse()`` unwinds them in reverse order."""

    def __init__(self, transforms: Sequence[FITSTransform]) -> None:
        self.transforms = list(transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int) -> FITSTransform:
        return self.transforms[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x

    def __repr__(self) -> str:
        inner = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n])"


# ---------------------------------------------------------------------------
# Stretches (stateless, always invertible)
# ---------------------------------------------------------------------------


class ArcsinhStretch(FITSTransform):
    """Lupton+ (2004) arcsinh stretch — the standard for high-DR astronomy.

    ``forward`` computes ``arcsinh(a * x) / arcsinh(a)``, which maps
    [0, 1] → [0, 1] for non-negative inputs when *a* is tuned to the
    data range.  For general inputs the output is not clamped.

    Parameters
    ----------
    a : float
        Softening parameter.  Smaller values = more linear near zero.
    """

    def __init__(self, a: float = 1.0) -> None:
        self.a = float(a)
        self._norm = math.asinh(self.a) if self.a > 0 else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return safe_arcsinh(x, self.a).div_(self._norm)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x.double() * self._norm).div_(self.a).to(x.dtype)

    def __repr__(self) -> str:
        return f"ArcsinhStretch(a={self.a})"


class LogStretch(FITSTransform):
    """Logarithmic stretch, safe for heavy-tailed flux distributions.

    .. note::
       Negative values are silently clamped to zero.  If your data may
       contain negatives (e.g. after sky subtraction), consider applying
       :class:`BackgroundSubtract` with an appropriate sky estimate first.

    Parameters
    ----------
    a : float
        Scale factor applied before the log.  Larger ``a`` compresses the
        low-flux region more gently.
    eps : float
        Floor value to prevent ``log(0)``.
    """

    def __init__(self, a: float = 1000.0, eps: float = 1e-9) -> None:
        self.a = float(a)
        self.eps = float(eps)
        self._norm = math.log10(1.0 + self.a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp_min(x, 0.0)
        return (
            safe_log(1.0 + self.a * x_clamped, eps=self.eps)
            .div_(math.log(10))
            .div_(self._norm)
        )

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        val = torch.pow(10.0, x.double() * self._norm).sub_(1.0)
        return val.div_(self.a).to(orig_dtype)

    def __repr__(self) -> str:
        return f"LogStretch(a={self.a}, eps={self.eps})"


class SqrtStretch(FITSTransform):
    """Square-root stretch — stabilises Poisson variance.

    .. note::
       Negative values are silently clamped to zero.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(x, 0.0))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x.pow(2)

    def __repr__(self) -> str:
        return "SqrtStretch()"


# ---------------------------------------------------------------------------
# Normalizers (data-dependent — cache state for inverse)
# ---------------------------------------------------------------------------


class ZScaleNormalize(FITSTransform):
    """IRAF zscale auto-contrast normalisation.

    ``forward`` maps data to [0, 1] using dynamically computed limits.
    ``inverse`` uses the limits from the most recent forward pass.
    """

    def __init__(self, contrast: float = 0.25, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.contrast = float(contrast)
        self.dim = tuple(dim)
        self._last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1, z2 = zscale_limits(x, contrast=self.contrast, dim=self.dim)
        self._last_state = (z1, z2)
        return (x - z1).div_(z2 - z1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._last_state is None:
            raise RuntimeError(
                "ZScaleNormalize.inverse() requires a prior forward() pass "
                "to capture the per-image limits."
            )
        z1, z2 = self._last_state
        return x.mul_(z2 - z1).add_(z1)

    def __repr__(self) -> str:
        return f"ZScaleNormalize(contrast={self.contrast}, dim={self.dim})"


class RobustNormalize(FITSTransform):
    """Normalise by subtracting the median and dividing by MAD-derived std.

    ``forward`` → ~zero median, unit MAD scale.
    ``inverse`` reverses using the cached statistics.
    """

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)
        self._last_med: Optional[torch.Tensor] = None
        self._last_std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        med, std = estimate_background(x, dim=self.dim)
        self._last_med = med
        self._last_std = std
        return (x - med).div_(torch.clamp_min(std, 1e-9))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._last_med is None or self._last_std is None:
            raise RuntimeError(
                "RobustNormalize.inverse() requires a prior forward() pass."
            )
        return x.mul_(self._last_std).add_(self._last_med)

    def __repr__(self) -> str:
        return f"RobustNormalize(dim={self.dim})"


class BackgroundSubtract(FITSTransform):
    """Subtract the estimated background (median)."""

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)
        self._last_bg: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bg, _ = estimate_background(x, dim=self.dim)
        self._last_bg = bg
        return x - bg

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._last_bg is None:
            raise RuntimeError(
                "BackgroundSubtract.inverse() requires a prior forward() pass."
            )
        return x + self._last_bg

    def __repr__(self) -> str:
        return f"BackgroundSubtract(dim={self.dim})"


class PercentileClipNormalize(FITSTransform):
    """Clip to [lower_pct, upper_pct] percentile range, then normalise to [0, 1].

    Parameters
    ----------
    lower_pct : float
        Lower percentile (0–100).
    upper_pct : float
        Upper percentile (0–100).
    dim :
        Dimensions along which percentiles are computed jointly.
    """

    def __init__(
        self,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
        dim: Tuple[int, ...] = (-2, -1),
    ) -> None:
        self.lower_pct = lower_pct / 100.0
        self.upper_pct = upper_pct / 100.0
        self.dim = tuple(dim)
        self._last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            lower = _quantile(x, self.lower_pct, self.dim)
            upper = _quantile(x, self.upper_pct, self.dim)

        self._last_state = (lower, upper)
        clipped = torch.clamp(x, lower, upper)
        denom = torch.where(upper == lower, torch.ones_like(upper), upper - lower)
        return (clipped - lower).div_(denom)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._last_state is None:
            raise RuntimeError(
                "PercentileClipNormalize.inverse() requires a prior forward() pass."
            )
        lower, upper = self._last_state
        return x.mul_(upper - lower).add_(lower)

    def __repr__(self) -> str:
        return (
            f"PercentileClipNormalize("
            f"lower_pct={self.lower_pct * 100:.0f}, "
            f"upper_pct={self.upper_pct * 100:.0f}, "
            f"dim={self.dim})"
        )


class MinMaxNormalize(FITSTransform):
    """Normalise to [0, 1] using per-image min / max."""

    def __init__(self, dim: Tuple[int, ...] = (-2, -1)) -> None:
        self.dim = tuple(dim)
        self._last_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vmin = _amin(x, self.dim)
            vmax = _amax(x, self.dim)
            # Data-relative epsilon to avoid float32 underflow on constant images.
            _eps = torch.maximum(
                torch.tensor(1e-6, device=x.device, dtype=vmin.dtype),
                vmin.abs() * 1e-6,
            )
            vmax = torch.where(vmin == vmax, vmin + _eps, vmax)
        self._last_state = (vmin, vmax)
        return (x - vmin).div_(vmax - vmin)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._last_state is None:
            raise RuntimeError(
                "MinMaxNormalize.inverse() requires a prior forward() pass."
            )
        vmin, vmax = self._last_state
        return x.mul_(vmax - vmin).add_(vmin)

    def __repr__(self) -> str:
        return f"MinMaxNormalize(dim={self.dim})"


# ---------------------------------------------------------------------------
# FITS metadata-aware transforms
# ---------------------------------------------------------------------------


class FITSHeaderScale(FITSTransform):
    """Apply or remove BSCALE/BZERO scaling using FITS header keywords.

    ``forward`` applies the scaling tensor → physical (BSCALE * tensor + BZERO).
    ``inverse`` removes it: (physical − BZERO) / BSCALE.

    Parameters
    ----------
    bscale : float
        FITS BSCALE keyword value.  Default 1.0.
    bzero : float
        FITS BZERO keyword value.  Default 0.0.

    Example
    -------
    >>> header = {"BSCALE": 0.5, "BZERO": 100.0}
    >>> scaler = FITSHeaderScale.from_header(header)
    >>> physical = scaler(raw_counts)   # raw → physical
    >>> raw = scaler.inverse(physical)  # physical → raw
    """

    def __init__(self, bscale: float = 1.0, bzero: float = 0.0) -> None:
        self.bscale = float(bscale)
        self.bzero = float(bzero)

    @classmethod
    def from_header(cls, header: dict) -> FITSHeaderScale:
        """Construct from a FITS header dict-like object."""
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
        return cls(bscale=bscale, bzero=bzero)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bscale == 1.0 and self.bzero == 0.0:
            return x
        result = x.to(torch.float32)
        if self.bscale != 1.0:
            result = result.mul_(self.bscale)
        if self.bzero != 0.0:
            result = result.add_(self.bzero)
        return result.to(x.dtype) if x.dtype != torch.float32 else result

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.bscale == 1.0 and self.bzero == 0.0:
            return x
        result = x.to(torch.float32)
        if self.bzero != 0.0:
            result = result.sub_(self.bzero)
        if self.bscale != 1.0:
            result = result.div_(self.bscale)
        return result.to(x.dtype) if x.dtype != torch.float32 else result

    def __repr__(self) -> str:
        return f"FITSHeaderScale(bscale={self.bscale}, bzero={self.bzero})"


def _fit_poly_continuum(
    x: torch.Tensor, order: int = 3, n_sigma: float = 2.0, max_iter: int = 3
) -> torch.Tensor:
    """Fit a low-order polynomial continuum via iterative sigma-clipping."""
    n, length = x.shape
    t = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)
    A = torch.stack([t**k for k in range(order + 1)], dim=1)  # [length, order+1]

    mask = torch.ones(n, length, dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        # Weighted least squares per spectrum
        coeffs = torch.zeros(n, order + 1, device=x.device, dtype=x.dtype)
        for i in range(n):
            mi = mask[i]
            if mi.sum() <= order:
                mi = torch.ones(length, dtype=torch.bool, device=x.device)
            Am = A[mi]
            ym = x[i][mi]
            try:
                coeffs[i] = torch.linalg.lstsq(Am, ym.unsqueeze(1)).solution.squeeze(1)
            except RuntimeError:
                # Fallback: use all points (singular / underdetermined)
                coeffs[i] = torch.linalg.lstsq(A, x[i].unsqueeze(1)).solution.squeeze(1)

        continuum = (A @ coeffs.T).T  # [n, length]
        residuals = x - continuum
        # Compute std only on currently-unmasked pixels (masked outliers
        # would inflate the std and prevent convergence).
        masked_res = torch.where(mask, residuals, torch.zeros_like(residuals))
        count = mask.float().sum(dim=1, keepdim=True)
        mean_res = masked_res.sum(dim=1, keepdim=True) / torch.clamp_min(count, 1.0)
        var = torch.where(
            mask, (residuals - mean_res) ** 2, torch.zeros_like(residuals)
        ).sum(dim=1, keepdim=True) / torch.clamp_min(count, 1.0)
        std = torch.sqrt(torch.clamp_min(var, 0.0))
        new_mask = residuals.abs() < n_sigma * torch.clamp_min(std, 1e-9)
        if torch.equal(new_mask, mask):
            break
        mask = new_mask

    return continuum


def _resample_spectrum(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Resample spectrum by a scale factor using linear interpolation."""
    shape_in = x.shape
    x_2d = x.reshape(-1, shape_in[-1])
    length = x_2d.shape[1]
    new_length = max(2, int(length * scale))
    new_grid = torch.linspace(0, length - 1, new_length, device=x.device, dtype=x.dtype)
    orig_grid = torch.arange(length, device=x.device, dtype=x.dtype)
    # Interpolate using linear interpolation
    out = _linear_interp_1d(x_2d, orig_grid, new_grid)
    return out.reshape(*shape_in[:-1], new_length)


def _linear_interp_1d(
    y: torch.Tensor, x_orig: torch.Tensor, x_new: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of y(x) at x_new points."""
    idx = torch.searchsorted(x_orig, x_new)
    idx = torch.clamp(idx, 1, len(x_orig) - 1)
    x0 = x_orig[idx - 1]
    x1 = x_orig[idx]
    y0 = y[:, idx - 1]
    y1 = y[:, idx]
    frac = (x_new - x0) / torch.clamp_min(x1 - x0, 1e-30)
    return y0 + (y1 - y0) * frac.unsqueeze(0)


# ---------------------------------------------------------------------------
# Spectral transforms (1D — not in torch/torchvision)
# ---------------------------------------------------------------------------


class ContinuumNormalize(FITSTransform):
    """Normalise a spectrum by fitting and dividing by its continuum.

    Fits a low-order polynomial to the flux array (iteratively rejecting
    absorption/emission features via sigma-clipping), then divides the
    spectrum by the fitted continuum.  Operates along the last dimension.

    ``inverse`` multiplies back by the cached continuum fit.

    Parameters
    ----------
    order : int
        Polynomial order for the continuum fit (default 3).
    n_sigma : float
        Sigma-clipping threshold for rejecting spectral features during
        the continuum fit (default 2.0).
    max_iter : int
        Maximum number of sigma-clipping iterations (default 3).
    """

    def __init__(self, order: int = 3, n_sigma: float = 2.0, max_iter: int = 3) -> None:
        self.order = int(order)
        self.n_sigma = float(n_sigma)
        self.max_iter = int(max_iter)
        self._continuum: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fit continuum and divide."""
        shape_in = x.shape
        # Work on 2D: flatten leading dims to [N, length]
        x_2d = x.reshape(-1, shape_in[-1])

        with torch.no_grad():
            continuum = _fit_poly_continuum(
                x_2d, order=self.order, n_sigma=self.n_sigma, max_iter=self.max_iter
            )
        self._continuum = continuum.reshape(shape_in)

        denom = torch.clamp_min(self._continuum.abs(), 1e-30)
        return x / denom

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._continuum is None:
            raise RuntimeError(
                "ContinuumNormalize.inverse() requires a prior forward() pass."
            )
        return x * self._continuum

    def __repr__(self) -> str:
        return (
            f"ContinuumNormalize(order={self.order}, "
            f"n_sigma={self.n_sigma}, max_iter={self.max_iter})"
        )


class DopplerShift(FITSTransform):
    """Apply a redshift or blueshift to spectral data via linear interpolation.

    Resamples the last dimension by a factor ``1 + z``, where *z* is the
    redshift (positive = redshifted, negative = blueshifted).  Flux is
    conserved per bin via normalisation.

    ``inverse`` applies the opposite shift (``-z / (1 + z)``).

    Parameters
    ----------
    z : float
        Redshift.  Positive values stretch the spectrum (redshift).
    """

    def __init__(self, z: float = 0.0) -> None:
        self.z = float(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.z == 0.0:
            return x
        return _resample_spectrum(x, 1.0 + self.z)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.z == 0.0:
            return x
        return _resample_spectrum(x, 1.0 / (1.0 + self.z))

    def __repr__(self) -> str:
        return f"DopplerShift(z={self.z})"


# ---------------------------------------------------------------------------
# Time-domain transforms (not in torch/torchvision)
# ---------------------------------------------------------------------------


class PhaseFold(FITSTransform):
    """Fold a periodic time series by period into phase space.

    Maps each time step ``t`` to phase ``(t / period) % 1`` and then
    sorts/resamples onto a uniform phase grid.  Operates along the last
    dimension.

    ``inverse`` is not available — folding is lossy (many-to-one).

    Parameters
    ----------
    period : float
        Folding period in the same units as the time axis.
    n_bins : int
        Number of uniform phase bins for the output (default 64).
    t0 : float
        Phase zero-point offset (default 0).
    """

    def __init__(self, period: float = 1.0, n_bins: int = 64, t0: float = 0.0) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.period = float(period)
        self.n_bins = int(n_bins)
        self.t0 = float(t0)
        self._orig_length: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold into phase bins."""
        self._orig_length = x.shape[-1]
        shape_in = x.shape
        x_2d = x.reshape(-1, shape_in[-1])
        n_samples = x_2d.shape[0]
        length = x_2d.shape[1]

        # Time grid
        t = torch.arange(length, device=x.device, dtype=x.dtype)
        phase = ((t - self.t0) / self.period) % 1.0

        # Bin edges and indices
        edges = torch.linspace(
            0.0, 1.0, self.n_bins + 1, device=x.device, dtype=x.dtype
        )
        bin_idx = torch.bucketize(phase, edges[:-1]) - 1
        # Clamp out-of-range values
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)

        # Scatter sum into bins
        folded = torch.zeros(n_samples, self.n_bins, device=x.device, dtype=x.dtype)
        counts = torch.zeros(self.n_bins, device=x.device, dtype=x.dtype)
        for b in range(self.n_bins):
            mask = bin_idx == b
            folded[:, b] = x_2d[:, mask].sum(dim=1)
            counts[b] = mask.sum()

        # Normalise by counts (mean per bin)
        folded = folded / torch.clamp_min(counts.unsqueeze(0), 1.0)

        return folded.reshape(*shape_in[:-1], self.n_bins)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "PhaseFold.inverse() is not available — folding is many-to-one."
        )

    def __repr__(self) -> str:
        return f"PhaseFold(period={self.period}, n_bins={self.n_bins}, t0={self.t0})"


# ---------------------------------------------------------------------------
# Hyperspectral transforms (not in torch/torchvision)
# ---------------------------------------------------------------------------


class SpectralBinning(FITSTransform):
    """Bin adjacent spectral channels to reduce spectral resolution.

    Replaces groups of *factor* adjacent channels along *dim* with their
    mean (flux-conserving) or sum.  Trailing partial bins are dropped.

    ``inverse`` upsamples via nearest-neighbour repeat, dividing by
    *factor* when ``mode="sum"`` to conserve absolute flux.

    Parameters
    ----------
    factor : int
        Number of adjacent channels to bin together (>= 1).
    mode : str
        Reduction: ``"mean"`` (default, flux-conserving) or ``"sum"``.
    dim : int
        Spectral dimension to bin along (default -1).
    """

    def __init__(self, factor: int = 2, mode: str = "mean", dim: int = -1) -> None:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        if mode not in ("mean", "sum"):
            raise ValueError("mode must be 'mean' or 'sum'")
        self.factor = int(factor)
        self.mode = mode
        self.dim = int(dim)
        self._orig_length: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        length = x.shape[dim]
        trimmed = length - (length % self.factor)
        self._orig_length = length

        # Slice off trailing partial channels
        slices = [slice(None)] * ndim
        slices[dim] = slice(0, trimmed)
        x_trimmed = x[tuple(slices)]

        # Reshape to introduce factor dimension and reduce
        new_shape = list(x_trimmed.shape)
        new_shape.insert(dim + 1, self.factor)
        new_shape[dim] = trimmed // self.factor
        x_reshaped = x_trimmed.reshape(new_shape)

        if self.mode == "mean":
            return x_reshaped.mean(dim=dim + 1)
        return x_reshaped.sum(dim=dim + 1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        # Nearest-neighbour repeat: expand then reshape
        shape = list(x.shape)
        shape.insert(dim + 1, self.factor)
        x_repeated = x.unsqueeze(dim + 1).expand(shape)
        out = x_repeated.reshape(
            *x_repeated.shape[:dim],
            x.shape[dim] * self.factor,
            *x_repeated.shape[dim + 2 :],
        )

        # For "sum" mode, divide to recover per-pixel flux
        if self.mode == "sum":
            out = out / self.factor

        # Pad to original length if trailing bins were dropped
        if self._orig_length is not None and out.shape[dim] < self._orig_length:
            pad_shape = list(out.shape)
            pad_shape[dim] = self._orig_length - out.shape[dim]
            padding = torch.zeros(pad_shape, device=out.device, dtype=out.dtype)
            out = torch.cat([out, padding], dim=dim)

        return out

    def __repr__(self) -> str:
        return (
            f"SpectralBinning(factor={self.factor}, mode={self.mode!r}, dim={self.dim})"
        )


def _build_spline_basis(
    n_points: int, n_knots: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build a cubic B-spline basis matrix with evenly-spaced knots.

    Returns a matrix of shape ``[n_points, n_knots]`` where each column
    is a cubic B-spline basis function evaluated at the *n_points*
    positions.
    """
    # Knot positions: extend 2 beyond the range for cubic support
    t = torch.linspace(-2, n_knots + 1, n_knots + 4, device=device, dtype=dtype)
    x = torch.linspace(0, n_knots - 1, n_points, device=device, dtype=dtype)
    B = torch.zeros(n_points, n_knots, device=device, dtype=dtype)

    for k in range(n_knots):
        # Cubic B-spline basis function centred at knot k
        B[:, k] = _bspline_cubic(x, t[k : k + 5])

    return B


def _basis_order1(
    points: torch.Tensor, t_i: torch.Tensor, t_ip1: torch.Tensor
) -> torch.Tensor:
    """Order-1 B-spline basis (piecewise constant)."""
    return ((points >= t_i) & (points < t_ip1)).float()


def _basis_order2(
    points: torch.Tensor,
    t_i: torch.Tensor,
    t_ip1: torch.Tensor,
    t_ip2: torch.Tensor,
) -> torch.Tensor:
    """Order-2 B-spline basis (linear)."""
    b1 = (
        _basis_order1(points, t_i, t_ip1)
        * (points - t_i)
        / torch.clamp_min(t_ip1 - t_i, 1e-30)
    )
    b2 = (
        _basis_order1(points, t_ip1, t_ip2)
        * (t_ip2 - points)
        / torch.clamp_min(t_ip2 - t_ip1, 1e-30)
    )
    return b1 + b2


def _basis_order3(
    points: torch.Tensor,
    t_i: torch.Tensor,
    t_ip1: torch.Tensor,
    t_ip2: torch.Tensor,
    t_ip3: torch.Tensor,
) -> torch.Tensor:
    """Order-3 B-spline basis (quadratic)."""
    b1 = (
        _basis_order2(points, t_i, t_ip1, t_ip2)
        * (points - t_i)
        / torch.clamp_min(t_ip2 - t_i, 1e-30)
    )
    b2 = (
        _basis_order2(points, t_ip1, t_ip2, t_ip3)
        * (t_ip3 - points)
        / torch.clamp_min(t_ip3 - t_ip1, 1e-30)
    )
    return b1 + b2


def _bspline_cubic(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate a cubic B-spline with given knot vector at positions x.

    Uses the Cox-de Boor recursion for order 4 (cubic).
    *knots* should have 5 elements: ``[t_0, t_1, t_2, t_3, t_4]``.
    Returns 0 outside ``[t_0, t_4]``.
    """
    t0, t1, t2, t3, t4 = knots[0], knots[1], knots[2], knots[3], knots[4]
    b1 = _basis_order3(x, t0, t1, t2, t3) * (x - t0) / torch.clamp_min(t3 - t0, 1e-30)
    b2 = _basis_order3(x, t1, t2, t3, t4) * (t4 - x) / torch.clamp_min(t4 - t1, 1e-30)
    return b1 + b2


def _fit_spline_continuum(
    x: torch.Tensor,
    n_knots: int = 10,
    n_sigma: float = 2.0,
    max_iter: int = 3,
) -> torch.Tensor:
    """Fit a cubic B-spline continuum with iterative sigma-clipping."""
    n, length = x.shape
    B = _build_spline_basis(length, n_knots, x.device, x.dtype)

    # Add ridge penalty for numerical stability
    ridge = 1e-6 * torch.eye(n_knots, device=x.device, dtype=x.dtype)

    mask = torch.ones(n, length, dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        coeffs = torch.zeros(n, n_knots, device=x.device, dtype=x.dtype)
        for i in range(n):
            mi = mask[i]
            if mi.sum() <= n_knots:
                mi = torch.ones(length, dtype=torch.bool, device=x.device)
            Bm = B[mi]
            ym = x[i][mi]
            # Weighted least squares with ridge
            BtB_m = Bm.T @ Bm + ridge
            Bty_m = Bm.T @ ym
            try:
                coeffs[i] = torch.linalg.solve(BtB_m, Bty_m)
            except RuntimeError:
                # Fallback: use all points
                BtB = B.T @ B + ridge
                coeffs[i] = torch.linalg.solve(BtB, B.T @ x[i])

        continuum = (B @ coeffs.T).T  # [n, length]
        residuals = x - continuum
        # Sigma-clip on unmasked pixels
        masked_res = torch.where(mask, residuals, torch.zeros_like(residuals))
        count = mask.float().sum(dim=1, keepdim=True)
        mean_res = masked_res.sum(dim=1, keepdim=True) / torch.clamp_min(count, 1.0)
        var = torch.where(
            mask, (residuals - mean_res) ** 2, torch.zeros_like(residuals)
        ).sum(dim=1, keepdim=True) / torch.clamp_min(count, 1.0)
        std = torch.sqrt(torch.clamp_min(var, 0.0))
        new_mask = residuals.abs() < n_sigma * torch.clamp_min(std, 1e-9)
        if torch.equal(new_mask, mask):
            break
        mask = new_mask

    return continuum


class ContinuumRemoval(FITSTransform):
    """Remove spectral continuum (baseline) from reflectance spectra.

    Fits a low-order polynomial or cubic B-spline to the spectrum
    and **subtracts** the fit, leaving absorption/emission features as
    positive or negative residuals around zero.

    This is distinct from :class:`ContinuumNormalize`, which **divides**
    by the continuum to normalise to ~1.  Use ``ContinuumRemoval`` for
    additive baseline correction (common in reflectance spectroscopy);
    use ``ContinuumNormalize`` for multiplicative normalisation.

    ``inverse`` adds the cached continuum back.

    Parameters
    ----------
    method : str
        ``"polynomial"`` (default) or ``"spline"``.
    order : int
        Polynomial order when ``method="polynomial"`` (default 3).
    n_knots : int
        Number of evenly-spaced knots when ``method="spline"`` (default 10).
    n_sigma : float
        Sigma-clip threshold for rejecting spectral features during
        the continuum fit (default 2.0).
    max_iter : int
        Maximum sigma-clipping iterations (default 3).
    """

    def __init__(
        self,
        method: str = "polynomial",
        order: int = 3,
        n_knots: int = 10,
        n_sigma: float = 2.0,
        max_iter: int = 3,
    ) -> None:
        if method not in ("polynomial", "spline"):
            raise ValueError("method must be 'polynomial' or 'spline'")
        self.method = method
        self.order = int(order)
        self.n_knots = int(n_knots)
        self.n_sigma = float(n_sigma)
        self.max_iter = int(max_iter)
        self._baseline: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape_in = x.shape
        x_2d = x.reshape(-1, shape_in[-1])

        with torch.no_grad():
            if self.method == "polynomial":
                baseline = _fit_poly_continuum(
                    x_2d,
                    order=self.order,
                    n_sigma=self.n_sigma,
                    max_iter=self.max_iter,
                )
            else:
                baseline = _fit_spline_continuum(
                    x_2d,
                    n_knots=self.n_knots,
                    n_sigma=self.n_sigma,
                    max_iter=self.max_iter,
                )

        self._baseline = baseline.reshape(shape_in)
        return x - self._baseline

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._baseline is None:
            raise RuntimeError(
                "ContinuumRemoval.inverse() requires a prior forward() pass."
            )
        return x + self._baseline

    def __repr__(self) -> str:
        if self.method == "polynomial":
            return (
                f"ContinuumRemoval(method='polynomial', order={self.order}, "
                f"n_sigma={self.n_sigma}, max_iter={self.max_iter})"
            )
        return (
            f"ContinuumRemoval(method='spline', n_knots={self.n_knots}, "
            f"n_sigma={self.n_sigma}, max_iter={self.max_iter})"
        )


class BandMath(FITSTransform):
    """Apply arithmetic band ratios and indices to multi-spectral data.

    Applies a user-supplied function along a specified *band_dim*.  The
    function receives a tuple of tensors — one per band slice along
    *band_dim* — and returns the result.  This gives dimension-agnostic
    band access: ``lambda b: (b[1] - b[0]) / (b[1] + b[0])`` for NDVI.

    ``inverse`` is not available — band arithmetic is lossy.

    Parameters
    ----------
    func : callable
        Function ``(tuple[Tensor, ...]) -> Tensor`` that takes a tuple of
        band-slice tensors and returns the arithmetic result.
    band_dim : int
        Dimension containing spectral bands (default 0 for ``[C, H, W]``).

    Examples
    --------
    >>> # NDVI: (NIR - Red) / (NIR + Red), NIR=band 3, Red=band 2
    >>> ndvi = BandMath(lambda b: (b[3] - b[2]) / (b[3] + b[2] + 1e-8))
    >>>
    >>> # WBI (Water Band Index): R900 / R970
    >>> wbi = BandMath(lambda b: b[0] / (b[1] + 1e-8), band_dim=-3)
    """

    def __init__(self, func, band_dim: int = 0) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func
        self.band_dim = int(band_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        dim = self.band_dim if self.band_dim >= 0 else ndim + self.band_dim
        # Unbind along band dimension for dimension-agnostic access
        bands = torch.unbind(x, dim=dim)
        return self.func(bands)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "BandMath.inverse() is not available — band arithmetic is lossy."
        )

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", repr(self.func))
        return f"BandMath(func={name}, band_dim={self.band_dim})"


# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------


class SigmaClip(FITSTransform):
    """Iterative sigma-clipping outlier rejection.

    Iteratively computes the mean and standard deviation over *dim*,
    masks values outside ``[mean - n_sigma*std, mean + n_sigma*std]``,
    and replaces them with the final mean.  Stops when no new pixels
    are clipped or *max_iter* is reached.

    ``inverse`` is not available — clipped values are irrecoverable.

    Parameters
    ----------
    n_sigma : float
        Number of standard deviations for the clipping threshold.
    max_iter : int
        Maximum number of clipping iterations.
    dim :
        Dimensions along which stats are computed independently.
    fill : str
        Replacement strategy: ``"mean"`` (default) or ``"median"``.
    """

    def __init__(
        self,
        n_sigma: float = 3.0,
        max_iter: int = 5,
        dim: Tuple[int, ...] = (-2, -1),
        fill: str = "mean",
    ) -> None:
        self.n_sigma = float(n_sigma)
        self.max_iter = int(max_iter)
        self.dim = tuple(dim)
        if fill not in ("mean", "median"):
            raise ValueError("fill must be 'mean' or 'median'")
        self.fill = fill
        self._last_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iteratively sigma-clip outliers and fill with mean or median."""
        ndim = x.ndim
        dims: tuple[int, ...] = ()
        if len(self.dim) > 0:
            dims = _normalize_dims(ndim, self.dim)

        with torch.no_grad():
            mask = torch.ones_like(x, dtype=torch.bool)
            for _ in range(self.max_iter):
                # Count and sum of unmasked pixels per group
                good_x = torch.where(mask, x, torch.zeros_like(x))
                good_cnt = mask.float()

                if len(dims) > 0:
                    x_flat = _flatten_dims(good_x, dims)
                    c_flat = _flatten_dims(good_cnt, dims)
                    total_sum = x_flat.sum(dim=-1, keepdim=True)
                    total_cnt = c_flat.sum(dim=-1, keepdim=True)
                    mean_v = total_sum / torch.clamp_min(total_cnt, 1.0)
                    # Broadcast mean back to original shape
                    mean_v_full = _unflatten_result(mean_v, x.shape, dims)
                    diff_sq = torch.where(
                        mask, (x - mean_v_full) ** 2, torch.zeros_like(x)
                    )
                    d_flat = _flatten_dims(diff_sq, dims)
                    var = d_flat.sum(dim=-1, keepdim=True) / torch.clamp_min(
                        total_cnt, 1.0
                    )
                    std_v_full = _unflatten_result(
                        torch.sqrt(torch.clamp_min(var, 0.0)), x.shape, dims
                    )
                else:
                    cnt = good_cnt.sum()
                    mean_v_full = torch.full_like(
                        x, good_x.sum() / max(cnt.item(), 1.0)
                    )
                    diff_sq = torch.where(
                        mask, (x - mean_v_full) ** 2, torch.zeros_like(x)
                    )
                    var = diff_sq.sum() / max(cnt.item(), 1.0)
                    std_v_full = torch.full_like(x, math.sqrt(max(var.item(), 0.0)))

                new_mask = (x >= mean_v_full - self.n_sigma * std_v_full) & (
                    x <= mean_v_full + self.n_sigma * std_v_full
                )
                if torch.equal(new_mask, mask):
                    break
                mask = new_mask

            self._last_mask = mask

            # Fill clipped values with per-group mean or median
            if self.fill == "mean":
                good_x_final = torch.where(mask, x, torch.zeros_like(x))
                good_c_final = mask.float()
                if len(dims) > 0:
                    xf = _flatten_dims(good_x_final, dims)
                    cf = _flatten_dims(good_c_final, dims)
                    fill_val = _unflatten_result(
                        xf.sum(dim=-1, keepdim=True)
                        / torch.clamp_min(cf.sum(dim=-1, keepdim=True), 1.0),
                        x.shape,
                        dims,
                    )
                else:
                    cnt = good_c_final.sum()
                    fill_val = good_x_final.sum() / max(cnt.item(), 1.0)
            else:
                # Median fill: use the existing _median helper
                fill_val = _median(
                    torch.where(
                        mask,
                        x,
                        torch.tensor(float("inf"), device=x.device, dtype=x.dtype),
                    ),
                    dims if dims else (-1,),
                )
                # Replace inf (all-masked groups) with 0
                fill_val = torch.where(
                    torch.isinf(fill_val), torch.zeros_like(fill_val), fill_val
                )

            return torch.where(mask, x, fill_val)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "SigmaClip.inverse() is not available — clipped values are irrecoverable."
        )

    def __repr__(self) -> str:
        return (
            f"SigmaClip(n_sigma={self.n_sigma}, max_iter={self.max_iter}, "
            f"dim={self.dim}, fill={self.fill!r})"
        )


# ---------------------------------------------------------------------------
# Header-aware normalization
# ---------------------------------------------------------------------------


class FITSHeaderNormalize(FITSTransform):
    """Auto-detect and apply normalization from FITS header keywords.

    Inspects BITPIX, BSCALE, and BZERO to determine the best
    normalization strategy:

    - **Integer types** (BITPIX 8/16/32): scales to [0, 1] using the
      integer range, optionally compensating for BZERO offset.
    - **Float types** (BITPIX -32/-64): applies no scaling by default
      (floats are already in physical units).  Set *scale_floats=True*
      to normalize to [0, 1] via min-max.

    ``inverse`` reverses the normalization using the cached parameters.

    Parameters
    ----------
    header : dict
        FITS header dict-like with BITPIX, BSCALE, BZERO keywords.
    scale_floats : bool
        If True, min-max normalize floating-point data.  Default False.
    """

    # BITPIX → (dtype, signed, bits)
    _BITPIX_MAP: dict[int, tuple[torch.dtype, bool, int]] = {
        8: (torch.uint8, False, 8),
        16: (torch.int16, True, 16),
        32: (torch.int32, True, 32),
        64: (torch.int64, True, 64),
        -32: (torch.float32, False, 32),
        -64: (torch.float64, False, 64),
    }

    def __init__(self, header: dict, scale_floats: bool = False) -> None:
        self.bitpix = int(header.get("BITPIX", -32))
        self.bscale = float(header.get("BSCALE", 1.0))
        self.bzero = float(header.get("BZERO", 0.0))
        self.scale_floats = bool(scale_floats)

        info = self._BITPIX_MAP.get(self.bitpix)
        self._is_integer = info is not None and info[1]
        self._is_unsigned = info is not None and not info[1] and self.bitpix > 0
        self._bits = info[2] if info else 32
        self._in_range: tuple[float, float] | None = None

        # Pre-compute the physical value range for integer types
        if self._is_integer:
            raw_min = -(2 ** (self._bits - 1))
            raw_max = (2 ** (self._bits - 1)) - 1
            phys_min = raw_min * self.bscale + self.bzero
            phys_max = raw_max * self.bscale + self.bzero
            self._in_range = (phys_min, phys_max)
        elif self._is_unsigned and self.bitpix == 8:
            phys_min = self.bzero
            phys_max = 255.0 * self.bscale + self.bzero
            self._in_range = (phys_min, phys_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_integer or self._is_unsigned:
            vmin, vmax = self._in_range  # type: ignore[misc]
            if vmax == vmin:
                return torch.zeros_like(x)
            return (x - vmin) / (vmax - vmin)
        if self.scale_floats:
            vmin = x.min()
            vmax = x.max()
            self._in_range = (float(vmin.item()), float(vmax.item()))
            if vmax == vmin:
                return torch.zeros_like(x)
            return (x - vmin) / (vmax - vmin)
        # Float types, no scaling requested — identity
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_integer or self._is_unsigned or self.scale_floats:
            if self._in_range is None:
                raise RuntimeError(
                    "FITSHeaderNormalize.inverse() requires a prior forward() pass "
                    "when scale_floats=True."
                )
            vmin, vmax = self._in_range
            return x * (vmax - vmin) + vmin
        return x

    def __repr__(self) -> str:
        return (
            f"FITSHeaderNormalize(bitpix={self.bitpix}, "
            f"bscale={self.bscale}, bzero={self.bzero}, "
            f"scale_floats={self.scale_floats})"
        )
