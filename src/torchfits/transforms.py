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


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------


class Gaussian2D(FITSTransform):
    """Convolve with a 2D Gaussian kernel (PSF smoothing).

    Operates on the last two spatial dimensions.  Supports elliptical
    kernels via separate *sigma_x* and *sigma_y* and an optional rotation
    angle *theta* (degrees, counter-clockwise).

    ``inverse`` is not available — deconvolution is ill-posed.

    Parameters
    ----------
    sigma : float
        Gaussian sigma (pixels).  When *sigma_y* is not given, used for
        both axes (circular PSF).
    sigma_y : float or None
        Y-axis sigma.  Defaults to *sigma*.
    theta : float
        Rotation angle in degrees (counter-clockwise).  Only meaningful
        when the kernel is elliptical.
    kernel_size : int or None
        Kernel width/height in pixels.  Default ``None`` auto-computes
        ``ceil(6 * max(sigma_x, sigma_y)) | 1`` (odd).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        sigma_y: float | None = None,
        theta: float = 0.0,
        kernel_size: int | None = None,
    ) -> None:
        self.sigma = float(sigma)
        self.sigma_y = float(sigma_y) if sigma_y is not None else self.sigma
        self.theta = float(theta)
        self._kernel: torch.Tensor | None = None
        if kernel_size is not None:
            self.kernel_size = int(kernel_size)
        else:
            s = max(self.sigma, self.sigma_y)
            self.kernel_size = max(3, int(math.ceil(6.0 * s)) | 1)
        self._build_kernel()

    def _build_kernel(self) -> None:
        """Pre-compute the Gaussian kernel tensor."""
        ksize = self.kernel_size
        ax = torch.linspace(-(ksize - 1) / 2.0, (ksize - 1) / 2.0, ksize)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")

        theta_rad = math.radians(self.theta)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        xr = xx * cos_t + yy * sin_t
        yr = -xx * sin_t + yy * cos_t

        kernel = torch.exp(-0.5 * ((xr / self.sigma) ** 2 + (yr / self.sigma_y) ** 2))
        kernel /= kernel.sum()
        # Store as [1, 1, H, W] for F.conv2d
        self._kernel = kernel.reshape(1, 1, ksize, ksize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian convolution to the last two spatial dims."""
        if self._kernel is None:
            self._build_kernel()
        kernel = self._kernel.to(device=x.device, dtype=x.dtype)

        # Reshape to [N, C, H, W] for conv2d
        shape_in = x.shape
        x_4d = x.reshape(-1, 1, *shape_in[-2:])
        padding = self.kernel_size // 2
        out = torch.nn.functional.conv2d(x_4d, kernel, padding=padding, groups=1)
        return out.reshape(shape_in)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "Gaussian2D.inverse() is not available — deconvolution is ill-posed. "
            "If needed, use an approximate method (e.g. Wiener filter) externally."
        )

    def __repr__(self) -> str:
        return (
            f"Gaussian2D(sigma={self.sigma}, sigma_y={self.sigma_y}, "
            f"theta={self.theta}, kernel_size={self.kernel_size})"
        )


class Downsample(FITSTransform):
    """Downsample (rebin) spatial dimensions by an integer factor.

    Uses mean pooling to preserve surface brightness (flux-conserving).
    Operates on the last two dimensions.  ``inverse`` upsamples via
    nearest-neighbour interpolation.

    Parameters
    ----------
    factor : int
        Integer downsampling factor per spatial axis.
    mode : str
        Pooling mode: ``"mean"`` (flux-conserving, default) or ``"max"``.
    """

    def __init__(self, factor: int = 2, mode: str = "mean") -> None:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        if mode not in ("mean", "max"):
            raise ValueError("mode must be 'mean' or 'max'")
        self.factor = int(factor)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        shape_in = x.shape
        # Flatten leading dims to [N, C, H, W] for pooling
        x_4d = x.reshape(-1, 1, *shape_in[-2:])
        pool = (
            torch.nn.functional.avg_pool2d
            if self.mode == "mean"
            else torch.nn.functional.max_pool2d
        )
        out = pool(x_4d, kernel_size=self.factor, stride=self.factor)
        return out.reshape(*shape_in[:-2], *out.shape[-2:])

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample back to original resolution (nearest-neighbour)."""
        if self.factor == 1:
            return x
        shape_in = x.shape
        x_4d = x.reshape(-1, 1, *shape_in[-2:])
        out = torch.nn.functional.interpolate(
            x_4d, scale_factor=self.factor, mode="nearest"
        )
        return out.reshape(*shape_in[:-2], *out.shape[-2:])

    def __repr__(self) -> str:
        return f"Downsample(factor={self.factor}, mode={self.mode!r})"


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
