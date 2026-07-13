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
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.linalg
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-dim reduction helpers (compatible with PyTorch builds that lack
# tuple dim support in median / quantile / amin / amax).
# ---------------------------------------------------------------------------


def _normalize_dims(ndim: int, dim: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert negative dims to positive and return sorted unique dims."""
    return tuple(sorted({d if d >= 0 else ndim + d for d in dim}))


def _get_valid_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Combine an optional explicit mask with an implicit NaN mask.

    Returns a boolean tensor where ``True`` indicates a valid (non-NaN,
    non-masked) element.  When *mask* is ``None``, the result is simply
    ``~torch.isnan(x)``.
    """
    valid = ~torch.isnan(x)
    if mask is not None:
        valid = valid & mask
    return valid


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


def _median(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.median over tuple dim."""
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(
        valid,
        x,
        torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
    )
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.nanmedian(t, dim=d, keepdim=k).values
    )


def _amin(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.amin over tuple dim."""
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(
        valid,
        x,
        torch.tensor(float("inf"), dtype=x.dtype, device=x.device),
    )
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.amin(t, dim=d, keepdim=k)
    )


def _amax(
    x: torch.Tensor,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.amax over tuple dim."""
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(
        valid,
        x,
        torch.tensor(float("-inf"), dtype=x.dtype, device=x.device),
    )
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.amax(t, dim=d, keepdim=k)
    )


def _quantile(
    x: torch.Tensor,
    q: float,
    dim: Tuple[int, ...],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware torch.quantile over tuple dim."""
    valid = _get_valid_mask(x, mask)
    x_clean = torch.where(
        valid,
        x,
        torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
    )
    return _reduce_keepdim(
        x_clean, dim, lambda t, d, k: torch.nanquantile(t, q, dim=d, keepdim=k)
    )


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------


def _upcast_for_precision(x: torch.Tensor) -> torch.Tensor:
    """Upcast to float64 for numerical stability, skipping if already float64.

    For float16/bfloat16, use float32 (sufficient precision, avoids
    unnecessary 4× memory doubling from float16→float64).
    """
    if x.dtype == torch.float64:
        return x
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x.double()


def safe_arcsinh(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``arcsinh(scale * x)`` using float64 internally.

    This preserves precision across the large dynamic range typical of
    astronomical images (LSST / SDSS convention).
    """
    orig_dtype = x.dtype
    out = torch.arcsinh(_upcast_for_precision(x) * scale)
    return out.to(orig_dtype)


def safe_log(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute ``log(x)`` with a floor at *eps* to avoid -inf.

    Uses float64 internally for precision.
    """
    orig_dtype = x.dtype
    out = torch.log(torch.clamp_min(_upcast_for_precision(x), eps))
    return out.to(orig_dtype)


def estimate_background(
    x: torch.Tensor,
    dim: Tuple[int, ...] = (-2, -1),
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust background estimator: median and MAD-based dispersion.

    Parameters
    ----------
    mask :
        Optional boolean mask where ``True`` indicates a valid pixel.
        Masked-out pixels (and any NaN values) are excluded from the
        median and MAD computation.

    Returns
    -------
    med : Tensor
        Per-pixel-group median (keepdim=True).
    std_approx : Tensor
        MAD × 1.4826 ≈ standard deviation of the background.
    """
    with torch.no_grad():
        med = _median(x, dim, mask=mask)
        mad = _median(torch.abs(x - med), dim, mask=mask)
        std_approx = mad.mul_(1.4826)
    return med, std_approx


def zscale_limits(
    x: torch.Tensor,
    contrast: float = 0.25,
    dim: Tuple[int, ...] = (-2, -1),
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IRAF-style zscale auto-contrast limits (fast proxy).

    Parameters
    ----------
    mask :
        Optional boolean mask where ``True`` indicates a valid pixel.
        Masked-out pixels (and any NaN values) are excluded from the
        median, MAD, min, and max computations.

    Returns (z1, z2) clipped to [vmin, vmax] with a fallback when the image
    is constant (z1 == z2).
    """
    with torch.no_grad():
        med, std = estimate_background(x, dim=dim, mask=mask)
        z1 = med - (std / max(contrast, 1e-5))
        z2 = med + (std / max(contrast, 1e-5))

        vmin = _amin(x, dim, mask=mask)
        vmax = _amax(x, dim, mask=mask)
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

    All transforms accept an optional ``mask`` parameter
    (``torch.Tensor | None``) on both :meth:`forward` and
    :meth:`inverse`.  The mask is a boolean tensor where ``True``
    indicates a valid pixel.  Transforms that compute statistics
    (median, min, max, etc.) use the mask to exclude invalid
    pixels; pointwise transforms can safely ignore it.
    """

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise NotImplementedError

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise NotImplementedError

    def __call__(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        return self.forward(x, mask=mask)


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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for t in self.transforms:
            x = t(x, mask=mask)
        return x

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for t in reversed(self.transforms):
            x = t.inverse(x, mask=mask)
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return safe_arcsinh(x, self.a).div_(self._norm)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (
            torch.sinh(_upcast_for_precision(x) * self._norm).div_(self.a).to(x.dtype)
        )

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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_clamped = torch.clamp_min(x, 0.0)
        return (
            safe_log(1.0 + self.a * x_clamped, eps=self.eps)
            .div_(math.log(10))
            .div_(self._norm)
        )

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        val = torch.pow(10.0, _upcast_for_precision(x) * self._norm).sub_(1.0)
        return val.div_(self.a).to(orig_dtype)

    def __repr__(self) -> str:
        return f"LogStretch(a={self.a}, eps={self.eps})"


class SqrtStretch(FITSTransform):
    """Square-root stretch — stabilises Poisson variance.

    .. note::
       Negative values are silently clamped to zero.
    """

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(x, 0.0))

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        z1, z2 = zscale_limits(x, contrast=self.contrast, dim=self.dim, mask=mask)
        self._last_state = (z1, z2)
        return (x - z1).div_(z2 - z1)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        med, std = estimate_background(x, dim=self.dim, mask=mask)
        self._last_med = med
        self._last_std = std
        return (x - med).div_(torch.clamp_min(std, 1e-9))

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bg, _ = estimate_background(x, dim=self.dim, mask=mask)
        self._last_bg = bg
        return x - bg

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            lower = _quantile(x, self.lower_pct, self.dim, mask=mask)
            upper = _quantile(x, self.upper_pct, self.dim, mask=mask)

        self._last_state = (lower, upper)
        clipped = torch.clamp(x, lower, upper)
        denom = torch.where(upper == lower, torch.ones_like(upper), upper - lower)
        return (clipped - lower).div_(denom)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            vmin = _amin(x, self.dim, mask=mask)
            vmax = _amax(x, self.dim, mask=mask)
            # Data-relative epsilon to avoid float32 underflow on constant images.
            _eps = torch.maximum(
                torch.tensor(1e-6, device=x.device, dtype=vmin.dtype),
                vmin.abs() * 1e-6,
            )
            vmax = torch.where(vmin == vmax, vmin + _eps, vmax)
        self._last_state = (vmin, vmax)
        return (x - vmin).div_(vmax - vmin)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.bscale == 1.0 and self.bzero == 0.0:
            return x
        result = x.to(torch.float32)
        if self.bscale != 1.0:
            result = result.mul_(self.bscale)
        if self.bzero != 0.0:
            result = result.add_(self.bzero)
        return result.to(x.dtype) if x.dtype != torch.float32 else result

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
    """Fit a low-order polynomial continuum via iterative sigma-clipping.

    Uses batched normal equations (``torch.linalg.solve`` on the whole
    batch at once) instead of a per-spectrum ``torch.linalg.lstsq`` loop,
    reducing the Python-loop overhead from O(n) lstsq calls to a single
    batched solve.
    """
    n, length = x.shape
    t = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)
    A = torch.stack([t**k for k in range(order + 1)], dim=1)  # [length, order+1]
    ridge = 1e-6 * torch.eye(order + 1, device=x.device, dtype=x.dtype)

    mask = torch.ones(n, length, dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        # Ensure each spectrum has enough unmasked points; reset
        # masks that are too sparse before the batched solve.
        counts = mask.sum(dim=1)  # [n]
        too_few = counts <= order
        if too_few.any():
            mask = mask.clone()
            mask[too_few] = True

        # Batched normal equations: solve (A^T W A) c = A^T W y
        # where W = diag(mask) for each spectrum.  A is shared across
        # all spectra, so we broadcast and use bmm.
        A_exp = A.unsqueeze(0)  # [1, length, order+1]
        mask_f = mask.unsqueeze(2).to(x.dtype)  # [n, length, 1]
        A_masked = A_exp * mask_f  # [n, length, order+1]
        AtA = torch.bmm(A_masked.transpose(1, 2), A_masked) + ridge
        Aty = torch.bmm(A_masked.transpose(1, 2), x.unsqueeze(2))  # [n, order+1, 1]
        try:
            coeffs = torch.linalg.solve(AtA, Aty).squeeze(2)  # [n, order+1]
        except RuntimeError:
            # Fallback for singular matrices (rare with ridge)
            coeffs = torch.zeros(n, order + 1, device=x.device, dtype=x.dtype)
            for i in range(n):
                try:
                    coeffs[i] = torch.linalg.solve(AtA[i], Aty[i]).squeeze(1)
                except RuntimeError:
                    pass  # Leave zeros

        continuum = (A @ coeffs.T).T  # [n, length]
        residuals = x - continuum
        # Compute std only on currently-unmasked pixels (masked outliers
        # would inflate the std and prevent convergence).
        count = mask_f.sum(dim=1)  # [n, 1]
        mean_res = (residuals * mask_f.squeeze(2)).sum(
            dim=1, keepdim=True
        ) / torch.clamp_min(count, 1.0)
        var = ((residuals - mean_res) ** 2 * mask_f.squeeze(2)).sum(
            dim=1, keepdim=True
        ) / torch.clamp_min(count, 1.0)
        std = torch.sqrt(torch.clamp_min(var, 0.0))
        new_mask = residuals.abs() < n_sigma * torch.clamp_min(std, 1e-9)
        if torch.equal(new_mask, mask):
            break
        mask = new_mask

    return continuum


def _to_pt_mode(mode: str) -> str:
    """Map user-facing mode names to PyTorch-native function modes."""
    return {"linear": "bilinear", "nearest": "nearest", "cubic": "bicubic"}[mode]


def _to_interpolate_2d_mode(mode: str) -> tuple[str, dict]:
    """Map user mode to 2-D ``F.interpolate`` mode and kwargs."""
    if mode == "cubic":
        return "bicubic", {"align_corners": True}
    if mode == "area":
        return "area", {}
    raise ValueError(f"_to_interpolate_2d_mode expects 'cubic' or 'area', got {mode!r}")


def _resample_1d(
    y: torch.Tensor,
    x_old: torch.Tensor,
    x_new: torch.Tensor,
    *,
    mode: str = "linear",
) -> torch.Tensor:
    """Resample 1-D data along the last dimension at arbitrary positions.

    Uses PyTorch's native :func:`torch.nn.functional.interpolate` and
    :func:`torch.nn.functional.grid_sample` for maximum speed on both
    CPU and GPU.  Falls back to ``searchsorted``-based interpolation
    only for truly irregular ``x_old`` grids where the torch-native
    functions cannot be used.

    ``y`` has shape ``[..., L_src]`` — values at positions ``x_old``.
    ``x_old`` has shape ``[L_src]`` — source-grid coordinates (must be
    monotonically increasing).
    ``x_new`` has shape ``[L_dst]`` — target positions.

    Returns a tensor of shape ``[..., L_dst]``.

    Parameters
    ----------
    mode : str
        ``"linear"`` (default, bilinear in torch), ``"nearest"``,
        ``"cubic"`` (bicubic in torch), or ``"area"`` (flux-conserving
        box average — ideal for preserving narrow emission/absorption
        lines during resampling).

    Notes
    -----
    This is the engine behind :class:`DopplerShift` and any transform
    that resamples spectral axes.  ``"area"`` mode is recommended for
    spiky spectroscopy data because it conserves flux per output bin
    without smearing narrow features across neighboring pixels.

    Path selection (in order):

    1. **x_old uniform + mode ≠ area → F.grid_sample**
       Fastest path: normalizes x_new to [-1..1], builds a 4-D grid
       tensor, and calls ``F.grid_sample(bilinear|nearest|bicubic)``.

    2. **Both grids uniform → F.interpolate**
       Uses ``F.interpolate`` (1-D for linear/nearest, 2-D reshape
       trick for cubic/area) — the simplest torch-native path.

    3. **Irregular x_old or mode=area → searchsorted fallback**
       Falls back to index-based interpolation (linear, nearest,
       cubic Catmull–Rom, or box-average area).
    """
    if x_new.numel() == 0:
        return y[..., :0]

    if mode not in ("linear", "nearest", "cubic", "area"):
        raise ValueError(
            f"mode must be 'linear', 'nearest', 'cubic', or 'area', got {mode!r}"
        )

    shape_in = y.shape
    L_src = shape_in[-1]
    L_dst = x_new.shape[0]

    if L_src == 0:
        return y[..., :0]

    y_2d = y.reshape(-1, L_src)  # [N, L_src]
    if L_src == 1:
        # Single-point source: broadcast to all output positions.
        return y_2d[:, :1].expand(-1, L_dst).reshape(*shape_in[:-1], L_dst)

    # ---- fast path: x_old is uniform → use F.grid_sample ----
    if L_src >= 2:
        dx = x_old[1] - x_old[0]
        _eps = max(1e-12, abs(dx.item()) * 1e-6)
        mid = L_src // 2
        checks = [1, mid, mid + 1, L_src - 1] if L_src >= 4 else [1]
        is_uniform = True
        for idx_check in checks:
            if (
                idx_check < L_src
                and abs((x_old[idx_check] - x_old[idx_check - 1] - dx).item()) > _eps
            ):
                is_uniform = False
                break
        if is_uniform:
            is_uniform = torch.allclose(
                x_old[1:] - x_old[:-1],
                dx.expand(L_src - 1),
                atol=_eps,
            )
    else:
        is_uniform = False

    if is_uniform and mode != "area":
        # F.grid_sample assumes the input tensor is on a uniform grid.
        # Reshape [N, L_src] → [N, 1, 1, L_src] (4-D).
        y_4d = y_2d.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, L_src]

        # Normalize x_new to [-1, 1] (PyTorch's grid_sample convention).
        x0, x1 = x_old[0], x_old[-1]
        denom = (x1 - x0).clamp_min(1e-30)
        x_norm = 2.0 * (x_new - x0) / denom - 1.0

        # Build grid: [N, 1, L_dst, 2] — x is the spectral coordinate, y=0.
        grid_x = x_norm.unsqueeze(0).expand(y_2d.shape[0], -1)  # [N, L_dst]
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # [N, 1, L_dst, 2]

        out = F.grid_sample(
            y_4d,
            grid,
            mode=_to_pt_mode(mode),
            padding_mode="border",
            align_corners=True,
        )  # [N, 1, 1, L_dst]
        return out.reshape(*shape_in[:-1], L_dst)

    # ---- uniform → uniform path (both sides regular) — use F.interpolate ----
    # Check if x_new is also uniform (which means we can use F.interpolate).
    if L_dst >= 2:
        dx_new = x_new[1] - x_new[0]
        _eps_new = max(1e-12, abs(dx_new.item()) * 1e-6)
        mid_new = L_dst // 2
        checks_new = [1, mid_new, mid_new + 1, L_dst - 1] if L_dst >= 4 else [1]
        is_new_uniform = True
        for idx_check in checks_new:
            if (
                idx_check < L_dst
                and abs((x_new[idx_check] - x_new[idx_check - 1] - dx_new).item())
                > _eps_new
            ):
                is_new_uniform = False
                break
        if is_new_uniform:
            is_new_uniform = torch.allclose(
                x_new[1:] - x_new[:-1],
                dx_new.expand(L_dst - 1),
                atol=_eps_new,
            )
    else:
        is_new_uniform = L_dst <= 1

    if is_uniform and is_new_uniform:
        if mode in ("linear", "nearest"):
            y_3d = y_2d.unsqueeze(1)  # [N, 1, L_src]
            out = F.interpolate(
                y_3d,
                size=L_dst,
                mode=mode,
                **({"align_corners": True} if mode == "linear" else {}),
            )  # [N, 1, L_dst]
        else:
            # cubic or area: reshape to 2-D for PyTorch's 2D interpolate.
            # Shape [N, 1, 1, L_src] — height=1 "image".
            y_4d = y_2d.unsqueeze(1).unsqueeze(1)
            pt_mode, kwargs = _to_interpolate_2d_mode(mode)
            out = F.interpolate(y_4d, size=(1, L_dst), mode=pt_mode, **kwargs)
        return out.reshape(*shape_in[:-1], L_dst)

    # ---- fallback: searchsorted for irregular x_old or area mode ----
    idx = torch.searchsorted(x_old, x_new)
    idx = idx.clamp(1, L_src - 1)

    if mode == "nearest":
        x_lo = x_old[idx - 1]
        x_hi = x_old[idx]
        pick_left = (x_new - x_lo).abs() <= (x_hi - x_new).abs()
        near_idx = torch.where(pick_left, idx - 1, idx)
        return y_2d[:, near_idx].reshape(*shape_in[:-1], L_dst)

    if mode == "area":
        # NOTE: O(N × L_dst) Python loops here; prefer uniform grids
        # (`_resample_scale`) for area-mode performance on large datasets.
        # Flux-conserving box average for irregular grids.
        # For each output bin at x_new[j], average all input pixels that
        # overlap the interval [x_new[j] - half_width, x_new[j] + half_width].
        half = (
            (x_new[1] - x_new[0]).abs() / 2.0
            if L_dst >= 2
            else torch.tensor(1.0, device=x_new.device, dtype=x_new.dtype)
        )
        out = torch.zeros(y_2d.shape[0], L_dst, device=y.device, dtype=y.dtype)
        for j in range(L_dst):
            lo = x_new[j] - half
            hi = x_new[j] + half
            # Find all input pixels overlapping [lo, hi]
            ilo = torch.searchsorted(x_old, lo).clamp(0, L_src - 1)
            ihi = torch.searchsorted(x_old, hi).clamp(1, L_src)
            # For each spectrum, sum over overlapping pixels and divide
            # by the fractional overlap width.
            for b in range(y_2d.shape[0]):
                _ilo, _ihi = ilo.item(), ihi.item()
                if _ihi <= _ilo:
                    # Degenerate bin: take nearest neighbor
                    _clo = torch.searchsorted(x_old, x_new[j]).clamp(0, L_src - 1)
                    out[b, j] = y_2d[b, _clo]
                else:
                    out[b, j] = y_2d[b, _ilo:_ihi].mean()
        return out.reshape(*shape_in[:-1], L_dst)

    if mode == "linear":
        x_lo = x_old[idx - 1]
        x_hi = x_old[idx]
        y_lo = y_2d[:, idx - 1]
        y_hi = y_2d[:, idx]
        frac = (x_new - x_lo) / (x_hi - x_lo).clamp_min(1e-30)
        return (y_lo + (y_hi - y_lo) * frac.unsqueeze(0)).reshape(*shape_in[:-1], L_dst)

    # mode == "cubic" — Catmull–Rom fallback
    im2 = (idx - 2).clamp(0, L_src - 1)
    im1 = (idx - 1).clamp(0, L_src - 1)
    ip1 = (idx + 1).clamp(0, L_src - 1)
    at_left = idx <= 1
    at_right = idx >= L_src - 1
    interior = ~(at_left | at_right)
    xm1 = x_old[im1]
    x0v = x_old[idx]
    dx_seg = (x0v - xm1).clamp_min(1e-30)
    t = ((x_new - xm1) / dx_seg).unsqueeze(0).clamp(0, 1)
    t2, t3 = t * t, t * t * t
    p0, p1 = y_2d[:, im2], y_2d[:, im1]
    p2, p3 = y_2d[:, idx], y_2d[:, ip1]
    cubic = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    x_lo, x_hi = x_old[idx - 1], x_old[idx]
    y_lo, y_hi = y_2d[:, idx - 1], y_2d[:, idx]
    frac_l = (x_new - x_lo) / (x_hi - x_lo).clamp_min(1e-30)
    linear = y_lo + (y_hi - y_lo) * frac_l.unsqueeze(0)
    out = torch.where(interior.unsqueeze(0), cubic, linear)
    return out.reshape(*shape_in[:-1], L_dst)


def _resample_scale(
    x: torch.Tensor,
    scale: float,
    *,
    mode: str = "linear",
) -> torch.Tensor:
    """Resample the last dimension of *x* by a factor *scale*.

    Uses :func:`torch.nn.functional.interpolate` (torch-native C++ / CUDA)
    for maximum throughput on uniform-grid resampling.

    Parameters
    ----------
    mode : str
        ``"linear"`` (default), ``"nearest"``, ``"cubic"`` (bicubic),
        or ``"area"`` (flux-conserving box average — recommended for
        spiky spectra with narrow emission/absorption lines).
    """
    shape_in = x.shape
    x_2d = x.reshape(-1, shape_in[-1])
    L_src = x_2d.shape[1]
    L_dst = max(2, int(L_src * scale))

    if mode not in ("linear", "nearest", "cubic", "area"):
        raise ValueError(
            f"mode must be 'linear', 'nearest', 'cubic', or 'area', got {mode!r}"
        )

    if mode in ("linear", "nearest"):
        y_3d = x_2d.unsqueeze(1)  # [N, 1, L_src]
        out = F.interpolate(
            y_3d,
            size=L_dst,
            mode=mode,
            **({"align_corners": True} if mode == "linear" else {}),
        )  # [N, 1, L_dst]
    else:
        # cubic → bicubic, area → area — both need 2-D reshape.
        y_4d = x_2d.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, L_src]
        pt_mode, kwargs = _to_interpolate_2d_mode(mode)
        out = F.interpolate(y_4d, size=(1, L_dst), mode=pt_mode, **kwargs)

    return out.reshape(*shape_in[:-1], L_dst)


# Backward-compatible aliases
_resample_spectrum = _resample_scale  # old name


def _linear_interp_1d(
    y: torch.Tensor, x_orig: torch.Tensor, x_new: torch.Tensor
) -> torch.Tensor:
    """Backward-compatible wrapper around :func:`_resample_1d`."""
    return _resample_1d(y, x_orig, x_new, mode="linear")


# ---------------------------------------------------------------------------
# FITS table-aware transforms (TSCAL/TZERO/TNULL)
# ---------------------------------------------------------------------------


class FITSScaleColumns(FITSTransform):
    """Apply or remove TSCAL/TZERO scaling to table column tensors.

    Reads TSCAL and TZERO keywords for each column from a FITS table header
    and applies ``physical = TSCAL * stored + TZERO``.  Columns with default
    values (TSCAL=1.0, TZERO=0.0) are passed through unchanged.

    ``forward`` applies scaling: stored → physical.
    ``inverse`` removes it: ``(physical - TZERO) / TSCAL``.

    Parameters
    ----------
    header : dict
        FITS table header dict-like with TTYPE*/TFORM*/TSCAL*/TZERO* keywords.

    Example
    -------
    >>> header = {"TFIELDS": 2, "TTYPE1": "FLUX", "TFORM1": "E",
    ...            "TSCAL1": 0.001, "TZERO1": 0.0,
    ...            "TTYPE2": "MAG", "TFORM2": "E",
    ...            "TSCAL2": 1.0, "TZERO2": 25.0}
    >>> scaler = FITSScaleColumns.from_header(header)
    >>> physical = scaler({"FLUX": raw_flux, "MAG": raw_mag})
    >>> raw = scaler.inverse(physical)
    """

    def __init__(self, scales: dict[str, tuple[float, float]]) -> None:
        """
        Parameters
        ----------
        scales : dict[str, tuple[float, float]]
            Mapping of column name → (TSCAL, TZERO).
        """
        self.scales: dict[str, tuple[float, float]] = {
            name: (float(ts), float(tz))
            for name, (ts, tz) in scales.items()
            if ts != 1.0 or tz != 0.0
        }

    @classmethod
    def from_header(cls, header: dict) -> FITSScaleColumns:
        """Construct from a FITS table header dict-like object."""
        from .fits_schema import iter_table_columns  # noqa: PLC0415

        scales: dict[str, tuple[float, float]] = {}
        for col in iter_table_columns(header):
            tscal = float(col.tscal) if col.tscal is not None else 1.0
            tzero = float(col.tzero) if col.tzero is not None else 0.0
            scales[col.name] = (tscal, tzero)
        return cls(scales)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.scales:
            return x
        out = dict(x)
        for name, (tscal, tzero) in self.scales.items():
            if name not in out:
                continue
            val = out[name]
            if tscal == 1.0 and tzero == 0.0:
                continue
            result = val.to(torch.float32)
            if tscal != 1.0:
                result = result.mul_(tscal)
            if tzero != 0.0:
                result = result.add_(tzero)
            out[name] = result.to(val.dtype) if val.dtype != torch.float32 else result
        return out

    def inverse(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.scales:
            return x
        out = dict(x)
        for name, (tscal, tzero) in self.scales.items():
            if name not in out:
                continue
            val = out[name]
            if tscal == 1.0 and tzero == 0.0:
                continue
            result = val.to(torch.float32)
            if tzero != 0.0:
                result = result.sub_(tzero)
            if tscal != 1.0:
                result = result.div_(tscal)
            out[name] = result.to(val.dtype) if val.dtype != torch.float32 else result
        return out

    def __repr__(self) -> str:
        items = ", ".join(
            f"{n!r}: ({ts}, {tz})" for n, (ts, tz) in sorted(self.scales.items())
        )
        return f"FITSScaleColumns({{{items}}})"


class TNullToNan(FITSTransform):
    """Replace FITS TNULL sentinel values with NaN.

    Reads TNULL keywords from a FITS table header and replaces the
    corresponding sentinel values in each tensor column with NaN.
    Integer columns are promoted to float32 so NaN can be represented.

    ``inverse`` is not available — null replacement is lossy.

    Parameters
    ----------
    header : dict
        FITS table header dict-like with TTYPE*/TNULL* keywords.

    Example
    -------
    >>> header = {"TFIELDS": 1, "TTYPE1": "FLUX", "TFORM1": "J", "TNULL1": -999}
    >>> nuller = TNullToNan.from_header(header)
    >>> clean = nuller({"FLUX": torch.tensor([1, -999, 3], dtype=torch.int32)})
    >>> # FLUX is now float32 with NaN at position 1
    """

    def __init__(self, nulls: dict[str, float]) -> None:
        """
        Parameters
        ----------
        nulls : dict[str, float]
            Mapping of column name → TNULL value.
        """
        self.nulls: dict[str, float] = {name: float(v) for name, v in nulls.items()}

    @classmethod
    def from_header(cls, header: dict) -> TNullToNan:
        """Construct from a FITS table header dict-like object."""
        from .fits_schema import column_tnull_map  # noqa: PLC0415

        nulls = column_tnull_map(header)
        return cls(nulls)

    def forward(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.nulls:
            return x
        out = dict(x)
        for name, tnull in self.nulls.items():
            if name not in out:
                continue
            val = out[name]
            # Promote integer columns to float32 so NaN is representable
            if val.dtype not in (torch.float32, torch.float64):
                val = val.to(torch.float32)
            null_mask = val.eq(tnull)
            out[name] = torch.where(
                null_mask,
                torch.tensor(float("nan"), dtype=val.dtype, device=val.device),
                val,
            )
        return out

    def inverse(
        self, x: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError(
            "TNullToNan.inverse() is not available — null replacement is lossy."
        )

    def __repr__(self) -> str:
        items = ", ".join(f"{n!r}: {v}" for n, v in sorted(self.nulls.items()))
        return f"TNullToNan({{{items}}})"


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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    ``inverse`` applies the opposite shift (``-z / (1 + z)``), interpolating
    the forward-resampled values back to the original grid positions.

    Parameters
    ----------
    z : float
        Redshift.  Positive values stretch the spectrum (redshift).
    """

    def __init__(self, z: float = 0.0) -> None:
        self.z = float(z)
        self._orig_length: int | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.z == 0.0:
            return x
        self._orig_length = x.shape[-1]
        return _resample_scale(x, 1.0 + self.z)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.z == 0.0:
            return x
        orig_len = self._orig_length
        if orig_len is None:
            raise RuntimeError(
                "DopplerShift.inverse() requires a prior forward() pass."
            )
        shape_in = x.shape
        x_2d = x.reshape(-1, shape_in[-1])
        forward_len = x_2d.shape[1]
        # The forward pass resampled the original spectrum (at integer
        # positions 0..orig_len-1) to forward_len points uniformly
        # spanning [0, orig_len-1].  To invert, we interpolate those
        # forward values back onto the original integer grid.
        forward_grid = torch.linspace(
            0, orig_len - 1, forward_len, device=x.device, dtype=x.dtype
        )
        orig_grid = torch.arange(orig_len, device=x.device, dtype=x.dtype)
        out = _resample_1d(x_2d, forward_grid, orig_grid)
        return out.reshape(*shape_in[:-1], orig_len)

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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fold into phase bins."""
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

        # Scatter sum into bins — vectorized via scatter_add_ + bincount
        folded = torch.zeros(n_samples, self.n_bins, device=x.device, dtype=x.dtype)
        bin_idx_exp = bin_idx.unsqueeze(0).expand(n_samples, -1)
        folded.scatter_add_(1, bin_idx_exp, x_2d)
        counts = torch.bincount(bin_idx, minlength=self.n_bins).to(x.dtype)

        # Normalise by counts (mean per bin)
        folded = folded / torch.clamp_min(counts.unsqueeze(0), 1.0)

        return folded.reshape(*shape_in[:-1], self.n_bins)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.band_dim if self.band_dim >= 0 else ndim + self.band_dim
        # Unbind along band dimension for dimension-agnostic access
        bands = torch.unbind(x, dim=dim)
        return self.func(bands)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise RuntimeError(
            "BandMath.inverse() is not available — band arithmetic is lossy."
        )

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", repr(self.func))
        return f"BandMath(func={name}, band_dim={self.band_dim})"


# ---------------------------------------------------------------------------
# Continuum / baseline estimators (all use additive decomposition:
#   Original = Estimate + Residuals  →  invertible)
# ---------------------------------------------------------------------------


class GlobalScalarNorm(FITSTransform):
    """Normalise by dividing by a global scalar statistic.

    The simplest linear transform — used by virtually all astronomical
    foundation models (AstroCLIP, SpecFormer, SpecHub) as the only
    preprocessing step.  A neural network's first layer can implicitly
    un-learn this through gradient descent.

    ``inverse`` multiplies by the cached scalar.

    Parameters
    ----------
    stat : str
        Statistic to compute: ``"median"`` (default, robust), ``"max"``,
        ``"mean"``, or ``"rms"``.
    dim :
        Dimensions over which to compute the statistic.  Default ``None``
        (all dims — a single scalar for the whole tensor).
    """

    def __init__(
        self, stat: str = "median", dim: Optional[Tuple[int, ...]] = None
    ) -> None:
        if stat not in ("median", "max", "mean", "rms"):
            raise ValueError("stat must be 'median', 'max', 'mean', or 'rms'")
        self.stat = stat
        self.dim = dim
        self._scalar: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        dim = self.dim if self.dim is not None else tuple(range(x.ndim))
        with torch.no_grad():
            if self.stat == "median":
                scalar = _median(x, dim, mask=mask)
            elif self.stat == "max":
                scalar = _amax(x, dim, mask=mask)
            elif self.stat == "mean":
                if mask is not None:
                    valid = _get_valid_mask(x, mask)
                    x_clean = torch.where(
                        valid,
                        x.float(),
                        torch.zeros_like(x.float()),
                    )
                    count = valid.float().sum(dim=dim, keepdim=True)
                    scalar = x_clean.sum(dim=dim, keepdim=True) / torch.clamp_min(
                        count, 1.0
                    )
                    scalar = scalar.to(x.dtype)
                else:
                    scalar = x.float().mean(dim=dim, keepdim=True).to(x.dtype)
            else:  # rms
                if mask is not None:
                    valid = _get_valid_mask(x, mask)
                    x_clean = torch.where(
                        valid,
                        x.float() ** 2,
                        torch.zeros_like(x.float()),
                    )
                    count = valid.float().sum(dim=dim, keepdim=True)
                    scalar = torch.sqrt(
                        x_clean.sum(dim=dim, keepdim=True) / torch.clamp_min(count, 1.0)
                    ).to(x.dtype)
                else:
                    scalar = torch.sqrt(
                        (x.float() ** 2).mean(dim=dim, keepdim=True)
                    ).to(x.dtype)
        self._scalar = scalar
        return x / torch.clamp_min(scalar, 1e-30)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._scalar is None:
            raise RuntimeError(
                "GlobalScalarNorm.inverse() requires a prior forward() pass."
            )
        return x * self._scalar

    def __repr__(self) -> str:
        return f"GlobalScalarNorm(stat={self.stat!r}, dim={self.dim})"


def _sg_coeffs(window_length: int, polyorder: int, deriv: int = 0) -> torch.Tensor:
    """Compute Savitzky–Golay filter coefficients.

    Returns a 1D tensor of length *window_length* with the convolution
    coefficients for derivative order *deriv*.
    """
    if window_length % 2 == 0 or window_length < 3:
        raise ValueError("window_length must be odd and >= 3")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    half = window_length // 2
    # Build Vandermonde matrix: x values relative to window centre
    x = torch.arange(-half, half + 1, dtype=torch.float64)
    A = torch.stack([x**k for k in range(polyorder + 1)], dim=1)  # [W, P+1]
    # Target: unit impulse at the window centre for derivative order *deriv*.
    # For deriv=0 (smoothing), we want the convolution to reproduce the
    # central value of a polynomial, which is solved by lstsq.
    y = torch.zeros(window_length, dtype=torch.float64)
    y[half] = 1.0
    # Solve A @ c ≈ y to get polynomial coefficients, then evaluate at all
    # window positions to produce the full convolution kernel of length W.
    c = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)  # [P+1]
    coeffs = A @ c  # [W]
    return coeffs.float()


class SavitzkyGolayFilter(FITSTransform):
    """Savitzky–Golay polynomial smoothing filter.

    Convolves the data along *dim* with pre-computed SG coefficients.
    The filter is additive: ``Original = Smoothed + Residuals``,
    so ``inverse`` recovers the original by re-adding the residuals.

    This is the standard smoothing method in laboratory spectroscopy
    (UV/VIS/NIR) and is fully information-preserving when residuals
    are retained.

    Parameters
    ----------
    window_length : int
        Odd window length in samples (>= 3).
    polyorder : int
        Polynomial order for the local fit (< window_length).
    dim : int
        Dimension to filter along (default -1).

    Notes
    -----
    The filter is applied via ``F.conv1d``, which is efficient on GPU.
    Edge values are padded by reflecting the boundary.
    """

    def __init__(
        self, window_length: int = 7, polyorder: int = 3, dim: int = -1
    ) -> None:
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.dim = int(dim)
        # Pre-compute SG coefficients once
        coeffs = _sg_coeffs(window_length, polyorder)
        # Reshape for F.conv1d: [out_channels, in_channels/groups, kernel]
        self._coeffs_1d = coeffs.view(1, 1, -1)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.window_length < 3:
            return x
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        pad = self.window_length // 2

        # Move filtering dim to last position for conv1d
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]

        # Pad and convolve
        x_padded = torch.nn.functional.pad(
            x_flat.unsqueeze(1), (pad, pad), mode="reflect"
        )  # [N, 1, L+2*pad]
        smoothed = torch.nn.functional.conv1d(
            x_padded,
            self._coeffs_1d.to(device=x.device, dtype=x.dtype),
        ).squeeze(1)  # [N, L]

        smoothed = smoothed.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - smoothed
        return smoothed

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "SavitzkyGolayFilter.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"SavitzkyGolayFilter(window_length={self.window_length}, "
            f"polyorder={self.polyorder}, dim={self.dim})"
        )


class RunningPercentile(FITSTransform):
    """Running percentile continuum estimator.

    Computes the *percentile*-th percentile in a sliding window along
    *dim*, producing a smooth upper-envelope continuum.  This is the
    standard quick-look continuum method in many spectroscopic surveys.

    The transform is additive: ``Original = Continuum + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    percentile : float
        Percentile to compute in each window (0–100).  Default 90 gives
        the upper envelope while ignoring narrow absorption lines.
    window_size : int
        Sliding window size in samples.  Must be odd and >= 3.
    dim : int
        Dimension to filter along (default -1).
    """

    def __init__(
        self, percentile: float = 90.0, window_size: int = 21, dim: int = -1
    ) -> None:
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("window_size must be odd and >= 3")
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        self.percentile = float(percentile)
        self.window_size = int(window_size)
        self.dim = int(dim)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        pad = self.window_size // 2
        q = self.percentile / 100.0

        # Move filter dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]

        with torch.no_grad():
            # Pad edges with reflection for continuity
            x_padded = torch.nn.functional.pad(
                x_flat, (pad, pad), mode="reflect"
            )  # [N, L+2*pad]

            # Unfold into windows: [N, L, window_size]
            windows = x_padded.unfold(-1, self.window_size, 1)

            # Compute percentile along the window dimension
            continuum = torch.quantile(
                windows.float(), q, dim=-1, interpolation="linear"
            ).to(x.dtype)

        continuum = continuum.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - continuum
        return continuum

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "RunningPercentile.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"RunningPercentile(percentile={self.percentile}, "
            f"window_size={self.window_size}, dim={self.dim})"
        )


class UpperEnvelopeContinuum(FITSTransform):
    """Upper-envelope continuum estimation via local-maxima interpolation.

    Finds local maxima in sliding windows, then interpolates between them
    to produce a smooth continuum.  This approximates the alpha-shape
    / convex-hull method used by RASSINE (Cretignier et al. 2020) but
    is implemented entirely in PyTorch.

    The transform is additive: ``Original = Continuum + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    window : int
        Half-width for local-maximum detection.  A point is a local max
        if it is the largest in [i-window, i+window].  Larger values
        produce a smoother (less concave) continuum.
    smooth : float
        Optional Gaussian sigma for smoothing the final continuum.
        Default 0 (no smoothing).
    dim : int
        Dimension to operate along (default -1).
    """

    def __init__(self, window: int = 11, smooth: float = 0.0, dim: int = -1) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.smooth = float(smooth)
        self.dim = int(dim)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        # Move operating dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]
        n_spectra, length = x_flat.shape

        with torch.no_grad():
            # Pad for local-max search at edges
            x_padded = torch.nn.functional.pad(
                x_flat, (self.window, self.window), mode="reflect"
            )  # [N, L+2*W]
            windows = x_padded.unfold(-1, 2 * self.window + 1, 1)  # [N, L, 2W+1]
            # A point is a local max if it equals the window maximum
            is_local_max = x_flat == windows.max(dim=-1).values  # [N, L]

            # Vectorized upper envelope via cummax-based nearest local max lookup.
            # For each position, find the nearest local max to the left and right,
            # then linearly interpolate between their values.
            positions = torch.arange(length, device=x.device, dtype=x.dtype)
            pos_exp = positions.unsqueeze(0).expand(n_spectra, -1)  # [N, L]

            # Nearest local max to the left: forward-fill local max positions.
            # Set non-local-max positions to -inf, then cummax gives the running
            # maximum position (i.e., nearest local max to the left).
            lm_positions = torch.where(
                is_local_max, pos_exp, torch.full_like(pos_exp, float("-inf"))
            )
            left_max_pos, _ = torch.cummax(lm_positions, dim=1)  # [N, L]

            # Nearest local max to the right: reverse, cummax, reverse back.
            rev_pos = length - 1 - pos_exp
            rev_lm = torch.where(
                is_local_max.flip(1),
                rev_pos.flip(1),
                torch.full_like(rev_pos.flip(1), float("-inf")),
            )
            rev_cummax, _ = torch.cummax(rev_lm, dim=1)
            right_max_pos = (length - 1) - rev_cummax.flip(1)  # [N, L]

            # Clean up inf/-inf: where no left max exists, use right; vice versa.
            left_max_pos = torch.where(
                torch.isinf(left_max_pos), right_max_pos, left_max_pos
            )
            right_max_pos = torch.where(
                torch.isinf(right_max_pos), left_max_pos, right_max_pos
            )
            # If both were inf (no local maxima at all), clamp to 0.
            left_max_pos = torch.where(
                torch.isinf(left_max_pos), torch.zeros_like(left_max_pos), left_max_pos
            )
            right_max_pos = torch.where(
                torch.isinf(right_max_pos),
                torch.zeros_like(right_max_pos),
                right_max_pos,
            )

            # Gather values at nearest left/right local max positions.
            left_idx = left_max_pos.long().clamp(0, length - 1)
            right_idx = right_max_pos.long().clamp(0, length - 1)
            left_vals = torch.gather(x_flat, 1, left_idx)  # [N, L]
            right_vals = torch.gather(x_flat, 1, right_idx)  # [N, L]

            # Linear interpolation between left and right local max values.
            denom = torch.clamp_min(right_max_pos - left_max_pos, 1e-30)
            frac = (pos_exp - left_max_pos) / denom
            continuum_vec = left_vals + (right_vals - left_vals) * frac

            # Fallback for spectra with < 2 local maxima: use global max.
            has_enough = is_local_max.sum(dim=1) >= 2  # [N]
            max_vals = x_flat.max(dim=1, keepdim=True).values  # [N, 1]
            continuum = torch.where(
                has_enough.unsqueeze(1),
                continuum_vec,
                max_vals.expand(-1, length),
            )

            # Optional Gaussian smoothing
            if self.smooth > 0:
                half = int(math.ceil(3.0 * self.smooth))
                t_kernel = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
                kernel = torch.exp(-0.5 * (t_kernel / self.smooth) ** 2)
                kernel = kernel / kernel.sum()
                kernel_1d = kernel.view(1, 1, -1)
                cont_padded = torch.nn.functional.pad(
                    continuum.unsqueeze(1), (half, half), mode="reflect"
                )
                continuum = torch.nn.functional.conv1d(
                    cont_padded, kernel_1d.to(device=x.device, dtype=x.dtype)
                ).squeeze(1)

        continuum = continuum.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - continuum
        return continuum

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "UpperEnvelopeContinuum.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"UpperEnvelopeContinuum(window={self.window}, "
            f"smooth={self.smooth}, dim={self.dim})"
        )


def _haar_dwt_1d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-level 1D Haar discrete wavelet transform.

    Returns (approx, detail) where approx has half the length of x
    and detail holds the high-frequency coefficients.
    """
    # Ensure even length
    length = x.shape[-1]
    if length % 2 != 0:
        x = x[..., : length - 1]
        length -= 1
    # Average and difference
    approx = (x[..., 0::2] + x[..., 1::2]) / 2.0
    detail = (x[..., 0::2] - x[..., 1::2]) / 2.0
    return approx, detail


def _haar_idwt_1d(approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Inverse single-level 1D Haar DWT."""
    length = approx.shape[-1] * 2
    x = torch.zeros(
        *approx.shape[:-1], length, device=approx.device, dtype=approx.dtype
    )
    x[..., 0::2] = approx + detail
    x[..., 1::2] = approx - detail
    return x


class WaveletDecompose(FITSTransform):
    """Multi-level Haar wavelet decomposition.

    Decomposes the signal along *dim* into *levels* of approximation +
    detail coefficients.  The output stacks ``[approx_L, detail_L, ...,
    detail_1]`` along *dim*, preserving all information for a perfect
    reconstruction.

    This is a fully invertible frequency split — the approximation
    coefficients capture the broadband continuum, while detail
    coefficients capture narrow spectral features.  Neural networks
    can learn to attend to either frequency band independently.

    ``inverse`` reconstructs the original signal from the coefficients.

    Parameters
    ----------
    levels : int
        Number of decomposition levels (1–8).  Level 1 splits into
        approx (half-length) + detail; each subsequent level further
        splits the approximation.
    dim : int
        Dimension to decompose along (default -1).

    Notes
    -----
    Uses the Haar wavelet (simplest, fastest, and most common in
    astro-ML for continuum/feature separation).  The transform is
    orthogonal (up to the ``sqrt(2)`` scaling factor), so it is
    numerically stable and gradient-safe.
    """

    def __init__(self, levels: int = 3, dim: int = -1) -> None:
        if levels < 1 or levels > 8:
            raise ValueError("levels must be in [1, 8]")
        self.levels = int(levels)
        self.dim = int(dim)
        self._orig_shape: tuple[int, ...] | None = None
        self._padded: bool = False

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        self._orig_shape = x.shape

        # Move working dim to last position
        x_w = x.movedim(dim, -1)
        shape_prefix = x_w.shape[:-1]

        # Pad to make length divisible by 2^levels
        length = x_w.shape[-1]
        target = ((length + (1 << self.levels) - 1) >> self.levels) << self.levels
        self._padded = target > length
        if self._padded:
            pad_amount = target - length
            x_w = torch.nn.functional.pad(x_w, (0, pad_amount), mode="reflect")
            self._pad_amount = pad_amount

        # Multi-level decomposition
        coeffs: list[torch.Tensor] = []
        current = x_w.reshape(-1, x_w.shape[-1])  # [N, L]
        for _ in range(self.levels):
            approx, detail = _haar_dwt_1d(current)
            coeffs.append(detail.reshape(*shape_prefix, detail.shape[-1]))
            current = approx  # continue decomposing the approximation
        coeffs.append(current.reshape(*shape_prefix, current.shape[-1]))  # final approx

        # Stack [approx_L, detail_L, ..., detail_1] along working dim.
        # coeffs = [detail_1, ..., detail_L, approx_L] in order of
        # increasing level.  Reverse to get decreasing frequency.
        result = torch.cat(coeffs[::-1], dim=-1)
        return result.movedim(-1, dim)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._orig_shape is None:
            raise RuntimeError(
                "WaveletDecompose.inverse() requires a prior forward() pass."
            )
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        x_w = x.movedim(dim, -1)
        shape_prefix = x_w.shape[:-1]
        length = x_w.shape[-1]

        # Split into coefficient bands
        # The layout is [approx_L, detail_L, ..., detail_1]
        # Approx length = ceil(original / 2^L), but padded to power of 2
        target = (
            (self._orig_shape[dim] + (1 << self.levels) - 1) >> self.levels
        ) << self.levels
        n_padded = target  # length after padding
        # Approx has n_padded / 2^levels elements
        approx_len = n_padded >> self.levels

        # Split: first approx_len elements are the final approximation,
        # then detail_L (same length), detail_{L-1} (2x), ..., detail_1 (2^{L-1}x)
        coeffs_flat = x_w.reshape(-1, length)  # [N, L]
        positions = [approx_len]
        for lev in range(self.levels - 1, -1, -1):
            positions.append(positions[-1] + (approx_len << (self.levels - lev - 1)))

        # Verify: positions[-1] should equal length
        splits = torch.split(
            coeffs_flat,
            [positions[0]]
            + [positions[i + 1] - positions[i] for i in range(len(positions) - 1)],
            dim=-1,
        )  # type: ignore[arg-type]
        approx = splits[0]  # final approx
        details = list(splits[1:])  # [detail_L, ..., detail_1] — deepest first

        # Reconstruct from coarsest to finest
        current = approx
        for detail in details:
            current = _haar_idwt_1d(current, detail)

        # Remove padding if any
        if self._padded:
            current = current[..., : self._orig_shape[dim]]

        current = current.reshape(*shape_prefix, current.shape[-1])
        return current.movedim(-1, dim)

    def __repr__(self) -> str:
        return f"WaveletDecompose(levels={self.levels}, dim={self.dim})"


def _build_d2_diagonals(
    n: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 3 upper diagonals of the symmetric pentadiagonal D^T D matrix.

    D is the (n-2)×n second-difference operator.  D^T D is n×n pentadiagonal.
    This returns only the 3 non-zero upper diagonals (the matrix is symmetric),
    reducing storage from O(n²) to O(n) and enabling O(n) banded Cholesky.

    Returns
    -------
    d0 : Tensor, shape [n]
        Main diagonal.
    d1 : Tensor, shape [n-1]
        First super-diagonal (offset +1).
    d2 : Tensor, shape [n-2]
        Second super-diagonal (offset +2).
    """
    if n < 4:
        # For n < 4 the second-difference penalty vanishes (D has < 2 rows).
        d0 = torch.zeros(n, device=device, dtype=dtype)
        d1 = torch.zeros(max(n - 1, 0), device=device, dtype=dtype)
        d2 = torch.zeros(max(n - 2, 0), device=device, dtype=dtype)
        return d0, d1, d2

    d0 = torch.full((n,), 6.0, device=device, dtype=dtype)
    d0[0] = 1.0
    d0[1] = 5.0
    d0[n - 2] = 5.0
    d0[n - 1] = 1.0

    d1 = torch.full((n - 1,), -4.0, device=device, dtype=dtype)
    d1[0] = -2.0
    d1[n - 2] = -2.0

    d2 = torch.ones(n - 2, device=device, dtype=dtype)
    return d0, d1, d2


def _banded_chol_solve_batched_impl(
    w: torch.Tensor,
    lam_d0: torch.Tensor,
    lam_d1: torch.Tensor,
    lam_d2: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Solve ``(diag(W) + λD^T D) z = b`` via banded Cholesky factorization.

    The matrix is symmetric positive-definite pentadiagonal (bandwidth 2).
    The Cholesky factorization and triangular solves are O(n) per spectrum —
    a dramatic improvement over the O(n³) dense ``torch.linalg.solve``.

    All inputs must be in the same dtype (typically float64 for stability).
    The loop over spectrum positions is sequential, but each step is
    vectorized over the batch (spectra) dimension.

    This is the pure-Python implementation used as fallback when
    ``torch.jit.script`` is unavailable.  Prefer ``_banded_chol_solve_batched``
    which wraps this with JIT compilation for ~5-10× speedup on large spectra.

    Parameters
    ----------
    w : Tensor, shape [N, L]
        Diagonal weights (the W matrix as a 2D tensor).
    lam_d0 : Tensor, shape [L]
        λ * main diagonal of D^T D (precomputed, shared across iterations).
    lam_d1 : Tensor, shape [L-1]
        λ * first super-diagonal of D^T D.
    lam_d2 : Tensor, shape [L-2]
        λ * second super-diagonal of D^T D.
    b : Tensor, shape [N, L]
        Right-hand side.

    Returns
    -------
    z : Tensor, shape [N, L]
        Solution.
    """
    n_batch, length = w.shape

    # --- Build A's diagonals: A = diag(W) + λD² ---
    # Only a0 changes per iteration; a1, a2 are fixed (= lam_d1, lam_d2).
    a0 = w + lam_d0.unsqueeze(0)  # [N, L]
    a1 = lam_d1.unsqueeze(0).expand(n_batch, -1)  # [N, L-1]
    a2 = lam_d2.unsqueeze(0).expand(n_batch, -1)  # [N, L-2]

    # --- Cholesky factorization: A = L L^T, L has bandwidth 2 ---
    l0 = torch.empty_like(a0)  # [N, L]   main diagonal of L
    l1 = torch.empty_like(a1)  # [N, L-1] first sub-diagonal of L
    l2 = torch.empty_like(a2)  # [N, L-2] second sub-diagonal of L

    # j = 0
    l0[:, 0] = torch.sqrt(torch.clamp_min(a0[:, 0], 1e-30))
    l1[:, 0] = a1[:, 0] / l0[:, 0]
    if length > 2:
        l2[:, 0] = a2[:, 0] / l0[:, 0]

    # j = 1
    if length > 1:
        l0[:, 1] = torch.sqrt(torch.clamp_min(a0[:, 1] - l1[:, 0] ** 2, 1e-30))
        l1[:, 1] = (a1[:, 1] - l2[:, 0] * l1[:, 0]) / l0[:, 1]
        if length > 3:
            l2[:, 1] = a2[:, 1] / l0[:, 1]

    # j = 2 .. n-3 (interior points)
    for j in range(2, length - 2):
        l0[:, j] = torch.sqrt(
            torch.clamp_min(a0[:, j] - l1[:, j - 1] ** 2 - l2[:, j - 2] ** 2, 1e-30)
        )
        l1[:, j] = (a1[:, j] - l2[:, j - 1] * l1[:, j - 1]) / l0[:, j]
        l2[:, j] = a2[:, j] / l0[:, j]

    # j = n-2
    if length > 2:
        l0[:, length - 2] = torch.sqrt(
            torch.clamp_min(
                a0[:, length - 2] - l1[:, length - 3] ** 2 - l2[:, length - 4] ** 2,
                1e-30,
            )
        )
        l1[:, length - 2] = (
            a1[:, length - 2] - l2[:, length - 3] * l1[:, length - 3]
        ) / l0[:, length - 2]

    # j = n-1
    l0[:, length - 1] = torch.sqrt(
        torch.clamp_min(
            a0[:, length - 1] - l1[:, length - 2] ** 2 - l2[:, length - 3] ** 2, 1e-30
        )
    )

    # --- Forward substitution: L y = b ---
    y = torch.empty_like(b)
    y[:, 0] = b[:, 0] / l0[:, 0]
    if length > 1:
        y[:, 1] = (b[:, 1] - l1[:, 0] * y[:, 0]) / l0[:, 1]
    for j in range(2, length):
        y[:, j] = (
            b[:, j] - l1[:, j - 1] * y[:, j - 1] - l2[:, j - 2] * y[:, j - 2]
        ) / l0[:, j]

    # --- Backward substitution: L^T z = y ---
    z = torch.empty_like(y)
    z[:, length - 1] = y[:, length - 1] / l0[:, length - 1]
    if length > 1:
        z[:, length - 2] = (
            y[:, length - 2] - l1[:, length - 2] * z[:, length - 1]
        ) / l0[:, length - 2]
    for j in range(length - 3, -1, -1):
        z[:, j] = (y[:, j] - l1[:, j] * z[:, j + 1] - l2[:, j] * z[:, j + 2]) / l0[:, j]

    return z


# JIT-compile the banded solver to eliminate Python for-loop overhead.
# torch.jit.script pushes the three sequential loops (Cholesky, forward
# substitution, backward substitution) into C++ with zero code changes.
# Profiling shows this eliminates ~94% of the runtime for large L.
# Falls back to pure Python if `torch.jit.script` is unavailable.
try:
    _banded_chol_solve_batched = torch.jit.script(_banded_chol_solve_batched_impl)  # type: ignore[assignment]
except (RuntimeError, TypeError, AttributeError):
    import warnings

    warnings.warn(
        "torch.jit.script() unavailable; _banded_chol_solve_batched "
        "will use pure Python (slower for large spectra).",
        stacklevel=2,
    )
    _banded_chol_solve_batched = _banded_chol_solve_batched_impl  # type: ignore[assignment]


class AsymmetricLeastSquares(FITSTransform):
    """Asymmetric Least Squares baseline correction (Eilers 2003).

    Iteratively fits a smooth baseline that hugs either the lower or upper
    envelope of the signal by differentially weighting points above vs below
    the baseline.  This is the standard method in Raman/NIR spectroscopy for
    automated baseline removal and is fully information-preserving via
    additive decomposition.

    The algorithm solves ``(W + λ D^T D) z = W y`` at each iteration,
    where *W* is a diagonal weight matrix with weights determined by
    *p* and *envelope*:

    - ``envelope="lower"`` (default): ``w_i = p`` if ``y_i > z_i``,
      ``1 − p`` otherwise.  Baseline hugs absorption features.
    - ``envelope="upper"``: ``w_i = 1 − p`` if ``y_i > z_i``,
      ``p`` otherwise.  Baseline hugs emission features.

    The transform is additive: ``Original = Baseline + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    lam : float
        Smoothness parameter.  Larger values produce a stiffer baseline.
        Typical range: 1e2 to 1e9 (default 1e5).
    p : float
        Asymmetry parameter in (0, 1).  Smaller values make the baseline
        hug the target envelope more aggressively.
        Typical range: 0.001 to 0.1 (default 0.01).
    max_iter : int
        Maximum number of reweighting iterations (default 10).
    dim : int
        Dimension to operate along (default -1).
    envelope : str
        Which envelope to hug: ``"lower"`` (default) for absorption
        features (Raman/NIR), ``"upper"`` for emission features
        (stellar absorption spectroscopy).

    References
    ----------
    Eilers, P. H. C. (2003). "A Perfect Smoother."
    Analytical Chemistry, 75(14), 3631–3636.
    """

    def __init__(
        self,
        lam: float = 1e5,
        p: float = 0.01,
        max_iter: int = 10,
        dim: int = -1,
        envelope: str = "lower",
    ) -> None:
        if lam <= 0:
            raise ValueError("lam must be > 0")
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if envelope not in ("lower", "upper"):
            raise ValueError("envelope must be 'lower' or 'upper'")
        self.lam = float(lam)
        self.p = float(p)
        self.max_iter = int(max_iter)
        self.dim = int(dim)
        self.envelope = envelope
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        # Move working dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]
        n_spectra, length = x_flat.shape

        with torch.no_grad():
            if length < 4:
                # D² penalty vanishes for n < 4: (W + 0) z = W y  →  z = y.
                baseline = x_flat.clone()
            else:
                # Precompute λ * D² diagonals once (shared across all
                # spectra and all reweighting iterations).
                d0, d1, d2 = _build_d2_diagonals(length, x.device, torch.float64)
                lam_d0 = self.lam * d0
                lam_d1 = self.lam * d1
                lam_d2 = self.lam * d2

                # Work in float64 for numerical stability with large λ.
                y = x_flat.double()  # [N, L]
                z = y.clone()  # initial estimate = signal

                for _ in range(self.max_iter):
                    # Weights: determined by envelope mode.
                    # lower: p for points above baseline (ignore peaks),
                    #        1-p for points below (hug absorption troughs).
                    # upper: 1-p for points above (hug emission peaks),
                    #        p for points below (ignore troughs).
                    if self.envelope == "lower":
                        w = torch.where(y > z, self.p, 1.0 - self.p)
                    else:
                        w = torch.where(y > z, 1.0 - self.p, self.p)

                    # RHS: W y (element-wise since W is diagonal)
                    b = w * y  # [N, L]

                    # Solve (W + λD²) z_new = W y via batched banded Cholesky.
                    # O(N*L) per iteration vs O(N*L³) for the dense solve.
                    z_new = _banded_chol_solve_batched(w, lam_d0, lam_d1, lam_d2, b)

                    # Check convergence across all spectra (same semantics
                    # as the old per-spectrum allclose with atol=1e-6, rtol=1e-5).
                    if torch.allclose(z_new, z, atol=1e-6):
                        z = z_new
                        break
                    z = z_new

                baseline = z.to(x.dtype)

        baseline = baseline.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - baseline
        return baseline

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "AsymmetricLeastSquares.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"AsymmetricLeastSquares(lam={self.lam}, p={self.p}, "
            f"max_iter={self.max_iter}, dim={self.dim}, "
            f"envelope={self.envelope!r})"
        )


class AlphaShapeContinuum(FITSTransform):
    """Alpha-shape continuum via morphological closing.

    Computes an upper-envelope continuum using morphological closing
    (dilation followed by erosion), which naturally follows the spectral
    peaks while bridging narrow absorption features.  This is a
    practical approximation to the full alpha-shape algorithm used by
    RASSINE (Cretignier et al. 2020), implemented entirely in PyTorch
    using unfold + max/min operations.

    Unlike :class:`UpperEnvelopeContinuum` (which uses local-max detection
    + interpolation), morphological closing produces a guaranteed upper
    envelope that is always >= the original signal.

    The transform is additive: ``Original = Continuum + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    half_window : int
        Half-width of the structuring element in samples.  Larger values
        bridge wider absorption features.  Default 15.
    iterations : int
        Number of closing operations.  Each iteration applies
        dilation→erosion, progressively smoothing the continuum.
        Default 1.
    dim : int
        Dimension to operate along (default -1).

    References
    ----------
    Cretignier, M. et al. (2020). "RASSINE: Interactive tool for
    normalising stellar spectra." Astronomy & Astrophysics, 640, A42.
    """

    def __init__(
        self, half_window: int = 15, iterations: int = 1, dim: int = -1
    ) -> None:
        if half_window < 1:
            raise ValueError("half_window must be >= 1")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.half_window = int(half_window)
        self.iterations = int(iterations)
        self.dim = int(dim)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        window_size = 2 * self.half_window + 1

        # Move working dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]
        pad = self.half_window

        with torch.no_grad():
            continuum = x_flat
            for _ in range(self.iterations):
                # Dilation: running max
                padded = torch.nn.functional.pad(continuum, (pad, pad), mode="reflect")
                windows = padded.unfold(-1, window_size, 1)  # [N, L, W]
                dilated = windows.max(dim=-1).values  # [N, L]
                # Erosion: running min of the dilated signal
                padded = torch.nn.functional.pad(dilated, (pad, pad), mode="reflect")
                windows = padded.unfold(-1, window_size, 1)  # [N, L, W]
                continuum = windows.min(dim=-1).values  # [N, L]

        continuum = continuum.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - continuum
        return continuum

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "AlphaShapeContinuum.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"AlphaShapeContinuum(half_window={self.half_window}, "
            f"iterations={self.iterations}, dim={self.dim})"
        )


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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Iteratively sigma-clip outliers and fill with mean or median.

        Optimised to minimise per-iteration allocations: uses a single
        pre-allocated masked-copy buffer and in-place arithmetic instead
        of allocating fresh ``torch.zeros_like`` / ``torch.where`` tensors
        each iteration.
        """
        ndim = x.ndim
        dims: tuple[int, ...] = ()
        if len(self.dim) > 0:
            dims = _normalize_dims(ndim, self.dim)

        with torch.no_grad():
            # Combine user mask with NaN mask if provided.
            if mask is not None:
                internal_mask = _get_valid_mask(x, mask)
            else:
                internal_mask = torch.ones_like(x, dtype=torch.bool)
            # Pre-allocate working buffers for masked values and zeros
            # to avoid per-iteration torch.zeros_like allocations.
            masked_buf = x.clone()
            zeros_buf = torch.zeros_like(x)
            for _ in range(self.max_iter):
                # Zero out masked-out positions, sum, and count.
                torch.where(internal_mask, x, zeros_buf, out=masked_buf)
                mask_f = internal_mask.to(x.dtype)

                if len(dims) > 0:
                    x_flat = _flatten_dims(masked_buf, dims)
                    c_flat = _flatten_dims(mask_f, dims)
                    total_sum = x_flat.sum(dim=-1, keepdim=True)
                    total_cnt = c_flat.sum(dim=-1, keepdim=True)
                    mean_v = total_sum / torch.clamp_min(total_cnt, 1.0)
                    mean_v_full = _unflatten_result(mean_v, x.shape, dims)
                    # Compute variance using the same buffer
                    masked_buf.sub_(mean_v_full).pow_(2)
                    torch.where(internal_mask, masked_buf, zeros_buf, out=masked_buf)
                    d_flat = _flatten_dims(masked_buf, dims)
                    var = d_flat.sum(dim=-1, keepdim=True) / torch.clamp_min(
                        total_cnt, 1.0
                    )
                    std_v_full = _unflatten_result(
                        torch.sqrt(torch.clamp_min(var, 0.0)), x.shape, dims
                    )
                    # Restore buffer for next iteration
                    masked_buf.copy_(x)
                else:
                    cnt = mask_f.sum()
                    mean_scalar = (masked_buf.sum() / max(cnt.item(), 1.0)).item()
                    mean_v_full = x.new_full(x.shape, mean_scalar)
                    masked_buf.sub_(mean_scalar).pow_(2)
                    torch.where(internal_mask, masked_buf, zeros_buf, out=masked_buf)
                    var = masked_buf.sum() / max(cnt.item(), 1.0)
                    std_scalar = math.sqrt(max(var.item(), 0.0))
                    std_v_full = x.new_full(x.shape, std_scalar)
                    masked_buf.copy_(x)

                new_mask = (x >= mean_v_full - self.n_sigma * std_v_full) & (
                    x <= mean_v_full + self.n_sigma * std_v_full
                )
                new_mask = new_mask & internal_mask
                if torch.equal(new_mask, internal_mask):
                    break
                internal_mask = new_mask

            self._last_mask = internal_mask

            # Fill clipped values with per-group mean or median
            if self.fill == "mean":
                torch.where(internal_mask, x, zeros_buf, out=masked_buf)
                mask_f = internal_mask.to(x.dtype)
                if len(dims) > 0:
                    xf = _flatten_dims(masked_buf, dims)
                    cf = _flatten_dims(mask_f, dims)
                    fill_val = _unflatten_result(
                        xf.sum(dim=-1, keepdim=True)
                        / torch.clamp_min(cf.sum(dim=-1, keepdim=True), 1.0),
                        x.shape,
                        dims,
                    )
                else:
                    cnt = mask_f.sum()
                    fill_val = masked_buf.sum() / max(cnt.item(), 1.0)
            else:
                # Median fill: use the existing _median helper
                fill_val = _median(
                    torch.where(
                        internal_mask,
                        x,
                        torch.tensor(float("inf"), device=x.device, dtype=x.dtype),
                    ),
                    dims if dims else (-1,),
                    mask=internal_mask,
                )
                # Replace inf (all-masked groups) with 0
                fill_val = torch.where(
                    torch.isinf(fill_val), torch.zeros_like(fill_val), fill_val
                )

            return torch.where(internal_mask, x, fill_val)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise RuntimeError(
            "SigmaClip.inverse() is not available — clipped values are irrecoverable."
        )

    def __repr__(self) -> str:
        return (
            f"SigmaClip(n_sigma={self.n_sigma}, max_iter={self.max_iter}, "
            f"dim={self.dim}, fill={self.fill!r})"
        )


class AsymmetricSigmaClip(FITSTransform):
    """Simple one-pass asymmetric sigma-clipping outlier rejection.

    Computes per-group median and MAD-derived std (:func:`estimate_background`),
    then replaces values outside ``[median - n_low*std, median + n_high*std]``
    with the per-group median.  Non-iterative — faster and simpler than the
    full :class:`SigmaClip`, and supports different thresholds for the lower
    and upper tails.

    ``inverse`` is not available — clipped values are irrecoverable.

    Parameters
    ----------
    n_low : float
        Number of std deviations below median to clip (default 3.0).
        Set higher to preserve more faint pixels.
    n_high : float
        Number of std deviations above median to clip (default 3.0).
        Set higher to preserve more bright pixels.
    dim :
        Dimensions along which stats are computed independently.
        Default ``(-2, -1)`` for per-image clipping.

    Examples
    --------
    >>> # Clip negative outliers aggressively, preserve bright sources
    >>> clip = AsymmetricSigmaClip(n_low=5.0, n_high=3.0)
    >>>
    >>> # Per-spectrum clipping along the spectral axis
    >>> clip = AsymmetricSigmaClip(n_low=2.5, n_high=2.5, dim=(-1,))
    """

    def __init__(
        self,
        n_low: float = 3.0,
        n_high: float = 3.0,
        dim: Tuple[int, ...] = (-2, -1),
    ) -> None:
        if n_low <= 0 or n_high <= 0:
            raise ValueError("n_low and n_high must be > 0")
        self.n_low = float(n_low)
        self.n_high = float(n_high)
        self.dim = tuple(dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            med, std = estimate_background(x, dim=self.dim)
            lower = med - self.n_low * std
            upper = med + self.n_high * std
            clip_mask = (x >= lower) & (x <= upper)
            return torch.where(clip_mask, x, med)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise RuntimeError(
            "AsymmetricSigmaClip.inverse() is not available — "
            "clipped values are irrecoverable."
        )

    def __repr__(self) -> str:
        return (
            f"AsymmetricSigmaClip(n_low={self.n_low}, "
            f"n_high={self.n_high}, dim={self.dim})"
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._is_integer or self._is_unsigned:
            vmin, vmax = self._in_range  # type: ignore[misc]
            if vmax == vmin:
                return torch.zeros_like(x)
            return (x - vmin) / (vmax - vmin)
        if self.scale_floats:
            vmin = _amin(x, tuple(range(x.ndim)), mask=mask)
            vmax = _amax(x, tuple(range(x.ndim)), mask=mask)
            self._in_range = (float(vmin.item()), float(vmax.item()))
            if vmax == vmin:
                return torch.zeros_like(x)
            return (x - vmin) / (vmax - vmin)
        # Float types, no scaling requested — identity
        return x

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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


__all__ = [
    "FITSTransform",
    "Compose",
    "ArcsinhStretch",
    "LogStretch",
    "SqrtStretch",
    "ZScaleNormalize",
    "RobustNormalize",
    "BackgroundSubtract",
    "PercentileClipNormalize",
    "MinMaxNormalize",
    "FITSHeaderScale",
    "FITSScaleColumns",
    "TNullToNan",
    "ContinuumNormalize",
    "DopplerShift",
    "PhaseFold",
    "SpectralBinning",
    "ContinuumRemoval",
    "BandMath",
    "GlobalScalarNorm",
    "SavitzkyGolayFilter",
    "RunningPercentile",
    "UpperEnvelopeContinuum",
    "WaveletDecompose",
    "AsymmetricLeastSquares",
    "AlphaShapeContinuum",
    "SigmaClip",
    "AsymmetricSigmaClip",
    "FITSHeaderNormalize",
    "safe_arcsinh",
    "safe_log",
    "estimate_background",
    "zscale_limits",
]
