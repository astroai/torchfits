from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .base import FITSTransform
from .continuum import _fit_spline_continuum

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

        continuum: torch.Tensor = (A @ coeffs.T).T  # [n, length]
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
    _map: dict[str, str] = {
        "linear": "bilinear",
        "nearest": "nearest",
        "cubic": "bicubic",
    }
    return _map[mode]



def _to_interpolate_2d_mode(mode: str) -> tuple[str, dict[str, bool]]:
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
        denom = x1 - x0
        # Guard against near-zero span while preserving sign for
        # descending grids (e.g. wavelength in air → vacuum).
        if abs(denom.item()) < 1e-30:
            denom = torch.tensor(1e-30, device=denom.device, dtype=denom.dtype)
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

    def __init__(self, func: Callable[..., torch.Tensor], band_dim: int = 0) -> None:
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


