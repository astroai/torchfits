"""Robust float → int16 linear quantization (images + table columns).

Linear min→max packing onto BITPIX=16 / TFORM=I wastes codes on rare extremes
when the value distribution is skewed. Prefer native float storage for science;
when int16 size is mandatory, use :func:`quantize_int16_robust` (percentile bulk
range + clip) and write explicit BSCALE/BZERO or TSCAL/TZERO.

ponytail: global min→max (poloka FitsImage::Write) is intentionally not offered
as a default — it is the failure mode this helper avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

# Match poloka's ±2 margin from hard int16 limits (overflow safety).
_SHRT_MIN_EFF = -32766
_SHRT_MAX_EFF = 32765
_SPAN = _SHRT_MAX_EFF - _SHRT_MIN_EFF  # 65531


@dataclass(frozen=True)
class QuantizeInt16Result:
    """Packed int16 codes plus FITS linear scale keywords."""

    codes: Tensor
    scale: float
    zero: float
    lo: float
    hi: float
    n_clipped: int


@dataclass(frozen=True)
class QuantizeOptions:
    lo_q: float = 0.1
    hi_q: float = 99.9
    keep_zero: bool = False


def parse_quantize_options(spec: Any) -> QuantizeOptions | None:
    """Normalize ``quantize=`` for a single array/column.

    Accepts ``None``, ``\"robust\"``, ``True``, or a mapping with ``lo_q`` /
    ``hi_q`` / ``keep_zero``.
    """
    if spec is None or spec is False:
        return None
    if spec is True or spec == "robust":
        return QuantizeOptions()
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"lo_q", "hi_q", "keep_zero"}
        if unknown:
            raise TypeError(
                f"quantize option dict has unknown keys {sorted(unknown)!r}; "
                "expected lo_q, hi_q, keep_zero"
            )
        lo_q = float(spec.get("lo_q", 0.1))
        hi_q = float(spec.get("hi_q", 99.9))
        keep_zero = bool(spec.get("keep_zero", False))
        if not (0.0 <= lo_q < hi_q <= 100.0):
            raise ValueError(
                f"quantize lo_q/hi_q must satisfy 0 <= lo_q < hi_q <= 100, "
                f"got {lo_q!r}, {hi_q!r}"
            )
        return QuantizeOptions(lo_q=lo_q, hi_q=hi_q, keep_zero=keep_zero)
    raise TypeError(
        "quantize must be None, True, 'robust', or a dict with lo_q/hi_q/keep_zero"
    )


def parse_image_quantize_spec(spec: Any) -> QuantizeOptions | None:
    """Parse image ``write(..., quantize=)`` (options dict, not column map)."""
    return parse_quantize_options(spec)


def _is_floating_column(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(value.is_floating_point())
    arr = np.asarray(value)
    return bool(np.issubdtype(arr.dtype, np.floating))


def parse_table_quantize_spec(
    spec: Any,
    columns: list[str],
    data: Mapping[str, Any] | None = None,
) -> dict[str, QuantizeOptions]:
    """Parse table ``write(..., quantize=)`` into per-column options.

    - ``\"robust\"`` / ``True`` → all *floating* columns (integer columns skipped)
    - ``{name: spec, ...}`` → only named columns (must be floating)
    """
    if spec is None or spec is False:
        return {}
    if spec is True or spec == "robust":
        opts = QuantizeOptions()
        if data is None:
            return {name: opts for name in columns}
        return {
            name: opts
            for name in columns
            if name in data and _is_floating_column(data[name])
        }
    if not isinstance(spec, Mapping):
        raise TypeError(
            "table quantize must be None, True, 'robust', or a dict of column specs"
        )
    # Disambiguate image-style option dict vs per-column map.
    keys = set(spec)
    if keys and keys <= {"lo_q", "hi_q", "keep_zero"}:
        parsed_opts = parse_quantize_options(spec)
        if parsed_opts is None:
            return {}
        if data is None:
            return {name: parsed_opts for name in columns}
        return {
            name: parsed_opts
            for name in columns
            if name in data and _is_floating_column(data[name])
        }

    out: dict[str, QuantizeOptions] = {}
    for name, col_spec in spec.items():
        key = str(name)
        if key not in columns:
            raise KeyError(f"quantize column {key!r} not in data columns {columns!r}")
        parsed = parse_quantize_options(col_spec)
        if parsed is None:
            continue
        if data is not None and not _is_floating_column(data[key]):
            raise TypeError(
                f"quantize column {key!r} must be floating-point, "
                f"got dtype={getattr(data[key], 'dtype', type(data[key]))}"
            )
        out[key] = parsed
    return out


def _as_work_flat(
    values: Tensor | np.ndarray,
) -> tuple[Any, tuple[int, ...], torch.device]:
    """Return contiguous float flat view (float32 preferred), shape, device.

    float32 inputs stay float32 (less bandwidth than float64 upcast). float16 /
    bfloat16 promote to float32; float64 stays float64.
    """
    if isinstance(values, Tensor):
        if values.numel() == 0:
            raise ValueError("quantize_int16_robust: empty array")
        if not values.is_floating_point():
            raise TypeError(
                f"quantize_int16_robust requires floating values, got dtype={values.dtype}"
            )
        work_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        host = values.detach().to(device="cpu", dtype=work_dtype).reshape(-1)
        return host.numpy(), tuple(values.shape), values.device

    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError("quantize_int16_robust: empty array")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            f"quantize_int16_robust requires floating values, got dtype={arr.dtype}"
        )
    flat: Any
    if arr.dtype == np.dtype(np.float64):
        flat = np.ascontiguousarray(arr, dtype=np.float64).reshape(-1)
    else:
        flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    return flat, arr.shape, torch.device("cpu")


def _percentile_sample(finite: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    """Finite samples used for percentile bounds.

    Large arrays with interior percentiles use a deterministic strided sample
    (~128k points) — rare extremes are usually excluded, which matches the
    robust goal and avoids an O(n log n) full partition.
    """
    n = int(finite.size)
    # Exact endpoints need the full population min/max.
    if lo_q <= 0.0 and hi_q >= 100.0:
        return finite
    if n <= 262_144:
        return finite
    step = max(1, n // 131_072)
    return finite[::step]


def _pack_codes(physical: np.ndarray, scale: float, zero: float) -> np.ndarray:
    """Round physical → int16 codes; clip to effective short range."""
    inv = 1.0 / scale
    codes_f = np.empty(physical.shape, dtype=np.float64)
    np.subtract(physical, zero, out=codes_f)
    codes_f *= inv
    np.rint(codes_f, out=codes_f)
    np.clip(codes_f, _SHRT_MIN_EFF, _SHRT_MAX_EFF, out=codes_f)
    return codes_f.astype(np.int16, copy=False)


def quantize_int16_robust(
    values: Tensor | np.ndarray,
    *,
    lo_q: float = 0.1,
    hi_q: float = 99.9,
    keep_zero: bool = False,
) -> QuantizeInt16Result:
    """Pack float values to int16 with robust linear BSCALE/BZERO (or TSCAL/TZERO).

    ``lo_q`` / ``hi_q`` are percentiles over finite flattened samples. Values
    outside ``[lo, hi]`` (and non-finite samples) are clipped before rounding.
    Shape is preserved. Endpoint identity: ``lo`` → code ``-32766``, ``hi`` →
    ``32765`` (poloka ±2 margin).
    """
    if not (0.0 <= lo_q < hi_q <= 100.0):
        raise ValueError(
            f"lo_q/hi_q must satisfy 0 <= lo_q < hi_q <= 100, got {lo_q!r}, {hi_q!r}"
        )

    flat, shape, device = _as_work_flat(values)
    finite_mask = np.isfinite(flat)
    # Avoid a copy when every sample is finite (common image/table path).
    finite = flat if bool(finite_mask.all()) else flat[finite_mask]
    if finite.size == 0:
        raise ValueError("quantize_int16_robust: no finite values to quantize")

    if keep_zero:
        # Weight/mask path: force BZERO=0; negatives clip to 0 (poloka KEEPZERO).
        positive = finite[finite > 0.0]
        if positive.size == 0:
            codes_np = np.zeros(flat.shape, dtype=np.int16)
            codes = torch.from_numpy(codes_np).reshape(shape)
            if device.type != "cpu":
                codes = codes.to(device)
            return QuantizeInt16Result(
                codes=codes, scale=1.0, zero=0.0, lo=0.0, hi=0.0, n_clipped=0
            )
        sample = _percentile_sample(positive, 0.0, hi_q)
        hi = float(np.percentile(sample, hi_q))
        if hi <= 0.0:
            hi = float(np.max(positive))
        scale = hi / float(_SHRT_MAX_EFF)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        zero = 0.0
        lo = 0.0
        clipped = np.empty_like(flat, dtype=np.float64)
        np.clip(flat, 0.0, hi, out=clipped)
        if not finite_mask.all():
            clipped[~finite_mask] = 0.0
    else:
        sample = _percentile_sample(finite, lo_q, hi_q)
        lo, hi = (float(x) for x in np.percentile(sample, (lo_q, hi_q)))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("quantize_int16_robust: non-finite percentile bounds")
        if hi <= lo:
            scale = 1.0
            zero = lo
            codes_np = np.zeros(flat.shape, dtype=np.int16)
            n_clipped = int((~finite_mask).sum())
            codes = torch.from_numpy(codes_np).reshape(shape)
            if device.type != "cpu":
                codes = codes.to(device)
            return QuantizeInt16Result(
                codes=codes,
                scale=scale,
                zero=zero,
                lo=lo,
                hi=hi,
                n_clipped=n_clipped,
            )
        scale = (hi - lo) / float(_SPAN)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        zero = lo - scale * float(_SHRT_MIN_EFF)
        clipped = np.empty_like(flat, dtype=np.float64)
        np.clip(flat, lo, hi, out=clipped)
        if not finite_mask.all():
            clipped[~finite_mask] = lo

    codes_np = _pack_codes(clipped, scale, zero)
    in_range = finite_mask & (flat >= lo) & (flat <= hi)
    n_clipped = int((~in_range).sum())

    codes = torch.from_numpy(codes_np).reshape(shape)
    if device.type != "cpu":
        codes = codes.to(device)
    return QuantizeInt16Result(
        codes=codes,
        scale=float(scale),
        zero=float(zero),
        lo=float(lo),
        hi=float(hi),
        n_clipped=n_clipped,
    )


def quantize_int16_minmax(values: Tensor | np.ndarray) -> QuantizeInt16Result:
    """Poloka-style global min→max pack (for tests / comparison only)."""
    flat, shape, device = _as_work_flat(values)
    finite_mask = np.isfinite(flat)
    finite = flat if bool(finite_mask.all()) else flat[finite_mask]
    if finite.size == 0:
        raise ValueError("quantize_int16_minmax: no finite values")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        scale = 1.0
        zero = lo
        codes_np = np.zeros(flat.shape, dtype=np.int16)
    else:
        scale = (hi - lo) / float(_SPAN)
        zero = lo - scale * float(_SHRT_MIN_EFF)
        clipped = np.empty_like(flat, dtype=np.float64)
        np.clip(flat, lo, hi, out=clipped)
        if not finite_mask.all():
            clipped[~finite_mask] = lo
        codes_np = _pack_codes(clipped, scale, zero)
    codes = torch.from_numpy(codes_np).reshape(shape)
    if device.type != "cpu":
        codes = codes.to(device)
    return QuantizeInt16Result(
        codes=codes,
        scale=float(scale),
        zero=float(zero),
        lo=lo,
        hi=hi,
        n_clipped=0,
    )


def dequantize_int16(
    codes: Tensor, scale: float, zero: float, *, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Apply physical = scale * code + zero."""
    return codes.to(dtype=dtype) * float(scale) + float(zero)
