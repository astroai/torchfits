"""``torchfits arith`` — CFITSIO-style image arithmetic (scalar or image–image)."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torchfits
from torchfits._io_engine.paths import cfitsio_base_path

if TYPE_CHECKING:
    import torch

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    add_hdu_arg,
    add_jobs_arg,
    add_split_arg,
    configure_torch_jobs,
    drop_data_range_cards,
    ensure_unique_basenames,
    _hdu_width,
    ensure_unique_split_stems,
    hdu_type_name,
    reject_same_path,
    resolve_file_jobs,
    run_file_jobs,
    selected_hdu_indices,
)

_OPS: dict[str, Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]] = {
    "add": lambda t, v: t + v,
    "sub": lambda t, v: t - v,
    "mul": lambda t, v: t * v,
    "div": lambda t, v: t / v,
}

_DTYPE_CHOICES = ("auto", "float32", "float64")


# Integer scalars inside int64 stay exact Python ints (the overflow-safe
# integer path below handles them); larger ones (``1e20``) stay floats and
# saturate to the output dtype through the float64 path.
_EXACT_INT_LIMIT = 2**63


def _parse_scalar(raw: str) -> float | int:
    """``--value``: an integer literal stays an exact int, anything else a float."""
    text = raw.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {raw!r}") from exc


def _promote_scalar(value: float | int) -> float | int:
    """Keep integer scalars exact so integer adds stay in integer arithmetic.

    Huge integral values (``1e20``) become floats: they take the float64 path
    and saturate to the output dtype instead of overflowing int64.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value if abs(value) < _EXACT_INT_LIMIT else float(value)
    if not math.isfinite(value) or abs(value) >= _EXACT_INT_LIMIT:
        return value
    as_int = int(value)
    return as_int if float(as_int) == value else value


_INT32_RANGE = (-(2**31), 2**31 - 1)


def _needs_exact_int(
    left: torch.Tensor, right: torch.Tensor | int, out_dtype: torch.dtype
) -> bool:
    """True when an int64 accumulator could overflow (64-bit operand or output)."""
    import torch

    wide = (torch.int64, torch.uint64)
    if left.dtype in wide or out_dtype in wide:
        return True
    if isinstance(right, torch.Tensor):
        return right.dtype in wide
    return not _INT32_RANGE[0] <= int(right) <= _INT32_RANGE[1]


_PY_OPS: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


def _exact_int_op(
    op: str,
    left: torch.Tensor,
    right: torch.Tensor | int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    """Exact integer add/sub/mul saturated into ``out_dtype``.

    Elements whose result provably fits int64 -- a float64 estimate below
    2**62 and both operands representable in int64 -- are computed in int64;
    only the rest (overflowing results, uint64 values >= 2**63) are computed
    with Python integers. Returns the result and the number of saturated
    elements.
    """
    import numpy as np
    import torch

    a = left.detach().cpu().contiguous().numpy()
    b: Any
    if isinstance(right, torch.Tensor):
        b = right.detach().cpu().contiguous().numpy()
        shape = np.broadcast_shapes(a.shape, b.shape)
    else:
        b = int(right)
        shape = a.shape
    i64 = np.iinfo(np.int64)
    info = torch.iinfo(out_dtype)
    lo, hi = int(info.min), int(info.max)

    with np.errstate(all="ignore"):
        b_f = b.astype(np.float64) if isinstance(b, np.ndarray) else float(b)
        est = _OPS[op](a.astype(np.float64), b_f)  # type: ignore[arg-type]
        safe = np.abs(np.broadcast_to(est, shape)) < 2.0**62
        if a.dtype == np.uint64:
            safe = safe & (a <= np.uint64(i64.max))
        if isinstance(b, np.ndarray) and b.dtype == np.uint64:
            safe = safe & (b <= np.uint64(i64.max))
        b64 = b.astype(np.int64) if isinstance(b, np.ndarray) else np.int64(b)
        exact = np.broadcast_to(_OPS[op](a.astype(np.int64), b64), shape)  # type: ignore[arg-type]

    np_out = torch.empty(0, dtype=out_dtype).numpy().dtype
    lo64, hi64 = max(lo, int(i64.min)), min(hi, int(i64.max))
    out = np.empty(shape, dtype=np_out)
    in_range = (exact >= lo64) & (exact <= hi64)
    n_clipped = int(np.count_nonzero(safe & ~in_range))
    out[safe] = np.clip(exact[safe], lo64, hi64).astype(np_out)

    unsafe = np.flatnonzero(~safe)
    if unsafe.size:
        a_flat = np.broadcast_to(a, shape).reshape(-1)
        b_flat = (
            np.broadcast_to(b, shape).reshape(-1) if isinstance(b, np.ndarray) else None
        )
        py_op = _PY_OPS[op]
        values = []
        for flat_index in unsafe.tolist():
            rhs = int(b_flat[flat_index]) if b_flat is not None else b
            value = py_op(int(a_flat[flat_index]), rhs)
            if value < lo or value > hi:
                n_clipped += 1
                value = lo if value < lo else hi
            values.append(value)
        out.reshape(-1)[unsafe] = np.array(values, dtype=np_out)
    return torch.from_numpy(out), n_clipped


def _two_sum(x: Any, y: Any) -> tuple[Any, Any]:
    """Knuth TwoSum: ``s + e == x + y`` exactly (float64, no overflow)."""
    s = x + y
    bv = s - x
    av = s - bv
    return s, (x - av) + (y - bv)


def _split(x: Any) -> tuple[Any, Any]:
    c = 134217729.0 * x  # 2**27 + 1 (Dekker/Veltkamp split)
    hi = c - (c - x)
    return hi, x - hi


def _two_product(x: Any, y: Any) -> tuple[Any, Any]:
    """Dekker TwoProduct: ``p + e == x * y`` exactly (float64, no FMA)."""
    p = x * y
    xh, xl = _split(x)
    yh, yl = _split(y)
    return p, ((xh * yh - p) + xh * yl + xl * yh) + xl * yl


def _exact_float_operand_op(
    op: str,
    left: torch.Tensor,
    right: torch.Tensor | float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int]:
    """Integer image ``op`` float operand into an integer ``out_dtype``, exactly.

    The result is the exact rational value (pixel and float operand taken
    at their exact values), truncated toward zero and saturated to
    ``out_dtype``; ``+inf``/``-inf`` results saturate. A NaN result (a NaN
    operand, or ``0 * inf``) has no integer value and is refused with a
    :class:`UsageError`.

    Elements whose pixel and float64 estimate are below 2**53 / 2**52 are
    resolved vectorized from an error-free TwoSum/TwoProduct; elements with a
    result past 2**65 saturate by sign; only the rest (64-bit pixels near the
    type limits) use Python :class:`~fractions.Fraction` arithmetic.

    Returns the result, the number of saturated elements and the number of
    in-range elements whose fractional part was dropped.
    """
    from fractions import Fraction

    import numpy as np
    import torch

    a = left.detach().cpu().contiguous().numpy()
    if isinstance(right, torch.Tensor):
        b: Any = right.detach().cpu().to(torch.float64).contiguous().numpy()
        shape = np.broadcast_shapes(a.shape, b.shape)
    else:
        b = np.float64(right)
        shape = a.shape
    info = torch.iinfo(out_dtype)
    lo, hi = int(info.min), int(info.max)
    np_out = torch.empty(0, dtype=out_dtype).numpy().dtype

    af = np.broadcast_to(a.astype(np.float64), shape)
    bf = np.broadcast_to(b, shape)
    with np.errstate(all="ignore"):
        if op == "mul":
            est = af * bf
        else:
            est = af + (bf if op == "add" else -bf)
    nan = np.isnan(est)
    if nan.any():
        raise UsageError(
            f"arith: {int(np.count_nonzero(nan))} result pixel(s) are NaN "
            f"(NaN operand or 0 * inf) and cannot be stored in an {out_dtype} "
            "image; use --dtype float32/float64"
        )

    abs_est = np.abs(est)
    trusted = (np.abs(af) <= 2.0**53) & (abs_est < 2.0**52)
    huge = abs_est >= 2.0**65  # includes +-inf: saturates whatever the rounding
    out = np.empty(shape, dtype=np_out)
    n_clipped = 0
    n_frac = 0

    if trusted.any():
        x = af[trusted]
        y = bf[trusted]
        with np.errstate(all="ignore"):
            if op == "mul":
                # x == 0 gives an exact 0 (y finite here); skip the split,
                # which would overflow for a huge y.
                y_safe = np.where(x == 0, 0.0, y)
                p, e = _two_product(x, y_safe)
            else:
                p, e = _two_sum(x, y if op == "add" else -y)
        t = np.trunc(p)
        integral = t == p
        # p is integral: the exact p + e lies just below/above it.
        t = np.where(integral & (p > 0) & (e < 0), t - 1, t)
        t = np.where(integral & (p < 0) & (e > 0), t + 1, t)
        fractional = ~integral | (e != 0)
        ti = t.astype(np.int64)
        in_range = (ti >= lo) & (ti <= hi) if hi <= 2**63 - 1 else (ti >= lo)
        n_clipped += int(np.count_nonzero(~in_range))
        n_frac += int(np.count_nonzero(fractional & in_range))
        clipped = np.clip(ti, max(lo, -(2**63)), min(hi, 2**63 - 1))
        out[trusted] = clipped.astype(np_out)

    if huge.any():
        out[huge & (est > 0)] = hi
        out[huge & (est < 0)] = lo
        n_clipped += int(np.count_nonzero(huge))

    rest = np.flatnonzero(~(trusted | huge))
    if rest.size:
        a_flat = np.broadcast_to(a, shape).reshape(-1)
        b_flat = bf.reshape(-1)
        values = []
        for flat_index in rest.tolist():
            pixel = Fraction(int(a_flat[flat_index]))
            operand = Fraction(float(b_flat[flat_index]))
            if op == "mul":
                exact = pixel * operand
            elif op == "add":
                exact = pixel + operand
            else:
                exact = pixel - operand
            value = int(exact)  # truncates toward zero
            if value < lo or value > hi:
                n_clipped += 1
                value = lo if value < lo else hi
            elif value != exact:
                n_frac += 1
            values.append(value)
        out.reshape(-1)[rest] = np.array(values, dtype=np_out)
    return torch.from_numpy(out), n_clipped, n_frac


def _saturate_to(t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Cast *t* to *dtype* clamping to the representable range.

    Returns the cast tensor and the number of elements that required clamping
    (i.e. wrapped values a naive cast would have corrupted). *t* holds
    integral values (an int64 accumulator, or float64 products of <= 32-bit
    integers); float operands go through :func:`_exact_float_operand_op`.
    """
    import torch

    if not t.dtype.is_floating_point and dtype.is_floating_point:
        return t.to(dtype), 0
    info = torch.iinfo(dtype)
    clipped = torch.clamp(t, info.min, info.max)
    n_clipped = int((clipped != t).sum())
    return clipped.to(dtype), n_clipped


def _compute(
    op: str,
    op_fn: Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor | float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run one arithmetic op without silent integer wraparound/truncation.

    Integer inputs are accumulated exactly (int64, Python ints for 64-bit
    overflow, exact rationals for a float operand) and saturating-cast back
    to ``out_dtype``; out-of-range pixels and dropped fractions raise a
    RuntimeWarning instead of wrapping silently, and NaN results are refused.
    """
    import torch

    if out_dtype.is_floating_point:
        # Cast an image operand B too: float32 A (or --dtype float32) with a
        # float64 B must not be promoted to a float64 result.
        if isinstance(right, torch.Tensor):
            right = right.to(out_dtype)
        return op_fn(left.to(out_dtype), right)

    if left.dtype.is_floating_point:
        # Already float: native semantics are exact enough for preview math.
        return op_fn(left, right)

    # Integer input: accumulate wide so 65535 + 1 cannot wrap uint16.
    if isinstance(right, float) or (
        isinstance(right, torch.Tensor) and right.dtype.is_floating_point
    ):
        # A float operand: compute the exact rational result, then truncate
        # and saturate (float64 alone rounds 64-bit pixels and can carry a
        # result across an integer boundary before the truncation).
        out, n_clipped, n_frac = _exact_float_operand_op(op, left, right, out_dtype)
        _warn_fractional(n_frac, out_dtype)
        _warn_saturated(n_clipped, out_dtype)
        return out
    else:
        right_t = right if isinstance(right, torch.Tensor) else None
        if _needs_exact_int(left, right, out_dtype):
            # 64-bit operands or output: an int64 accumulator would wrap and
            # float64 would round, so compute exactly and saturate.
            out, n_clipped = _exact_int_op(op, left, right, out_dtype)
            _warn_saturated(n_clipped, out_dtype)
            return out
        elif op == "mul" and out_dtype != torch.int64:
            # Integer mul can overflow even the int64 accumulator (e.g. a
            # uint32 image squared reaches 2**64) and wrap to garbage.
            # float64 is exact for every product that survives saturation into
            # any <= 32-bit FITS integer dtype and saturates cleanly beyond it.
            result = op_fn(
                left.to(torch.float64),
                right_t.to(torch.float64) if right_t is not None else float(right),
            )
        else:
            acc = left.to(torch.int64)
            acc_r = right_t.to(torch.int64) if right_t is not None else right
            result = op_fn(acc, acc_r)
    out, n_clipped = _saturate_to(result, out_dtype)
    _warn_saturated(n_clipped, out_dtype)
    return out


def _warn_fractional(n_frac: int, out_dtype: torch.dtype) -> None:
    if n_frac:
        warnings.warn(
            f"arith: {n_frac} pixel value(s) lost fractional parts casting "
            f"to {out_dtype} (use --dtype float32/float64 to keep full precision)",
            RuntimeWarning,
            stacklevel=3,
        )


def _warn_saturated(n_clipped: int, out_dtype: torch.dtype) -> None:
    if n_clipped:
        warnings.warn(
            f"arith: {n_clipped} pixel value(s) outside {out_dtype} range were "
            "saturated (use --dtype float32/float64 to keep full precision)",
            RuntimeWarning,
            stacklevel=3,
        )


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "arith",
        help="image ±×÷ by a scalar or second image",
        description=(
            "CFITSIO-style imarith: image±×÷scalar or image±×÷image. "
            "-j = PyTorch intra-op threads; -J = parallel file workers for multi-A. "
            "Same-shape multi-HDU inputs are stacked for ATen ops."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "operand A path(s); with no --value and exactly two paths plus -o, "
            "the second path is operand B"
        ),
    )
    parser.add_argument(
        "--op", required=True, choices=tuple(_OPS), help="arithmetic op"
    )
    parser.add_argument(
        "--value",
        type=_parse_scalar,
        default=None,
        help="scalar operand B (mutually exclusive with image B)",
    )
    parser.add_argument(
        "--dtype",
        choices=_DTYPE_CHOICES,
        default="auto",
        help=(
            "output/compute dtype: 'auto' keeps operand A's dtype (div on an "
            "integer A gives float64), warning on saturation or fractional "
            "truncation; float32/float64 cast both operands to that dtype, "
            "compute, and store in it"
        ),
    )
    parser.add_argument(
        "--operand2",
        default=None,
        help="image operand B path (for multi-A + --out-dir, or instead of 2nd positional)",
    )
    add_hdu_arg(
        parser,
        help="HDU indices on A (default: all image HDUs)",
    )
    parser.add_argument(
        "--hdu2",
        default=None,
        help="HDU index on image B (default: same as each A HDU, else 0)",
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs when processing multiple A inputs",
    )
    add_split_arg(parser)
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _parse_hdu2(hdu2: str | None) -> int | None:
    if hdu2 is None:
        return None
    try:
        return int(hdu2)
    except ValueError as exc:
        raise UsageError(f"invalid --hdu2: {hdu2!r}") from exc


def _resolve_operands(
    args: argparse.Namespace,
) -> tuple[list[str], float | int | None, str | None]:
    paths = [str(p) for p in args.paths]
    value = args.value
    operand2 = args.operand2
    if value is not None and operand2 is not None:
        raise UsageError("use either --value or --operand2, not both")

    if value is not None:
        return paths, value, None

    if operand2 is not None:
        return paths, None, str(operand2)

    # image–image shorthand: arith a.fits b.fits --op mul -o out.fits
    if len(paths) == 2 and args.out and not args.out_dir:
        return [paths[0]], None, paths[1]

    if len(paths) >= 2 and args.out_dir:
        raise UsageError(
            "multi-file A with image B requires --operand2 "
            "(or use --value for a scalar)"
        )

    raise UsageError(
        "operand B required: pass --value FLOAT, --operand2 PATH, "
        "or exactly two paths with -o/--out"
    )


def _image_indices(path: str, hdu: str | None) -> list[int]:
    try:
        with torchfits.open(path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu, path=path)
            return [
                index
                for index in indices
                if hdu_type_name(hdul[index].header, hdul[index]) == "IMAGE"
            ]
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc


def _read_image(path: str, index: int) -> tuple[torch.Tensor, Any]:
    import torch

    tensor = torchfits.read_tensor(path, hdu=index)
    if not isinstance(tensor, torch.Tensor):
        raise IoError(f"{path}:{index} read_tensor did not return a tensor")
    header = torchfits.read_header(path, index)
    return tensor, header


def _b_image(
    operand2: str,
    a_index: int,
    hdu2: int | None,
) -> tuple[torch.Tensor, Any]:
    if hdu2 is not None:
        b_index = hdu2
    else:
        try:
            with torchfits.open(operand2) as hdul:
                b_index = a_index if 0 <= a_index < len(hdul) else 0
        except Exception as exc:
            raise IoError(f"{operand2}: {exc}") from exc
    return _read_image(operand2, b_index)


def _bunit(header: Any) -> str:
    value = header.get("BUNIT") if header is not None else None
    return "" if value is None else str(value).strip()


def _result_header_units(
    op: str,
    header: Any,
    *,
    scalar: float | int | None,
    b_header: Any,
    where: str,
) -> None:
    """Keep BUNIT only where the result is still in it (no unit algebra).

    ``add`` / ``sub`` keep the unit; two images with different non-empty
    BUNIT cannot be added or subtracted (usage error). ``mul`` / ``div``
    by an image, or by a scalar other than 1, give a product unit
    torchfits does not derive, so BUNIT is dropped.
    """
    if op in ("add", "sub"):
        if b_header is not None:
            unit_a, unit_b = _bunit(header), _bunit(b_header)
            if unit_a and unit_b and unit_a != unit_b:
                raise UsageError(
                    f"{where}: cannot {op} images with different BUNIT "
                    f"({unit_a!r} vs {unit_b!r})"
                )
        return
    if b_header is None and scalar == 1:
        return
    if "BUNIT" in header:
        del header["BUNIT"]


def _resolve_out_dtype(spec: str, left: torch.Tensor) -> torch.dtype:
    import torch

    if spec == "float32":
        return torch.float32
    if spec == "float64":
        return torch.float64
    return left.dtype


def _apply_op(
    op: str,
    left: torch.Tensor,
    right: torch.Tensor | float,
    dtype_spec: str = "auto",
) -> torch.Tensor:
    import torch

    if op == "div" and not isinstance(right, torch.Tensor) and right == 0:
        raise UsageError("division by zero")
    if op == "div" and isinstance(right, torch.Tensor) and bool((right == 0).any()):
        raise UsageError("division by zero")
    out_dtype = _resolve_out_dtype(dtype_spec, left)
    if op == "div" and dtype_spec == "auto":
        # Fractional quotients must survive: integral inputs compute in
        # float64 rather than truncating through an integer output dtype.
        out_dtype = left.dtype if left.dtype.is_floating_point else torch.float64
    return _compute(op, _OPS[op], left, right, out_dtype)


def _arith_one_file(
    path_a: str,
    *,
    op: str,
    dtype_spec: str = "auto",
    value: float | int | None,
    operand2: str | None,
    hdu: str | None,
    hdu2: int | None,
    out_path: str | None,
    out_dir: Path | None,
    split: str,
    batch_inputs: tuple[str, ...] = (),
) -> None:
    import torch

    indices = _image_indices(path_a, hdu)
    if not indices:
        raise IoError(f"{path_a}: no image HDUs to process")

    tensors: list[torch.Tensor] = []
    headers: list[Any] = []
    for index in indices:
        tensor, header = _read_image(path_a, index)
        tensors.append(tensor)
        headers.append(header)

    rights: list[torch.Tensor | float]
    b_headers: list[Any] = [None] * len(tensors)
    if value is not None:
        scalar_b: torch.Tensor | float = _promote_scalar(value)
        rights = [scalar_b] * len(tensors)
    else:
        assert operand2 is not None
        if hdu2 is not None:
            # --hdu2 pins operand B to one HDU: read it once, reuse per A HDU.
            b_fixed, b_header = _b_image(operand2, indices[0], hdu2)
            rights = [b_fixed] * len(tensors)
            b_headers = [b_header] * len(tensors)
        else:
            b_images = [_b_image(operand2, index, hdu2) for index in indices]
            rights = [tensor for tensor, _ in b_images]
            b_headers = [header for _, header in b_images]
        for index, left, right in zip(indices, tensors, rights, strict=True):
            if not isinstance(right, torch.Tensor):
                continue
            if left.shape != right.shape:
                raise UsageError(
                    f"shape mismatch at HDU {index}: {tuple(left.shape)} vs "
                    f"{tuple(right.shape)}"
                )

    for index, header, b_header in zip(indices, headers, b_headers, strict=True):
        drop_data_range_cards(header)
        _result_header_units(
            op,
            header,
            scalar=value,
            b_header=b_header,
            where=f"{path_a}:{index}",
        )

    shapes = {tuple(t.shape) for t in tensors}
    # Stacking is only an execution shortcut: HDUs of different dtypes would
    # be promoted to a common dtype by torch.stack and change an HDU's output
    # dtype, so they are processed one by one.
    dtypes = {t.dtype for t in tensors}
    can_stack = len(shapes) == 1 and len(dtypes) == 1 and len(tensors) > 1
    if can_stack and value is not None:
        results = list(
            _apply_op(op, torch.stack(tensors), scalar_b, dtype_spec).unbind(0)
        )
    elif (
        can_stack
        and value is None
        and all(isinstance(right, torch.Tensor) for right in rights)
    ):
        b_tensors = [right for right in rights if isinstance(right, torch.Tensor)]
        if (
            len({tuple(right.shape) for right in b_tensors}) == 1
            and len({right.dtype for right in b_tensors}) == 1
        ):
            results = list(
                _apply_op(
                    op, torch.stack(tensors), torch.stack(b_tensors), dtype_spec
                ).unbind(0)
            )
        else:
            results = [
                _apply_op(op, left, right, dtype_spec)
                for left, right in zip(tensors, rights, strict=True)
            ]
    else:
        results = [
            _apply_op(op, left, right, dtype_spec)
            for left, right in zip(tensors, rights, strict=True)
        ]

    if split == "hdu":
        if out_dir is None:
            raise UsageError("--split hdu requires --out-dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        width = _hdu_width(indices)
        stem = Path(cfitsio_base_path(path_a)).stem
        for index, result, header in zip(indices, results, headers, strict=True):
            dest = out_dir / f"{stem}_hdu{index:0{width}d}.fits"
            # A generated split name may coincide with another input of the
            # batch (re-splitting into the same directory): refuse instead of
            # silently clobbering an unrelated input.
            for other in batch_inputs:
                reject_same_path(other, str(dest))
            torchfits.write_tensor(str(dest), result, header=header, overwrite=True)
        return

    if out_path is None:
        raise UsageError("output path required (-o/--out)")
    if len(results) == 1:
        torchfits.write_tensor(out_path, results[0], header=headers[0], overwrite=True)
    else:
        torchfits.write(
            out_path,
            [
                {"data": result, "header": header}
                for result, header in zip(results, headers, strict=True)
            ],
            overwrite=True,
        )


def _planned_destinations(
    a_paths: list[str],
    *,
    out: str | None,
    out_dir: Path | None,
    split: str,
    hdu: str | None,
) -> list[str]:
    if split == "file":
        if out_dir is not None:
            return [
                str(out_dir / Path(cfitsio_base_path(path)).name) for path in a_paths
            ]
        return [str(out)] if out is not None else []
    if out_dir is None:
        return []
    destinations: list[str] = []
    for path in a_paths:
        indices = _image_indices(path, hdu)
        width = _hdu_width(indices)
        stem = Path(cfitsio_base_path(path)).stem
        destinations.extend(
            str(out_dir / f"{stem}_hdu{index:0{width}d}.fits") for index in indices
        )
    return destinations


def run(args: argparse.Namespace) -> int:
    a_paths, value, operand2 = _resolve_operands(args)
    if args.out and args.out_dir:
        raise UsageError("use either --out-dir or -o/--out, not both")
    if len(a_paths) > 1 and not args.out_dir:
        raise UsageError("multiple A inputs require --out-dir")
    if len(a_paths) == 1 and not args.out and args.split != "hdu":
        raise UsageError("output path required (-o/--out)")
    if args.split == "hdu" and not args.out_dir:
        raise UsageError("--split hdu requires --out-dir")
    if args.split == "hdu" and args.out:
        raise UsageError("--split hdu uses --out-dir (not -o/--out)")
    if value is not None and args.hdu2 is not None:
        raise UsageError("--hdu2 applies only to image operand B")

    hdu2 = _parse_hdu2(args.hdu2)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        ensure_unique_basenames(a_paths, directory=out_dir)
        if args.split == "hdu":
            ensure_unique_split_stems(a_paths, directory=out_dir)

    destinations = _planned_destinations(
        a_paths,
        out=args.out,
        out_dir=out_dir,
        split=args.split,
        hdu=args.hdu,
    )
    sources = [*a_paths, *([operand2] if operand2 is not None else [])]
    for destination in destinations:
        for source in sources:
            reject_same_path(source, destination)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(a_paths))
    configure_torch_jobs(int(args.jobs))

    def _one(path_a: str) -> None:
        if out_dir is not None and args.split == "file":
            # Match ensure_unique_basenames: strip CFITSIO [section] suffixes.
            out_path = str(out_dir / Path(cfitsio_base_path(path_a)).name)
        else:
            out_path = args.out
        _arith_one_file(
            path_a,
            op=args.op,
            dtype_spec=args.dtype,
            value=value,
            operand2=operand2,
            hdu=args.hdu,
            hdu2=hdu2,
            out_path=out_path,
            out_dir=out_dir,
            split=args.split,
            batch_inputs=tuple(a_paths),
        )

    try:
        run_file_jobs(a_paths, _one, file_jobs, torch_runtime=True)
    except UsageError:
        raise
    except ZeroDivisionError as exc:
        raise UsageError("division by zero") from exc
    except IoError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
