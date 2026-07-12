#!/usr/bin/env python3
"""Benchmark: batched normal-equations vs per-spectrum lstsq for _fit_poly_continuum.

Sweeps over (n_spectra, length, order, max_iter) and reports:
- Wall time per variant
- Speedup (lstsq_ms / batched_ms, i.e. how many × faster batched is)
- Numerical agreement (allclose check with max diff)
- Whether both solvers converged to the same mask

Both variants use the *same* sigma-clip logic (batched, mean-subtracted std)
so the only difference is the linear solver: normal equations (bmm+solve) vs
per-spectrum lstsq (QR/SVD).  This isolates the algebraic performance gap.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchfits.transforms import _fit_poly_continuum  # noqa: E402


# ---------------------------------------------------------------------------
# Reference: uses the SAME sigma-clip as the batched version, but per-spectrum
# lstsq for the linear solve step.  This isolates the solver performance gap.
# ---------------------------------------------------------------------------


def _fit_poly_continuum_lstsq(
    x: torch.Tensor,
    order: int = 3,
    n_sigma: float = 2.0,
    max_iter: int = 3,
) -> torch.Tensor:
    """Reference using per-spectrum lstsq with batched sigma-clip."""
    n, length = x.shape
    t = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)
    A = torch.stack([t**k for k in range(order + 1)], dim=1)  # [length, order+1]

    mask = torch.ones(n, length, dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        counts = mask.sum(dim=1)  # [n]
        too_few = counts <= order
        if too_few.any():
            mask = mask.clone()
            mask[too_few] = True

        # Per-spectrum lstsq solve (the ONLY difference from batched version)
        coeffs = torch.zeros(n, order + 1, device=x.device, dtype=x.dtype)
        for i in range(n):
            mi = mask[i]
            Am = A[mi]
            ym = x[i][mi]
            try:
                coeffs[i] = torch.linalg.lstsq(Am, ym.unsqueeze(1)).solution.squeeze(1)
            except RuntimeError:
                pass  # Leave zeros

        continuum = (A @ coeffs.T).T  # [n, length]
        residuals = x - continuum

        # Same batched sigma-clip as the normal-equations version
        mask_f = mask.unsqueeze(2).to(x.dtype)  # [n, length, 1]
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


# ---------------------------------------------------------------------------
# Benchmark grid
# ---------------------------------------------------------------------------

# (label, n_spectra, length, order, max_iter)
GRID: list[tuple[str, int, int, int, int]] = [
    # --- Tiny: single spectrum ---
    ("single_order1", 1, 500, 1, 3),
    ("single_order3", 1, 500, 3, 3),
    ("single_order5", 1, 500, 5, 3),
    # --- Small batch ---
    ("small_order1", 16, 500, 1, 3),
    ("small_order3", 16, 500, 3, 3),
    ("small_order5", 16, 500, 5, 3),
    # --- Medium batch, longer spectra ---
    ("medium_order1", 64, 2000, 1, 3),
    ("medium_order3", 64, 2000, 3, 3),
    ("medium_order5", 64, 2000, 5, 3),
    # --- Large batch ---
    ("large_order1", 256, 2000, 1, 3),
    ("large_order3", 256, 2000, 3, 3),
    ("large_order5", 256, 2000, 5, 3),
    # --- Many iterations (heavier sigma-clip) ---
    ("many_iter_order3", 64, 2000, 3, 10),
    # --- High order (stress-test conditioning) ---
    ("high_order7", 64, 2000, 7, 3),
    ("high_order9", 64, 2000, 9, 3),
    # --- Wide batch, short spectra ---
    ("wide_order3", 512, 256, 3, 3),
    ("wide_order5", 512, 256, 5, 3),
]

SEED = 42


def _sync(device: torch.device) -> None:
    """Synchronize the device so perf_counter captures GPU kernel time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_method(
    fn,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[float, float]:
    """Run *fn* with warmup, returning (median, std) over timed iterations.

    *fn* is called once per iteration and is responsible for generating
    fresh input data so that caching / allocator reuse don't contaminate
    the measurement.
    """
    times: list[float] = []
    for i in range(warmup + iterations):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            times.append(elapsed)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    return median, variance**0.5


def _generate_data(
    n: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate synthetic spectra with a polynomial continuum + noise + outliers."""
    torch.manual_seed(seed)
    t = torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)
    continuum = (1.0 + 0.5 * t - 0.3 * t**2 + 0.1 * t**3).unsqueeze(0)
    noise = 0.1 * torch.randn(n, length, device=device, dtype=dtype)
    amp = 0.5 + torch.rand(n, 1, device=device, dtype=dtype)
    x = amp * continuum + noise
    outlier_mask = torch.rand(n, length, device=device) < 0.05
    x[outlier_mask] += torch.randn(outlier_mask.sum(), device=device, dtype=dtype) * 3.0
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default cpu).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per case.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Timed iterations per case.",
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        dest="use_float64",
        help="Use float64 instead of float32 (default).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Regex filter on case label.",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip lstsq runs for large cases where per-spectrum is very slow.",
    )
    parser.add_argument(
        "--skip-lstsq-threshold-ms",
        type=float,
        default=500.0,
        help="Quick estimate; skip lstsq if single-run > this ms (default 500).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.use_float64 else torch.float32

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    case_filter = re.compile(args.filter) if args.filter else None

    # Header
    header = (
        f"{'case':<22s} {'N':>5s} {'L':>7s} {'ord':>4s} {'iter':>5s} "
        f"{'batched_ms':>11s} {'lstsq_ms':>11s} {'speedup':>8s} "
        f"{'match':>6s} {'max_diff':>10s} {'note':>12s}"
    )
    print(header)
    print("-" * len(header))

    for label, n, length, order_, max_iter_ in GRID:
        if case_filter and not case_filter.search(label):
            continue

        # --- Batched (normal equations) ---
        def run_batched() -> torch.Tensor:
            x = _generate_data(n, length, device, dtype, SEED + 0)
            return _fit_poly_continuum(x, order=order_, n_sigma=2.0, max_iter=max_iter_)

        batched_median, _ = _time_method(
            run_batched, args.warmup, args.iterations, device
        )

        # --- Per-spectrum lstsq ---
        lstsq_median: Optional[float] = None
        note = ""

        def run_lstsq() -> torch.Tensor:
            x = _generate_data(n, length, device, dtype, SEED + 1)
            return _fit_poly_continuum_lstsq(
                x, order=order_, n_sigma=2.0, max_iter=max_iter_
            )

        if args.skip_slow:
            # Quick single-run estimate
            x_est = _generate_data(n, length, device, dtype, SEED + 9999)
            _sync(device)
            t0 = time.perf_counter()
            _fit_poly_continuum_lstsq(
                x_est, order=order_, n_sigma=2.0, max_iter=max_iter_
            )
            _sync(device)
            est_ms = (time.perf_counter() - t0) * 1000.0
            if est_ms > args.skip_lstsq_threshold_ms:
                note = f"skip-est={est_ms:.0f}ms"
            else:
                lstsq_median, _ = _time_method(
                    run_lstsq, 0, max(1, args.iterations // 2), device
                )
        else:
            lstsq_median, _ = _time_method(
                run_lstsq, args.warmup, args.iterations, device
            )

        # --- Correctness: both on same input ---
        x_ref = _generate_data(n, length, device, dtype, SEED + 2)
        with torch.no_grad():
            result_batched = _fit_poly_continuum(
                x_ref, order=order_, n_sigma=2.0, max_iter=max_iter_
            )
            result_lstsq = _fit_poly_continuum_lstsq(
                x_ref, order=order_, n_sigma=2.0, max_iter=max_iter_
            )
        match = torch.allclose(result_batched, result_lstsq, atol=1e-5, rtol=1e-3)
        max_diff = (result_batched - result_lstsq).abs().max().item()

        speedup_str = (
            f"{lstsq_median / batched_median:.1f}x"
            if lstsq_median and batched_median > 0
            else "-"
        )
        lstsq_str = f"{lstsq_median * 1000:9.1f}" if lstsq_median else "      skip"
        match_str = "OK" if match else "MISMATCH"

        print(
            f"{label:<22s} {n:>5d} {length:>7d} {order_:>4d} {max_iter_:>5d} "
            f"{batched_median * 1000:9.1f}  {lstsq_str:>9s}  {speedup_str:>6s}  "
            f"{match_str:>6s}  {max_diff:8.1e}  {note:<12s}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
