"""Shared wall-clock timers with optional peak RSS / CUDA alloc samples."""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, Callable

import numpy as np

try:
    import psutil

    _PROC = psutil.Process()
except Exception:  # pragma: no cover - optional in minimal envs
    psutil = None  # type: ignore[assignment]
    _PROC = None


def _rss_mb() -> float | None:
    if _PROC is None:
        return None
    try:
        return float(_PROC.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _cuda_alloc_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return None


def _cuda_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


class _RssPeakSampler:
    """Background RSS peak tracker for the duration of a timed call."""

    def __init__(self, *, interval_s: float = 0.001) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._peak: float | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_RssPeakSampler":
        self._peak = _rss_mb()
        self._stop.clear()
        if _PROC is None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="bench-rss-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Capture a final sample in case peak occurred after last tick.
        last = _rss_mb()
        if last is not None:
            self._peak = last if self._peak is None else max(self._peak, last)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            sample = _rss_mb()
            if sample is None:
                continue
            self._peak = sample if self._peak is None else max(self._peak, sample)

    @property
    def peak_mb(self) -> float | None:
        return self._peak


def time_median(
    fn: Callable[[], Any],
    *,
    runs: int,
    warmup: int,
    sync_device: str | None = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Return (median_s, peak_rss_mb, peak_cuda_mb, error)."""

    def _sync() -> None:
        if not sync_device:
            return
        try:
            import torch

            if sync_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif sync_device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception:
            return

    for _ in range(max(0, warmup)):
        try:
            _ = fn()
            _sync()
        except Exception as exc:
            return None, None, None, str(exc)

    times: list[float] = []
    rss_peaks: list[float] = []
    cuda_peaks: list[float] = []
    for _ in range(max(1, runs)):
        gc.collect()
        _cuda_reset()
        with _RssPeakSampler() as sampler:
            t0 = time.perf_counter()
            try:
                _ = fn()
                _sync()
            except Exception as exc:
                return None, None, None, str(exc)
            elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if sampler.peak_mb is not None:
            rss_peaks.append(sampler.peak_mb)
        cu = _cuda_alloc_mb()
        if cu is not None:
            cuda_peaks.append(cu)

    if not times:
        return None, None, None, "no_samples"
    peak_rss = float(max(rss_peaks)) if rss_peaks else None
    peak_cuda = float(max(cuda_peaks)) if cuda_peaks else None
    return float(np.median(times)), peak_rss, peak_cuda, None


def time_medians_interleaved(
    methods: dict[str, Callable[[], Any]],
    *,
    runs: int,
    warmup: int,
    sync_device: str | None = None,
) -> dict[str, tuple[float | None, float | None, float | None, str | None]]:
    """Round-robin timing; returns name -> (median_s, peak_rss_mb, peak_cuda_mb, err)."""

    def _sync() -> None:
        if not sync_device:
            return
        try:
            import torch

            if sync_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif sync_device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception:
            return

    names = list(methods.keys())
    samples: dict[str, list[float]] = {name: [] for name in names}
    rss_peaks: dict[str, list[float]] = {name: [] for name in names}
    cuda_peaks: dict[str, list[float]] = {name: [] for name in names}
    errors: dict[str, str | None] = {name: None for name in names}
    # Soft-skip peers that fail warmup; do not poison the whole interleaved set.
    for name in names:
        for _ in range(max(0, warmup)):
            try:
                methods[name]()
                _sync()
            except Exception as exc:
                errors[name] = str(exc)
                break

    for _ in range(max(1, runs)):
        gc.collect()
        order = names[:]
        np.random.default_rng().shuffle(order)
        for name in order:
            if errors[name] is not None:
                continue
            _cuda_reset()
            with _RssPeakSampler() as sampler:
                try:
                    t0 = time.perf_counter()
                    methods[name]()
                    _sync()
                    samples[name].append(time.perf_counter() - t0)
                except Exception as exc:
                    errors[name] = str(exc)
                    continue
            if sampler.peak_mb is not None:
                rss_peaks[name].append(sampler.peak_mb)
            cu = _cuda_alloc_mb()
            if cu is not None:
                cuda_peaks[name].append(cu)

    out: dict[str, tuple[float | None, float | None, float | None, str | None]] = {}
    for name in names:
        if errors[name] is not None or not samples[name]:
            out[name] = (None, None, None, errors[name] or "no_samples")
        else:
            peak_rss = float(max(rss_peaks[name])) if rss_peaks[name] else None
            peak_cuda = float(max(cuda_peaks[name])) if cuda_peaks[name] else None
            out[name] = (float(np.median(samples[name])), peak_rss, peak_cuda, None)
    return out
