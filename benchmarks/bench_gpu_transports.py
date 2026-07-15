#!/usr/bin/env python3
"""GPU I/O transport rows for bench-all (MPS on macOS, CUDA on Linux).

Emits normalized rows with metadata.io_transport set so
scripts/render_bench_iopath_table.py can fill GPU transport rows.

``device="cuda"`` / ``device="mps"`` always decode on the host first
(CFITSIO / astropy / fitsio read into host RAM), then copy with
``.to(device)``. There is no native disk→GPU path (no GPUDirect Storage).

* mmap on  -> ``disk→RAM→GPU`` (page-cache / mmap decode + H2D copy)
* mmap off -> ``disk→CPU→GPU`` (buffered host decode + H2D copy)
"""

from __future__ import annotations

import json
import platform
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitsio  # noqa: E402
import torchfits  # noqa: E402
from benchmarks.bench_fits_io import FITSBenchmarkSuite, _strict_patch_astropy  # noqa: E402
from benchmarks.bench_timing import time_median, time_medians_interleaved  # noqa: E402
from astropy.io import fits as astropy_fits  # noqa: E402

_BENCH_HOST = socket.gethostname()


def default_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _median_time(
    fn, warmup: int, iters: int, device: str
) -> tuple[float | None, float | None, float | None]:
    median, peak_rss, peak_cuda, err = time_median(
        fn, runs=iters, warmup=warmup, sync_device=device
    )
    if err is not None:
        return None, None, None
    return median, peak_rss, peak_cuda


def _median_times_interleaved(
    labeled_fns: list[tuple[str, Any]],
    warmup: int,
    iters: int,
    device: str,
) -> dict[str, tuple[float | None, float | None, float | None]]:
    """Round-robin GPU timings so the first library is not favored."""
    # Soft-skip methods that fail warmup (shared timer aborts on first failure).
    alive: dict[str, Any] = {}
    for label, fn in labeled_fns:
        try:
            for _ in range(max(1, warmup)):
                fn()
                _sync(device)
            alive[label] = fn
        except Exception as exc:
            print(f"[bench-gpu] skip warmup {label}: {exc}", flush=True)
    if not alive:
        return {label: (None, None, None) for label, _ in labeled_fns}
    timed = time_medians_interleaved(alive, runs=iters, warmup=0, sync_device=device)
    out: dict[str, tuple[float | None, float | None, float | None]] = {}
    for label, _ in labeled_fns:
        if label not in timed:
            out[label] = (None, None, None)
        else:
            median, peak_rss, peak_cuda, err = timed[label]
            out[label] = (None, None, None) if err else (median, peak_rss, peak_cuda)
    return out


def run_gpu_transport_rows(
    *,
    run_id: str,
    device: str | None = None,
    iterations: int = 10,
    warmup: int = 3,
    quick: bool = False,
    use_mmap: bool = True,
    case_filter: str = "",
    operation_filter: str = "",
) -> list[dict[str, Any]]:
    device = device or default_device()
    if device == "cpu":
        return []

    if device == "cuda" and not torch.cuda.is_available():
        return []
    if device == "mps" and not torch.backends.mps.is_available():
        return []

    op_rx = re.compile(operation_filter) if operation_filter else None

    def _want_ops(*candidates: str) -> bool:
        if op_rx is None:
            return True
        return any(op_rx.search(c) for c in candidates)

    # Probe real operation names (and short aliases) so filters like
    # ``repeated_cutouts`` still enable the cutout sections.
    want_read_full = _want_ops("read_full")
    want_cutout = _want_ops(
        "cutout",
        "cutout_100x100",
        "repeated_cutouts",
        "repeated_cutouts_50x_100x100",
    )

    data_dir = Path(tempfile.mkdtemp(prefix="torchfits_bench_gpu_"))
    suite = FITSBenchmarkSuite(
        output_dir=data_dir,
        use_mmap=use_mmap,
        profile="user",
    )
    _strict_patch_astropy(suite)

    try:
        files = suite.create_test_files()
        files = {
            k: v
            for k, v in files.items()
            if (not k.startswith("table_")) and ("wcs" not in k.lower())
        }
        if quick:
            files = {
                k: v
                for k, v in files.items()
                if k in {"tiny_int16_2d", "mef_small", "compressed_rice_1"}
            }
        if case_filter:
            rx = re.compile(case_filter)
            files = {k: v for k, v in files.items() if rx.search(k)}
            print(
                f"[bench-gpu] case filter {case_filter!r} -> {len(files)} file(s)",
                flush=True,
            )
            if not files:
                return []

        rows: list[dict[str, Any]] = []
        mmap_target = "on" if use_mmap else "off"
        transport = "disk\u2192RAM\u2192GPU" if use_mmap else "disk\u2192CPU\u2192GPU"

        # 1. Full image reads
        if want_read_full:
            for name, path in sorted(files.items()):
                file_type = suite._get_file_type(name)
                hdu = 1 if file_type in {"compressed", "mef", "multi_mef"} else 0
                case_id = f"{name}::read_full_gpu"
                size_mb = path.stat().st_size / (1024 * 1024)
                print(
                    f"[bench-gpu] case={name} file_type={file_type} runs={iterations}",
                    flush=True,
                )

                # Methods to benchmark
                def tf_read(p=path, h=hdu, um=use_mmap):
                    return torchfits.read(
                        str(p), hdu=h, mmap=um, device=device, scale_on_device=True
                    )

                def tf_specialized_read(p=path, h=hdu, um=use_mmap):
                    return torchfits.read_tensor(str(p), hdu=h, mmap=um, device=device)

                def tf_dtype_fair_read(p=path, h=hdu, um=use_mmap):
                    return torchfits.read_tensor(
                        str(p), hdu=h, mmap=um, device=device, raw_scale=True
                    )

                def fitsio_torch_read(p=path, h=hdu):
                    arr = fitsio.read(str(p), ext=h)
                    return torch.from_numpy(suite._ensure_native_endian_numpy(arr)).to(
                        device
                    )

                def astropy_torch_read(p=path, h=hdu):
                    return suite._astropy_to_torch(p, h).to(device)

                methods = [
                    ("torchfits", "torchfits_device", "smart", "smart", tf_read),
                    (
                        "fitsio",
                        "fitsio_torch_device",
                        "smart",
                        "smart",
                        fitsio_torch_read,
                    ),
                    (
                        "astropy",
                        "astropy_torch_device",
                        "smart",
                        "smart",
                        astropy_torch_read,
                    ),
                    (
                        "torchfits",
                        "torchfits_specialized_device",
                        "specialized",
                        "specialized",
                        tf_specialized_read,
                    ),
                    (
                        "torchfits",
                        "torchfits_dtype_fair_device",
                        "specialized",
                        "dtype_fair",
                        tf_dtype_fair_read,
                    ),
                ]

                # Interleaved timing — sequential order favored later methods (fitsio)
                # on MPS microbenches when torchfits always ran first.
                timed = _median_times_interleaved(
                    [(method, fn) for _lib, method, _fam, _mode, fn in methods],
                    warmup,
                    iterations,
                    device,
                )

                for library, method, family, mode, _fn in methods:
                    t, peak_rss, peak_cuda = timed.get(method, (None, None, None))
                    if t is None:
                        continue
                    comparable = True
                    skip_reason = ""
                    if library == "fitsio" and mmap_target == "on":
                        comparable = False
                        skip_reason = "fitsio_no_mmap: not comparable under mmap-on"
                    rows.append(
                        {
                            "run_id": run_id,
                            "domain": "fits",
                            "suite": "fits_gpu",
                            "case_id": case_id,
                            "case_label": f"{name} [read_full @ {device}]",
                            "operation": "read_full",
                            "family": family,
                            "library": library,
                            "method": method,
                            "mode": mode,
                            "status": "OK",
                            "skip_reason": skip_reason,
                            "comparable": comparable,
                            "mmap_target": mmap_target,
                            "host": _BENCH_HOST,
                            "time_s": t,
                            "peak_rss_mb": peak_rss,
                            "peak_cuda_alloc_mb": peak_cuda,
                            "throughput": "",
                            "unit": "MB/s",
                            "size_mb": size_mb,
                            "n_points": "",
                            "metadata": json.dumps(
                                {"device": device, "io_transport": transport}
                            ),
                        }
                    )

        # 2. Cutout 100x100 reads
        if want_cutout:
            targets = [
                ("multi_mef_10ext", 5, "uncompressed"),
                ("compressed_rice_1", 1, "compressed"),
            ]
            x1, y1, x2, y2 = 100, 100, 200, 200
            for name, hdu, compression in targets:
                path = files.get(name)
                if path is None:
                    continue
                case_id = f"{name}::cutout_100x100_gpu"
                size_mb = path.stat().st_size / (1024 * 1024)
                print(
                    f"[bench-gpu] case={name} cutout=100x100 runs={iterations}",
                    flush=True,
                )

                def tf_cutout(p=path, h=hdu):
                    # Open once like fitsio.FITS(...); read_subset alone re-opens each call.
                    with torchfits.open_subset_reader(
                        str(p), hdu=h, device=device
                    ) as reader:
                        return reader.read_subset(x1, y1, x2, y2)

                def fitsio_cutout(p=path, h=hdu):
                    with fitsio.FITS(str(p)) as handle:
                        arr = suite._ensure_native_endian_numpy(handle[h][y1:y2, x1:x2])
                        return torch.from_numpy(arr).to(device)

                def astropy_cutout(p=path, h=hdu, um=use_mmap):
                    with astropy_fits.open(p, memmap=um) as hdul:
                        arr = suite._ensure_native_endian_numpy(
                            np.array(hdul[h].section[y1:y2, x1:x2], copy=True)
                        )
                        return torch.from_numpy(arr).to(device)

                methods = [
                    ("torchfits", "torchfits_device", "smart", "smart", tf_cutout),
                    ("fitsio", "fitsio_torch_device", "smart", "smart", fitsio_cutout),
                    (
                        "astropy",
                        "astropy_torch_device",
                        "smart",
                        "smart",
                        astropy_cutout,
                    ),
                    (
                        "torchfits",
                        "torchfits_specialized_device",
                        "specialized",
                        "specialized",
                        tf_cutout,
                    ),
                ]

                timed = _median_times_interleaved(
                    [(method, fn) for _lib, method, _fam, _mode, fn in methods],
                    warmup,
                    iterations,
                    device,
                )
                for library, method, family, mode, _fn in methods:
                    t, peak_rss, peak_cuda = timed.get(method, (None, None, None))
                    if t is None:
                        continue
                    comparable = True
                    skip_reason = ""
                    if library == "fitsio" and mmap_target == "on":
                        comparable = False
                        skip_reason = "fitsio_no_mmap: not comparable under mmap-on"
                    rows.append(
                        {
                            "run_id": run_id,
                            "domain": "fits",
                            "suite": "fits_gpu",
                            "case_id": case_id,
                            "case_label": f"{name} [cutout_100x100 @ {device}]",
                            "operation": "cutout_100x100",
                            "family": family,
                            "library": library,
                            "method": method,
                            "mode": mode,
                            "status": "OK",
                            "skip_reason": skip_reason,
                            "comparable": comparable,
                            "mmap_target": mmap_target,
                            "host": _BENCH_HOST,
                            "time_s": t,
                            "peak_rss_mb": peak_rss,
                            "peak_cuda_alloc_mb": peak_cuda,
                            "throughput": "",
                            "unit": "MB/s",
                            "size_mb": size_mb,
                            "n_points": "",
                            "metadata": json.dumps(
                                {"device": device, "io_transport": transport}
                            ),
                        }
                    )

        # 3. Repeated Cutouts 50x 100x100 reads on GPU
        path = files.get("medium_float32_2d") if want_cutout else None
        if path is None and want_cutout:
            for k in sorted(files.keys()):
                if "2d" in k:
                    path = files[k]
                    break
        if path is not None:
            file_type = suite._get_file_type(path.stem)
            hdu = 1 if file_type in {"compressed", "mef", "multi_mef"} else 0

            with fitsio.FITS(str(path)) as f:
                header = f[hdu].read_header()
                naxis1 = header.get("NAXIS1", 1024)
                naxis2 = header.get("NAXIS2", 1024)

            cutout_size = min(100, naxis1 // 2, naxis2 // 2)
            if cutout_size < 2:
                cutout_size = 2

            case_id = f"repeated_cutouts_50x_{cutout_size}x{cutout_size}_gpu"
            size_mb = path.stat().st_size / (1024 * 1024)
            print(
                f"[bench-gpu] case=repeated_cutouts_50x_{cutout_size}x{cutout_size} runs={iterations}",
                flush=True,
            )

            coords_rng = np.random.default_rng(42)
            cutouts_coords = []
            for _ in range(50):
                x1 = int(coords_rng.integers(0, max(1, naxis1 - cutout_size)))
                y1 = int(coords_rng.integers(0, max(1, naxis2 - cutout_size)))
                cutouts_coords.append((x1, y1, x1 + cutout_size, y1 + cutout_size))

            def tf_repeated_cutout_persistent(p=path):
                with torchfits.open_subset_reader(
                    str(p), hdu=hdu, device=device
                ) as reader:
                    results = []
                    for x1, y1, x2, y2 in cutouts_coords:
                        results.append(reader.read_subset(x1, y1, x2, y2))
                    return results

            def fitsio_repeated_cutout(p=path):
                with fitsio.FITS(str(p)) as handle:
                    results = []
                    for x1, y1, x2, y2 in cutouts_coords:
                        arr = suite._ensure_native_endian_numpy(
                            handle[hdu][y1:y2, x1:x2]
                        )
                        results.append(torch.from_numpy(arr).to(device))
                    return results

            def astropy_repeated_cutout(p=path, um=use_mmap):
                with astropy_fits.open(p, memmap=um) as hdul:
                    results = []
                    for x1, y1, x2, y2 in cutouts_coords:
                        arr = suite._ensure_native_endian_numpy(
                            np.array(hdul[hdu].section[y1:y2, x1:x2], copy=True)
                        )
                        results.append(torch.from_numpy(arr).to(device))
                    return results

            methods = [
                (
                    "torchfits",
                    "torchfits_device",
                    "smart",
                    "smart",
                    # Match fitsio's open-once FITS handle: persistent subset reader.
                    tf_repeated_cutout_persistent,
                ),
                (
                    "fitsio",
                    "fitsio_torch_device",
                    "smart",
                    "smart",
                    fitsio_repeated_cutout,
                ),
                (
                    "astropy",
                    "astropy_torch_device",
                    "smart",
                    "smart",
                    astropy_repeated_cutout,
                ),
                (
                    "torchfits",
                    "torchfits_specialized_device",
                    "specialized",
                    "specialized",
                    tf_repeated_cutout_persistent,
                ),
            ]

            timed = _median_times_interleaved(
                [(method, fn) for _lib, method, _fam, _mode, fn in methods],
                warmup,
                iterations,
                device,
            )
            for library, method, family, mode, _fn in methods:
                t, peak_rss, peak_cuda = timed.get(method, (None, None, None))
                if t is None:
                    continue
                comparable = True
                skip_reason = ""
                if library == "fitsio" and mmap_target == "on":
                    comparable = False
                    skip_reason = "fitsio_no_mmap: not comparable under mmap-on"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "fits_gpu",
                        "case_id": case_id,
                        "case_label": f"repeated_cutouts_50x_{cutout_size}x{cutout_size} @ {device}",
                        "operation": f"repeated_cutouts_50x_{cutout_size}x{cutout_size}",
                        "family": family,
                        "library": library,
                        "method": method,
                        "mode": mode,
                        "status": "OK",
                        "skip_reason": skip_reason,
                        "comparable": comparable,
                        "mmap_target": mmap_target,
                        "host": _BENCH_HOST,
                        "time_s": t,
                        "peak_rss_mb": peak_rss,
                        "peak_cuda_alloc_mb": peak_cuda,
                        "throughput": "",
                        "unit": "MB/s",
                        "size_mb": size_mb,
                        "n_points": "",
                        "metadata": json.dumps(
                            {"device": device, "io_transport": transport}
                        ),
                    }
                )

        return rows
    finally:
        suite.cleanup()


def main() -> int:
    import argparse

    from benchmarks.bench_contract import RESULT_COLUMNS, write_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()
    rows = run_gpu_transport_rows(run_id=args.run_id, device=args.device)
    write_csv(args.output, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} GPU transport rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
