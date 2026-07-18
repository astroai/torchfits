#!/usr/bin/env python3
"""CFHT MegaCam MEF cutout benchmark: torchfits vs fitsio on Rice .fz.

Addresses the prior measurement deficit:
- Peers share one family so rankings are meaningful (not torchfits-only).
- Throughput uses cutout payload bytes, not whole-file size.
- ``torchfits_materialize`` decompresses the plane once then slices in memory
  (fair vs uncompressed synthetic; isolates Rice cost from cutout API cost).
- ZNAXIS* sizes for tile-compressed HDUs; ``read_header`` (not deprecated).
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
from pathlib import Path
from typing import Any

import fitsio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchfits  # noqa: E402

from benchmarks.bench_contract import (  # noqa: E402
    RESULT_COLUMNS,
    annotate_rankings,
    make_run_id,
    write_csv,
)
from benchmarks.bench_timing import time_medians_interleaved  # noqa: E402
from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

_BENCH_HOST = socket.gethostname()
_DEFAULT_DATA = ROOT / "benchmarks_data" / "cfht_megacam"
_CUTOUTS_PER_HDU = 40
_CUTOUT_SIZE = 256


def _discover_mef_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    paths = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and re.search(r"\.fits(\.fz|\.gz)?$", p.name, re.I)
    )
    return paths[:10]


def _effective_image_size(header: Any) -> tuple[int, int]:
    """Prefer ZNAXIS* on tile-compressed / ZIMAGE HDUs; else NAXIS*."""
    z1 = int(header.get("ZNAXIS1", 0) or 0)
    z2 = int(header.get("ZNAXIS2", 0) or 0)
    if z1 > 0 and z2 > 0:
        return z1, z2
    n1 = int(header.get("NAXIS1", 0) or 0)
    n2 = int(header.get("NAXIS2", 0) or 0)
    return n1, n2


def _image_hdus(path: Path) -> list[int]:
    hdus: list[int] = []
    with fitsio.FITS(str(path)) as handle:
        for idx in range(len(handle)):
            try:
                header = handle[idx].read_header()
                n1, n2 = _effective_image_size(header)
                if n1 >= 16 and n2 >= 16:
                    hdus.append(idx)
            except Exception:
                continue
    return hdus


def _cutout_coords(
    naxis1: int, naxis2: int, *, n: int, seed: int
) -> list[tuple[int, int, int, int]]:
    rng = np.random.default_rng(seed)
    size = min(_CUTOUT_SIZE, max(8, min(naxis1, naxis2) // 4))
    coords: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        x1 = int(rng.integers(0, max(1, naxis1 - size)))
        y1 = int(rng.integers(0, max(1, naxis2 - size)))
        coords.append((x1, y1, x1 + size, y1 + size))
    return coords


def _cutout_payload_mb(
    coords: list[tuple[int, int, int, int]], *, itemsize: int
) -> float:
    pixels = sum(max(0, x2 - x1) * max(0, y2 - y1) for x1, y1, x2, y2 in coords)
    return (pixels * itemsize) / (1024.0 * 1024.0)


def run_megacam_cutout_rows(
    *,
    run_id: str,
    data_dir: Path,
    profile: str = "user",
    runs: int | None = None,
    warmup: int | None = None,
    max_files: int = 10,
    max_hdus: int = 4,
) -> list[dict[str, Any]]:
    if runs is None:
        runs = 3 if profile == "user" else 5
    if warmup is None:
        warmup = 1 if profile == "user" else 2

    files = _discover_mef_files(data_dir)[:max_files]
    if not files:
        print(
            f"[megacam] no FITS under {data_dir}; run scripts/fetch_cfht_megacam_sample.sh",
            flush=True,
        )
        return []

    rows: list[dict[str, Any]] = []
    for path in files:
        hdus = _image_hdus(path)
        if not hdus:
            print(f"[megacam] skip {path.name}: no 2D image HDUs", flush=True)
            continue
        for hdu in hdus[:max_hdus]:
            header = torchfits.read_header(str(path), hdu=hdu)
            naxis1, naxis2 = _effective_image_size(header)
            if naxis1 < 16 or naxis2 < 16:
                continue
            compressed = bool(
                int(header.get("ZNAXIS1", 0) or 0)
                and int(header.get("ZNAXIS2", 0) or 0)
            )
            coords = _cutout_coords(
                naxis1, naxis2, n=_CUTOUTS_PER_HDU, seed=hdu + len(path.name)
            )
            # Probe dtype once for payload accounting (outside timed path).
            probe = torchfits.read_subset(str(path), hdu=hdu, x1=0, y1=0, x2=8, y2=8)
            itemsize = int(
                np.dtype(np.asarray(probe.detach().cpu().numpy()).dtype).itemsize
            )
            size_mb = _cutout_payload_mb(coords, itemsize=itemsize)
            case_id = f"{path.stem}::hdu{hdu}::megacam_cutouts"
            case_label = (
                f"{path.name} ext {hdu} [{len(coords)} cutouts"
                f"{'; rice' if compressed else ''}]"
            )

            def torchfits_naive(p=path, h=hdu, c=coords) -> list[Any]:
                out = []
                for x1, y1, x2, y2 in c:
                    out.append(
                        torchfits.read_subset(str(p), hdu=h, x1=x1, y1=y1, x2=x2, y2=y2)
                    )
                return out

            def torchfits_cached(p=path, h=hdu, c=coords) -> list[Any]:
                with torchfits.open_subset_reader(str(p), hdu=h) as reader:
                    return [reader.read_subset(x1, y1, x2, y2) for x1, y1, x2, y2 in c]

            def fitsio_cached(p=path, h=hdu, c=coords) -> list[Any]:
                with fitsio.FITS(str(p)) as handle:
                    ext = handle[h]
                    return [np.array(ext[y1:y2, x1:x2]) for x1, y1, x2, y2 in c]

            def torchfits_materialize(p=path, h=hdu, c=coords) -> list[Any]:
                # Decompress / read full plane once, then host slices (no per-cutout I/O).
                plane = torchfits.read_tensor(str(p), hdu=h).detach().cpu().numpy()
                return [np.array(plane[y1:y2, x1:x2]) for x1, y1, x2, y2 in c]

            print(
                f"[megacam] case={case_id} runs={runs} "
                f"payload={size_mb:.3f}MB compressed={compressed}",
                flush=True,
            )
            timed = time_medians_interleaved(
                {
                    "torchfits_naive": torchfits_naive,
                    "torchfits_cached": torchfits_cached,
                    "fitsio_cached": fitsio_cached,
                    "torchfits_materialize": torchfits_materialize,
                },
                runs=runs,
                warmup=warmup,
            )
            # Peer family: open-once subset APIs (torchfits vs fitsio).
            # Materialize is a separate family (different algorithm: full plane).
            # Naive re-open is non-comparable (pathological baseline).
            method_meta = (
                ("torchfits_naive", "torchfits", "baseline", False),
                ("torchfits_cached", "torchfits", "specialized", True),
                ("fitsio_cached", "fitsio", "specialized", True),
                ("torchfits_materialize", "torchfits", "materialize", True),
            )
            for method, library, family, comparable in method_meta:
                t_val, peak_rss, peak_cuda, err = timed[method]
                status = "OK" if t_val is not None else "FAILED"
                rows.append(
                    {
                        "run_id": run_id,
                        "domain": "fits",
                        "suite": "megacam_cutouts",
                        "case_id": case_id,
                        "case_label": case_label,
                        "operation": "megacam_repeated_cutouts",
                        "family": family,
                        "library": library,
                        "method": method,
                        "mode": family,
                        "status": status,
                        "skip_reason": "" if err is None else str(err),
                        "comparable": comparable and status == "OK",
                        "mmap_target": "n/a",
                        "host": _BENCH_HOST,
                        "time_s": t_val,
                        "peak_rss_mb": peak_rss,
                        "peak_cuda_alloc_mb": peak_cuda,
                        "throughput": (size_mb / t_val) if t_val else None,
                        "unit": "MB/s",
                        "size_mb": size_mb,
                        "n_points": len(coords),
                        "metadata": json.dumps(
                            {
                                "hdu": hdu,
                                "cutouts": len(coords),
                                "source": path.name,
                                "znaxis": compressed,
                                "naxis1": naxis1,
                                "naxis2": naxis2,
                                "payload_mb": size_mb,
                            }
                        ),
                    }
                )
    annotate_rankings(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-hdus", type=int, default=4)
    args = parser.parse_args()
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = run_megacam_cutout_rows(
        run_id=run_id,
        data_dir=args.data_dir,
        profile=args.profile,
        max_files=args.max_files,
        max_hdus=args.max_hdus,
    )
    out_csv = run_dir / "megacam_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"Wrote {len(rows)} megacam rows to {out_csv}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
