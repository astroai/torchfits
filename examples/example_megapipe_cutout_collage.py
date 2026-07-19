"""Example: RGB cutout collage from the CFHTLS-Deep D1 MegaPipe g/r/i mosaics.

Scans ``benchmarks_data/cfht_megapipe`` for the D1 mosaics + SExtractor
catalog. Skips cleanly if absent — fetch via
``bash scripts/fetch_cfht_megapipe_sample.sh`` (~5.3 GB).

Gallery stamps use ``MAG_AUTO ∈ [17, 22]`` (pretty galaxies). Timing uses a
separate random box list, one subprocess per backend so an earlier reader
cannot silently warm the OS page cache for the next.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import megapipe_dir  # noqa: E402

import torch  # noqa: E402

import torchfits  # noqa: E402
from torchfits.cli.rgb import lupton_rgb, write_rgb_image  # noqa: E402

FETCH_CMD = "bash scripts/fetch_cfht_megapipe_sample.sh"
SIZE = 64
# Collage for the gallery: bright sources, full tile size (no thumbnail crush).
DEFAULT_COLLAGE_N = 64
# Timing pass can use more boxes without bloating the PNG.
DEFAULT_TIMING_N = 1000
Box = tuple[int, int, int, int]


def _fast_mode() -> bool:
    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
    )


def _parse_cat(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse a SExtractor ``.cat``: ``#`` comment lines define column names."""
    col_index: dict[str, int] = {}
    xs: list[float] = []
    ys: list[float] = []
    mags: list[float] = []
    x_col = y_col = mag_col = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.match(r"#\s*(\d+)\s+(\S+)", line)
                if m:
                    col_index[m.group(2)] = int(m.group(1)) - 1
                continue
            if x_col is None:
                x_col, y_col = col_index.get("X_IMAGE"), col_index.get("Y_IMAGE")
                mag_col = col_index.get("MAG_AUTO")
                if x_col is None or y_col is None or mag_col is None:
                    raise ValueError(
                        f"X_IMAGE/Y_IMAGE/MAG_AUTO not found in {path} header"
                    )
            parts = line.split()
            xs.append(float(parts[x_col]))
            ys.append(float(parts[y_col]))
            mags.append(float(parts[mag_col]))
    x = torch.tensor(xs, dtype=torch.float64) - 1.0  # SExtractor 1-based -> 0-based
    y = torch.tensor(ys, dtype=torch.float64) - 1.0
    mag = torch.tensor(mags, dtype=torch.float64)
    return x, y, mag


def _select_boxes(
    x: torch.Tensor,
    y: torch.Tensor,
    mag: torch.Tensor,
    n: int,
    size: int,
    naxis1: int,
    naxis2: int,
    *,
    mode: str,
    seed: int = 0,
    mag_lo: float = 17.0,
    mag_hi: float = 22.0,
) -> list[Box]:
    """Build pixel boxes.

    ``mode``:
      - ``gallery``: MAG_AUTO in ``[mag_lo, mag_hi]`` (pretty galaxies, not
        saturated stars that wash out Lupton)
      - ``timing``: random valid sources
    """
    margin = size  # generous buffer so the box always stays in-mosaic
    valid = (
        (x >= margin)
        & (x < naxis1 - margin)
        & (y >= margin)
        & (y < naxis2 - margin)
        & torch.isfinite(mag)
        & (mag < 90.0)  # SExtractor sentinel
    )
    if mode == "gallery":
        valid = valid & (mag >= mag_lo) & (mag <= mag_hi)
    elif mode != "timing":
        raise ValueError(f"unknown select mode: {mode}")
    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        raise RuntimeError(f"no catalog sources for mode={mode}")
    g = torch.Generator().manual_seed(seed)
    chosen = idx[torch.randperm(idx.numel(), generator=g)][:n]
    half = size // 2
    return [
        (cx - half, cy - half, cx - half + size, cy - half + size)
        for cx, cy in ((int(x[i]), int(y[i])) for i in chosen.tolist())
    ]


def _grid_dims(n: int) -> tuple[int, int]:
    cols = max(1, round(math.sqrt(n)))
    rows = math.ceil(n / cols)
    return rows, cols


def _build_collage(
    rgb_list: list[torch.Tensor], rows: int, cols: int, size: int
) -> torch.Tensor:
    canvas = torch.zeros((rows * size, cols * size, 3), dtype=torch.float64)
    for i, rgb in enumerate(rgb_list):
        r, c = divmod(i, cols)
        canvas[r * size : (r + 1) * size, c * size : (c + 1) * size, :] = rgb
    return canvas


def _assert_visible(rgb: torch.Tensor, *, label: str) -> None:
    """Fail loudly if a gallery RGB is near-black or washed-out."""
    u8 = (rgb.clamp(0, 1) * 255).reshape(-1, 3).mean(dim=-1)
    mean = float(u8.mean())
    p50 = float(torch.quantile(u8, 0.50))
    p90 = float(torch.quantile(u8, 0.90))
    if mean < 8.0 or p90 < 25.0:
        raise RuntimeError(
            f"{label}: RGB looks near-black (mean={mean:.1f}, p90={p90:.1f}). "
            "Check lupton_rgb stretch / source selection — do not ship this PNG."
        )
    if mean > 140.0 and p50 > 120.0:
        raise RuntimeError(
            f"{label}: RGB looks washed-out (mean={mean:.1f}, p50={p50:.1f}). "
            "Use fainter MAG_AUTO / larger stretch — do not ship this PNG."
        )
    print(f"  visibility OK ({label}): mean={mean:.1f} p50={p50:.1f} p90={p90:.1f}")


def _print_timing_table(path: Path, boxes: list[Box]) -> list[tuple[str, float]]:
    """Time each backend in its own subprocess.

    Setup (import + open + one warm cutout) is outside the timed loop so we
    measure cutout throughput, not extension import cost.
    """
    workers = {
        "torchfits.read_subset": (
            "import torchfits\n"
            "torchfits.read_subset(path, 0, *boxes[0])\n",
            "for b in boxes:\n"
            "    torchfits.read_subset(path, 0, *b)\n",
        ),
        "torchfits.open_subset_reader": (
            "import torchfits\n"
            "reader = torchfits.open_subset_reader(path, hdu=0)\n"
            "reader.read_subset(*boxes[0])\n",
            "for b in boxes:\n"
            "    reader.read_subset(*b)\n"
            "reader.close()\n",
        ),
        "fitsio": (
            "import fitsio\n"
            "import numpy as np\n"
            "handle = fitsio.FITS(path)\n"
            "ext = handle[0]\n"
            "x1,y1,x2,y2 = boxes[0]\n"
            "_ = np.array(ext[y1:y2, x1:x2], copy=True)\n",
            "for x1, y1, x2, y2 in boxes:\n"
            "    _ = np.array(ext[y1:y2, x1:x2], copy=True)\n"
            "handle.close()\n",
        ),
        "astropy": (
            "from astropy.io import fits as af\n"
            "import numpy as np\n"
            "hdul = af.open(path, memmap=True)\n"
            "data = hdul[0].data\n"
            "x1,y1,x2,y2 = boxes[0]\n"
            "_ = np.array(data[y1:y2, x1:x2], copy=True)\n",
            "for x1, y1, x2, y2 in boxes:\n"
            "    _ = np.array(data[y1:y2, x1:x2], copy=True)\n"
            "hdul.close()\n",
        ),
    }

    rows: list[tuple[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        box_path = Path(tmp) / "boxes.json"
        box_path.write_text(json.dumps(boxes))
        for name, (setup, body) in workers.items():
            script = (
                "import json, time\n"
                f"path = {str(path)!r}\n"
                f"boxes = [tuple(b) for b in json.loads(open({str(box_path)!r}).read())]\n"
                "try:\n"
                + "".join(f"    {line}\n" for line in setup.splitlines())
                + "except ImportError as exc:\n"
                "    print('IMPORT_ERROR', exc)\n"
                "    raise SystemExit(2) from exc\n"
                "t0 = time.perf_counter()\n"
                + body
                + "print(time.perf_counter() - t0)\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 2:
                print(f"  ({name} not installed, skipped)")
                continue
            if proc.returncode != 0:
                err = (
                    (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:200]
                )
                print(f"  ({name} failed: {err})")
                continue
            secs = float(proc.stdout.strip().splitlines()[-1])
            rows.append((name, secs))

    n = len(boxes)
    print(
        f"\ntiming: {n} {SIZE}x{SIZE} cutouts, G band, one subprocess per backend "
        f"(setup+warm outside timer; owned copies):"
    )
    for name, secs in rows:
        print(f"  {name:<30s} {secs:7.3f}s  ({secs / n * 1e3:6.3f} ms/cutout)")
    return rows


def main() -> int:
    if _fast_mode():
        print("SKIP: TORCHFITS_EXAMPLE_FAST=1")
        return 0

    mp_dir = megapipe_dir()
    g_path = mp_dir / "D1.IQ.G.fits"
    r_path = mp_dir / "D1.IQ.R.fits"
    i_path = mp_dir / "D1.IQ.I.fits"
    cat_path = mp_dir / "D1.IQ.G.cat"
    missing = [p.name for p in (g_path, r_path, i_path, cat_path) if not p.is_file()]
    if missing:
        print(f"SKIP: missing {missing} under {mp_dir}. Fetch via: {FETCH_CMD}")
        return 0

    collage_n = int(os.environ.get("MEGAPIPE_COLLAGE_N", DEFAULT_COLLAGE_N))
    timing_n = int(os.environ.get("MEGAPIPE_N_CUTOUTS", DEFAULT_TIMING_N))

    print(f"catalog: {cat_path.name}")
    x, y, mag = _parse_cat(cat_path)
    print(f"  sources: {x.numel()}")

    header = torchfits.read_header(str(g_path), hdu=0)
    naxis1, naxis2 = int(header["NAXIS1"]), int(header["NAXIS2"])

    collage_boxes = _select_boxes(
        x, y, mag, collage_n, SIZE, naxis1, naxis2, mode="gallery", seed=0
    )
    timing_boxes = _select_boxes(
        x, y, mag, timing_n, SIZE, naxis1, naxis2, mode="timing", seed=1
    )
    print(
        f"  collage: {len(collage_boxes)} MAG_AUTO in [17,22] "
        f"(size={SIZE}); timing: {len(timing_boxes)} random boxes"
    )

    # Time peers before collage I/O so the printed table is not a warm-cache gift.
    timing_rows = _print_timing_table(g_path, timing_boxes)

    t0 = time.perf_counter()
    with (
        torchfits.open_subset_reader(str(g_path), hdu=0) as gr,
        torchfits.open_subset_reader(str(r_path), hdu=0) as rr,
        torchfits.open_subset_reader(str(i_path), hdu=0) as ir,
    ):
        g_cuts = [gr.read_subset(*b) for b in collage_boxes]
        r_cuts = [rr.read_subset(*b) for b in collage_boxes]
        i_cuts = [ir.read_subset(*b) for b in collage_boxes]
    print(
        f"  read {3 * len(collage_boxes)} cutouts via open_subset_reader in "
        f"{time.perf_counter() - t0:.2f}s"
    )

    # Astropy-default stretch=5 suits mag~17–22 MegaPipe stamps.
    rgb_list = [
        lupton_rgb(i_c, r_c, g_c, Q=8.0, stretch=5.0)
        for g_c, r_c, i_c in zip(g_cuts, r_cuts, i_cuts, strict=True)
    ]

    rows, cols = _grid_dims(len(rgb_list))
    canvas = _build_collage(rgb_list, rows, cols, SIZE)
    _assert_visible(canvas, label="megapipe collage")

    out_path = (
        Path(__file__).resolve().parent / "output" / "megapipe_cutout_collage.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgb_image(str(out_path), canvas.float())
    print(
        f"wrote {out_path} ({rows}x{cols} grid, {canvas.shape[1]}x{canvas.shape[0]} px)"
    )
    if timing_rows:
        best = min(timing_rows, key=lambda r: r[1])
        print(f"fastest timing backend this run: {best[0]} ({best[1]:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
