"""Example: stack multiple exposures (M13 blue frames) into a mean image.

Skips cleanly if the M13 samples aren't cached (fetch via
``bash scripts/fetch_example_samples.sh``).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402

NAMES = [f"m13_blue_000{i}" for i in range(1, 6)]


def main() -> int:
    paths = [try_ensure_sample(name) for name in NAMES]
    if any(p is None for p in paths):
        print(
            "SKIP: m13_blue_0001..5 not cached. "
            "Fetch via: bash scripts/fetch_example_samples.sh"
        )
        return 0

    frames = [torchfits.read_tensor(str(p), hdu=0).float() for p in paths]
    for name, frame in zip(NAMES, frames):
        print(f"{name}: shape={tuple(frame.shape)} mean={frame.mean().item():.2f}")

    stack = torch.stack(frames, dim=0)
    mean_frame = stack.mean(dim=0)
    print(f"stack: shape={tuple(stack.shape)}")
    print(
        f"mean frame: shape={tuple(mean_frame.shape)} mean={mean_frame.mean().item():.2f}"
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        out_path = fh.name
    try:
        torchfits.write_tensor(out_path, mean_frame, overwrite=True)
        roundtrip = torchfits.read_tensor(out_path)
        print(f"wrote mean stack: {out_path} shape={tuple(roundtrip.shape)}")
    finally:
        os.unlink(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
