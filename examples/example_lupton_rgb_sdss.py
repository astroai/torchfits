"""Example: Lupton+ (2004) asinh RGB from reprojected SDSS g/r/i cutouts.

Astropy's convention maps the reddest band to R: ``lupton_rgb(i, r, g)``.
The sample images ship ``.fits.bz2``; CFITSIO doesn't decompress bzip2, so
this example inflates them to a temp ``.fits`` file first. Skips cleanly if
the samples aren't cached (fetch via ``bash scripts/fetch_example_samples.sh``).
"""

from __future__ import annotations

import bz2
import contextlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torch  # noqa: E402

import torchfits  # noqa: E402
from torchfits.cli.rgb import write_rgb_image  # noqa: E402
from torchfits.transforms import lupton_rgb  # noqa: E402


@contextlib.contextmanager
def _decompressed(path: Path):
    if path.suffix != ".bz2":
        yield path
        return
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        fh.write(bz2.decompress(path.read_bytes()))
        tmp = fh.name
    try:
        yield Path(tmp)
    finally:
        os.unlink(tmp)


def main() -> int:
    bands = {}
    for band in ("g", "r", "i"):
        p = try_ensure_sample(f"sdss_lupton_{band}")
        if p is None:
            print(
                f"SKIP: sample 'sdss_lupton_{band}' not cached. "
                "Fetch via: bash scripts/fetch_example_samples.sh"
            )
            return 0
        bands[band] = p

    with (
        _decompressed(bands["g"]) as g_path,
        _decompressed(bands["r"]) as r_path,
        _decompressed(bands["i"]) as i_path,
    ):
        g = torchfits.read_tensor(str(g_path), hdu=0).float()
        r = torchfits.read_tensor(str(r_path), hdu=0).float()
        i = torchfits.read_tensor(str(i_path), hdu=0).float()

    print(f"bands: g={tuple(g.shape)} r={tuple(r.shape)} i={tuple(i.shape)}")

    # stretch=0.15: this SDSS sample's object fluxes sit near ~0.1–1 counts;
    # Astropy tutorials often use 0.5, but that leaves this field near-black in a
    # browser. The mapping itself is Astropy-parity (see lupton_rgb).
    rgb = lupton_rgb(i, r, g, Q=8.0, stretch=0.15)
    luma = (rgb.clamp(0, 1) * 255).mean(dim=-1)
    print(
        f"lupton_rgb: shape={tuple(rgb.shape)} "
        f"min={float(rgb.min()):.3f} max={float(rgb.max()):.3f} "
        f"mean={float(luma.mean()):.1f} p90={float(torch.quantile(luma, 0.90)):.1f}"
    )
    if float(luma.mean()) < 8.0 or float(torch.quantile(luma, 0.90)) < 20.0:
        raise RuntimeError(
            "lupton_rgb SDSS PNG looks near-black — refuse to write gallery asset"
        )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "lupton_rgb_sdss.png"
    write_rgb_image(str(png_path), rgb)
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
