"""
Example: fpack-style tile compression controls, in Python and on the CLI.

``torchfits.write`` (and ``write_tensor`` / ``insert_hdu`` / ``replace_hdu``)
take the same knobs as ``fpack``:

| Python keyword          | CLI (``torchfits compress``) | fpack  |
|-------------------------|------------------------------|--------|
| ``tile_shape=(r, c)``   | ``-t c,r`` (FITS axis order) | ``-t`` |
| ``quantize_level=q``    | ``-q q``                     | ``-q`` |
| ``quantize_method=``    | ``--quantize-method``        |        |
| ``dither_seed=n``       | ``--dither-seed n``          |        |
| ``int_to_float=True``   | ``--int-to-float``           | ``-i2f`` |

Note the axis order: ``tile_shape`` is in tensor order (rows, cols), ``-t``
in FITS order (NAXIS1 first). Synthetic data only; checks are assertions.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import torch

import torchfits


def _sky(seed: int = 0, n: int = 512) -> torch.Tensor:
    """Float sky: flat background + Gaussian noise + a few bright sources."""
    g = torch.Generator().manual_seed(seed)
    img = 100.0 + 5.0 * torch.randn(n, n, generator=g)
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    for cy, cx in ((100, 120), (300, 400), (420, 60)):
        img += 2000.0 * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
    img[10, 10] = float("nan")  # a bad pixel survives every codec as NaN
    return img.float()


def _cli(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "torchfits.cli", *args], check=True)


def main() -> None:
    sky = _sky()
    with tempfile.TemporaryDirectory(prefix="torchfits_fpack_") as tmp:

        def path(name: str) -> str:
            return os.path.join(tmp, name)

        raw = path("raw.fits")
        torchfits.write(raw, sky, overwrite=True)
        print(f"uncompressed float32: {os.path.getsize(raw):>9,} bytes")

        # 1. Lossless: GZIP_1 keeps floats exact by default (the other codecs
        #    quantize them; GZIP_2 is lossless with quantize_level=0).
        #    tile_shape sets the tile size in tensor order (rows, cols).
        torchfits.write(
            path("gzip.fits"),
            sky,
            compress="GZIP_1",
            tile_shape=(128, 128),
            overwrite=True,
        )
        gz = torchfits.read(path("gzip.fits"), hdu=1)
        assert torch.allclose(gz, sky, rtol=0, atol=0, equal_nan=True)  # bit-exact
        hdr = torchfits.read_header(path("gzip.fits"), hdu=1)
        assert (hdr["ZTILE1"], hdr["ZTILE2"]) == (128, 128)
        print(
            f"GZIP_1 128x128 tiles: {os.path.getsize(path('gzip.fits')):>9,} bytes (lossless)"
        )

        # 2. Lossy RICE_1: quantize_level q sets the step to (tile noise / q);
        #    a fixed dither_seed makes repeated writes byte-identical.
        for name in ("rice_a.fits", "rice_b.fits"):
            torchfits.write(
                path(name),
                sky,
                compress="RICE_1",
                quantize_level=16,
                dither_seed=42,
                overwrite=True,
            )
        with open(path("rice_a.fits"), "rb") as a, open(path("rice_b.fits"), "rb") as b:
            assert a.read() == b.read()
        rice = torchfits.read(path("rice_a.fits"), hdu=1)
        ok = ~torch.isnan(sky)
        assert torch.isnan(rice[10, 10])
        err = (rice - sky)[ok].abs().max().item()
        assert err < 5.0 / 16  # within one step (noise sigma 5 / q)
        hdr = torchfits.read_header(path("rice_a.fits"), hdu=1)
        assert hdr["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_1" and hdr["ZDITHER0"] == 42
        print(
            f"RICE_1 q=16 seed=42:  {os.path.getsize(path('rice_a.fits')):>9,} bytes "
            f"(max error {err:.3f}, reproducible)"
        )

        # quantize_level < 0 is an absolute step; subtractive_dither_2 also
        # keeps exact zeros.
        torchfits.write(
            path("abs.fits"),
            sky,
            compress="RICE_1",
            quantize_level=-0.5,
            quantize_method="subtractive_dither_2",
            dither_seed=1,
            overwrite=True,
        )
        err = (torchfits.read(path("abs.fits"), hdu=1) - sky)[ok].abs().max().item()
        assert err <= 0.25 + 1e-3
        print(
            f"RICE_1 step 0.5:      {os.path.getsize(path('abs.fits')):>9,} bytes "
            f"(max error {err:.3f})"
        )

        # 3. int_to_float (fpack -i2f): integer counts stored as quantized
        #    float32. Integer images are otherwise always lossless.
        counts = sky.nan_to_num(100.0).round().to(torch.int32)
        torchfits.write(
            path("i2f.fits"),
            counts,
            compress="RICE_1",
            int_to_float=True,
            quantize_level=-0.5,
            dither_seed=3,
            overwrite=True,
        )
        restored = torchfits.read(path("i2f.fits"), hdu=1)
        assert restored.dtype == torch.float32
        assert (restored - counts).abs().max().item() <= 0.25 + 1e-3
        print(
            f"int32 -> i2f step 0.5: {os.path.getsize(path('i2f.fits')):>8,} bytes "
            f"(read back as {restored.dtype})"
        )

        # 4. The same controls on the CLI. -t is in FITS axis order (NAXIS1
        #    first): "-t 256,64" is 64 rows x 256 columns per tile.
        _cli(
            "compress",
            raw,
            path("cli.fits.fz"),
            "--algorithm",
            "RICE_1",
            "-t",
            "256,64",
            "-q",
            "16",
            "--dither-seed",
            "42",
        )
        hdr = torchfits.read_header(path("cli.fits.fz"), hdu=1)
        assert (hdr["ZTILE1"], hdr["ZTILE2"]) == (256, 64)
        torchfits.write(path("counts.fits"), counts, overwrite=True)
        _cli(
            "compress",
            path("counts.fits"),
            path("counts.fits.fz"),
            "--algorithm",
            "RICE_1",
            "--int-to-float",
            "-q",
            "4",
        )
        _cli("decompress", path("counts.fits.fz"), path("counts_back.fits"))
        assert torchfits.read_shape(path("counts_back.fits"), hdu=0)[0] == -32
        print("torchfits compress -t/-q/--dither-seed/--int-to-float: OK")

    print("All compression checks passed.")


if __name__ == "__main__":
    main()
