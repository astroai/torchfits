"""``convert --to png`` row order: FITS row 0 is the bottom of the image.

PNG scanlines run top-down, so the default (``--origin lower``) must flip
rows to match ds9 / matplotlib ``origin="lower"``; ``--origin upper`` keeps
array row 0 on top.
"""

from __future__ import annotations

import struct
import zlib

import torch

import torchfits


def _png_rows(path) -> list[bytes]:
    data = path.read_bytes()
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    return [raw[i * stride + 1 : (i + 1) * stride] for i in range(height)]


def _bright_bottom_row_fits(tmp_path):
    img = torch.zeros(8, 6)
    img[0, :] = 100.0  # FITS row 0 = southern edge for a standard WCS
    path = tmp_path / "img.fits"
    torchfits.write(str(path), img, overwrite=True)
    return path


def test_convert_png_default_origin_lower_puts_row0_at_bottom(tmp_path):
    from torchfits.cli.main import main

    src = _bright_bottom_row_fits(tmp_path)
    out = tmp_path / "out.png"
    assert main(["convert", str(src), "-o", str(out), "--to", "png"]) == 0
    rows = _png_rows(out)
    assert sum(rows[-1]) > sum(rows[0])


def test_convert_png_origin_upper_keeps_row0_on_top(tmp_path):
    from torchfits.cli.main import main

    src = _bright_bottom_row_fits(tmp_path)
    out = tmp_path / "out.png"
    assert (
        main(["convert", str(src), "-o", str(out), "--to", "png", "--origin", "upper"])
        == 0
    )
    rows = _png_rows(out)
    assert sum(rows[0]) > sum(rows[-1])
