"""
Example: N-D table cells (``TDIMn``) -- PSF models, postage stamps, cubes per row.

Writing a column of shape ``(N, d1, d2, ...)`` stores each row's cell flat with
``TFORM = '<d1*d2*...><code>'`` and ``TDIM = '(..., d2, d1)'`` (FITS lists the
fastest axis first). Readers return such a column flat by default; pass
``shape_cells=True`` to get ``(N, d1, d2, ...)`` back, on ``read_torch``,
``scan_torch``, ``read``, ``scan``, ``reader``, ``schema`` and the converters.

Synthetic data only; every claim is an assertion.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

import torchfits

N = 64


def main() -> None:
    rng = np.random.default_rng(3)
    psf = rng.normal(size=(N, 5, 7)).astype(np.float32)  # numpy (N, 5, 7)
    stamps = torch.arange(N * 3 * 16 * 16, dtype=torch.int16).reshape(N, 3, 16, 16)

    with tempfile.TemporaryDirectory(prefix="torchfits_tdim_") as tmp:
        path = os.path.join(tmp, "cells.fits")
        torchfits.table.write(
            path,
            {"ID": np.arange(N, dtype=np.int32), "PSF": psf, "STAMP": stamps},
            overwrite=True,
        )
        hdr = torchfits.read_header(path, hdu=1)
        assert (hdr["TFORM2"], hdr["TDIM2"]) == ("35E", "(7,5)")
        assert (hdr["TFORM3"], hdr["TDIM3"]) == ("768I", "(16,16,3)")
        print(f"PSF   TFORM={hdr['TFORM2']:<5} TDIM={hdr['TDIM2']}")
        print(f"STAMP TFORM={hdr['TFORM3']:<5} TDIM={hdr['TDIM3']}")

        # Default: vector columns read flat, as before.
        flat = torchfits.table.read_torch(path)
        assert flat["PSF"].shape == (N, 35)

        # shape_cells=True: the cell shape you wrote comes back.
        shaped = torchfits.table.read_torch(path, shape_cells=True)
        assert shaped["PSF"].shape == (N, 5, 7)
        assert shaped["STAMP"].shape == (N, 3, 16, 16)
        assert np.array_equal(shaped["PSF"].numpy(), psf)
        assert torch.equal(shaped["STAMP"], stamps)
        print(f"read_torch(shape_cells=True): STAMP {tuple(shaped['STAMP'].shape)}")

        for batch in torchfits.table.scan_torch(path, shape_cells=True, batch_size=16):
            assert batch["STAMP"].shape == (16, 3, 16, 16)

        # Arrow: nested fixed-size lists, which Polars reads as an Array column.
        table = torchfits.table.read(path, columns=["PSF"], shape_cells=True)
        cell = np.asarray(table.column("PSF")[0].as_py(), dtype=np.float32)
        assert cell.shape == (5, 7) and np.array_equal(cell, psf[0])
        print(f"table.read(shape_cells=True): {table.schema.field('PSF').type}")

        # Streaming writes take N-D batches too; the first batch fixes the cell.
        streamed = os.path.join(tmp, "streamed.fits")
        with torchfits.table.writer(streamed, overwrite=True) as writer:
            for start in range(0, N, 16):
                writer.write({"STAMP": stamps[start : start + 16]})
        assert torchfits.read_header(streamed, hdu=1)["TDIM1"] == "(16,16,3)"
        back = torchfits.table.read_torch(streamed, shape_cells=True)["STAMP"]
        assert torch.equal(back, stamps)
        print("table.writer: N-D batches stored with TDIM")

    print("All TDIM checks passed.")


if __name__ == "__main__":
    main()
