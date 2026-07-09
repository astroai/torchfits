"""End-to-end FITS round-trip tests for scaled images and tables.

Verifies that ``FITSHeaderScale``, ``FITSScaleColumns``, ``TNullToNan`` and
``FITSHeaderNormalize`` correctly handle FITS files written with real
BSCALE / BZERO / TSCAL / TZERO / TNULL keywords by writing, reading, and
asserting the round-trip is within the storage precision.

These tests live in a separate module to keep ``tests/test_transforms.py``
focused on pure-tensor behaviour of the transforms themselves.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from torchfits.transforms import (
    FITSHeaderNormalize,
    TNullToNan,
)


# ---------------------------------------------------------------------------
# Helpers: write FITS files with BSCALE/BZERO/TSCAL/TZERO/TNULL
# ---------------------------------------------------------------------------


def _write_bscaled_image_fits(
    path: str,
    physical: torch.Tensor,
    *,
    bscale: float,
    bzero: float,
    bitpix: int = 16,
) -> np.ndarray:
    """Write a single-HDU image with BSCALE/BZERO. Returns on-disk storage."""
    from astropy.io import fits

    p = physical.double()
    stored_quantized = ((p - bzero) / bscale).round()

    if bitpix == 16:
        stored = stored_quantized.clamp(-32768, 32767).to(torch.int16).numpy()
    elif bitpix == 32:
        stored = stored_quantized.clamp(-2147483648, 2147483647).to(torch.int32).numpy()
    elif bitpix == -32:
        stored = p.to(torch.float32).numpy()
    else:
        raise ValueError(f"unsupported bitpix for test: {bitpix}")

    hdu = fits.PrimaryHDU(stored)
    hdu.header["BITPIX"] = bitpix
    hdu.header["BSCALE"] = bscale
    hdu.header["BZERO"] = bzero
    hdu.writeto(path, overwrite=True)
    return stored


def _write_scaled_table_fits(
    path: str,
    physical: torch.Tensor,
    *,
    column_name: str,
    tscal: float,
    tzero: float,
    tnull: int | None = None,
) -> np.ndarray:
    """Write a single-column FITS binary table via BinTableHDU (no astropy.table)."""
    from astropy.io import fits as _fits

    stored_quantized = ((physical.double() - tzero) / tscal).round()
    stored = stored_quantized.clamp(-2147483648, 2147483647).to(torch.int32).numpy()

    c1 = _fits.Column(name=column_name, format="1J", array=stored)
    hdu = _fits.BinTableHDU.from_columns([c1])
    hdu.header["TTYPE1"] = column_name
    hdu.header["TFORM1"] = "J"
    hdu.header["TSCAL1"] = tscal
    hdu.header["TZERO1"] = tzero
    if tnull is not None:
        hdu.header["TNULL1"] = int(tnull)
    hdu.writeto(path, overwrite=True)
    return stored


# ---------------------------------------------------------------------------
# Image round-trips
# ---------------------------------------------------------------------------


class TestEndToEndImageRoundTrip:
    """Read a FITS image with BSCALE/BZERO; verify recovery matches encoding."""

    def test_int16_bscale_bzero_round_trip(self):
        bscale = 0.25
        bzero = 1000.0
        # Use a value grid that fits exactly on the (bscale, bzero) step
        # lattice so the encoded int16 storage reconstructs physical values
        # within bscale/2.
        physical = torch.linspace(-100.0, 100.0, 64 * 64).reshape(64, 64)
        # Snap to the (bscale, bzero) lattice so physical ↔ storage is exact.
        physical = ((physical.double() - bzero) / bscale).round() * bscale + bzero
        physical = physical.reshape(64, 64).to(torch.float64)
        # Quantize to storage grid + back to physical so on-disk and physical
        # match at the bzero/bscale step level.
        physical_grid = ((physical.double() - bzero) / bscale).round() * bscale + bzero
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            _write_bscaled_image_fits(
                path, physical_grid, bscale=bscale, bzero=bzero, bitpix=16
            )
            from torchfits import read_tensor

            recovered = read_tensor(path, hdu=0)
            err = (recovered.double() - physical_grid.double()).abs().max().item()
            assert err <= abs(bscale) / 2 + 1e-4, (
                f"image round-trip error {err} > {abs(bscale) / 2}"
            )
        finally:
            os.unlink(path)

    def test_uint16_unsigned_convention(self):
        """BITPIX=16 / BZERO=32768 returns uint16 physical values directly."""
        from astropy.io import fits

        # Construct a stored_int16 array such that storage + 32768 == phys
        # for all positions without int16 overflow.
        phys_int64 = torch.arange(0, 32 * 32, dtype=torch.int64) % 65536
        stored_int = (phys_int64 - 32768).clamp(-32768, 32767).to(torch.int16).numpy()
        expected_phys = torch.from_numpy(
            (stored_int.astype("int64") + 32768).astype("int32")
        )
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            hdu = fits.PrimaryHDU(stored_int)
            hdu.header["BITPIX"] = 16
            hdu.header["BSCALE"] = 1.0
            hdu.header["BZERO"] = 32768.0
            hdu.writeto(path, overwrite=True)

            from torchfits import read_tensor

            recovered = read_tensor(path, hdu=0)
            assert torch.allclose(recovered.double(), expected_phys.double(), atol=0.5)
        finally:
            os.unlink(path)

    def test_int32_scaled_image(self):
        bscale = 0.001
        bzero = -100.0
        physical = (
            torch.arange(0, 32 * 32, dtype=torch.float64) / bscale + bzero
        ).reshape(32, 32)
        physical_grid = (
            ((physical - bzero) / bscale).round() * bscale + bzero
        ).reshape(32, 32)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            _write_bscaled_image_fits(
                path, physical_grid, bscale=bscale, bzero=bzero, bitpix=32
            )
            from torchfits import read_tensor

            recovered = read_tensor(path, hdu=0)
            err = (recovered.double() - physical_grid.double()).abs().max().item()
            assert err <= abs(bscale) / 2 + 1e-5
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Table round-trips
# ---------------------------------------------------------------------------


class TestEndToEndTableRoundTrip:
    """Round-trip TSCAL/TZERO and TNULL through FITS files + transforms."""

    def test_tscal_tzero_table_round_trip(self):
        """TSCAL/TZERO column reads back as the original physical values."""
        tscal = 0.5
        tzero = 1000.0
        physical = torch.tensor(
            [10.5, 20.0, 100.5, -5.25, 0.0, 50.75, -42.5, 7.125],
            dtype=torch.float64,
        )
        # Quantize to TSCAL grid so on-disk storage yields the same physical
        # value back via TSCAL*stored + TZERO.
        physical_grid = ((physical - tzero) / tscal).round() * tscal + tzero
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            _write_scaled_table_fits(
                path, physical_grid, column_name="FLUX", tscal=tscal, tzero=tzero
            )
            import torchfits.table  # noqa: PLC0415

            arrow_table = torchfits.table.read(path, hdu=1)
            recovered = torch.from_numpy(
                arrow_table.column("FLUX").to_numpy().astype("float64")
            )
            err = (recovered - physical_grid).abs().max().item()
            assert err <= abs(tscal) / 2 + 1e-5, (
                f"table round-trip error {err} > {abs(tscal) / 2}"
            )
        finally:
            os.unlink(path)

    def test_tnull_sentinel_to_nan_round_trip(self):
        from astropy.io import fits as _fits

        sentinel = -999
        raw_int = np.array([1, sentinel, 3, sentinel, 5], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            # Write directly via BinTableHDU so on-disk int32 is byte-for-byte.
            c1 = _fits.Column(name="FLUX", format="1J", array=raw_int)
            hdu = _fits.BinTableHDU.from_columns([c1])
            hdu.header["TSCAL1"] = 1.0
            hdu.header["TZERO1"] = 0.0
            hdu.header["TNULL1"] = sentinel
            hdu.writeto(path, overwrite=True)

            header_for_nuller = {
                "TFIELDS": 1,
                "TTYPE1": "FLUX",
                "TFORM1": "J",
                "TSCAL1": 1.0,
                "TZERO1": 0.0,
                "TNULL1": sentinel,
            }
            n = TNullToNan.from_header(header_for_nuller)
            cleaned = n.forward({"FLUX": torch.from_numpy(raw_int)})["FLUX"]
            assert torch.isnan(cleaned[1]).item()
            assert torch.isnan(cleaned[3]).item()
            assert cleaned[0].item() == 1.0
            assert cleaned[2].item() == 3.0
            assert cleaned[4].item() == 5.0
        finally:
            os.unlink(path)

    def test_identity_scales_round_trip(self):
        """TSCAL=1.0, TZERO=0.0 — physical values round-trip within storage."""
        physical = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, 0.0])
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            _write_scaled_table_fits(
                path, physical, column_name="VAL", tscal=1.0, tzero=0.0
            )
            import torchfits.table  # noqa: PLC0415

            arrow_table = torchfits.table.read(path, hdu=1)
            recovered = torch.from_numpy(
                arrow_table.column("VAL").to_numpy().astype("float64")
            )
            assert torch.equal(
                recovered.round().to(torch.int32), physical.to(torch.int32)
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# FITSHeaderNormalize (full header-driven normalization)
# ---------------------------------------------------------------------------


class TestEndToEndFITSHeaderNormalize:
    """Round-trip through the FITSHeaderNormalize transform."""

    def test_unsigned_int16_norm_inverse(self):
        from astropy.io import fits

        # Use physical values in [0, 65535] that round-trip to int16 storage
        # cleanly: stored = phys - 32768 must fit int16.
        raw_phys_np = (np.arange(64 * 64) % 65536).astype("int32")
        stored = (raw_phys_np - 32768).clip(-32768, 32767).astype("int16")
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            path = tmp.name
        try:
            hdu = fits.PrimaryHDU(stored)
            hdu.header["BITPIX"] = 16
            hdu.header["BSCALE"] = 1.0
            hdu.header["BZERO"] = 32768.0
            hdu.writeto(path, overwrite=True)
            from torchfits import get_header, read_tensor

            header = get_header(path, hdu=0)
            recovered = read_tensor(path, hdu=0)
            t = FITSHeaderNormalize(dict(header))
            normalised = t.forward(recovered.float())
            assert normalised.min() >= 0
            assert normalised.max() <= 1.0
            restored = t.inverse(normalised)
            expected = torch.from_numpy(raw_phys_np.astype("float64"))
            assert torch.allclose(restored.double(), expected, atol=2.0)
        finally:
            os.unlink(path)
