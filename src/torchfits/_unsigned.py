"""FITS unsigned-integer convention detection and conversion.

FITS has no native unsigned integer types.  Conceptually unsigned values
(uint16, uint32) or signed bytes (int8) are stored as signed integers with
a BZERO/TZERO offset:

    physical_value = BSCALE * storage_value + BZERO

where the *unsigned* convention is :math:`\\text{BSCALE}=1.0` and

* :math:`\\text{BITPIX}=16,\\;\\text{BZERO}=32768`          → uint16
* :math:`\\text{BITPIX}=32,\\;\\text{BZERO}=2147483648`     → uint32
* :math:`\\text{BITPIX}=8,\\;\\text{BZERO}=-128`            → int8  (signed byte)

For binary tables the same convention uses TSCAL=1.0 and TZERO with
TFORM code ``I`` (16-bit) or ``J`` (32-bit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch


@dataclass(frozen=True)
class UnsignedConvention:
    """A known FITS unsigned-integer or signed-byte pseudo-type.

    Attributes
    ----------
    bscale : float
        BSCALE value.
    bzero : float
        BZERO offset.
    bitpix : int
        Corresponding BITPIX value.
    storage_dtype : torch.dtype
        On-disk FITS dtype (signed).
    target_dtype : torch.dtype
        Logical dtype after applying the offset.
    offset : int
        Integer offset to add when converting storage → logical.
    tform_code : str | None
        Binary-table TFORM code (``"I"``, ``"J"``) or ``None`` for images.
    """

    bscale: float
    bzero: float
    bitpix: int
    storage_dtype: torch.dtype
    target_dtype: torch.dtype
    offset: int
    tform_code: str | None = None

    # ------------------------------------------------------------------
    # Pre-defined known conventions
    # ------------------------------------------------------------------

    UINT16: ClassVar[UnsignedConvention]
    UINT32: ClassVar[UnsignedConvention]
    SBYTE: ClassVar[UnsignedConvention]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        bscale: float = 1.0,
        bzero: float = 0.0,
        bitpix: int | None = None,
        tform_code: str | None = None,
    ) -> UnsignedConvention | None:
        """Detect unsigned convention from image or table-column parameters.

        Parameters
        ----------
        bscale : float
            BSCALE / TSCAL value.
        bzero : float
            BZERO / TZERO value.
        bitpix : int | None
            Image BITPIX.  Required for image-format detection; ignored
            when *tform_code* is provided.
        tform_code : str | None
            Binary-table TFORM code (``"I"`` or ``"J"``).  When set the
            detection uses table-column rules instead of image BITPIX rules.

        Returns
        -------
        UnsignedConvention or None
            The matching convention, or *None* if none is recognised.
        """
        if bscale != 1.0:
            return None
        if tform_code is not None:
            if tform_code.upper() == "I" and bzero == 32768.0:
                return UnsignedConvention.UINT16
            if tform_code.upper() == "J" and bzero == 2147483648.0:
                return UnsignedConvention.UINT32
            return None
        # Image-format detection
        if bitpix == 16 and bzero == 32768.0:
            return UnsignedConvention.UINT16
        if bitpix == 32 and bzero == 2147483648.0:
            return UnsignedConvention.UINT32
        if bitpix == 8 and bzero == -128.0:
            return UnsignedConvention.SBYTE
        return None

    @staticmethod
    def detect_from_header(header: dict[str, Any]) -> UnsignedConvention | None:
        """Detect unsigned convention from an image-HDU header.

        Inspects ``BITPIX``, ``BSCALE``, ``BZERO``.
        """
        bitpix = int(header.get("BITPIX", -32))
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
        return UnsignedConvention.detect(bscale=bscale, bzero=bzero, bitpix=bitpix)

    @staticmethod
    def conventions() -> list[UnsignedConvention]:
        """Return all known unsigned conventions."""
        return [
            UnsignedConvention.UINT16,
            UnsignedConvention.UINT32,
            UnsignedConvention.SBYTE,
        ]

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_unsigned(
        self, data: torch.Tensor, *, device: str | None = None
    ) -> torch.Tensor:
        """Convert FITS storage tensor to logical unsigned dtype.

        The conversion adds *offset* and widens the dtype to avoid
        overflow where necessary.
        """
        if device is not None and device != "cpu":
            data = data.to(device=device)
        if self.target_dtype == torch.uint16:
            return (data.to(torch.int32) + self.offset).to(torch.uint16)
        if self.target_dtype == torch.uint32:
            return (data.to(torch.int64) + self.offset).to(torch.uint32)
        if self.target_dtype == torch.int8:
            return (data.to(torch.int16) + self.offset).to(torch.int8)
        return data.to(torch.int64).add_(self.offset).to(dtype=self.target_dtype)

    def to_storage(self, data: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Convert logical unsigned tensor to FITS storage (signed) dtype.

        Returns the converted tensor and a dict of header cards to set
        (``BSCALE``, ``BZERO``) so the file round-trips correctly.
        """
        if self.target_dtype == torch.uint16:
            raw = (data.to(torch.int32) - self.offset).to(torch.int16)
            return raw, {"BSCALE": 1.0, "BZERO": float(self.offset)}
        if self.target_dtype == torch.uint32:
            raw = (data.to(torch.int64) - self.offset).to(torch.int32)
            return raw, {"BSCALE": 1.0, "BZERO": float(self.offset)}
        if self.target_dtype == torch.int8:
            raw = (data.to(torch.int16) - self.offset).to(torch.uint8)
            return raw, {"BSCALE": 1.0, "BZERO": float(self.offset)}
        msg = f"UnsignedConvention has no storage mapping for {self.target_dtype}"
        raise TypeError(msg)


# -----------------------------------------------------------------------
# Fill pre-defined class-var conventions
# -----------------------------------------------------------------------
UnsignedConvention.UINT16 = UnsignedConvention(
    bscale=1.0,
    bzero=32768.0,
    bitpix=16,
    storage_dtype=torch.int16,
    target_dtype=torch.uint16,
    offset=32768,
    tform_code="I",
)
UnsignedConvention.UINT32 = UnsignedConvention(
    bscale=1.0,
    bzero=2147483648.0,
    bitpix=32,
    storage_dtype=torch.int32,
    target_dtype=torch.uint32,
    offset=2147483648,
    tform_code="J",
)
UnsignedConvention.SBYTE = UnsignedConvention(
    bscale=1.0,
    bzero=-128.0,
    bitpix=8,
    storage_dtype=torch.uint8,
    target_dtype=torch.int8,
    offset=-128,
    tform_code=None,
)
