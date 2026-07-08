"""
Core FITS data type support and checksum handling.
"""

from __future__ import annotations


class ChecksumVerifier:
    """CFITSIO-backed FITS checksum helpers.

    FITS checksums are defined over the HDU bytes, not just an already-materialized
    ndarray. Use the file/HDU methods for exact validation.
    """

    @staticmethod
    def _validate_hdu(hdu: int) -> int:
        if isinstance(hdu, bool) or not isinstance(hdu, int):
            raise TypeError("hdu must be a non-negative integer")
        if hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        return int(hdu)

    @staticmethod
    def write_hdu_checksums(path: str, hdu: int = 0) -> None:
        """Compute and write DATASUM/CHECKSUM for one HDU."""
        import torchfits._C as cpp

        cpp.write_hdu_checksums(str(path), ChecksumVerifier._validate_hdu(hdu))

    @staticmethod
    def verify_hdu_checksums(path: str, hdu: int = 0) -> dict[str, int | bool]:
        """Verify DATASUM/CHECKSUM for one HDU using CFITSIO semantics."""
        import torchfits._C as cpp

        datastatus, hdustatus = cpp.verify_hdu_checksums(
            str(path), ChecksumVerifier._validate_hdu(hdu)
        )
        data_i = int(datastatus)
        hdu_i = int(hdustatus)
        return {
            "datastatus": data_i,
            "hdustatus": hdu_i,
            "ok": data_i == 1 and hdu_i == 1,
        }
