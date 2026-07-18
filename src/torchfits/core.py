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
    def verify_hdu_checksums(path: str, hdu: int = 0) -> dict[str, int | bool | str]:
        """Verify DATASUM/CHECKSUM for one HDU using CFITSIO semantics.

        CFITSIO ``ffvcks`` status codes:
        - ``0`` — checksum keywords absent (nothing to verify)
        - ``1`` — checksum present and correct
        - ``-1`` — checksum present but incorrect (corrupt)

        Returns a dict with ``datastatus``, ``hdustatus``, ``ok``, and
        ``status`` (a human-readable string: ``"ok"``, ``"no_checksums"``,
        or ``"fail"``).
        """
        import torchfits._C as cpp

        datastatus, hdustatus = cpp.verify_hdu_checksums(
            str(path), ChecksumVerifier._validate_hdu(hdu)
        )
        data_i = int(datastatus)
        hdu_i = int(hdustatus)

        # Determine human-readable status from the two sub-statuses.
        if data_i == 0 and hdu_i == 0:
            status_str = "no_checksums"
            ok = True  # Not corrupt — just no keywords to verify.
        elif data_i == 1 and hdu_i == 1:
            status_str = "ok"
            ok = True
        else:
            status_str = "fail"
            ok = False

        return {
            "datastatus": data_i,
            "hdustatus": hdu_i,
            "ok": ok,
            "status": status_str,
        }
