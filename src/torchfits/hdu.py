"""
Core HDU classes for torchfits.

This module re-exports from ``_hdu/`` sub-modules:
- HDUList: Container for multiple HDUs
- TensorHDU: Image/cube data with lazy loading
- TableHDU: Tabular data with torch-frame integration
- TableHDURef: Lazy file-backed table handle
- Header: FITS header management
- Card: FITS header card
"""

from __future__ import annotations

# -- card / header -----------------------------------------------------------------

from ._hdu.card import Card as Card

from ._hdu.header import Header as Header

# -- image / cube HDU --------------------------------------------------------------

from ._hdu.tensor_hdu import TensorHDU as TensorHDU

# -- table HDU ---------------------------------------------------------------------

from ._hdu.table_hdu import TableHDU as TableHDU

from ._hdu.table_hdu_ref import TableHDURef as TableHDURef

# -- HDU list ----------------------------------------------------------------------

from ._hdu.hdu_list import HDUList as HDUList
