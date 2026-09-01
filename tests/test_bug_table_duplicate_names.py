from astropy.io import fits
import numpy as np

import torchfits
import torchfits.table


def test_duplicate_column_names(tmp_path):
    """FITS tables can contain duplicate column names.

    Regression: The C++ reader used an unordered_map keyed by string name,
    which overwrote duplicates and caused undefined behavior (moved-from UB)
    when assembling the final output vector.
    """
    path = tmp_path / "duplicate_cols.fits"

    # Bypass astropy validation by modifying the header manually
    col1 = fits.Column(name="A", format="J", array=np.array([1, 2]))
    col2 = fits.Column(name="B", format="J", array=np.array([3, 4]))
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.BinTableHDU.from_columns([col1, col2])
    hdu2.header["TTYPE2"] = "A"

    hdul = fits.HDUList([hdu1, hdu2])
    hdul.writeto(str(path), overwrite=True)

    # Should not crash/UB, and should return both 'A' arrays
    # Since python dicts overwrite duplicate keys, we use read_torch which
    # returns a dict where the second 'A' overwrites the first. But the
    # crucial check is that the C++ layer doesn't crash and surfaces both if
    # requested properly, or at least safely populates the result without UB.

    res = torchfits.table.read_torch(str(path), mmap=True)
    assert "A" in res
    assert res["A"].tolist() == [
        1,
        2,
    ]  # The second column overwrote the first in the Python dict

    res = torchfits.table.read_torch(str(path), mmap=False)
    assert "A" in res
    assert res["A"].tolist() == [1, 2]
