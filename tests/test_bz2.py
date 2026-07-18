import pytest
import torchfits


def test_bz2_error():
    with pytest.raises(ValueError, match="CFITSIO does not support .bz2"):
        torchfits.read("test.fits.bz2")
    with pytest.raises(ValueError, match="CFITSIO does not support .bz2"):
        torchfits.read_tensor("test.fits.bz2")
    with pytest.raises(ValueError, match="CFITSIO does not support .bz2"):
        torchfits.read_subset("test.fits.bz2", hdu=0, x1=0, y1=0, x2=1, y2=1)
