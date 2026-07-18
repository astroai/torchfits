import torch
import torchfits


def test_read_subset_3d(tmp_path):
    path = str(tmp_path / "test_3d.fits")
    data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    # FITS shape will be NAXIS1=4, NAXIS2=3, NAXIS3=2
    torchfits.write(path, data)

    # Read subset: x: 1 to 3, y: 0 to 2
    # Should return shape (2, 2, 2)
    subset = torchfits.read_subset(path, hdu=0, x1=1, y1=0, x2=3, y2=2)
    assert subset.shape == (2, 2, 2)

    # data[z, y, x]
    # subset should be data[:, 0:2, 1:3]
    expected = data[:, 0:2, 1:3]
    torch.testing.assert_close(subset, expected)


def test_subset_reader_3d(tmp_path):
    path = str(tmp_path / "test_3d.fits")
    data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    torchfits.write(path, data)

    with torchfits.open_subset_reader(path, hdu=0) as reader:
        subset = reader(1, 0, 3, 2)
        assert subset.shape == (2, 2, 2)
        expected = data[:, 0:2, 1:3]
        torch.testing.assert_close(subset, expected)


def test_read_subset_zero_size_box_keeps_valid_dim(tmp_path):
    """A degenerate box on one axis must not collapse the other axis to 0.

    Regression: read_subset used to always report shape (..., 0, 0) when
    either width or height collapsed to zero, discarding the size of the
    still-valid axis (e.g. a zero-width, full-height cutout).
    """
    path = str(tmp_path / "test_2d.fits")
    data = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    torchfits.write(path, data)

    zero_width = torchfits.read_subset(path, hdu=0, x1=5, y1=0, x2=5, y2=10)
    assert zero_width.shape == (10, 0)

    zero_height = torchfits.read_subset(path, hdu=0, x1=0, y1=5, x2=10, y2=5)
    assert zero_height.shape == (0, 10)

    both_zero = torchfits.read_subset(path, hdu=0, x1=5, y1=5, x2=5, y2=5)
    assert both_zero.shape == (0, 0)

    with torchfits.open_subset_reader(path, hdu=0) as reader:
        assert reader(5, 0, 5, 10).shape == (10, 0)
        assert reader(0, 5, 10, 5).shape == (0, 10)
