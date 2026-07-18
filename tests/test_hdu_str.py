import torch
import torchfits


def test_read_tensor_string_hdu(tmp_path):
    path = str(tmp_path / "test.fits")
    data = torch.ones((10, 10))
    torchfits.write(path, data, header={"EXTNAME": "MYDATA"})

    tensor = torchfits.read_tensor(path, hdu="MYDATA")
    assert tensor.shape == (10, 10)

    subset = torchfits.read_subset(path, hdu="MYDATA", x1=0, y1=0, x2=5, y2=5)
    assert subset.shape == (5, 5)
