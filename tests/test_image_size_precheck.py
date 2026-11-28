"""An image header that declares more data than the file holds.

A corrupt header claiming a huge NAXIS product on a small file made every
image read allocate the declared size first: a bare ``MemoryError``
(``std::bad_alloc``), and cutouts failed with CFITSIO's "tried to move past
end of file". The declared data extent is now checked against the file
before allocating, like the table readers do.
"""

from __future__ import annotations

import pytest

import torchfits


def _card(key: str, value: str) -> str:
    return f"{key:<8}= {value:>20}".ljust(80)


def _write_lying_header(path, naxis1: int, naxis2: int, payload: int = 2880) -> None:
    header = "".join(
        [
            _card("SIMPLE", "T"),
            _card("BITPIX", "-32"),
            _card("NAXIS", "2"),
            _card("NAXIS1", str(naxis1)),
            _card("NAXIS2", str(naxis2)),
            "END".ljust(80),
        ]
    ).ljust(2880)
    path.write_bytes(header.encode("ascii") + b"\0" * payload)


READERS = {
    "read": lambda p: torchfits.read(p),
    "read_nommap": lambda p: torchfits.read(p, mmap=False),
    "read_tensor": lambda p: torchfits.read_tensor(p),
    "read_tensor_nommap": lambda p: torchfits.read_tensor(p, mmap=False),
    "subset": lambda p: torchfits.read_subset(p, 0, 0, 0, 10, 10),
    "open_slice": lambda p: torchfits.open(p)[0].data[0:10, 0:10],
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_huge_declared_image_on_small_file_raises_truncated(tmp_path, reader):
    path = tmp_path / "huge.fits"
    _write_lying_header(path, 200000, 200000)
    with pytest.raises(RuntimeError, match="FITS image is truncated") as info:
        READERS[reader](str(path))
    assert not isinstance(info.value, MemoryError)
    assert "5760" in str(info.value)
    torchfits.clear_file_cache()


def test_cutout_inside_the_present_bytes_of_a_truncated_image_raises(tmp_path):
    # Like the table readers, the whole declared data unit must be present:
    # a cutout of the first rows of a file cut short is refused too.
    path = tmp_path / "short.fits"
    _write_lying_header(path, 100, 100, payload=2880)  # needs 40000 bytes
    with pytest.raises(RuntimeError, match="FITS image is truncated"):
        torchfits.read_subset(str(path), 0, 0, 0, 5, 5)
    torchfits.clear_file_cache()
