"""insert/replace/delete_hdu accept files with bytes after the last HDU.

CFITSIO ignores anything after the last HDU that does not start with an
``XTENSION`` card; the byte-level splicer used to fail on it ("truncated FITS
header" or an HDU-count mismatch). The trailing bytes are kept verbatim at the
end of the edited file.
"""

from __future__ import annotations

import pytest
import torch

import torchfits
from torchfits._io_engine._hdu_splice import scan_hdus

TRAILERS = {
    "garbage80": b"not a FITS card, just junk".ljust(80, b"#"),
    "zero_block": b"\0" * 2880,
    "space_block": b" " * 2880,
}


def _make(path, trailer):
    hdus = [
        torchfits.TensorHDU(torch.empty(0)),
        torchfits.TensorHDU(torch.full((3, 3), 1.0), {"EXTNAME": "A"}),
        torchfits.TensorHDU(torch.full((2, 2), 2.0), {"EXTNAME": "B"}),
    ]
    torchfits.write(str(path), torchfits.HDUList(hdus), overwrite=True)
    body = path.read_bytes()
    path.write_bytes(body + trailer)
    return len(body)


@pytest.mark.parametrize("kind", sorted(TRAILERS))
def test_scan_stops_at_trailing_bytes(tmp_path, kind):
    path = tmp_path / "t.fits"
    size = _make(path, TRAILERS[kind])
    spans = scan_hdus(str(path))
    assert len(spans) == 3
    assert spans[-1][2] == size


@pytest.mark.parametrize("kind", sorted(TRAILERS))
@pytest.mark.parametrize("op", ["insert_mid", "insert_end", "replace", "delete"])
def test_edit_keeps_trailing_bytes(tmp_path, kind, op):
    trailer = TRAILERS[kind]
    path = tmp_path / "t.fits"
    _make(path, trailer)
    p = str(path)
    if op == "insert_mid":
        torchfits.insert_hdu(p, torch.full((2,), 5.0), 1, header={"EXTNAME": "NEW"})
        names = ["", "NEW", "A", "B"]
    elif op == "insert_end":
        torchfits.insert_hdu(p, torch.full((2,), 5.0), 3, header={"EXTNAME": "NEW"})
        names = ["", "A", "B", "NEW"]
    elif op == "replace":
        torchfits.replace_hdu(p, "A", torch.full((4,), 7.0))
        names = ["", "A", "B"]
    else:
        torchfits.delete_hdu(p, "A")
        names = ["", "B"]
    raw = path.read_bytes()
    assert raw.endswith(trailer)
    assert (len(raw) - len(trailer)) % 2880 == 0
    with torchfits.open(p) as hdul:
        assert len(hdul) == len(names)
    for i, name in enumerate(names[1:], start=1):
        assert str(torchfits.read_header(p, hdu=i).get("EXTNAME", "")).strip() == name
    assert torch.equal(torchfits.read(p, hdu="B"), torch.full((2, 2), 2.0))
