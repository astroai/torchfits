"""One error type per lookup failure, on every public route.

Before, the same mistake raised a different type depending on the route: an
unknown EXTNAME was RuntimeError from ``read`` and ValueError from
``read_header``; an index past the end was OSError or RuntimeError.

Now an absent HDU (unknown EXTNAME, index negative or past the end) raises
``torchfits.HDUNotFoundError`` -- a KeyError, IndexError and ValueError at
once, so every earlier ``except`` keeps working.
"""

from __future__ import annotations

import os
import pickle

import pytest
import torch

import torchfits
from torchfits import HDUNotFoundError
from torchfits.data import FitsImageDataset


@pytest.fixture()
def mef(tmp_path):
    path = str(tmp_path / "mef.fits")
    torchfits.write(
        path,
        [
            torch.zeros(4, 4),
            {"data": torch.ones(4, 4), "header": {"EXTNAME": "SCI"}},
        ],
    )
    return path


def _reader_read(p, h):
    with torchfits.open_subset_reader(p, h) as reader:
        return reader(0, 0, 1, 1)


# name -> callable(path, hdu). Every route that takes an HDU by index or name.
ROUTES = {
    "read": lambda p, h: torchfits.read(p, hdu=h),
    "read_tensor": lambda p, h: torchfits.read_tensor(p, hdu=h),
    "read_header": lambda p, h: torchfits.read_header(p, h),
    "read_shape": lambda p, h: torchfits.read_shape(p, h),
    "read_nrows": lambda p, h: torchfits.read_nrows(p, h),
    "read_keys": lambda p, h: torchfits.read_keys(p, ["NAXIS"], hdu=h),
    "read_colnames": lambda p, h: torchfits.read_colnames(p, h),
    "read_hdu_type": lambda p, h: torchfits.read_hdu_type(p, h),
    "read_extname": lambda p, h: torchfits.read_extname(p, h),
    "read_table_info": lambda p, h: torchfits.read_table_info(p, h),
    "read_subset": lambda p, h: torchfits.read_subset(p, h, 0, 0, 1, 1),
    "open_subset_reader": _reader_read,
    "open_table_reader": lambda p, h: torchfits.open_table_reader(p, h),
    "read_batch": lambda p, h: torchfits.read_batch([p], hdu=h),
    "read_batch[strict]": lambda p, h: torchfits.read_batch([p], hdu=h, strict=True),
    "read_hdus": lambda p, h: torchfits.read_hdus(p, [h]),
    "verify_checksums": lambda p, h: torchfits.verify_checksums(p, h),
    "write_checksums": lambda p, h: torchfits.write_checksums(p, h),
    "FitsImageDataset": lambda p, h: FitsImageDataset([p], hdu=h)[0],
}
INDEX_ONLY_ROUTES = {
    # A list read takes one integer HDU; a name there is an argument error.
    "read[list]": lambda p, h: torchfits.read([p, p], hdu=h),
}
NEGATIVE_ROUTES = {**ROUTES, **INDEX_ONLY_ROUTES}
OUT_OF_RANGE: dict[str, int] = {}


def _assert_hdu_not_found(exc_info, path, needle):
    exc = exc_info.value
    assert type(exc) is HDUNotFoundError, repr(exc)
    assert isinstance(exc, KeyError)
    assert isinstance(exc, IndexError)
    assert isinstance(exc, ValueError)
    message = str(exc)
    assert os.path.basename(path) in message, message
    assert needle in message, message
    assert not message.startswith(('"', "'")), "message is repr-quoted"


@pytest.mark.parametrize("hdu", ["NOPE", "SCI,2"])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_unknown_extname(mef, route, hdu):
    before = open(mef, "rb").read()
    with pytest.raises(HDUNotFoundError) as exc_info:
        ROUTES[route](mef, hdu)
    _assert_hdu_not_found(exc_info, mef, repr(hdu))
    assert open(mef, "rb").read() == before


@pytest.mark.parametrize("route", sorted({**ROUTES, **INDEX_ONLY_ROUTES}))
def test_index_past_the_end(mef, route):
    index = OUT_OF_RANGE.get(route, 7)
    before = open(mef, "rb").read()
    with pytest.raises(HDUNotFoundError) as exc_info:
        {**ROUTES, **INDEX_ONLY_ROUTES}[route](mef, index)
    _assert_hdu_not_found(exc_info, mef, f"HDU index {index} out of range")
    if route != "read[list]":
        assert "(2 HDUs)" in str(exc_info.value)
    assert open(mef, "rb").read() == before


@pytest.mark.parametrize("route", sorted(NEGATIVE_ROUTES))
def test_negative_index(mef, route):
    with pytest.raises(HDUNotFoundError) as exc_info:
        NEGATIVE_ROUTES[route](mef, -1)
    _assert_hdu_not_found(exc_info, mef, "non-negative")


@pytest.mark.parametrize("builtin", [KeyError, IndexError, ValueError, LookupError])
def test_earlier_except_clauses_still_catch_it(mef, builtin):
    try:
        torchfits.read_header(mef, "NOPE")
    except builtin as exc:
        assert isinstance(exc, HDUNotFoundError)
    else:  # pragma: no cover - the call must raise
        pytest.fail("read_header did not raise")


def test_attributes_and_pickling(mef):
    with pytest.raises(HDUNotFoundError) as exc_info:
        torchfits.read(mef, hdu="NOPE")
    exc = exc_info.value
    assert (exc.hdu, exc.path, exc.n_hdus) == ("NOPE", mef, 2)
    clone = pickle.loads(pickle.dumps(exc))
    assert type(clone) is HDUNotFoundError
    assert (str(clone), clone.hdu, clone.path, clone.n_hdus) == (
        str(exc),
        "NOPE",
        mef,
        2,
    )


def test_import_does_not_load_torch():
    import subprocess
    import sys

    code = (
        "import sys, torchfits; torchfits.HDUNotFoundError; "
        "assert 'torch' not in sys.modules; assert 'torchfits._C' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
