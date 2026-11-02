"""One error type per lookup failure, on every public route.

Before, the same mistake raised a different type depending on the route: an
unknown EXTNAME was RuntimeError from ``read``, ValueError from
``read_header`` and KeyError from ``open()[...]`` / ``table.read``; an index
past the end was OSError, RuntimeError or IndexError.

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
import torchfits.table as tft
from torchfits import HDUNotFoundError
from torchfits.data import FitsImageDataset
from torchfits.hdu import TableHDURef


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
    "open()[]": lambda p, h: torchfits.open(p)[h],
    "replace_hdu": lambda p, h: torchfits.replace_hdu(p, h, torch.zeros(2, 2)),
    "delete_hdu": lambda p, h: torchfits.delete_hdu(p, h),
    "table.read": lambda p, h: tft.read(p, hdu=h),
    "table.read_torch": lambda p, h: tft.read_torch(p, hdu=h),
    "table.scan": lambda p, h: list(tft.scan(p, hdu=h)),
    "table.schema": lambda p, h: tft.schema(p, hdu=h),
    "table.reader": lambda p, h: tft.reader(p, hdu=h),
    "table.append_rows": lambda p, h: tft.append_rows(p, {"x": torch.arange(1)}, hdu=h),
    "FitsImageDataset": lambda p, h: FitsImageDataset([p], hdu=h)[0],
}
INDEX_ONLY_ROUTES = {
    # A list read takes one integer HDU; a name there is an argument error.
    "read[list]": lambda p, h: torchfits.read([p, p], hdu=h),
    "insert_hdu": lambda p, h: torchfits.insert_hdu(p, torch.zeros(2, 2), index=h),
    "TableHDURef.read": lambda p, h: TableHDURef(source_path=p, source_hdu=h).read(),
}
# open()[] follows Python sequence rules: -1 is the last HDU, not an error.
NEGATIVE_ROUTES = {k: v for k, v in {**ROUTES, **INDEX_ONLY_ROUTES}.items()}
NEGATIVE_ROUTES.pop("open()[]")
# insert_hdu(index=len) appends, so "out of range" starts one further.
OUT_OF_RANGE = {"insert_hdu": 3}


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


def test_hdulist_negative_index_still_counts_from_the_end(mef):
    with torchfits.open(mef) as hdul:
        assert hdul[-1].header["EXTNAME"] == "SCI"
        with pytest.raises(HDUNotFoundError, match="HDU index -3 out of range"):
            hdul[-3]


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
