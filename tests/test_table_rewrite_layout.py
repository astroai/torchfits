"""Rewrites keep a source TFORMn only for the same column with the same cells.

The source TFORM used to be matched to new columns by position and type letter
only, so a 6J column replaced by a scalar int32, a 10A column replaced by
int64, or a reordered table were relabeled into corrupt or scrambled HDUs,
stale TDIMn survived, and an 'rX' bit column was corrupted by any edit.
"""

import numpy as np
import torch

import torchfits
import torchfits._C as C
from torchfits import HDUList, TableHDU


def _raw_cards(path, hdu):
    data = open(path, "rb").read()
    off = 0
    for idx in range(hdu + 1):
        cards = []
        while True:
            block = data[off : off + 2880]
            off += 2880
            chunk = [block[i : i + 80].decode("ascii") for i in range(0, 2880, 80)]
            cards.extend(chunk)
            if any(c.startswith("END     ") for c in chunk):
                break
        kv = {
            c[:8].strip(): c[10:].split("/")[0].strip().strip("'").strip()
            for c in cards
            if c[8:10] == "= "
        }
        if idx == hdu:
            return kv
        naxis = int(kv["NAXIS"])
        n = 0
        if naxis:
            n = 1
            for k in range(1, naxis + 1):
                n *= int(kv[f"NAXIS{k}"])
            n = (n + int(kv.get("PCOUNT", 0))) * abs(int(kv["BITPIX"])) // 8
        off += (n + 2879) // 2880 * 2880
    raise IndexError(hdu)


def _with_table(path, columns, header=None):
    torchfits.write(path, [torch.zeros(1, 1)])
    torchfits.insert_hdu(path, columns, index=1, header=header or {"EXTNAME": "CAT"})


def test_replace_vector_column_with_scalar(tmp_path):
    path = str(tmp_path / "a.fits")
    _with_table(path, {"a": torch.arange(12, dtype=torch.int32).reshape(2, 6)})
    torchfits.replace_hdu(path, 1, {"a": torch.tensor([5, 6], dtype=torch.int32)})
    assert _raw_cards(path, 1)["TFORM1"].lstrip("1") == "J"
    assert torchfits.read(path, hdu=1)["a"].tolist() == [5, 6]


def test_replace_string_column_with_int64(tmp_path):
    path = str(tmp_path / "b.fits")
    _with_table(path, {"name": ["abcdefghij", "k"]})
    torchfits.replace_hdu(path, "CAT", {"x": torch.tensor([5, 6])})
    assert _raw_cards(path, 1)["TFORM1"].lstrip("1") == "K"
    assert torchfits.read(path, hdu=1)["x"].tolist() == [5, 6]


def test_hdulist_with_stale_header(tmp_path):
    src = str(tmp_path / "c.fits")
    torchfits.write(src, {"a": torch.arange(12, dtype=torch.int32).reshape(4, 3)})
    old = torchfits.read_header(src, 1)
    out = str(tmp_path / "c2.fits")
    new = TableHDU({"a": torch.arange(4, dtype=torch.int32)}, header=old)
    torchfits.write(out, HDUList([TableHDU({"z": torch.zeros(4)}), new]))
    assert _raw_cards(out, 2)["TFORM1"].lstrip("1") == "J"
    assert torchfits.read(out, hdu=2)["a"].tolist() == [0, 1, 2, 3]


def test_reordered_columns_drop_stale_tdim(tmp_path):
    path = str(tmp_path / "d.fits")
    a = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    b = torch.tensor([1.0, 2.0])
    _with_table(path, {"a": a, "b": b}, header={"EXTNAME": "CAT", "TDIM1": "(3,2)"})
    torchfits.replace_hdu(path, 1, {"b": b, "a": a})
    cards = _raw_cards(path, 1)
    assert cards["TFORM1"].lstrip("1") == "E" and cards["TFORM2"] == "6E"
    assert "TDIM1" not in cards
    got = torchfits.read(path, hdu=1)
    assert tuple(got["b"].shape) == (2,) and torch.equal(got["a"], a)


def test_exotic_spellings_survive_unrelated_edit(tmp_path):
    path = str(tmp_path / "e.fits")
    C.write_fits_table(
        path,
        {
            "s": ["abc", "de"],
            "bits": torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool),
            "v": [np.array([1, 2, 3], dtype=np.int32), np.array([4], dtype=np.int32)],
            "j": torch.tensor([1, 2], dtype=torch.int32),
        },
        {"EXTNAME": "T"},
        False,
        {
            "s": {"format": "5A"},
            "bits": {"format": "3X"},
            "v": {},
            "j": {"format": "J"},
        },
        "binary",
    )
    before = torchfits.read(path, hdu=1)
    torchfits.insert_hdu(path, torch.ones(2, 2), index=2)
    cards = _raw_cards(path, 1)
    assert [cards[f"TFORM{i}"] for i in range(1, 5)] == ["5A", "3X", "1PJ(3)", "J"]
    assert cards["NAXIS1"] == str(5 + 1 + 8 + 4)
    after = torchfits.read(path, hdu=1)
    for key in ("s", "bits", "j"):
        assert torch.equal(before[key], after[key])
    assert all(torch.equal(x, y) for x, y in zip(before["v"], after["v"]))


def test_annotations_without_ttype_are_kept(tmp_path):
    path = str(tmp_path / "f.fits")
    hdu = TableHDU({"a": torch.zeros(2, 6)}, header={"TUNIT1": "m", "TDIM1": "(3,2)"})
    torchfits.write(path, HDUList([TableHDU({"z": torch.zeros(2)}), hdu]))
    cards = _raw_cards(path, 2)
    assert cards["TUNIT1"] == "m" and cards["TDIM1"] == "(3,2)"
