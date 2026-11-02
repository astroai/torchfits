"""int8 table columns use the FITS signed-byte convention (TFORM B, TZERO=-128).

They used to be written as plain 1B with no TZERO, so -5 read back as uint8 251;
astropy-style signed-byte columns read back as float64.
"""

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as C
import torchfits.table as table

VALUES = [-5, 0, 127, -128]


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


@pytest.mark.parametrize("kind", ["torch", "numpy"])
def test_int8_column_roundtrip(tmp_path, kind):
    path = str(tmp_path / "s.fits")
    arr = np.array(VALUES, dtype=np.int8)
    col = torch.from_numpy(arr) if kind == "torch" else arr
    torchfits.write(path, {"S": col})

    cards = _raw_cards(path, 1)
    assert cards["TFORM1"].lstrip("1") == "B"
    assert float(cards["TZERO1"]) == -128.0

    got = torchfits.read(path, hdu=1)["S"]
    assert got.dtype == torch.int8
    assert got.tolist() == VALUES
    via_mmap = C.read_fits_table_rows(path, 1, ["S"], 1, -1, True)["S"]
    assert via_mmap.dtype == torch.int8 and via_mmap.tolist() == VALUES
    arrow = table.read(path, hdu=1)
    assert str(arrow.schema.field("S").type) == "int8"
    assert arrow["S"].to_pylist() == VALUES
    assert table.read(path, hdu=1, where="S < 0")["S"].to_pylist() == [-5, -128]


def test_int8_column_survives_rewrite_and_mutation(tmp_path):
    path = str(tmp_path / "s.fits")
    torchfits.write(path, {"S": torch.tensor(VALUES, dtype=torch.int8)})
    torchfits.insert_hdu(path, torch.ones(2, 2), index=2)
    cards = _raw_cards(path, 1)
    assert cards["TFORM1"].lstrip("1") == "B"
    assert float(cards["TZERO1"]) == -128.0

    table.append_rows(path, {"S": torch.tensor([-1, 1], dtype=torch.int8)})
    table.update_rows(
        path, {"S": torch.tensor([-100], dtype=torch.int8)}, row_slice=slice(0, 1)
    )
    got = torchfits.read(path, hdu=1)["S"]
    assert got.dtype == torch.int8
    assert got.tolist() == [-100, 0, 127, -128, -1, 1]
