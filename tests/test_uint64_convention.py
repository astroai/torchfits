"""uint64 images and table columns use the FITS 4.0 BZERO/TZERO = 2**63 convention.

Reads used to go through float64, losing the low bits (5 read back as 0.0),
and writes raised. Values must now round-trip exactly on every read path.
"""

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as C
import torchfits.table as table

EDGES = [0, 1, 5, 2**53 + 1, 2**63 - 1, 2**63, 2**63 + 7, 2**64 - 1]


def _u64(values):
    return torch.from_numpy(np.array(values, dtype=np.uint64))


@pytest.mark.parametrize("use_mmap", [True, False])
def test_image_roundtrip(tmp_path, use_mmap):
    path = str(tmp_path / "img.fits")
    data = _u64(EDGES).reshape(2, 4)
    torchfits.write(path, data, overwrite=True)

    header = torchfits.read_header(path, 0)
    assert int(header["BITPIX"]) == 64
    assert float(header["BZERO"]) == 2.0**63

    got = torchfits.read(path, mmap=use_mmap)
    assert got.dtype == torch.uint64
    assert torch.equal(got, data)

    with torchfits.open(path) as hdul:
        assert hdul[0].data.dtype == torch.uint64

    cut = C.SubsetReader(path, 0).read(1, 0, 3, 2)
    assert cut.dtype == torch.uint64
    assert torch.equal(cut, data[:, 1:3])


@pytest.mark.parametrize("kind", ["torch", "numpy"])
def test_table_roundtrip(tmp_path, kind):
    path = str(tmp_path / "tab.fits")
    values = np.array(EDGES, dtype=np.uint64)
    col = torch.from_numpy(values) if kind == "torch" else values
    torchfits.write(path, {"U": col}, overwrite=True)

    got = torchfits.read(path, hdu=1)["U"]
    assert got.dtype == torch.uint64
    assert got.numpy().tolist() == values.tolist()

    via_mmap = C.read_fits_table_rows(path, 1, ["U"], 1, -1, True)["U"]
    assert via_mmap.numpy().tolist() == values.tolist()

    arrow = table.read(path, hdu=1)
    assert str(arrow.schema.field("U").type) == "uint64"
    assert arrow["U"].to_pylist() == values.tolist()


@pytest.mark.parametrize(
    ("where", "pick"),
    [
        ("U >= 9223372036854775808", lambda v: v >= 2**63),
        ("U < 9223372036854775808", lambda v: v < 2**63),
        ("U == 5", lambda v: v == 5),
        ("U != 5", lambda v: v != 5),
        ("U > -1", lambda v: True),
        ("U <= 2.5", lambda v: v <= 2.5),
        ("U > 18446744073709551614", lambda v: v > 2**64 - 2),
    ],
)
def test_table_where(tmp_path, where, pick):
    path = str(tmp_path / "tab.fits")
    torchfits.write(path, {"U": np.array(EDGES, dtype=np.uint64)}, overwrite=True)
    expected = [v for v in EDGES if pick(v)]
    for kwargs in ({}, {"backend": "torch"}):
        got = table.read(path, hdu=1, where=where, **kwargs)["U"]
        got = got.to_pylist() if hasattr(got, "to_pylist") else got.numpy().tolist()
        assert got == expected, (where, kwargs)


def _raw_cards(path, hdu):
    """Header cards of HDU *hdu* straight from the 2880-byte blocks."""
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
            c[:8].strip(): c[10:].split("/")[0].strip()
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


def _write_direct(path, col):
    torchfits.write(path, {"U": col})


def _write_compressed(path, col):
    torchfits.write(path, {"U": col}, compress=True)


def _write_hdulist(path, col):
    torchfits.write(path, torchfits.HDUList([torchfits.TableHDU({"U": col})]))


def _insert_into(path, col):
    torchfits.write(path, [torch.zeros(1, 1)])
    torchfits.insert_hdu(path, {"U": col}, index=1)


def _edit_other_hdu(path, col):
    torchfits.write(path, {"U": col})
    torchfits.insert_hdu(path, torch.ones(2, 2), index=2)
    torchfits.replace_hdu(path, 2, torch.zeros(2, 2))
    torchfits.delete_hdu(path, 2)


def _write_schema(path, col):
    table.write(
        path,
        {"U": col.numpy(), "X": np.arange(len(EDGES))},
        schema={"X": {"unit": "m"}},
    )


def _replace_column(path, col):
    table.write(path, {"U": np.zeros(len(EDGES), dtype=np.int64)})
    table.replace_column(path, "U", col.numpy())


def _then_edit_rows(path, col):
    torchfits.write(path, {"U": col})
    table.append_rows(path, {"U": col[:1]})
    table.update_rows(path, {"U": col[:1]}, (0, 1))
    table.delete_rows(path, (len(EDGES), len(EDGES) + 1))


def _then_insert_column(path, col):
    torchfits.write(path, {"U": col})
    table.insert_column(path, "X", np.arange(len(EDGES)))


@pytest.mark.parametrize(
    "writer",
    [
        _write_direct,
        _write_compressed,
        _write_hdulist,
        _insert_into,
        _edit_other_hdu,
        _write_schema,
        _replace_column,
        _then_edit_rows,
        _then_insert_column,
    ],
)
def test_table_tzero_card_is_exactly_two_pow_63(tmp_path, writer):
    # TZERO = 2**63 used to be clamped to LLONG_MAX, then written as the float
    # 9.2233720368547758E+18; FITS 4.0 §5.3 spells it as the exact integer.
    path = str(tmp_path / "u.fits")
    col = _u64(EDGES)
    writer(path, col)
    cards = _raw_cards(path, 1)
    assert cards["TZERO1"] == "9223372036854775808"
    assert float(cards["TSCAL1"]) == 1.0
    got = torchfits.read(path, hdu=1)["U"]
    assert got.dtype == torch.uint64
    assert got.tolist() == EDGES


def _image_direct(path, data):
    torchfits.write(path, data)


def _image_hdulist(path, data):
    torchfits.write(path, torchfits.HDUList([torchfits.TensorHDU(data)]))


def _image_insert(path, data):
    torchfits.write(path, torch.zeros(1))
    torchfits.insert_hdu(path, data, index=0)


def _image_replace(path, data):
    torchfits.write(path, torch.zeros(1))
    torchfits.replace_hdu(path, 0, data)


@pytest.mark.parametrize(
    "writer", [_image_direct, _image_hdulist, _image_insert, _image_replace]
)
def test_image_bzero_card_is_exactly_two_pow_63(tmp_path, writer):
    path = str(tmp_path / "u.fits")
    data = _u64(EDGES).reshape(2, 4)
    writer(path, data)
    cards = _raw_cards(path, 0)
    assert cards["BZERO"] == "9223372036854775808"
    assert float(cards["BSCALE"]) == 1.0
    got = torchfits.read(path)
    assert got.dtype == torch.uint64
    assert torch.equal(got, data)


def _respell(path, key, new):
    """Rewrite one card's value in place (same file length)."""
    raw = bytearray(open(path, "rb").read())
    for i in range(0, len(raw), 80):
        if raw[i : i + 8].decode("ascii").rstrip() == key:
            raw[i : i + 80] = f"{key:<8}= {new:>20}".ljust(80).encode("ascii")
            break
    else:
        raise KeyError(key)
    open(path, "wb").write(bytes(raw))


def test_legacy_float_offset_spelling_still_reads_as_uint64(tmp_path):
    # Files written by torchfits 1.3 carry the float spelling; keep reading them.
    img = str(tmp_path / "img.fits")
    data = _u64(EDGES)
    torchfits.write(img, data)
    _respell(img, "BZERO", "9.2233720368547758E+18")
    assert _raw_cards(img, 0)["BZERO"] == "9.2233720368547758E+18"
    for mmap in (True, False):
        assert torchfits.read(img, mmap=mmap).tolist() == EDGES
    with torchfits.open(img) as hdul:
        assert hdul[0].data.dtype == torch.uint64

    tab = str(tmp_path / "tab.fits")
    torchfits.write(tab, {"U": data})
    _respell(tab, "TZERO1", "9.2233720368547758E+18")
    assert torchfits.read(tab, hdu=1)["U"].tolist() == EDGES
    assert table.read_torch(tab, hdu=1)["U"].tolist() == EDGES
    assert table.read(tab, hdu=1)["U"].to_pylist() == EDGES
    assert str(table.schema(tab, hdu=1).field("U").type) == "uint64"
