import pytest
import numpy as np
from astropy.io import fits


@pytest.fixture
def fits_file(tmp_path):
    path = str(tmp_path / "test_filter.fits")
    n_rows = 1000

    # Create data
    # FLOAT column 'MAG' : 0..100
    # INT column 'ID' : 0..1000
    # STRING column 'LABEL': 'A', 'B' alternating

    mag = np.linspace(0, 100, n_rows, dtype=np.float32)
    ids = np.arange(n_rows, dtype=np.int32)
    short_col = np.arange(n_rows, dtype=np.int16)

    c1 = fits.Column(name="MAG", format="E", array=mag)
    c2 = fits.Column(name="ID", format="J", array=ids)
    c3 = fits.Column(name="SHORT_VAL", format="I", array=short_col)

    # Add a string column if possible, but let's start with numeric

    hdu = fits.BinTableHDU.from_columns([c1, c2, c3])
    hdu.writeto(path)
    return path


def test_filter_lt(fits_file):
    import torchfits.cpp

    # MAG < 50.0. linspace(0,100,1000): i * 100/999 < 50 => i <= 499 (500 rows).
    filters = [("MAG", "<", 50.0)]
    cols = ["ID", "MAG"]

    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, cols, filters)

    assert "ID" in data
    assert "MAG" in data

    ids = data["ID"]
    mags = data["MAG"]

    assert len(ids) == 500
    assert (mags < 50.0).all()
    assert len(mags) == 500


def test_filter_gt(fits_file):
    import torchfits.cpp

    filters = [("ID", ">", 800)]
    cols = ["ID"]

    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, cols, filters)
    ids = data["ID"]

    # 801 to 999 -> 199 items
    assert len(ids) == 199
    assert (ids > 800).all()


def test_table_read_integration(fits_file):
    import torchfits

    # Test integration via torchfits.table.read(where=...)
    # MAG < 50.0 should use fast path

    t = torchfits.table.read(fits_file, where="MAG < 50.0")
    assert len(t) == 500
    mags = t["MAG"].to_numpy()
    assert (mags < 50.0).all()

    # OR falls back to slow path: MAG < 10 (100 rows) OR MAG > 90 (100 rows)
    t_slow = torchfits.table.read(fits_file, where="MAG < 10.0 OR MAG > 90.0")
    assert len(t_slow) == 200
    mags_slow = t_slow["MAG"].to_numpy()
    assert ((mags_slow < 10.0) | (mags_slow > 90.0)).all()


def test_table_read_where_torch_backend(fits_file):
    import torchfits

    # MAG > 10 AND MAG < 20: indices 100..199 -> 100 rows
    t = torchfits.table.read(
        fits_file, where="MAG > 10.0 AND MAG < 20.0", backend="torch"
    )
    mags = t["MAG"].to_numpy()
    assert len(t) == 100
    assert ((mags > 10.0) & (mags < 20.0)).all()


def test_filter_eq(fits_file):
    import torchfits.cpp

    filters = [("ID", "==", 500)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["ID"], filters)
    assert len(data["ID"]) == 1
    assert data["ID"][0].item() == 500


def test_filter_compound(fits_file):
    import torchfits.cpp

    # ID > 100 AND ID < 200
    filters = [("ID", ">", 100), ("ID", "<", 200)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["ID"], filters)
    ids = data["ID"]

    assert len(ids) == 99  # 101 to 199
    assert ids.min() == 101
    assert ids.max() == 199


def test_where_preserves_tnull_nulls(tmp_path):
    """WHERE-filtered reads must still convert TNULL sentinels to Arrow null.

    Regression: the CPP-pushdown and torch-tensor-mask fast paths taken for
    simple WHERE predicates used to skip TNULL handling entirely, so a
    nullable column read back through where= leaked its raw sentinel value
    instead of Arrow null even with apply_fits_nulls=True (the default).
    """
    import torchfits.table as table

    path = str(tmp_path / "nulls.fits")
    n = 20
    ids = np.arange(n, dtype=np.int32)
    vals = np.arange(n, dtype=np.int32)
    vals[3] = -999
    vals[7] = -999
    c1 = fits.Column(name="ID", format="J", array=ids)
    c2 = fits.Column(name="VAL", format="J", array=vals, null=-999)
    fits.BinTableHDU.from_columns([c1, c2]).writeto(path)

    full = table.read(path, hdu=1, apply_fits_nulls=True)
    assert full.column("VAL").null_count == 2

    filtered = table.read(path, hdu=1, where="ID >= 0", apply_fits_nulls=True)
    assert filtered.column("VAL").null_count == 2
    assert filtered.column("VAL").to_pylist()[3] is None

    # Projected columns (WHERE references a column outside the projection)
    # go through the Arrow-filter fallback and must also preserve nulls.
    projected = table.read(
        path, hdu=1, columns=["VAL"], where="ID >= 0", apply_fits_nulls=True
    )
    assert projected.column("VAL").null_count == 2

    # Explicit opt-out must still be honored.
    disabled = table.read(path, hdu=1, where="ID >= 0", apply_fits_nulls=False)
    assert disabled.column("VAL").null_count == 0


def test_filter_short(fits_file):
    import torchfits.cpp

    # SHORT_VAL is int16. Filter on it.
    filters = [("SHORT_VAL", "==", 10)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["SHORT_VAL"], filters)
    assert len(data["SHORT_VAL"]) == 1
    assert data["SHORT_VAL"][0].item() == 10
