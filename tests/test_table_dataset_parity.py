"""Map-style ``FitsTableDataset`` and ``FitsTableIterableDataset`` agree.

Covers zero-column tables (TFIELDS = 0 with NAXIS2 rows) and the native
read fallback in ``_eager_table_columns``.
"""

from __future__ import annotations

import pytest

from torchfits.data import FitsTableDataset, FitsTableIterableDataset


def _card(text: str) -> bytes:
    return text.ljust(80).encode("ascii")


def _header(cards: list[str]) -> bytes:
    payload = b"".join(_card(c) for c in [*cards, "END"])
    return payload.ljust(-(-len(payload) // 2880) * 2880, b" ")


@pytest.fixture
def zero_column_table(tmp_path):
    """A BINTABLE with TFIELDS = 0 and six (zero-byte) rows."""
    path = tmp_path / "zero_col.fits"
    primary = _header(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
        ]
    )
    table = _header(
        [
            "XTENSION= 'BINTABLE'",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    0",
            "NAXIS2  =                    6",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "TFIELDS =                    0",
        ]
    )
    path.write_bytes(primary + table)
    return str(path)


def test_zero_column_table_rows_match_map_style(zero_column_table):
    mapped = FitsTableDataset(zero_column_table, hdu=1, return_label=False)
    assert len(mapped) == 6
    rows = list(FitsTableIterableDataset(zero_column_table, hdu=1))
    assert rows == [mapped[i] for i in range(len(mapped))] == [{}] * 6


@pytest.mark.parametrize(("batch_size", "n_batches"), [(4, 2), (6, 1), (1, 6)])
def test_zero_column_table_batches(zero_column_table, batch_size, n_batches):
    batches = list(
        FitsTableIterableDataset(
            zero_column_table, hdu=1, batch_size=batch_size, as_batches=True
        )
    )
    assert batches == [{}] * n_batches


def test_zero_column_table_shards_cover_every_row(zero_column_table):
    per_rank = [
        list(
            FitsTableIterableDataset(
                zero_column_table, hdu=1, batch_size=2, rank=r, world_size=4
            )
        )
        for r in range(4)
    ]
    assert [len(rows) for rows in per_rank] == [2, 2, 1, 1]


@pytest.fixture
def vla_and_string_table(tmp_path):
    import numpy as np

    import torchfits.table

    path = tmp_path / "vla.fits"
    torchfits.table.write(
        str(path),
        {
            "V": [
                np.array([1], np.int32),
                np.array([2, 3], np.int32),
                np.array([4], np.int32),
            ],
            "S": np.array(["ab", "cd", "ef"]),
            "X": np.arange(3, dtype=np.int32),
        },
        schema={"V": {"format": "1PJ"}},
        overwrite=True,
    )
    return str(path)


def test_vla_rows_have_the_same_types_in_both_datasets(vla_and_string_table, caplog):
    import torch

    with caplog.at_level("WARNING", logger="torchfits.data"):
        mapped = FitsTableDataset(vla_and_string_table, return_label=False)
    # The mmap reader refuses VLA; the handle reader serves it natively, so
    # no pyarrow fallback (which would turn "S" into str).
    assert not [r for r in caplog.records if "pyarrow" in r.getMessage()]
    streamed = list(FitsTableIterableDataset(vla_and_string_table))
    assert len(mapped) == len(streamed) == 3
    for i, row in enumerate(streamed):
        ref = mapped[i]
        assert ref.keys() == row.keys()
        for key in row:
            assert isinstance(ref[key], torch.Tensor), key
            assert ref[key].dtype == row[key].dtype
            assert torch.equal(ref[key], row[key])
    assert mapped[1]["V"].tolist() == [2, 3]
    assert mapped[0]["S"].dtype == torch.uint8


@pytest.mark.parametrize(
    "error", [MemoryError("oom"), TypeError("binding bug"), KeyError("bug")]
)
def test_unexpected_native_errors_are_not_swallowed(
    vla_and_string_table, monkeypatch, error
):
    import torchfits._C as cpp

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(cpp, "read_fits_table", boom)
    with pytest.raises(type(error)):
        FitsTableDataset(vla_and_string_table, return_label=False)


def test_native_read_failure_falls_back_with_warning(
    vla_and_string_table, monkeypatch, caplog
):
    import torchfits._C as cpp

    def refuse(*args, **kwargs):
        raise RuntimeError("layout not supported")

    monkeypatch.setattr(cpp, "read_fits_table", refuse)
    with caplog.at_level("WARNING", logger="torchfits.data"):
        mapped = FitsTableDataset(vla_and_string_table, return_label=False)
    assert len(mapped) == 3
    assert any("falling back to pyarrow" in r.getMessage() for r in caplog.records)


def test_column_format_lookup_only_tolerates_native_errors(
    vla_and_string_table, monkeypatch, caplog
):
    import torchfits

    ds = FitsTableIterableDataset(vla_and_string_table, as_batches=True)
    assert ds._non_numeric_columns() == {"S"}

    def bug(*args, **kwargs):
        raise TypeError("binding bug")

    monkeypatch.setattr(torchfits, "read_table_info", bug)
    with pytest.raises(TypeError):
        ds._non_numeric_columns()

    def unreadable(*args, **kwargs):
        raise OSError("unreadable header")

    monkeypatch.setattr(torchfits, "read_table_info", unreadable)
    with caplog.at_level("WARNING", logger="torchfits.data"):
        assert ds._non_numeric_columns() == set()
    assert any("column formats" in r.getMessage() for r in caplog.records)
