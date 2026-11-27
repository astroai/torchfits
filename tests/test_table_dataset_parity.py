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
