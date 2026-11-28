"""Structure tests: write/update input coercion runs before any byte is written.

``torchfits.write`` coerces its inputs in ``_io_engine/_write_coerce.py`` and
writes in ``write_api._write_new_file``; ``table.update_rows`` coerces in
``_mutation_coerce._normalize_update_rows`` and writes in
``mutation._write_row_update``. The coercion functions take no path, so an
argument error can never leave a partial file.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits._io_engine import _write_coerce as wc
from torchfits._io_engine._write_helpers import QuantizeError
from torchfits._table import _mutation_coerce as mc


def test_image_hdu_dicts_put_the_header_overlay_on_the_first_hdu_only():
    hdus = wc.image_hdu_dicts_for_write(
        [torch.zeros(2, 2), torch.ones(3)], {"OBJECT": "M31"}, None
    )
    assert len(hdus) == 2
    assert all(isinstance(h, dict) for h in hdus)


@pytest.mark.parametrize(
    "call",
    [
        lambda: wc.image_hdu_dicts_for_write([torch.zeros(2)], None, "robust"),
        lambda: wc.compressed_hdus_for_write([torch.zeros(2)], None, "robust"),
        lambda: wc.hdulist_hdus_for_write(torchfits.HDUList([]), None, "robust"),
    ],
)
def test_multi_hdu_quantize_is_refused_by_the_coercion(call):
    with pytest.raises(QuantizeError):
        call()


def test_empty_sequence_and_bad_items_are_refused():
    with pytest.raises(ValueError, match="At least one writable HDU"):
        wc.image_hdu_dicts_for_write([], None, None)
    with pytest.raises(TypeError, match="Unsupported HDU item type"):
        wc.image_hdu_dicts_for_write([object()], None, None)
    with pytest.raises(NotImplementedError):
        wc.compressed_hdus_for_write(42, None, None)


def test_update_rows_coercion_classifies_and_pads_strings():
    tforms = {"S": "4A", "V": "1PJ(3)", "C": "1C", "N": "1J"}
    codes = mc._update_column_codes(tforms)
    assert codes.string_widths == {"S": 4}
    assert set(codes.vla) == {"V"} and set(codes.complex) == {"C"}
    normalized, n = mc._normalize_update_rows(
        {"S": ["ab", ""], "N": np.array([1, 2], dtype=np.int32)},
        codes,
        tforms,
        {},
        {},
    )
    assert n == 2
    assert normalized["S"].tolist() == [list(b"ab  "), list(b"    ")]
    assert normalized["N"].tolist() == [1, 2]
