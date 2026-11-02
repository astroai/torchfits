"""Bad TNULL / TSCAL / TZERO schema values raise ValueError naming the field.

They used to reach the native writer and come back as an unnamed
RuntimeError ("schema numeric value must be an int or a numeric string").
"""

import numpy as np
import pytest

import torchfits
from torchfits import table


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "t.fits")
    table.write(p, {"X": np.arange(3)})
    return p


@pytest.mark.parametrize(
    ("kwargs", "card"),
    [
        ({"tnull": "not-an-int"}, "TNULL"),
        ({"tnull": 1.5}, "TNULL"),
        ({"tnull": [1]}, "TNULL"),
        ({"tnull": 2**70}, "TNULL"),
        ({"tscal": "x"}, "TSCAL"),
        ({"tzero": "x"}, "TZERO"),
    ],
)
@pytest.mark.parametrize("op", ["insert", "replace"])
def test_column_mutation_rejects_bad_numerics(path, kwargs, card, op):
    before = open(path, "rb").read()
    values = np.zeros(3, np.int16)
    with pytest.raises(ValueError, match=card) as info:
        if op == "insert":
            table.insert_column(path, "QUAL", values, format="I", **kwargs)
        else:
            table.replace_column(path, "X", values, format="I", **kwargs)
    field = next(iter(kwargs))
    assert field in str(info.value)
    assert open(path, "rb").read() == before


@pytest.mark.parametrize(
    ("meta", "card"),
    [
        ({"tnull": "bad"}, "TNULL"),
        ({"null": "bad"}, "TNULL"),
        ({"bscale": "bad"}, "TSCAL"),
        ({"bzero": [1]}, "TZERO"),
    ],
)
def test_table_write_schema_rejects_bad_numerics(tmp_path, meta, card):
    with pytest.raises(ValueError, match=card):
        table.write(
            str(tmp_path / "w.fits"), {"Q": np.zeros(3, np.int16)}, schema={"Q": meta}
        )


def test_integer_like_tnull_is_still_accepted(path):
    table.insert_column(path, "A", np.zeros(3, np.int16), format="I", tnull="7")
    table.insert_column(path, "B", np.zeros(3, np.int16), format="I", tnull=2.0)
    header = torchfits.read_header(path, 1)
    assert int(header["TNULL2"]) == 7
    assert int(header["TNULL3"]) == 2
