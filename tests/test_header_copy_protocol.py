"""Header survives pickle / copy.copy / copy.deepcopy with its card list intact."""

from __future__ import annotations

import copy
import pickle

import pytest
import torch

import torchfits
from torchfits import Card, Header


def _header() -> Header:
    return Header(
        [
            ("OBJECT", "M31", "target"),
            ("EXPTIME", 30.0, "s"),
            ("HISTORY", "h1", ""),
            ("HISTORY", "h2", ""),
        ]
    )


EXPECTED = (
    Card("OBJECT", "M31", "target"),
    Card("EXPTIME", 30.0, "s"),
    Card("HISTORY", "h1", ""),
    Card("HISTORY", "h2", ""),
)


@pytest.mark.parametrize(
    "clone",
    [
        lambda h: pickle.loads(pickle.dumps(h)),
        copy.copy,
        copy.deepcopy,
    ],
    ids=["pickle", "copy", "deepcopy"],
)
def test_clone_keeps_cards_and_leaves_original_alone(clone):
    original = _header()
    twin = clone(original)
    assert type(twin) is Header
    assert twin.cards == EXPECTED
    assert dict(twin) == dict(original)
    assert original.cards == EXPECTED
    twin["NEW"] = 1
    twin.add_history("h3")
    assert original.cards == EXPECTED
    assert "NEW" not in original


def test_read_header_result_pickles(tmp_path):
    path = tmp_path / "h.fits"
    torchfits.write(str(path), torch.zeros(2, 2), header=_header(), overwrite=True)
    hdr = torchfits.read_header(str(path))
    back = pickle.loads(pickle.dumps(hdr))
    assert back.cards == hdr.cards
    assert back.get_history() == ["h1", "h2"]


def test_copy_method_returns_header_with_cards():
    original = _header()
    twin = original.copy()
    assert type(twin) is Header
    assert twin.cards == EXPECTED
    twin["NEW"] = 1
    assert "NEW" not in original


def test_ior_updates_cards_and_reaches_the_file(tmp_path):
    hdr = _header()
    alias = hdr
    hdr |= {"NEW": (1, "added")}
    assert hdr is alias
    assert hdr.card("NEW") == Card("NEW", 1, "added")
    path = tmp_path / "ior.fits"
    torchfits.write(str(path), torch.zeros(2, 2), header=hdr, overwrite=True)
    assert torchfits.read_header(str(path))["NEW"] == 1


@pytest.mark.parametrize("merge", [lambda h: h | {"X": 1}, lambda h: {"X": 1} | h])
def test_or_returns_header_with_all_cards(merge):
    original = _header()
    merged = merge(original)
    assert type(merged) is Header
    assert merged.card("X") == Card("X", 1, "")
    assert merged.get_history() == ["h1", "h2"]
    assert merged.card("OBJECT").comment == "target"
    assert "X" not in original
