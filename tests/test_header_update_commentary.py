import pytest
from torchfits.hdu import Header

def test_header_update_commentary_replaces():
    h = Header()
    h["HISTORY"] = "old 1"
    h["HISTORY"] = "old 2"

    h2 = Header()
    h2["HISTORY"] = "new 1"
    h2["HISTORY"] = "new 2"

    h.update(h2)
    assert [c.value for c in h.cards if c.key == "HISTORY"] == ["new 1", "new 2"]

    h3 = Header()
    h3["HISTORY"] = "old 1"

    h3.update({"HISTORY": ["dict 1", "dict 2"]})
    assert [c.value for c in h3.cards if c.key == "HISTORY"] == ["dict 1", "dict 2"]

    h4 = Header()
    h4["COMMENT"] = "old comment"

    h4.update(COMMENT="new comment")
    assert [c.value for c in h4.cards if c.key == "COMMENT"] == ["new comment"]
