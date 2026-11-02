from torchfits import Header


def test_header_update_replaces_commentary():
    # r7c-22: replaying commentary cards appends HISTORY/COMMENT instead of replacing them.
    h = Header()
    h["HISTORY"] = "a"

    # 1. Update from dict
    h.update({"HISTORY": "b"})
    cards = [c.value for c in h.cards if c.key == "HISTORY"]
    assert cards == ["b"], f"Expected ['b'], got {cards}"

    # 2. Update from another header
    h2 = Header()
    h2["HISTORY"] = "c"
    h.update(h2)
    cards = [c.value for c in h.cards if c.key == "HISTORY"]
    assert cards == ["c"], f"Expected ['c'], got {cards}"

    # 3. Update from dict containing a list of comments
    h3 = Header()
    h3["HISTORY"] = "d"
    h3.update({"HISTORY": ["e", "f"]})
    cards = [c.value for c in h3.cards if c.key == "HISTORY"]
    assert cards == ["e", "f"], f"Expected ['e', 'f'], got {cards}"


def test_header_update_merges_multiple_commentary_cards_from_header():
    h = Header()
    h["HISTORY"] = "a"

    h2 = Header()
    h2.add_history("b")
    h2.add_history("c")

    h.update(h2)
    cards = [c.value for c in h.cards if c.key == "HISTORY"]
    assert cards == ["b", "c"], f"Expected ['b', 'c'], got {cards}"
