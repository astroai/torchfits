import torchfits


def test_commentary_cards_replace():
    header = torchfits.Header({"NAXIS": 0, "HISTORY": "h1"})
    # Updating with a dictionary should REPLACE HISTORY, not append
    header.update({"HISTORY": ["h2", "h3"]})
    assert header.get_history() == ["h2", "h3"]


def test_commentary_cards_replace_from_header():
    header = torchfits.Header({"NAXIS": 0, "HISTORY": "h1"})
    header2 = torchfits.Header({"NAXIS": 1, "HISTORY": ["h2", "h3"]})
    header.update(header2)
    assert header.get_history() == ["h2", "h3"]
