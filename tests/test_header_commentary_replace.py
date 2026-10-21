from torchfits import Header

def test_header_update_replaces_history():
    h = Header()
    h.add_history("old1")
    h.add_history("old2")

    h2 = Header()
    h2.add_history("new1")
    h2.add_history("new2")

    h.update(h2)
    assert h.get_history() == ["new1", "new2"]

def test_header_update_dict_replaces_history():
    h = Header()
    h.add_history("old")
    h.update({"HISTORY": ["new"]})
    assert h.get_history() == ["new"]

def test_header_setitem_replaces_history():
    h = Header()
    h.add_history("old")
    h["HISTORY"] = ["new"]
    assert h.get_history() == ["new"]
