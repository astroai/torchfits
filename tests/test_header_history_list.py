import torchfits

def test_write_header_with_list(tmp_path):
    # Tests that providing a list for commentary cards like HISTORY doesn't crash std::bad_cast
    out_file = str(tmp_path / "test_history.fits")
    header = {"HISTORY": ["line 1", "line 2"], "COMMENT": ["comment 1", "comment 2"]}

    # Should not raise exception
    torchfits.table.write(out_file, {"A": [1]}, header=header)

    # Verify
    h = torchfits.HDUList.fromfile(out_file)[1].header
    assert "line 1" in h.get_history()
    assert "line 2" in h.get_history()
    assert "comment 1" in h.get_comment()
    assert "comment 2" in h.get_comment()
