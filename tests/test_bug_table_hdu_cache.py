from torchfits.hdu import Header, TableHDU


def test_tablehdu_cache_invalidation():
    header = Header()
    header["TFIELDS"] = 1
    header["TTYPE1"] = "OLD_NAME"
    header["TFORM1"] = "A10" # string col

    hdu = TableHDU({}, header=header)
    assert hdu.string_columns == ["OLD_NAME"]

    # Mutate header
    header["TTYPE1"] = "NEW_NAME"
    assert hdu.string_columns == ["NEW_NAME"], f"Expected ['NEW_NAME'], got {hdu.string_columns}"
