from torchfits.hdu import Header, TableHDURef


def test_tablehduref_cache_invalidation():
    header = Header()
    header["TFIELDS"] = 1
    header["TTYPE1"] = "OLD_NAME"

    ref = TableHDURef(header=header)
    assert ref.columns == ["OLD_NAME"]

    header["TTYPE1"] = "NEW_NAME"
    assert ref.columns == ["NEW_NAME"], f"Expected ['NEW_NAME'], got {ref.columns}"
