import inspect
import warnings

import numpy as np

import torchfits
from torchfits import hdu, table, where


def test_hdu_and_table_public_surfaces_are_importable():
    assert hdu.DataView is not None
    assert hdu.TableDataAccessor is not None
    assert callable(table.read)
    assert callable(table.write)
    assert callable(table.clear_cache)
    assert torchfits.hdu is hdu
    assert "hdu" in torchfits.__all__
    assert "read_fast" not in torchfits.__all__
    assert "SpectralBinning" not in torchfits.__all__
    assert "can_use_mmap_row_path_for_full_read" not in table.__all__
    import torchfits.io as io

    assert not hasattr(io, "read_fast")
    assert "read_fast" not in io.__all__


def test_table_destination_readers_are_public():
    assert "read_arrow" in table.__all__
    assert "read_torch" in table.__all__
    assert table.read_arrow is table.read
    assert callable(table.read_torch)
    assert callable(torchfits.read_table)
    assert callable(torchfits.read_header)


def test_torch_frame_is_not_part_of_the_fits_hdu_surface():
    assert not hasattr(hdu, "TensorFrame")


def test_root_functions_preserve_real_signatures():
    import torchfits.io

    assert inspect.signature(torchfits.read) == inspect.signature(torchfits.io.read)
    assert torchfits.read is torchfits.io.read


def test_cpp_public_surface_is_explicit_and_resolves():
    import torchfits.cpp as cpp

    assert len(cpp.__all__) == len(set(cpp.__all__))
    assert all(hasattr(cpp, name) for name in cpp.__all__)
    assert "read_header_dict" in cpp.__all__
    assert "resolve_hdu_name_cached" in cpp.__all__
    assert "compute_stats" not in cpp.__all__


def test_where_public_surface_matches_table_predicate_semantics():
    ast = where.parse_where_expression("A > 1 AND B IS NOT NULL")
    mask = where.evaluate_where(
        ast, {"A": np.array([0, 2, 3]), "B": np.array([1, None, 4], dtype=object)}
    )
    np.testing.assert_array_equal(mask, np.array([False, False, True]))
    assert torchfits.where.parse_where_literal("42") == 42
    assert where.where_columns_from_ast(ast) == ["A", "B"]


def test_root_table_helpers_emit_deprecation_warning():
    path = "/nonexistent_torchfits_deprecation.fits"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for call in (
            lambda: torchfits.read_table(path),
            lambda: next(torchfits.stream_table(path)),
            lambda: torchfits.read_table_rows(path),
            lambda: torchfits.get_header(path),
            lambda: torchfits.get_batch_info([path]),
        ):
            try:
                call()
            except Exception:
                pass
        msgs = [
            str(item.message)
            for item in caught
            if issubclass(item.category, DeprecationWarning)
        ]
    assert any("read_table is deprecated" in m for m in msgs)
    assert any("stream_table is deprecated" in m for m in msgs)
    assert any("read_table_rows is deprecated" in m for m in msgs)
    assert any("get_header is deprecated" in m for m in msgs)
    assert any("get_batch_info is deprecated" in m for m in msgs)
