"""Read-policy unit tests."""

from torchfits._table_engine import (
    WhereStrategy,
    choose_where_read_plan,
    should_skip_cpp_for_where,
    validate_table_backend,
)


def test_should_skip_cpp_for_where():
    assert should_skip_cpp_for_where("auto", "MAG < 20") is True
    assert should_skip_cpp_for_where("auto", None) is False
    assert should_skip_cpp_for_where("cpp", "MAG < 20") is False


def test_choose_where_read_plan_auto_uses_cpp_unfiltered():
    plan = choose_where_read_plan(
        header={},
        header_ok=True,
        columns=None,
        backend="auto",
        n_rows=10_000,
    )
    assert plan.strategy == WhereStrategy.CPP_PUSHDOWN
    assert plan.unfiltered_backend == "cpp"


def test_cpp_numpy_backend_deprecated():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_table_backend("cpp_numpy") == "cpp"
    assert any(
        issubclass(w.category, DeprecationWarning) and "cpp_numpy" in str(w.message)
        for w in caught
    )
