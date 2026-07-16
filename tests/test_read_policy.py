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


def test_choose_where_read_plan_auto_mmap_uses_pushdown():
    """auto + mmap=True ⇒ documented native mmap-scan pushdown."""
    plan = choose_where_read_plan(
        header={},
        header_ok=True,
        columns=None,
        backend="auto",
        n_rows=1_000,
        mmap=True,
    )
    assert plan.strategy == WhereStrategy.CPP_PUSHDOWN
    assert plan.unfiltered_backend == "cpp"
    assert plan.cpp_pushdown_safe is True


def test_choose_where_read_plan_auto_buffered_uses_arrow_filter():
    """auto + mmap=False ⇒ read then tensor/Arrow filter (no size forks)."""
    for n_rows in (1_000, 100_000):
        plan = choose_where_read_plan(
            header={},
            header_ok=True,
            columns=None,
            backend="auto",
            n_rows=n_rows,
            mmap=False,
        )
        assert plan.strategy == WhereStrategy.ARROW_FILTER
        assert plan.unfiltered_backend == "cpp"


def test_choose_where_read_plan_cpp_uses_pushdown():
    plan = choose_where_read_plan(
        header={},
        header_ok=True,
        columns=None,
        backend="cpp",
        n_rows=10_000,
        mmap=True,
    )
    assert plan.strategy == WhereStrategy.CPP_PUSHDOWN


def test_cpp_numpy_backend_rejected():
    """The legacy 'cpp_numpy' backend alias was removed in 0.8.0."""
    import pytest

    with pytest.raises(ValueError, match="backend must be one of"):
        validate_table_backend("cpp_numpy")
