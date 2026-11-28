"""WHERE filtering helpers for table reads."""

from __future__ import annotations

import functools
import logging
import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import torch

from .. import fits_schema
from .._table.cache import _acquire_cpp_reader
from .._where import parse_where_expression, where_columns_from_ast
from .._table_engine import (
    WhereReadPlan,
    WhereStrategy,
    choose_where_read_plan,
)
from .._table.utils import _require_pyarrow
from .._string_decode import column_decode_context
from .._table.arrow_convert import (
    _pa_array,
    _tensor_to_arrow_array,
)
from ._read_schema import (
    _all_table_columns,
    _attach_fits_metadata,
    _can_use_mmap_row_path_for_full_read,
    _empty_table_with_schema,
)


logger = logging.getLogger(__name__)


def _not_null(tensor: Any, sentinel: Any) -> Any:
    """Rows of ``tensor`` that do not hold the decoded TNULL ``sentinel``.

    ``sentinel`` comes from :func:`fits_schema.column_null_sentinel_map`: the
    raw TNULL for unscaled columns, ``raw + TZERO`` for unsigned ones, and NaN
    for scaled columns (whose TNULL cells decode to NaN).
    """
    import math

    import torch

    if isinstance(sentinel, float) and math.isnan(sentinel):
        if tensor.dtype.is_floating_point:
            return ~torch.isnan(tensor)
        return torch.ones(tensor.shape, dtype=torch.bool)
    return torch.ne(tensor, sentinel)


# NOTE: ceiling — full-table torch WHERE materializes all selected columns before
# masking; above this row count fall through to chunked Arrow-filter instead.
_TORCH_WHERE_MAX_ROWS = 1_000_000


@functools.lru_cache(maxsize=128)
def _compile_where_to_simple_predicates(
    where: str,
) -> Optional[tuple[tuple[str, str, Any], ...]]:
    """Parse a where string into simple predicates (cached).

    Returns a tuple of (col, op, literal) triples, or None if the where
    clause cannot be reduced to simple predicates.  The tuple is immutable
    so the cached value cannot be corrupted by callers.
    """
    try:
        ast = parse_where_expression(where)
    except (ValueError, TypeError, KeyError, RuntimeError):
        return None

    predicates: list[tuple[str, str, Any]] = []

    def _visit(node: Any) -> bool:
        kind = node[0]
        if kind == "cmp":
            _, col, op, literal = node
            if op not in {"==", "!=", ">", ">=", "<", "<="}:
                return False
            if literal is None:
                return False
            predicates.append((col, op, literal))
            return True
        if kind == "between":
            _, col, low, high, negate = node
            if bool(negate) or low is None or high is None:
                return False
            predicates.append((col, ">=", low))
            predicates.append((col, "<=", high))
            return True
        if kind == "and":
            return _visit(node[1]) and _visit(node[2])
        return False

    if not _visit(ast):
        return None
    return tuple(predicates)


def _integer_literal(op: str, literal: Any) -> tuple[str, Any]:
    """Map a float literal compared with an integer column to an exact form.

    Returns ``(op, int_literal)``, or ``("const", bool)`` when the comparison
    is the same for every integer value (NaN, infinities, ``==`` against a
    non-integral value). Comparing in float would round large integers
    (``J > 16777216.5`` in float32, ``K == 2**53 + 0.0`` in float64).
    """
    import math

    if not isinstance(literal, float):
        return op, literal
    if math.isnan(literal):
        return "const", op == "!="
    if math.isinf(literal):
        above = literal > 0  # every value is below +inf, above -inf
        return "const", op in ({"<", "<=", "!="} if above else {">", ">=", "!="})
    low = math.floor(literal)
    if low == literal:
        return op, int(literal)
    # Exact integer bound for a non-integral literal.
    if op in {">", ">="}:
        return ">", low
    if op in {"<", "<="}:
        return "<", low + 1
    return "const", op == "!="


def _uint64_cmp_mask(tensor: torch.Tensor, op: str, literal: Any) -> torch.Tensor:
    """Compare a uint64 column exactly.

    torch has no CPU compare kernels for uint64, and widening to int64 would
    wrap values >= 2**63. Flipping the sign bit maps [0, 2**64) onto int64
    in order, so compare there against the literal mapped the same way.
    """
    import torch

    if op not in {"==", "!=", ">", ">=", "<", "<="}:
        raise ValueError(f"Unsupported where operator '{op}'")

    def constant(value: bool) -> torch.Tensor:
        return torch.full(tensor.shape, value, dtype=torch.bool)

    op, literal = _integer_literal(op, literal)
    if op == "const":
        return constant(literal)
    if isinstance(literal, bool) or not isinstance(literal, int):
        raise TypeError(f"Unsupported literal for uint64 column: {literal!r}")
    if literal < 0:
        return constant(op in {">", ">=", "!="})
    if literal >= 2**64:
        return constant(op in {"<", "<=", "!="})

    keys = tensor.contiguous().view(torch.int64) ^ torch.iinfo(torch.int64).min
    target = literal - 2**63
    fn = {
        "==": torch.eq,
        "!=": torch.ne,
        ">": torch.gt,
        ">=": torch.ge,
        "<": torch.lt,
        "<=": torch.le,
    }[op]
    return fn(keys, target)


def _torch_cmp_mask(tensor: torch.Tensor, op: str, literal: Any) -> torch.Tensor:
    import torch

    # Same literal coercions as the Arrow filter (_aligned_scalar): T/F/TRUE
    # for logical columns, a numeric word such as inf for float columns.
    if tensor.dtype == torch.bool:
        literal = _boolean_literal(literal, "")
    elif tensor.dtype.is_floating_point and isinstance(literal, str):
        try:
            literal = float(literal)
        except ValueError:
            raise ValueError(
                f"where compares a float column with {literal!r}, which is not a number"
            ) from None
    elif isinstance(literal, str):
        raise ValueError(
            f"where compares a numeric column with {literal!r}, which is not a number"
        )

    if tensor.dtype == torch.uint64:
        return _uint64_cmp_mask(tensor, op, literal)

    if not tensor.dtype.is_floating_point and tensor.dtype != torch.bool:
        if op not in {"==", "!=", ">", ">=", "<", "<="}:
            raise ValueError(f"Unsupported where operator '{op}'")
        op, literal = _integer_literal(op, literal)
        if op == "const":
            return torch.full(tensor.shape, bool(literal), dtype=torch.bool)
        if (
            tensor.dtype == torch.int64
            and isinstance(literal, int)
            and not isinstance(literal, bool)
            and not (-(2**63) <= literal < 2**63)
        ):
            above = literal > 0  # every int64 value is below / above it
            return torch.full(
                tensor.shape,
                op in ({"<", "<=", "!="} if above else {">", ">=", "!="}),
                dtype=torch.bool,
            )

    # PyTorch wraps an out-of-range Python int scalar to the tensor's dtype
    # (e.g. comparing an int16 column against 40000 becomes `> -25536`), which
    # silently flips the predicate. Promote the tensor to int64 so the literal
    # is compared at full width instead (matching the C++ pushdown, which now
    # also compares integers in int64). uint16/uint32 always compare at int64:
    # torch has no CPU compare kernels for those dtypes at all (r5a-07).
    if tensor.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.uint8,
        torch.uint16,
        torch.uint32,
    ):
        widen = tensor.dtype in (torch.uint16, torch.uint32)
        if isinstance(literal, int) and not isinstance(literal, bool):
            info = torch.iinfo(tensor.dtype)
            if not (info.min <= literal <= info.max):
                widen = True
        if widen:
            tensor = tensor.to(torch.int64)

    if op == "==":
        out = torch.eq(tensor, literal)
    elif op == "!=":
        out = torch.ne(tensor, literal)
    elif op == ">":
        out = torch.gt(tensor, literal)
    elif op == ">=":
        out = torch.ge(tensor, literal)
    elif op == "<":
        out = torch.lt(tensor, literal)
    elif op == "<=":
        out = torch.le(tensor, literal)
    else:
        raise ValueError(f"Unsupported where operator '{op}'")
    # Scaled TNULL/BLANK pixels are NaN; SQL three-valued logic never matches NULL.
    if op == "!=" and tensor.is_floating_point():
        out = out & ~torch.isnan(tensor)
    return out


def _try_torch_tensor_where_filter(
    *,
    pa: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    header: Any | None,
    apply_fits_nulls: bool = False,
    include_fits_metadata: bool = False,
) -> Any | None:
    """Buffered/mmap tensor read + torch mask + Arrow for simple numeric WHERE.

    Honors ``mmap`` via the existing C++ row readers (no secret mmap when
    ``mmap=False``). Skips Arrow conversion of the full column before filtering.
    """
    if row_slice is not None or rows is not None:
        return None
    predicates = _compile_where_to_simple_predicates(where)
    if predicates is None:
        return None

    # This is an optimisation: the caller falls through to the Arrow-side
    # filter when we return None. Like the native pushdown, it only runs when
    # torch is already loaded: an Arrow-only `where=` read must neither fail
    # without torch nor import it just to filter faster.
    if "torch" not in sys.modules:
        return None
    import torch
    import torchfits._C as cpp

    output_cols = columns
    if output_cols is None:
        if header is not None:
            # Name every column the reader can serve, TTYPE-less ones included:
            # a filtered read that returned a narrower set than the same read
            # without a filter made the result backend-dependent (TE-007).
            output_cols = [col.name for col in _all_table_columns(header)]
        else:
            return None

    # decode_bytes only matters for string/bit columns; numeric WHERE paths stay eligible.
    if decode_bytes and header is not None:
        selected = set(output_cols)
        for col in _all_table_columns(header, selected):
            if col.tform_info.is_string or col.tform_info.is_bit:
                return None
    elif header is None:
        return None

    pred_cols = [col for col, _op, _lit in predicates]
    read_cols: list[str] = []
    seen: set[str] = set()
    for name in list(output_cols) + pred_cols:
        if name not in seen:
            read_cols.append(name)
            seen.add(name)

    n_rows = 0
    if header is not None:
        try:
            n_rows = int(header.get("NAXIS2", 0) or 0)
        except (TypeError, ValueError):
            n_rows = 0
    if n_rows <= 0:
        try:
            n_rows = int(cpp.read_nrows(path, hdu))
        except (AttributeError, TypeError, RuntimeError, OSError, ValueError):
            n_rows = 0
    if n_rows > _TORCH_WHERE_MAX_ROWS:
        return None

    try:
        if mmap and _can_use_mmap_row_path_for_full_read(
            path, hdu, read_cols, header=header
        ):
            chunk = cpp.read_fits_table(path, hdu, read_cols, True)
        else:
            reader = _acquire_cpp_reader(path, hdu, cpp)
            chunk = reader.read_rows(read_cols, 1, -1)
    except (
        RuntimeError,
        OSError,
        MemoryError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        # A truncated file must surface as corruption, not as a silent
        # fallback to a slower engine that fails again downstream.
        if "truncat" in str(exc).lower():
            raise
        return None
    if not isinstance(chunk, dict) or not chunk:
        return None

    # Sentinel→null exclusion must happen BEFORE predicate evaluation so the
    # torch engine agrees with the Arrow engine (comparisons against NULL
    # never match) regardless of which strategy runs.
    tnull_map = (
        fits_schema.column_null_sentinel_map(header)
        if apply_fits_nulls and header is not None
        else {}
    )

    mask: torch.Tensor | None = None
    try:
        for pred_col, op, literal in predicates:
            tensor = chunk.get(pred_col)
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 1:
                # Vector cells: leave them to the Arrow filter, which refuses
                # them with a clear error instead of comparing elements.
                return None
            part = _torch_cmp_mask(tensor, op, literal)
            sentinel = tnull_map.get(pred_col)
            if sentinel is not None:
                part = part & _not_null(tensor, sentinel)
            mask = part if mask is None else (mask & part)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("torch WHERE mask build failed; falling back: %s", exc)
        return None
    if mask is None:
        return None

    # The TFORM tells a numeric byte ('B') vector from a character matrix:
    # both arrive as 2-D uint8 tensors.
    tforms = (
        {col.name: col.tform for col in _all_table_columns(header)}
        if header is not None
        else {}
    )
    arrays = []
    names_out = []
    for name in output_cols:
        value = chunk.get(name)
        if not isinstance(value, torch.Tensor):
            return None
        filtered = value[mask]
        with column_decode_context(name):
            arrays.append(
                _tensor_to_arrow_array(
                    pa,
                    filtered,
                    decode_bytes,
                    encoding,
                    strip,
                    null_sentinel=tnull_map.get(name),
                    fits_tform=tforms.get(name),
                )
            )
        names_out.append(name)

    if not arrays:
        # Preserve projected schema for empty output.
        empty = []
        for name in output_cols:
            value = chunk.get(name)
            if isinstance(value, torch.Tensor):
                empty.append(
                    _tensor_to_arrow_array(
                        pa,
                        value[:0],
                        decode_bytes,
                        encoding,
                        strip,
                        null_sentinel=tnull_map.get(name),
                        fits_tform=tforms.get(name),
                    )
                )
            else:
                # Non-tensor column (string/bit) — fall back to header schema.
                return _empty_table_with_schema(
                    pa,
                    path,
                    hdu,
                    output_cols,
                    decode_bytes,
                    include_fits_metadata,
                )
        # Preserve the projected schema *and* its FITS metadata: the
        # projected columns have no data rows, so the header is the only
        # place that metadata can come from (TE-005).
        return _empty_table_with_schema(
            pa, path, hdu, output_cols, decode_bytes, include_fits_metadata
        )
    # FITS writes logical values as T/F, and users reasonably also type
    # TRUE/1. Field metadata is attached afterwards: `from_arrays(names=...)`
    # alone would leave a zero-row result describing its columns less
    # completely than a result with rows (TE-005).
    return _attach_fits_metadata(
        pa,
        pa.Table.from_arrays(arrays, names=names_out),
        header,
        include_fits_metadata,
    )


# Arrow has no bool-vs-string kernel, so an uncoerced literal used to escape as a
# raw `pyarrow.lib.ArrowNotImplementedError: Function 'equal' has no kernel
# matching input types (bool, string)` and made boolean filtering impossible.
# The C++ pushdown engine already accepts these spellings.
_BOOLEAN_LITERALS = {
    "true": True,
    "t": True,
    "yes": True,
    "false": False,
    "f": False,
    "no": False,
}


def _boolean_literal(value: Any, column_name: str) -> bool:
    """Coerce a where literal for a logical (TFORM 'L') column."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    if isinstance(value, str):
        found = _BOOLEAN_LITERALS.get(value.strip().lower())
        if found is not None:
            return found
    raise ValueError(
        f"where compares logical column '{column_name}' with {value!r}, which "
        "is not a logical value; use TRUE/FALSE, T/F or 0/1"
    )


def _aligned_scalar(
    pa: Any, column: Any, value: Any, column_name: str = "", op: str = "=="
) -> tuple[str, Any]:
    """Build an Arrow scalar that matches the C++ pushdown comparison rule.

    Both engines must select identical rows for a given predicate. The C++
    mmap-filtered scan casts the literal to the column's storage precision
    before comparing, so the Arrow engine must do the same for float32
    columns instead of comparing against a float64 scalar (which would drop
    rows whose stored float32 equals the literal). Logical columns take a
    coerced bool so Arrow has a matching kernel.

    Integer columns compare exactly: a float literal becomes an integer bound
    and the scalar takes the column's type (an untyped scalar cannot hold a
    uint64 value >= 2**63, and int64 vs a float scalar casts the column to
    float). A literal outside the column's range answers the same for every
    row. Returns ``(op, scalar)``, or ``("const", bool)`` for such a constant.
    """
    if pa.types.is_boolean(column.type):
        return op, pa.scalar(_boolean_literal(value, column_name), type=column.type)
    if pa.types.is_floating(column.type):
        # float() also reads numeric words (inf, nan) like the torch mask.
        return op, pa.scalar(float(value), type=column.type)
    if pa.types.is_integer(column.type) and isinstance(value, (int, float)):
        op, value = _integer_literal(op, value)
        if op == "const":
            return op, value
        if not isinstance(value, bool):
            import numpy as np

            info = np.iinfo(column.type.to_pandas_dtype())
            if value < int(info.min):
                return "const", op in {">", ">=", "!="}
            if value > int(info.max):
                return "const", op in {"<", "<=", "!="}
            return op, pa.scalar(value, type=column.type)

    return op, pa.scalar(value)


def _where_mask_for_table(
    table: Any, where: str, parsed_ast: Any = None
) -> "np.ndarray":
    pa = _require_pyarrow()
    import pyarrow.compute as _pc

    pc: Any = _pc

    ast = parsed_ast if parsed_ast is not None else parse_where_expression(where)

    def _get_predicate_column(column_name: str) -> Any:
        if column_name not in table.column_names:
            raise ValueError(f"where references unknown column '{column_name}'")

        column = table[column_name]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            raise ValueError(f"where does not support list/VLA column '{column_name}'")
        if pa.types.is_fixed_size_list(column.type):
            raise ValueError(
                f"where does not support fixed-size vector column '{column_name}'"
            )
        return column

    def _exclude_float_nan(column: Any, mask: Any) -> Any:
        # Arrow comparison kernels follow IEEE: NaN != x is true. SQL
        # three-valued logic makes every comparison with NaN unknown.
        if not pa.types.is_floating(column.type):
            return mask
        return pc.and_(mask, pc.invert(pc.is_nan(column)))

    def _compare(column: Any, column_name: str, op: str, literal: Any) -> Any:
        """Nullable comparison mask; third-party errors become ValueError."""
        kernels = {
            "==": pc.equal,
            "!=": pc.not_equal,
            ">": pc.greater,
            ">=": pc.greater_equal,
            "<": pc.less,
            "<=": pc.less_equal,
        }
        if op not in kernels:
            raise ValueError(f"Unsupported where operator '{op}'")
        try:
            kind, scalar = _aligned_scalar(pa, column, literal, column_name, op)
            if kind == "const":
                # Same answer for every row; a null row stays unknown.
                return pc.if_else(
                    pc.is_valid(column),
                    pa.scalar(bool(scalar)),
                    pa.scalar(None, pa.bool_()),
                )
            return kernels[kind](column, scalar)
        except (
            pa.ArrowNotImplementedError,
            pa.ArrowInvalid,
            OverflowError,
            TypeError,
        ) as exc:
            # Never let a third-party error type escape: name the column and
            # operator so the caller can see what could not be compared.
            raise ValueError(
                f"where cannot compare column '{column_name}' ({column.type}) "
                f"with {literal!r} using '{op}': {exc}"
            ) from exc

    def _cmp_mask(column_name: str, op: str, literal: Any) -> Any:
        column = _get_predicate_column(column_name)

        if literal is None:
            if (
                pa.types.is_integer(column.type)
                or pa.types.is_floating(column.type)
                or pa.types.is_boolean(column.type)
            ):
                # Documented dialect and reference evaluator: numeric FITS
                # nulls are TNULLn / NaN, tested with IS NULL, not == / !=.
                raise ValueError(
                    f"where cannot compare numeric column '{column_name}' with "
                    f"NULL using '{op}'; use '{column_name} IS NULL' or "
                    f"'{column_name} IS NOT NULL'"
                )
            if op == "==":
                return pc.is_null(column)
            if op == "!=":
                return pc.invert(pc.is_null(column))
            raise ValueError("where comparisons with null only support == and !=")

        return _exclude_float_nan(column, _compare(column, column_name, op, literal))

    def _in_mask(column_name: str, literals: list[Any], negate: bool) -> Any:
        column = _get_predicate_column(column_name)
        non_null = [v for v in literals if v is not None]
        has_null = any(v is None for v in literals)

        if non_null and (
            pa.types.is_integer(column.type) or pa.types.is_floating(column.type)
        ):
            # Membership is equality with any listed value. Reusing the exact
            # comparison keeps a literal outside the column's range (or a
            # float on an integer column) a plain non-match, where an Arrow
            # value set would overflow or refuse the cast.
            mask = None
            for value in non_null:
                part = _compare(column, column_name, "==", value)
                mask = part if mask is None else pc.or_kleene(mask, part)
        elif non_null:
            if pa.types.is_boolean(column.type):
                value_set = pa.array(
                    [_boolean_literal(v, column_name) for v in non_null],
                    type=column.type,
                )
            elif pa.types.is_float32(column.type):
                value_set = pa.array([float(v) for v in non_null], type=column.type)
            else:
                value_set = _pa_array(pa, non_null)
            mask = pc.is_in(column, value_set=value_set)
        else:
            mask = _pa_array(pa, [False] * int(len(column)))

        if has_null:
            mask = pc.or_(pc.fill_null(mask, False), pc.is_null(column))
        # ``pc.is_in`` is a membership test rather than a comparison: it
        # answers a non-null False for a null input (unlike pc.equal, which
        # answers null). Left as-is, a surrounding `not` inverts that False
        # back into a match, so NOT (A IN (...)) would select the very rows
        # A NOT IN excludes. Re-mark the null rows as null -- not as False --
        # so the mask stays correct in both directions and the null is
        # resolved once, at the top of _where_mask_for_table.
        mask = pc.if_else(pc.is_valid(column), mask, pa.nulls(len(column), pa.bool_()))
        # NaN is a non-null value that IEEE calls "not in" every finite set;
        # SQL three-valued logic makes that match unknown, so exclude it in the
        # negated direction too.
        if negate:
            return _exclude_float_nan(column, pc.invert(mask))
        return mask

    def _between_mask(column_name: str, low: Any, high: Any, negate: bool) -> Any:
        column = _get_predicate_column(column_name)
        if low is None or high is None:
            raise ValueError("where BETWEEN does not support NULL bounds")
        ge = _compare(column, column_name, ">=", low)
        le = _compare(column, column_name, "<=", high)
        mask = pc.and_kleene(ge, le)
        # Nullable in both directions, for the same reason as _in_mask: the
        # null is resolved once, at the top, not here. A NaN is not inside the
        # interval under IEEE, so negation must drop it as well.
        if negate:
            return _exclude_float_nan(column, pc.invert(mask))
        return mask

    def _isnull_mask(column_name: str, negate: bool) -> Any:
        column = _get_predicate_column(column_name)
        # A float NaN counts as NULL, as in the reference evaluator.
        mask = pc.is_null(column, nan_is_null=True)
        mask = pc.fill_null(mask, False)
        if negate:
            return pc.invert(mask)
        return mask

    def _subtree_has_comparison(node: Any) -> bool:
        kind = node[0]
        if kind in {"cmp", "in", "between"}:
            return True
        if kind in {"and", "or"}:
            return _subtree_has_comparison(node[1]) or _subtree_has_comparison(node[2])
        if kind == "not":
            return _subtree_has_comparison(node[1])
        return False

    def _eval(node: Any) -> Any:
        """Evaluate one AST node under SQL three-valued logic.

        Every level returns a **nullable** mask and the single
        ``fill_null(..., False)`` below the tree is the only place nulls are
        resolved: an unknown (null) row is not selected. ``and_kleene`` /
        ``or_kleene`` implement Kleene logic (plain ``and_`` / ``or_``
        propagate null, so ``A IS NULL OR A > 3`` lost the null rows).

        Resolving nulls *inside* each node instead -- what this used to do --
        destroyed the information ``not`` needs: ``fill_null(equal, False)``
        turns "unknown" into "false", so ``NOT (X == 5)`` then selected the
        null row that ``X != 5`` excluded, and the NumPy reference evaluator
        disagreed. The dialect is unchanged (a null is excluded from every
        negated form, as announced in 1.1.0); only the engine now follows it.
        """
        kind = node[0]
        if kind == "cmp":
            return _cmp_mask(node[1], node[2], node[3])
        if kind == "in":
            return _in_mask(node[1], node[2], bool(node[3]))
        if kind == "between":
            return _between_mask(node[1], node[2], node[3], bool(node[4]))
        if kind == "isnull":
            return _isnull_mask(node[1], bool(node[2]))
        if kind == "and":
            return pc.and_kleene(_eval(node[1]), _eval(node[2]))
        if kind == "or":
            return pc.or_kleene(_eval(node[1]), _eval(node[2]))
        if kind == "not":
            # Comparisons already fill unknown to False, so a bare invert would
            # select the NaN rows (NOT false). Drop rows whose referenced
            # columns are null or NaN, matching NOT (X == v) ≡ X != v.
            inverted = pc.invert(pc.fill_null(_eval(node[1]), False))
            if _subtree_has_comparison(node[1]):
                for name in where_columns_from_ast(node[1]):
                    if name not in table.column_names:
                        continue
                    column = table[name]
                    valid = pc.is_valid(column)
                    if pa.types.is_floating(column.type):
                        not_nan = pc.fill_null(pc.invert(pc.is_nan(column)), True)
                        valid = pc.and_(valid, not_nan)
                    inverted = pc.and_(inverted, pc.fill_null(valid, False))
            return pc.fill_null(inverted, False)
        raise ValueError("Invalid where AST")

    return pc.fill_null(_eval(ast), False)  # type: ignore[no-any-return]


def _filter_table_with_where(pa: Any, table: Any, where: str) -> Any:
    # table.filter preserves schema for empty (all-false) masks; skip a sum(mask) pass.
    mask = _where_mask_for_table(table, where)
    return table.filter(mask)


# Predicate column codes the native scan compares (scalar numeric cells).
_NATIVE_PREDICATE_CODES = frozenset({"B", "I", "J", "K", "E", "D"})
_NATIVE_INTEGER_CODES = frozenset({"B", "I", "J", "K"})


def _native_layout_ok(header: Any, col: Any) -> bool:
    """Whether the native mmap gather serves *col* exactly as an unfiltered read.

    It copies raw cells (byte-swapped, with the unsigned / signed-byte TZERO
    conventions applied), so it cannot serve VLA cells or a column that the
    readers scale: any TSCAL/TZERO beyond those conventions, or a TNULL next
    to identity TSCAL/TZERO cards (those decode TNULL to NaN).
    """
    if col.tform_info.vla:
        return False
    has_scale_cards = (
        header.get(f"TSCAL{col.index}") is not None
        or header.get(f"TZERO{col.index}") is not None
    )
    if not has_scale_cards:
        return True
    offset = fits_schema._UNSIGNED_TZERO.get(col.tform_info.code)
    return col.tscal == 1.0 and offset is not None and col.tzero == float(offset)


def _native_where_filters(
    header: Any,
    where: str,
    output_columns: list[str],
    *,
    apply_fits_nulls: bool,
) -> Optional[list[tuple[str, str, Any]]]:
    """Native filter triples equivalent to *where*, or None to use the evaluator.

    Only a conjunction of comparisons (``AND`` / ``BETWEEN``) of a scalar
    numeric column with an int/float literal is pushed down; string, logical,
    vector, scaled and VLA predicates, ``OR`` / ``IN`` / ``NOT`` / ``IS NULL``
    and every other literal stay on the Arrow evaluator, which defines the
    dialect. Float literals on integer columns become the exact integer bound
    the evaluator uses (:func:`_integer_literal`). With ``apply_fits_nulls`` a
    TNULL cell is NULL and never satisfies a comparison, so each predicate
    column with a TNULL also gets a ``!= TNULL`` filter (in the decoded domain,
    as :func:`fits_schema.column_null_sentinel_map` defines it).

    *output_columns* must all be gathered natively too: a column the native
    gather cannot serve exactly (VLA, scaled) sends the read to the evaluator.
    """
    if header is None:
        return None
    if str(header.get("XTENSION", "")).strip().upper() != "BINTABLE":
        return None
    predicates = _compile_where_to_simple_predicates(where)
    if not predicates:
        return None
    by_name = {col.name: col for col in fits_schema.iter_table_columns(header)}
    for name in output_columns:
        col = by_name.get(name)
        if col is None or not _native_layout_ok(header, col):
            return None

    sentinels = fits_schema.column_null_sentinel_map(header) if apply_fits_nulls else {}
    filters: list[tuple[str, str, Any]] = []
    null_filtered: set[str] = set()
    for name, op, literal in predicates:
        col = by_name.get(name)
        if col is None:
            return None  # the evaluator raises the unknown-column error
        info = col.tform_info
        if (
            info.code not in _NATIVE_PREDICATE_CODES
            or info.repeat != 1
            or col.tdim
            or not _native_layout_ok(header, col)
        ):
            return None
        if isinstance(literal, bool) or not isinstance(literal, (int, float)):
            return None
        is_int_col = info.code in _NATIVE_INTEGER_CODES
        if isinstance(literal, float):
            if literal != literal or literal in (float("inf"), float("-inf")):
                return None
            if is_int_col:
                op, literal = _integer_literal(op, literal)
                if op == "const":
                    return None
        if isinstance(literal, int) and not (-(2**63) <= literal < 2**64):
            return None  # the binding takes int64 / uint64 integer literals
        if is_int_col and isinstance(literal, int):
            offset = fits_schema._UNSIGNED_TZERO.get(info.code)
            unsigned64 = (
                info.code == "K" and offset is not None and col.tzero == float(offset)
            )
            if not unsigned64 and literal >= 2**63:
                return None
        filters.append((name, op, literal))

        sentinel = sentinels.get(name)
        if sentinel is None or name in null_filtered:
            continue
        if not is_int_col or isinstance(sentinel, bool):
            # A float column's TNULL (non-standard) and NaN sentinels follow
            # the evaluator's own null rules; leave them to it.
            return None
        if not isinstance(sentinel, int) or not (-(2**63) <= sentinel < 2**64):
            return None
        filters.append((name, "!=", sentinel))
        null_filtered.add(name)
    return filters


def _native_where_output_columns(
    header: Any, columns: Optional[list[str]]
) -> list[str]:
    """The columns a native filtered read returns: the projection, else all."""
    if columns:
        return list(columns)
    return [col.name for col in _all_table_columns(header)]


def _native_read_failed(exc: BaseException) -> None:
    """Re-raise a native filtered-read failure that a fallback must not mask."""
    message = str(exc)
    if (
        "truncat" in message.lower()
        or "Invalid table geometry" in message
        or "Invalid VLA geometry" in message
    ):
        try:
            raise exc
        finally:
            del exc  # no exception -> traceback -> frame cycle (see _read_strategies)
    logger.debug("native WHERE read failed; using the evaluator: %s", exc)


def _try_native_where_read(
    *,
    pa: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    row_slice: Optional[slice | tuple[int, int]],
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    header: Any,
    apply_fits_nulls: bool,
    include_fits_metadata: bool,
) -> Any | None:
    """One native, torch-free filtered read of the row window, as Arrow.

    The predicate is evaluated natively over the window only (filter-within-
    window), the selected rows come back as raw buffers, and the Arrow
    conversion is the one the unfiltered raw read uses, so string decoding
    (``DecodeSpec``), TNULL nulls, unsigned columns and FITS metadata match an
    unfiltered read of the same rows. Returns None when the request is not
    native-eligible or the native reader refuses it; the caller then runs the
    Arrow evaluator.
    """
    from .utils import _normalize_row_slice

    if header is None:
        return None
    out_cols = _native_where_output_columns(header, columns)
    if not out_cols:
        return None
    filters = _native_where_filters(
        header, where, out_cols, apply_fits_nulls=apply_fits_nulls
    )
    if filters is None:
        return None
    start_row, num_rows = _normalize_row_slice(row_slice)
    if num_rows == 0:
        return None

    import torchfits._C as cpp

    try:
        chunk = cpp.read_fits_table_filtered_raw(
            path, hdu, out_cols, filters, start_row, num_rows
        )
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        _native_read_failed(exc)
        return None

    from ._read_scan import _raw_chunk_converter

    convert = _raw_chunk_converter(
        path,
        hdu,
        columns,
        header,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
    )
    return pa.Table.from_batches([convert(chunk)])


def _read_table_with_where(
    *,
    pa: Any,
    path: str,
    hdu: int,
    columns: Optional[list[str]],
    row_slice: Optional[slice | tuple[int, int]],
    rows: Optional[list[int]],
    where: str,
    batch_size: int,
    mmap: bool,
    decode_bytes: bool,
    encoding: str,
    strip: bool,
    include_fits_metadata: bool,
    apply_fits_nulls: bool,
    backend: str,
    header: Any = None,
) -> Any:
    import torchfits

    header_ok = False
    hdr: Any = {}
    n_rows = 0
    if header is not None:
        hdr = header
        try:
            n_rows = int(hdr.get("NAXIS2", 0))
        except (TypeError, ValueError):
            n_rows = 0
        header_ok = True
    else:
        try:
            hdr = torchfits.read_header(path, hdu)
            n_rows = int(hdr.get("NAXIS2", 0))
            header_ok = True
        except (ValueError, TypeError):
            n_rows = 0

    plan = (
        choose_where_read_plan(
            header=hdr,
            header_ok=header_ok,
            columns=columns,
            backend=backend,
            n_rows=n_rows,
            mmap=mmap,
        )
        if header_ok
        else WhereReadPlan(
            strategy=WhereStrategy.ARROW_FILTER,
            cpp_pushdown_safe=False,
            unfiltered_backend=backend,
        )
    )

    if plan.strategy == WhereStrategy.CPP_PUSHDOWN and rows is None:
        # Native, torch-free pushdown over the row window (filter-within-
        # window); a rows= scatter selection stays on the evaluator below.
        pushed = _try_native_where_read(
            pa=pa,
            path=path,
            hdu=hdu,
            columns=columns,
            where=where,
            row_slice=row_slice,
            decode_bytes=decode_bytes,
            encoding=encoding,
            strip=strip,
            header=hdr if header_ok else None,
            apply_fits_nulls=apply_fits_nulls,
            include_fits_metadata=include_fits_metadata,
        )
        if pushed is not None:
            return pushed

    torch_filtered = _try_torch_tensor_where_filter(
        pa=pa,
        path=path,
        hdu=hdu,
        columns=columns,
        where=where,
        row_slice=row_slice,
        rows=rows,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        header=hdr if header_ok else None,
        apply_fits_nulls=apply_fits_nulls,
        include_fits_metadata=include_fits_metadata,
    )
    if torch_filtered is not None:
        return torch_filtered

    # Read output ∪ predicate columns so WHERE can reference unprojected columns,
    # then drop hidden columns after filtering.
    read_columns = columns
    drop_after: list[str] = []
    if columns is not None:
        try:
            where_cols = where_columns_from_ast(parse_where_expression(where))
        except ValueError:
            where_cols = []
        if where_cols:
            seen = set(columns)
            merged = list(columns)
            for name in where_cols:
                if name not in seen:
                    merged.append(name)
                    drop_after.append(name)
                    seen.add(name)
            read_columns = merged

    # Lazy import: _read_scan depends on _read_where at call time only.
    from ._read_scan import _read_table_unfiltered  # noqa: F811

    base = _read_table_unfiltered(
        path=path,
        hdu=hdu,
        columns=read_columns,
        row_slice=row_slice,
        rows=rows,
        batch_size=batch_size,
        mmap=mmap,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        include_fits_metadata=include_fits_metadata,
        apply_fits_nulls=apply_fits_nulls,
        backend=plan.unfiltered_backend,
        header=hdr if header_ok else None,
    )
    filtered = _filter_table_with_where(pa, base, where)
    if drop_after:
        keep = [name for name in filtered.column_names if name not in set(drop_after)]
        return filtered.select(keep)
    return filtered
