"""ML-native FITS datasets and DataLoader integration.

Provides :class:`torch.utils.data.Dataset` and
:class:`~torch.utils.data.IterableDataset` implementations that wrap
``read_tensor`` / ``table.read``, plus ``fits_collate_fn`` and ``make_loader``.

Basic usage::

    from torchfits.data import FitsImageDataset, make_loader

    ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
    loader = make_loader(ds, batch_size=32, num_workers=4)
    for images, labels in loader:
        ...

See [API Reference — Data Module](api.md#data-module).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .._io_engine.device import move_table_columns, to_device
from .datasets import (
    BandInfo,
    FitsCubeDataset,
    FitsCubeIterableDataset,
    FitsImageDataset,
    FitsImageIterableDataset,
    FitsSpectrumDataset,
    FitsSpectrumIterableDataset,
    FitsStagedCutoutIterableDataset,
    FitsTensorDataset,
    FitsTensorIterableDataset,
    _check_return_label,
    _EpochSeed,
    _EpochStream,
    _resolve_rank_and_world_size,
    discover_bands,
)
from .remote import resolve_local_path

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FitsTableDataset — row-indexable table catalog
# ---------------------------------------------------------------------------


class FitsTableDataset(Dataset[Any]):
    """Map-style dataset for row-indexable FITS binary tables.

    Each ``__getitem__`` returns a ``(row_dict, label)`` tuple where *label*
    is a ``torch.long`` tensor (zero when no ``labels`` are provided). Pass
    ``return_label=False`` to get the row dict alone.
    Supports column projection and ``where=`` pushdown for filtered access.
    **Loads the full filtered table at ``__init__``** by default — small/medium
    catalogs only. ``lazy=True`` instead reads aligned blocks of
    ``block_rows`` rows on demand and keeps the last ``cache_blocks`` of them
    (per process, so per DataLoader worker): memory stays bounded for tables
    that do not fit in RAM, at the cost of slower uniformly random access.
    Use :class:`FitsTableIterableDataset` for streaming passes.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    hdu : int
        Table HDU index (default 1).
    columns : list[str] or None
        Column names to read.  None reads all columns.
    where : str or None
        FITS WHERE expression for row filtering (e.g. ``"MAG < 20"``).
    labels : list[int] or None
        Per-row integer labels (default 0 when omitted).
    transform : callable or None
        Optional transform applied to the row dict.
    device : str
        Torch device for tensors.
    mmap : bool or str
        Mmap policy.
    return_label : bool or None
        ``None`` (default) and ``True`` return ``(row_dict, label)``;
        ``False`` returns the row dict alone (and refuses ``labels``).
    lazy : bool
        Read rows on demand in cached blocks instead of loading the table at
        ``__init__`` (default False). With ``where=`` only the selected row
        numbers are kept in memory. ``mmap="auto"`` reads blocks with the
        buffered reader (one mmap per small block costs more than a pread).
    block_rows : int
        Rows per cached block in lazy mode (default 256).
    cache_blocks : int
        Blocks kept per process in lazy mode, least recently used evicted
        first (default 64).
    """

    def __init__(
        self,
        path: str,
        hdu: int = 1,
        columns: list[str] | None = None,
        where: str | None = None,
        labels: list[int] | None = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = "auto",
        *,
        return_label: bool | None = None,
        lazy: bool = False,
        block_rows: int = 256,
        cache_blocks: int = 64,
    ) -> None:
        self._return_label = _check_return_label(return_label, labels)
        self.path = resolve_local_path(path)
        self.hdu = hdu
        self.columns = columns
        self.where = where
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.lazy = bool(lazy)
        self._lazy_blocks: _TableBlockCache | None = None

        if self.lazy:
            self._data: dict[str, Any] = {}
            self._lazy_blocks = _TableBlockCache(
                self.path,
                hdu=self.hdu,
                columns=self.columns,
                where=self.where,
                device=self.device,
                mmap=self.mmap,
                block_rows=block_rows,
                cache_blocks=cache_blocks,
            )
            self._n_rows = len(self._lazy_blocks)
            self._init_labels(labels)
            return

        # Read all data once and index by row. The native table reader takes
        # an HDU index and reports a missing one as a bare RuntimeError: look
        # an EXTNAME up first and translate a missing HDU or file into the
        # typed errors every other reader raises (as lazy=True already does).
        from .._io_engine._hdu_lookup import call_with_lookup, resolve_hdu_name

        read_hdu = (
            resolve_hdu_name(self.path, self.hdu)
            if isinstance(self.hdu, str)
            else self.hdu
        )
        self._data = call_with_lookup(
            self.path,
            read_hdu,
            lambda: _eager_table_columns(
                self.path,
                hdu=read_hdu,
                columns=self.columns,
                where=self.where,
                device=self.device,
                mmap=self.mmap,
            ),
        )
        # Two authorities, and which one applies depends on the read:
        #   * the data itself, when a ``where=`` filter or a projection means
        #     the materialised columns are the whole truth;
        #   * the table's own NAXIS2, when the read is unfiltered. A
        #     zero-column BINTABLE materialises no columns at all, so the
        #     data route reported 0 rows for a table torchfits itself opens
        #     as ``num_rows = 6`` -- and ``ds[0]`` then handed back an empty
        #     dict instead of raising IndexError.
        import torchfits as _tf

        data_rows = 0
        for v in self._data.values():
            if isinstance(v, (torch.Tensor, list)):
                data_rows = len(v)
                break
        self._n_rows = (
            data_rows if self._data else int(_tf.read_nrows(self.path, read_hdu))
        )
        self._init_labels(labels)

    def _init_labels(self, labels: list[int] | None) -> None:
        if labels is not None:
            if len(labels) != self._n_rows:
                raise ValueError(
                    f"labels length {len(labels)} != n_rows {self._n_rows}"
                )
            self._labels: list[int] | None = list(labels)
        else:
            self._labels = None

    def __len__(self) -> int:
        return self._n_rows

    def __getitem__(self, idx: int) -> Any:
        if not -self._n_rows <= idx < self._n_rows:
            raise IndexError(f"index {idx} out of range for {self._n_rows} rows")
        if self._lazy_blocks is not None:
            row = self._lazy_blocks.row(idx % self._n_rows)
        else:
            row = {k: v[idx] for k, v in self._data.items()}
        if self.transform is not None:
            row = self.transform(row)
        if not self._return_label:
            return row
        label_val = 0 if self._labels is None else self._labels[idx]
        label = torch.tensor(label_val, dtype=torch.long)
        return row, label

    def __repr__(self) -> str:
        lazy = ", lazy=True" if self.lazy else ""
        return (
            f"FitsTableDataset(path={self.path!r}, hdu={self.hdu}, "
            f"n_rows={self._n_rows}{lazy})"
        )


class _TableBlockCache:
    """Rows of one table HDU read on demand in aligned blocks, LRU-cached.

    A block is ``block_rows`` consecutive table rows read with one native
    call; the last ``cache_blocks`` blocks stay in memory. The cache belongs
    to one process: a forked DataLoader worker starts from an empty cache
    (and a fresh lock) instead of sharing the parent's, and pickling (spawn
    workers) drops it. With ``where=`` the selected table row numbers are
    resolved once, a window at a time, so only those indices are resident.
    """

    _WHERE_WINDOW_ROWS = 65536

    def __init__(
        self,
        path: str,
        *,
        hdu: int,
        columns: list[str] | None,
        where: str | None,
        device: str,
        mmap: bool | str,
        block_rows: int,
        cache_blocks: int,
    ) -> None:
        for name, value in (("block_rows", block_rows), ("cache_blocks", cache_blocks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        import torchfits

        self.path = path
        self.hdu = hdu
        self.columns = list(columns) if columns else []
        self.device = device
        # One mmap of the whole file per small block costs more than a
        # pread of the block, so "auto" means the buffered reader here.
        self.use_mmap = mmap if isinstance(mmap, bool) else False
        _resolve_table_mmap(mmap)  # validates the value
        self.block_rows = block_rows
        self.cache_blocks = cache_blocks
        self.total_rows = int(torchfits.read_nrows(path, hdu))
        self.row_index: torch.Tensor | None = None
        if where is not None:
            self.row_index = self._where_rows(where)
        self._reset()

    def _reset(self) -> None:
        import os
        import threading
        from collections import OrderedDict

        self._blocks: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in ("_blocks", "_lock", "_pid"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._reset()

    def __len__(self) -> int:
        if self.row_index is not None:
            return int(self.row_index.numel())
        return self.total_rows

    def _where_rows(self, where: str) -> torch.Tensor:
        """0-based table rows selected by *where*, resolved a window at a time."""
        import torchfits
        from .._table._read_schema import _all_table_columns
        from .._table._read_where import _where_mask_for_table
        from .._where import parse_where_expression, where_columns_from_ast

        header = torchfits.read_header(self.path, self.hdu)
        pred_cols = where_columns_from_ast(parse_where_expression(where))
        known = {col.name for col in _all_table_columns(header)}
        for name in pred_cols:
            if name not in known:
                raise KeyError(f"Column '{name}' not found")
        picked: list[torch.Tensor] = []
        step = self._WHERE_WINDOW_ROWS
        for start in range(0, self.total_rows, step):
            n = min(step, self.total_rows - start)
            chunk = _native_table_read(
                self.path, self.hdu, pred_cols, self.use_mmap, window=(start, n)
            )
            if chunk:
                mask = _where_mask_for_chunk(
                    self.path, self.hdu, header, chunk, where, pred_cols
                )
            else:
                import numpy as np
                import torchfits.table

                pa_mask: Any = _where_mask_for_table(
                    torchfits.table.read(
                        self.path,
                        hdu=self.hdu,
                        columns=pred_cols,
                        row_slice=(start, start + n),
                        mmap=self.use_mmap,
                    ),
                    where,
                )
                mask = torch.from_numpy(
                    np.asarray(pa_mask.to_numpy(zero_copy_only=False), dtype=bool)
                )
            picked.append(mask.nonzero().flatten() + start)
        if not picked:
            return torch.zeros(0, dtype=torch.int64)
        return torch.cat(picked)

    def _read_block(self, block: int) -> dict[str, Any]:
        start = block * self.block_rows
        n = min(self.block_rows, self.total_rows - start)
        chunk = _native_table_read(
            self.path, self.hdu, self.columns, self.use_mmap, window=(start, n)
        )
        if chunk:
            return _move_table_chunk(chunk, self.device)
        return _pyarrow_table_columns(
            self.path,
            hdu=self.hdu,
            columns=self.columns or None,
            where=None,
            use_mmap=self.use_mmap,
            device=self.device,
            row_slice=(start, start + n),
        )

    def row(self, idx: int) -> dict[str, Any]:
        """Row *idx* (0-based, after any ``where=`` selection)."""
        import os

        if self._pid != os.getpid():
            self._reset()  # forked: never reuse the parent's cache or lock
        table_row = idx if self.row_index is None else int(self.row_index[idx])
        block, offset = divmod(table_row, self.block_rows)
        with self._lock:
            chunk = self._blocks.get(block)
            if chunk is not None:
                self._blocks.move_to_end(block)
        if chunk is None:
            # Read outside the lock: a slow block must not stall cache hits
            # from other threads. Two threads missing the same block both
            # read it; the second insert simply refreshes it.
            chunk = self._read_block(block)
            with self._lock:
                self._blocks[block] = chunk
                self._blocks.move_to_end(block)
                while len(self._blocks) > self.cache_blocks:
                    self._blocks.popitem(last=False)
        return {k: v[offset] for k, v in chunk.items()}


def _resolve_table_mmap(mmap: bool | str) -> bool:
    """Resolve ``mmap=`` the way image reads do: bool, or the string ``auto``."""
    if isinstance(mmap, str):
        if mmap.strip().lower() != "auto":
            raise ValueError("mmap must be bool or 'auto'")
        return True
    if not isinstance(mmap, bool):
        raise ValueError("mmap must be bool or 'auto'")
    return mmap


def _move_table_chunk(chunk: dict[str, Any], device: str) -> dict[str, Any]:
    # VLA columns (lists of row tensors) follow their sibling columns, as
    # they do in scan_torch / FitsTableIterableDataset.
    return move_table_columns(chunk, device)


def _eager_table_columns(
    path: str,
    *,
    hdu: int,
    columns: list[str] | None,
    where: str | None,
    device: str,
    mmap: bool | str,
) -> dict[str, Any]:
    """Load a full table into columnar storage (tensors or python lists).

    ``where=`` selects rows but never changes the representation: the
    selected rows come back exactly as an unfiltered read returns them
    (stored TNULL sentinels, raw ``uint8`` character bytes, VLA columns as
    lists of tensors -- the representation ``FitsTableIterableDataset``
    yields). Only when the native reader rejects the table does it fall back
    to pyarrow (logged at WARNING), where character columns become ``str``.
    """
    use_mmap = _resolve_table_mmap(mmap)
    col_list = list(columns) if columns else []
    pred_cols: list[str] = []
    drop: set[str] = set()
    header: Any = None
    if where is not None:
        # One read either way: the native predicate scan when it can serve the
        # request, else the projection plus the predicate columns, masked in
        # memory (the predicate columns used to be read a second time).
        import torchfits
        from .._table._read_schema import _all_table_columns
        from .._where import parse_where_expression, where_columns_from_ast

        header = torchfits.read_header(path, hdu)
        pred_cols = where_columns_from_ast(parse_where_expression(where))
        known = {col.name for col in _all_table_columns(header)}
        for name in pred_cols:
            if name not in known:
                raise KeyError(f"Column '{name}' not found")
        pushed = _native_filtered_columns(path, hdu, header, col_list, where, use_mmap)
        if pushed is not None:
            return _move_table_chunk(pushed, device)
        if col_list:
            drop = {name for name in pred_cols if name not in col_list}
            col_list = list(dict.fromkeys(col_list + pred_cols))
    chunk = _native_table_read(path, hdu, col_list, use_mmap)
    if chunk:
        if where is not None:
            row_mask = _where_mask_for_chunk(path, hdu, header, chunk, where, pred_cols)
            if drop:
                chunk = {k: v for k, v in chunk.items() if k not in drop}
            chunk = _select_table_rows(chunk, row_mask)
        return _move_table_chunk(chunk, device)
    return _pyarrow_table_columns(
        path, hdu=hdu, columns=columns, where=where, use_mmap=use_mmap, device=device
    )


def _native_table_read(
    path: str,
    hdu: int,
    col_list: list[str],
    use_mmap: bool,
    window: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Columns of the table (or of a ``(start, num_rows)`` window, 0-based start)
    from the native reader, or None when it rejects the table."""
    import torchfits._C as cpp
    from torchfits._table._read_scan import _reraise_invalid_geometry

    def read(mmap: bool) -> Any:
        if window is None:
            return cpp.read_fits_table(path, hdu, col_list, mmap)
        return cpp.read_fits_table_rows(
            path, hdu, col_list, window[0] + 1, window[1], mmap
        )

    # Only the native reader's own failures (RuntimeError / OSError, the
    # family _read_scan catches) select a fallback; anything else -- a
    # MemoryError, a binding TypeError -- is a bug and must surface.
    try:
        return dict(read(use_mmap))
    except (RuntimeError, OSError) as exc:
        _reraise_invalid_geometry(exc)
        failure: BaseException | None = exc
        if use_mmap:
            # The mmap reader refuses some layouts (e.g. VLA columns); the
            # handle reader serves them in the same representation the
            # iterable dataset's scan_torch uses (VLA -> list of tensors).
            _log.debug("mmap table read failed for %r, retrying: %s", path, exc)
            try:
                return dict(read(False))
            except (RuntimeError, OSError) as retry_exc:
                _reraise_invalid_geometry(retry_exc)
                failure = retry_exc
        _log.warning(
            "native table read failed for %r (%s); falling back to pyarrow, "
            "which returns character columns as str instead of uint8 bytes",
            path,
            failure,
        )
    return None


def _pyarrow_table_columns(
    path: str,
    *,
    hdu: int,
    columns: list[str] | None,
    where: str | None,
    use_mmap: bool,
    device: str,
    row_slice: tuple[int, int] | None = None,
) -> dict[str, Any]:
    import torchfits.table
    import pyarrow.types as pt

    pa_table = torchfits.table.read(
        path,
        hdu=hdu,
        columns=columns,
        where=where,
        row_slice=row_slice,
        mmap=use_mmap,
    )
    result: dict[str, Any] = {}
    for col_name in pa_table.column_names:
        col = pa_table.column(col_name)
        col_type = col.type
        if (
            pt.is_integer(col_type)
            or pt.is_floating(col_type)
            or pt.is_boolean(col_type)
        ):
            # Writable copy: Arrow buffers are often read-only for torch.from_numpy.
            result[col_name] = torch.as_tensor(
                col.to_numpy(zero_copy_only=False).copy()
            )
            if str(device) != "cpu":
                result[col_name] = to_device(result[col_name], device)
        else:
            result[col_name] = col.to_pylist()
    return result


def _native_filtered_columns(
    path: str,
    hdu: int,
    header: Any,
    columns: list[str],
    where: str,
    use_mmap: bool,
) -> dict[str, Any] | None:
    """*where* served by the native predicate scan, or None when it cannot be.

    The scan (``mmap=True`` only) gathers the selected rows as raw cells, in
    the representation ``read_fits_table`` gives an unfiltered read; requests
    it does not serve exactly (scaled or VLA columns, a predicate beyond a
    conjunction of numeric comparisons) return None for the in-memory mask.
    """
    if not use_mmap:
        return None
    import torchfits._C as cpp
    from .._table._read_where import (
        _native_read_failed,
        _native_where_filters,
        _native_where_output_columns,
    )

    out_cols = _native_where_output_columns(header, columns or None)
    if not out_cols:
        return None
    filters = _native_where_filters(header, where, out_cols, apply_fits_nulls=True)
    if filters is None:
        return None
    try:
        return dict(cpp.read_fits_table_filtered(path, hdu, out_cols, filters, 1, -1))
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        _native_read_failed(exc)
        return None


def _where_mask_for_chunk(
    path: str,
    hdu: int,
    header: Any,
    chunk: dict[str, Any],
    where: str,
    pred_cols: list[str],
) -> torch.Tensor:
    """Boolean row mask for *where* over an already-read columnar chunk.

    The predicate columns are converted to Arrow the way ``table.read``
    converts them (TNULL is NULL, character columns decoded as strict ASCII)
    and evaluated with its full predicate dialect.
    """
    import numpy as np
    import pyarrow as pa

    from .._string_decode import DecodeSpec
    from .._table._read_schema import (
        _build_fits_metadata,
        _column_tforms_for_decode,
        _unsigned_column_dtypes,
    )
    from .._table._read_where import _where_mask_for_table
    from .._table.arrow_convert import _chunk_to_record_batch
    from .._where import parse_where_expression

    pred_set = set(pred_cols)
    try:
        null_meta = _build_fits_metadata(path, hdu, pred_set, header=header)[0]
    except ValueError:
        null_meta = {}
    batch = _chunk_to_record_batch(
        {name: chunk[name] for name in pred_cols if name in chunk},
        True,
        DecodeSpec("ascii", "strict"),
        True,
        null_meta=null_meta,
        apply_fits_nulls=True,
        column_tforms=_column_tforms_for_decode(path, hdu, pred_set, header=header),
        unsigned_dtypes=_unsigned_column_dtypes(path, hdu, pred_set, header=header),
    )
    # A pyarrow BooleanArray (the annotation says ndarray).
    mask: Any = _where_mask_for_table(
        pa.Table.from_batches([batch]), where, parsed_ast=parse_where_expression(where)
    )
    values = mask.to_numpy(zero_copy_only=False)
    return torch.from_numpy(np.asarray(values, dtype=bool).copy())


def _select_table_rows(chunk: dict[str, Any], row_mask: torch.Tensor) -> dict[str, Any]:
    """Keep the rows of a columnar chunk where *row_mask* is True."""
    keep = row_mask.nonzero().flatten()
    picked = keep.tolist()
    out: dict[str, Any] = {}
    for name, value in chunk.items():
        if isinstance(value, torch.Tensor):
            # Advanced indexing, not index_select: torch < 2.14 has no
            # index_select kernel for uint16/uint32/uint64 columns.
            out[name] = value[keep.to(value.device)]
        elif isinstance(value, list):
            out[name] = [value[i] for i in picked]
        else:
            out[name] = value
    return out


def _row_from_torch_chunk(chunk: dict[str, Any], row_idx: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, value in chunk.items():
        if isinstance(value, torch.Tensor):
            row[name] = value[row_idx]
        elif isinstance(value, list):
            row[name] = value[row_idx]
        else:
            row[name] = value
    return row


# ---------------------------------------------------------------------------
# FitsTableIterableDataset — constant-memory table streaming
# ---------------------------------------------------------------------------


def _shard_row_range(
    total_rows: int, slot_id: int, total_slots: int
) -> tuple[int, int]:
    """Contiguous, near-equal ``[start, stop)`` row range for one worker slot.

    Unlike ``batch_idx % num_workers`` this is even when ``batch_size``
    dwarfs the row count, and it lets each worker skip unrelated byte ranges
    instead of decoding them.
    """
    if total_rows <= 0 or total_slots <= 1:
        return 0, max(0, total_rows)
    per_slot = total_rows // total_slots
    remainder = total_rows % total_slots
    start = slot_id * per_slot + min(slot_id, remainder)
    size = per_slot + (1 if slot_id < remainder else 0)
    return start, start + size


class FitsTableIterableDataset(_EpochStream, IterableDataset[Any]):
    """Iterable dataset streaming FITS table rows in constant memory.

    Each DataLoader worker (and each distributed rank) receives a
    **contiguous, near-equal row range** via ``row_slice``, so every row is
    seen exactly once per epoch and no worker decodes rows it will discard.

    Yields one ``dict[str, Tensor]`` per row by default. With
    ``as_batches=True`` it yields the raw ``dict[str, Tensor]`` chunks of
    ``row_slice``/``batch_size`` rows instead — the tensor-space path skips
    Arrow→Python per-row conversion and is markedly faster for wide numeric
    catalogs.

    With ``where=`` the window is applied *before* filtering, so a shard can
    hold a different number of surviving rows; ordering within a shard is
    preserved and no row is duplicated or dropped.
    """

    def __init__(
        self,
        path: str,
        hdu: int = 1,
        columns: list[str] | None = None,
        where: str | None = None,
        batch_size: int = 65536,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = "auto",
        shuffle_buffer_size: int | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        *,
        as_batches: bool = False,
    ) -> None:
        self.path = resolve_local_path(path)
        self.hdu = hdu
        self.columns = columns
        self.where = where
        self.batch_size = batch_size
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self._epoch_seed = _EpochSeed(seed)
        self.rank = rank
        self.world_size = world_size
        self.as_batches = bool(as_batches)

    def _non_numeric_columns(self) -> set[str]:
        """Character / bit columns that tensor-space batches cannot carry.

        ``scan_torch`` hands character columns back as raw ``uint8`` byte
        matrices, which are indistinguishable by dtype from a genuine ``uint8``
        image column. The schema's ``TFORM`` codes are not ambiguous, so use
        them to make ``as_batches=True`` behave the same on both scanner paths.
        """
        try:
            import torchfits

            info = torchfits.read_table_info(self.path, self.hdu)
        except (RuntimeError, OSError, ValueError) as exc:
            _log.warning(
                "could not read the column formats of %r hdu %r (%s); "
                "character columns stay in as_batches chunks as uint8 bytes",
                self.path,
                self.hdu,
                exc,
            )
            return set()
        names = list(info.get("colnames") or [])
        tforms = list(info.get("tforms") or [])
        dropped: set[str] = set()
        for name, tform in zip(names, tforms):
            code = "".join(ch for ch in str(tform) if ch.isalpha())
            if code and code[0].upper() in {"A", "X"}:
                dropped.add(str(name))
        return dropped

    def _row_window(self) -> tuple[int, int] | None:
        """This slot's ``(start, stop)`` row window, or None for the whole table."""
        rank, world_size = _resolve_rank_and_world_size(self.rank, self.world_size)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        total_slots = world_size * num_workers
        slot_id = rank * num_workers + worker_id
        if total_slots <= 1:
            return None

        import torchfits

        total_rows = int(torchfits.read_nrows(self.path, self.hdu))
        start, stop = _shard_row_range(total_rows, slot_id, total_slots)
        return (start, stop)

    def _generate(self) -> Iterator[dict[str, Any]]:
        window = self._row_window()
        if window is not None and window[1] <= window[0]:
            return

        skipped = self._non_numeric_columns() if self.as_batches else set()

        import torchfits.table

        # where= only selects rows: scan_torch reads each chunk's window once
        # and yields just its matching rows, in the exact representation of an
        # unfiltered scan (natively filtered when the predicate allows it).
        # Absolute row of the next chunk, and the end of the scanned range
        # (looked up lazily): a zero-column BINTABLE (TFIELDS = 0) yields
        # ``{}`` chunks that carry no row count, yet holds NAXIS2 rows, which
        # the map-style FitsTableDataset serves as empty row dicts.
        scan_pos = 0 if window is None else window[0]
        scan_end = None if window is None else window[1]
        for chunk in torchfits.table.scan_torch(
            self.path,
            hdu=self.hdu,
            columns=self.columns,
            row_slice=window,
            batch_size=self.batch_size,
            mmap=_resolve_table_mmap(self.mmap),
            device=self.device,
            where=self.where,
        ):
            if not chunk:
                if scan_end is None:
                    scan_end = int(torchfits.read_nrows(self.path, self.hdu))
                # scan_torch chunks are ``min(batch_size, remaining)`` rows.
                n_rows = max(0, min(self.batch_size, scan_end - scan_pos))
                scan_pos += n_rows
                if n_rows == 0:
                    continue
            else:
                n_rows = next(
                    (
                        v.shape[0] if isinstance(v, torch.Tensor) else len(v)
                        for v in chunk.values()
                        if isinstance(v, (torch.Tensor, list))
                    ),
                    0,
                )
                scan_pos += n_rows
            if self.as_batches:
                if skipped:
                    chunk = {k: v for k, v in chunk.items() if k not in skipped}
                if self.transform is not None:
                    chunk = self.transform(chunk)
                yield chunk
                continue
            for row_idx in range(n_rows):
                row = _row_from_torch_chunk(chunk, row_idx)
                if self.transform is not None:
                    row = self.transform(row)
                yield row

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.shuffle_buffer_size is not None and self.shuffle_buffer_size > 1:
            if self.as_batches:
                raise ValueError(
                    "shuffle_buffer_size requires per-row mode (as_batches=False)"
                )
        return super().__iter__()

    def __repr__(self) -> str:
        return (
            f"FitsTableIterableDataset(path={self.path!r}, hdu={self.hdu}, "
            f"batch_size={self.batch_size}, as_batches={self.as_batches})"
        )


# ---------------------------------------------------------------------------
# FitsCutoutDataset — patch training from cutout index table
# ---------------------------------------------------------------------------

CutoutSpec = tuple[str, int, int, int, int, int]
"""(path, hdu, x1, y1, x2, y2) half-open [x1,x2)x[y1,y2) exclusive upper bounds (matches read_subset)."""


class FitsCutoutDataset(Dataset[Any]):
    """Map-style dataset for fixed cutouts from one or more FITS images.

    Each ``__getitem__`` calls ``read_subset`` for one window. NOTE:
    same-path cutouts re-open the file each row; use ``open_subset_reader``
    when one mosaic dominates.

    A window that selects no pixels -- ``size <= 0`` in the 5-tuple form, or
    ``x2 <= x1`` / ``y2 <= y1`` in the 6-tuple form -- is rejected here rather
    than handed to ``read_subset``, which clamps it to an empty region. Such a
    sample is a valid 0-pixel tensor, so it stacks through
    :func:`fits_collate_fn` and reaches the model as a ``(B, 1, 0, 0)`` batch.
    A box running off *one* edge still clamps to the overlap, matching the
    ``torchfits cutout --box`` contract.
    """

    def __init__(
        self,
        cutouts: Sequence[CutoutSpec | tuple[str, int, int, int, int]],
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        add_channel_dim: bool = True,
    ) -> None:
        normalized: list[CutoutSpec] = []
        for spec in cutouts:
            if len(spec) == 5:
                path, hdu, x, y, size = spec
                normalized.append((path, hdu, x, y, x + int(size), y + int(size)))
            elif len(spec) == 6:
                normalized.append(tuple(spec))  # type: ignore[arg-type]
            else:
                raise ValueError(
                    "cutout must be (path, hdu, x, y, size) or "
                    "(path, hdu, x1, y1, x2, y2)"
                )
        if not normalized:
            # The same rule the sibling loaders already enforce:
            # ``_as_hdu_list`` refuses an empty ``hdu`` and ``_resolve_paths``
            # refuses an empty ``paths``, because an empty selection does not
            # fail on its own -- it builds a zero-length dataset, and whether
            # anything notices then depends on whether the caller passed
            # ``shuffle=`` to ``make_loader``.
            raise ValueError("cutouts must be non-empty; got an empty list")
        for path, hdu, x1, y1, x2, y2 in normalized:
            if x1 < 0 or y1 < 0:
                raise ValueError(
                    f"cutout origin must be non-negative; got ({x1}, {y1}) "
                    f"for {path!r} hdu={hdu!r}"
                )
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"cutout {path!r} hdu={hdu!r} selects no pixels: "
                    f"({x1}, {y1}) -> ({x2}, {y2}); a half-open window needs "
                    "x1 < x2 and y1 < y2"
                )
        self.cutouts = normalized
        self.transform = transform
        self.device = device
        self.add_channel_dim = add_channel_dim
        self.files = sorted({c[0] for c in self.cutouts})

    def __len__(self) -> int:
        return len(self.cutouts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from torchfits import read_subset

        path, hdu, x1, y1, x2, y2 = self.cutouts[idx]
        # HTTP(S) uncompressed 2D uses Range cutouts; compressed/vos fall back
        # inside read_subset. Do not force full prefetch here.
        image = read_subset(path, hdu, x1, y1, x2, y2)
        if str(self.device) != "cpu":
            image = to_device(image, self.device)
        if self.add_channel_dim and image.ndim == 2:
            image = image.unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"FitsCutoutDataset(n={len(self.cutouts)}, device={self.device!r})"


# ---------------------------------------------------------------------------
# fits_collate_fn — stack homogeneous tensors from batch of dicts/tensors
# ---------------------------------------------------------------------------


def _collate_dicts(samples: list[Any], prefix: str = "") -> dict[str, Any]:
    """Stack a list of (possibly nested) ``dict[str, Tensor]`` samples per key."""
    out: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        name = f"{prefix}{key}"
        if all(isinstance(v, dict) for v in values):
            out[key] = _collate_dicts(values, prefix=f"{name}.")
            continue
        if not all(isinstance(v, torch.Tensor) for v in values):
            raise ValueError(
                f"Cannot collate non-tensor column {name!r}. "
                f"Drop string/VLA columns or supply a custom collate_fn."
            )
        out[key] = torch.stack(values)
    return out


def fits_collate_fn(
    batch: list[Any],
) -> Any:
    """Collate a list of samples into a batch.

    - Lists of ``dict[str, Tensor]`` are stacked per key; nested dicts (e.g.
      multi-arm spectra ``{arm: {"flux": ...}}``) are collated recursively.
    - Lists of ``Tensor`` are stacked with ``torch.stack``.
    - Lists of ``(dict[str, Tensor], Tensor)`` (table row, label) pairs are
      stacked into ``(dict, labels)``.
    - Lists of ``(Tensor, Tensor)`` (image, label) pairs are stacked into
      ``(images, labels)`` tuples.
    - Ragged / VLA columns raise ``ValueError``.

    Parameters
    ----------
    batch : list
        List of samples from a torchfits dataset.

    Returns
    -------
    Collated batch (torch.Tensor, tuple, or dict).
    """
    if not batch:
        return batch

    first = batch[0]

    if isinstance(first, dict):
        return _collate_dicts(batch)

    if isinstance(first, tuple) and len(first) == 2:
        if isinstance(first[0], dict):
            # (dict[str, Tensor], Tensor) — table rows with labels
            stacked = _collate_dicts([s[0] for s in batch])
            labels = torch.stack([s[1] for s in batch])
            return stacked, labels
        # (Tensor, Tensor) — image, label pairs
        images = torch.stack([s[0] for s in batch])
        labels = torch.stack([s[1] for s in batch])
        return images, labels

    if isinstance(first, torch.Tensor):
        return torch.stack(batch)

    raise TypeError(f"Unsupported sample type for collation: {type(first)}")


# ---------------------------------------------------------------------------
# make_loader — DataLoader with cache optimisation
# ---------------------------------------------------------------------------


def make_loader(
    dataset: Dataset[Any] | IterableDataset[Any],
    batch_size: int | None = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    **loader_kwargs: Any,
) -> DataLoader[Any]:
    """Create a DataLoader with sensible defaults.

    Parameters
    ----------
    dataset : Dataset or IterableDataset
        A torchfits dataset instance.
    batch_size : int or None
        Batch size. ``None`` disables automatic batching: each sample is
        yielded as the dataset returns it (use it with datasets that already
        yield batches, such as ``as_batches=True``).
    shuffle : bool or None
        Whether to shuffle.  Defaults to True for map-style datasets.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Pin memory for faster CPU→GPU transfers.
    prefetch_factor : int
        Prefetch factor for multi-worker loading.
    drop_last : bool
        Drop the last incomplete batch.
    **loader_kwargs :
        Passed through to :class:`torch.utils.data.DataLoader`.

    Returns
    -------
    DataLoader
    """
    # Without automatic batching DataLoader calls collate_fn on each single
    # sample, not on a list of samples, so fits_collate_fn does not apply;
    # keep DataLoader's own per-sample conversion unless one was passed.
    default_collate = fits_collate_fn if batch_size is not None else None
    collate_fn = loader_kwargs.pop("collate_fn", default_collate)

    if shuffle is None:
        # Map-style datasets shuffle by default; IterableDataset must not
        shuffle = not isinstance(dataset, IterableDataset)

    # prefetch_factor is only valid with num_workers > 0
    if num_workers == 0:
        prefetch_factor = None  # type: ignore[assignment]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **loader_kwargs,
    )


__all__ = [
    "BandInfo",
    "CutoutSpec",
    "discover_bands",
    "FitsTensorDataset",
    "FitsTensorIterableDataset",
    "FitsImageDataset",
    "FitsImageIterableDataset",
    "FitsCubeDataset",
    "FitsCubeIterableDataset",
    "FitsSpectrumDataset",
    "FitsSpectrumIterableDataset",
    "FitsTableDataset",
    "FitsTableIterableDataset",
    "FitsCutoutDataset",
    "FitsStagedCutoutIterableDataset",
    "fits_collate_fn",
    "make_loader",
]
