"""Lazy file-backed table handle: metadata-only view with out-of-core streaming."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .header import Header
from .table_hdu import TableHDU


class _TableHDURefDataWrapper:
    def __init__(self, parent: "TableHDURef"):
        self._parent = parent

    def __getitem__(self, key: str) -> Any:
        return self._parent[key]

    def __contains__(self, key: str) -> bool:
        return key in self._parent.columns

    def keys(self):
        return self._parent.columns


class TableHDURef:
    def __init__(
        self,
        *,
        header: Optional[Header] = None,
        source_path: Optional[str] = None,
        source_hdu: Optional[int] = None,
        columns: Optional[List[str]] = None,
        row_slice: Optional[slice | tuple[int, int]] = None,
    ):
        self.header = header or Header()
        self._source_path = source_path
        self._source_hdu = source_hdu
        self._columns = columns[:] if columns else None
        self._row_slice = row_slice
        self._all_columns_cache = None

    def _require_source(self) -> tuple[str, int]:
        if not self._source_path or self._source_hdu is None:
            raise RuntimeError("This TableHDURef is not associated with a FITS file")
        return self._source_path, int(self._source_hdu)

    @property
    def num_rows(self) -> int:
        try:
            total = int(self.header.get("NAXIS2", 0))
        except Exception:
            total = 0
        if total <= 0:
            return 0
        if self._row_slice is None:
            return total
        if isinstance(self._row_slice, tuple):
            start, stop = self._row_slice
        else:
            start = 0 if self._row_slice.start is None else int(self._row_slice.start)
            stop = self._row_slice.stop
        start = int(start)
        if start < 0:
            start = 0
        if stop is None:
            return max(0, total - start)
        stop = int(stop)
        stop = min(stop, total)
        return max(0, stop - start)

    def __len__(self) -> int:
        return self.num_rows

    @property
    def columns(self) -> List[str]:
        if self._columns is not None:
            return list(self._columns)

        if self._all_columns_cache is not None:
            return list(self._all_columns_cache)
        try:
            n = int(self.header.get("TFIELDS", 0))
        except Exception:
            n = 0
        out: List[str] = []
        for i in range(1, n + 1):
            name = self.header.get(f"TTYPE{i}")
            if isinstance(name, str) and name:
                out.append(name)
            else:
                out.append(f"COL{i}")
        self._all_columns_cache = out
        return list(out)

    @property
    def string_columns(self) -> List[str]:
        from ..fits_schema import string_column_names

        selected = set(self._columns) if self._columns is not None else None
        return string_column_names(self.header, selected=selected)

    @property
    def schema(self) -> Dict[str, Any]:
        from ..fits_schema import build_table_schema_dict

        return build_table_schema_dict(
            self.header,
            selected_columns=self._columns,
        )

    def select(self, cols: List[str]) -> "TableHDURef":
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise TypeError("cols must be a list[str]")
        return TableHDURef(
            header=self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
            columns=cols,
            row_slice=self._row_slice,
        )

    def head(self, n: int) -> "TableHDURef":
        if n < 0:
            raise ValueError("n must be >= 0")
        return TableHDURef(
            header=self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
            columns=self._columns,
            row_slice=slice(0, n),
        )

    def filter(self, condition: str) -> "TableHDU":
        return self.materialize().filter(condition)

    def _normalize_row_slice(self, row_slice: Optional[slice | tuple[int, int]]):
        if row_slice is None:
            return 1, -1
        if isinstance(row_slice, tuple):
            if len(row_slice) != 2:
                raise ValueError("row_slice tuple must be (start, stop)")
            start, stop = row_slice
        else:
            start = 0 if row_slice.start is None else row_slice.start
            stop = row_slice.stop
            if row_slice.step not in (None, 1):
                raise ValueError("row_slice step is not supported")
        start = int(start)
        if start < 0:
            raise ValueError("row_slice start must be >= 0")
        if stop is None:
            return start + 1, -1
        stop = int(stop)
        if stop < start:
            return start + 1, 0
        return start + 1, stop - start

    def _is_ascii_table(self) -> bool:
        try:
            return str(self.header.get("XTENSION", "")).strip().upper() == "TABLE"
        except Exception:
            return False

    def read(
        self,
        *,
        columns: Optional[List[str]] = None,
        row_slice: Optional[slice | tuple[int, int]] = None,
        mmap: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        import torchfits

        path, hdu = self._require_source()
        if columns is None:
            columns = self._columns
        if row_slice is None:
            row_slice = self._row_slice
        start_row, num_rows = self._normalize_row_slice(row_slice)
        effective_mmap = bool(mmap)
        if effective_mmap and self._is_ascii_table():
            effective_mmap = False
        return torchfits.read(
            path,
            hdu=hdu,
            columns=columns,
            start_row=start_row,
            num_rows=num_rows,
            mmap=effective_mmap,
            device=device,
            cache_capacity=0,
        )

    def materialize(self, *, mmap: bool = True, device: str = "cpu") -> "TableHDU":
        data = self.read(mmap=mmap, device=device)
        return TableHDU(
            data,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def iter_rows(self, batch_size: int = 65536, *, mmap: bool = True):
        import torchfits

        path, hdu = self._require_source()
        start_row, num_rows = self._normalize_row_slice(self._row_slice)
        effective_mmap = bool(mmap)
        if effective_mmap and self._is_ascii_table():
            effective_mmap = False

        for chunk in torchfits.stream_table(
            path,
            hdu=hdu,
            columns=self._columns,
            start_row=start_row,
            num_rows=num_rows,
            chunk_rows=batch_size,
            mmap=effective_mmap,
        ):
            yield chunk

    def __getitem__(self, col_name: str) -> Any:
        data = self.read(columns=[col_name])
        if col_name not in data:
            raise KeyError(f"Column '{col_name}' not found")
        return data[col_name]

    def get_string_column(
        self, name: str, encoding: str = "ascii", strip: bool = True
    ) -> List[str]:
        value = self[name]
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"Column '{name}' is not a tensor string column")
        if value.dtype != torch.uint8 or value.dim() != 2:
            raise TypeError(
                f"Column '{name}' is not a uint8 (rows,width) encoded string column"
            )
        out: List[str] = []
        arr = value.detach().cpu().numpy()
        for row in arr:
            s = bytes(row.tolist()).decode(encoding, errors="ignore")
            if strip:
                s = s.rstrip(" \x00")
            out.append(s)
        return out

    def get_vla_column(self, name: str) -> List[Tensor]:
        value = self[name]
        if isinstance(value, list):
            return value
        raise KeyError(f"Column '{name}' is not a VLA list")

    def get_vla_lengths(self, name: str) -> List[int]:
        values = self.get_vla_column(name)
        lengths: List[int] = []
        for item in values:
            if isinstance(item, torch.Tensor):
                lengths.append(int(item.numel()))
            elif hasattr(item, "__len__"):
                lengths.append(len(item))
            else:
                lengths.append(1)
        return lengths

    @property
    def vla_lengths(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for col in self.schema.get("vla_columns", []):
            try:
                out[col] = self.get_vla_lengths(col)
            except Exception:
                continue
        return out

    @property
    def data(self):
        return _TableHDURefDataWrapper(self)

    def to_arrow(self, **kwargs):
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.read(
            path, hdu=hdu, columns=self._columns, row_slice=self._row_slice, **kwargs
        )

    def scan_arrow(self, **kwargs):
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.scan(
            path, hdu=hdu, columns=self._columns, row_slice=self._row_slice, **kwargs
        )

    def reader_arrow(self, **kwargs):
        import torchfits

        path, hdu = self._require_source()
        return torchfits.table.reader(
            path, hdu=hdu, columns=self._columns, row_slice=self._row_slice, **kwargs
        )

    def _refresh_file_view(self) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        header = Header(torchfits.get_header(path, hdu))
        return TableHDURef(header=header, source_path=path, source_hdu=hdu)

    def append_rows_file(self, rows: Dict[str, Any]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.append_rows(path, rows, hdu=hdu)
        return self._refresh_file_view()

    def insert_column_file(
        self,
        name: str,
        values: Any,
        *,
        index: Optional[int] = None,
        format: Optional[str] = None,
        unit: Optional[str] = None,
        dim: Optional[str] = None,
        tnull: Optional[Any] = None,
        tscal: Optional[float] = None,
        tzero: Optional[float] = None,
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.insert_column(
            path,
            name,
            values,
            hdu=hdu,
            index=index,
            format=format,
            unit=unit,
            dim=dim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )
        return self._refresh_file_view()

    def replace_column_file(
        self,
        name: str,
        values: Any,
        *,
        format: Optional[str] = None,
        unit: Optional[str] = None,
        dim: Optional[str] = None,
        tnull: Optional[Any] = None,
        tscal: Optional[float] = None,
        tzero: Optional[float] = None,
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.replace_column(
            path,
            name,
            values,
            hdu=hdu,
            format=format,
            unit=unit,
            dim=dim,
            tnull=tnull,
            tscal=tscal,
            tzero=tzero,
        )
        return self._refresh_file_view()

    def insert_rows_file(self, rows: Dict[str, Any], *, row: int) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.insert_rows(path, rows, row=row, hdu=hdu)
        return self._refresh_file_view()

    def delete_rows_file(
        self, row_slice: Union[int, slice, tuple[int, int]]
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.delete_rows(path, row_slice, hdu=hdu)
        return self._refresh_file_view()

    def update_rows_file(
        self,
        rows: Dict[str, Any],
        row_slice: Union[slice, tuple[int, int]],
        *,
        mmap: Union[bool, str] = "auto",
    ) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.update_rows(path, rows, row_slice=row_slice, hdu=hdu, mmap=mmap)
        return self._refresh_file_view()

    def rename_columns_file(self, mapping: Dict[str, str]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.rename_columns(path, mapping, hdu=hdu)
        return self._refresh_file_view()

    def drop_columns_file(self, columns: List[str]) -> "TableHDURef":
        import torchfits

        path, hdu = self._require_source()
        torchfits.table.drop_columns(path, columns, hdu=hdu)
        return self._refresh_file_view()

    def __repr__(self):
        name = self.header.get("EXTNAME", "TABLE")
        proj = f", cols={len(self.columns)}" if self._columns is not None else ""
        return f"TableHDURef(name='{name}', rows={self.num_rows}{proj})"
