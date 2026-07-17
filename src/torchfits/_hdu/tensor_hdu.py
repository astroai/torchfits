"""Image/cube HDU with lazy data loading and C++ backend integration."""

from __future__ import annotations

import html
from typing import Any, Iterator, Optional, Tuple, cast

from torch import Tensor

from .header import Header
from .dataview import DataView, _BITPIX_TO_DTYPE


class TensorHDU:
    def __init__(
        self,
        data: Optional[Tensor] = None,
        header: Optional[Header] = None,
        file_handle: Any = None,
        hdu_index: int = 0,
        source_path: Optional[str] = None,
    ):
        self._data = data
        self._header = header or Header()
        self._file_handle = file_handle
        self._hdu_index = hdu_index
        self._source_path = source_path
        self._data_view = DataView(file_handle, hdu_index) if file_handle else None

    @property
    def data(self) -> DataView:
        if self._data_view is None:
            raise ValueError("No file handle available")
        return self._data_view

    @property
    def header(self) -> Header:
        return self._header

    def to_tensor(self, device: str = "cpu") -> Tensor:
        if self._data is not None:
            return self._data.to(device)

        elif self._file_handle is not None:
            import torchfits._C as cpp

            return cast(
                Tensor, cpp.read_full(self._file_handle, self._hdu_index).to(device)
            )
        else:
            raise ValueError(
                "TensorHDU has no data available. "
                "Construct it with a file handle, tensor data, or a Header with a source HDU."
            )

    def chunks(self, chunk_size: Tuple[int, ...]) -> Iterator[Tensor]:
        import torchfits._C as cpp

        return cast(
            Iterator[Tensor],
            cpp.iter_chunks(self._file_handle, self._hdu_index, chunk_size),
        )

    def _get_shape_str(self) -> str:
        if self._data is not None:
            return str(tuple(self._data.shape))
        elif self._file_handle:
            try:
                naxis = int(self.header.get("NAXIS", 0))
            except (TypeError, ValueError):
                return "unknown"
            if naxis <= 0:
                return "()"
            dims = [str(self.header.get(f"NAXIS{i + 1}", 0)) for i in range(naxis)]
            return f"({', '.join(reversed(dims))})"
        return "()"

    def _get_dtype_str(self) -> str:
        if self._data is not None:
            return str(self._data.dtype).replace("torch.", "")
        elif self._file_handle:
            try:
                bitpix = int(self.header.get("BITPIX", 0))
            except (TypeError, ValueError):
                return "unknown"
            dtype = _BITPIX_TO_DTYPE.get(bitpix)
            if dtype is not None:
                return str(dtype).replace("torch.", "")
            return str(bitpix)
        return "unknown"

    def __repr__(self) -> str:
        name = self.header.get("EXTNAME", "PRIMARY")
        return f"TensorHDU(name='{name}', shape={self._get_shape_str()}, dtype={self._get_dtype_str()})"

    def _repr_html_(self) -> str:
        name = html.escape(str(self.header.get("EXTNAME", "PRIMARY")))
        shape = html.escape(self._get_shape_str())
        dtype = html.escape(self._get_dtype_str())
        return (
            "<table>"
            "<caption>TensorHDU</caption>"
            '<thead><tr><th scope="col">Name</th><th scope="col">Shape</th><th scope="col">Dtype</th></tr></thead>'
            f'<tbody><tr><th scope="row" style="font-weight: normal; text-align: left;">{name}</th><td>{shape}</td><td>{dtype}</td></tr></tbody></table>'
        )
