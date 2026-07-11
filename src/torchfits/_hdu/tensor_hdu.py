"""Image/cube HDU with lazy data loading and C++ backend integration."""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

from torch import Tensor

from .header import Header
from .dataview import DataView, _BITPIX_TO_DTYPE


class TensorHDU:
    def __init__(
        self,
        data: Optional[Tensor] = None,
        header: Optional[Header] = None,
        file_handle=None,
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

            return cpp.read_full(self._file_handle, self._hdu_index).to(device)
        else:
            raise ValueError(
                "TensorHDU has no data available. "
                "Construct it with a file handle, tensor data, or a Header with a source HDU."
            )

    def chunks(self, chunk_size: Tuple[int, ...]) -> Iterator[Tensor]:
        import torchfits._C as cpp

        return cpp.iter_chunks(self._file_handle, self._hdu_index, chunk_size)

    def stats(self) -> Dict[str, float]:
        import torchfits._C as cpp

        return cpp.compute_stats(self._file_handle, self._hdu_index)

    def _get_shape_str(self) -> str:
        if self._data is not None:
            return str(tuple(self._data.shape))
        elif self._file_handle:
            naxis = self.header.get("NAXIS", 0)
            if naxis == 0:
                return "()"
            dims = [str(self.header.get(f"NAXIS{i + 1}", 0)) for i in range(naxis)]
            return f"({', '.join(reversed(dims))})"
        return "()"

    def _get_dtype_str(self) -> str:
        if self._data is not None:
            return str(self._data.dtype).replace("torch.", "")
        elif self._file_handle:
            bitpix = self.header.get("BITPIX", 0)
            dtype = _BITPIX_TO_DTYPE.get(bitpix)
            if dtype is not None:
                return str(dtype).replace("torch.", "")
            return str(bitpix)
        return "unknown"

    def __repr__(self):
        name = self.header.get("EXTNAME", "PRIMARY")
        return f"TensorHDU(name='{name}', shape={self._get_shape_str()}, dtype={self._get_dtype_str()})"

    def _repr_html_(self):
        import html as pyhtml

        name = pyhtml.escape(str(self.header.get("EXTNAME", "PRIMARY")))
        shape = pyhtml.escape(self._get_shape_str())
        dtype = pyhtml.escape(self._get_dtype_str())

        html = [
            '<div tabindex="0" aria-label="TensorHDU Summary" style=\'max-height: 400px; overflow: auto; border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 1em;\'>',
            "<table style='border-collapse: collapse; width: 100%; margin: 0;'>",
            "<thead><tr>",
        ]

        headers = ["Property", "Value"]
        for h in headers:
            html.append(
                f'<th scope="col" style=\'text-align: left; padding: 8px; position: sticky; top: 0; '
                f"background-color: var(--theme-ui-colors-background, white); "
                f"border-bottom: 2px solid rgba(128, 128, 128, 0.3); z-index: 1;'>{h}</th>"
            )
        html.append("</tr></thead><tbody>")

        properties = [("Name", name), ("Shape", shape), ("Data Type", dtype)]
        for i, (prop, val) in enumerate(properties):
            html.append("<tr>")
            html.append(
                f"<th scope=\"row\" style='font-weight: normal; text-align: left; padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>{prop}</th>"
            )
            html.append(
                f"<td style='text-align: left; padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>{val}</td>"
            )
            html.append("</tr>")

        html.append("</tbody></table></div>")
        return "".join(html)
