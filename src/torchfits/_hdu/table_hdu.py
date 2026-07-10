"""Table HDU: FITS table as TensorFrame with raw data and column accessors."""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

try:
    from torch_frame import TensorFrame
    from torch_frame import stype as _torch_frame_stype

    HAS_TORCH_FRAME = True
except ImportError:
    HAS_TORCH_FRAME = False
    _torch_frame_stype = None

    class TensorFrame:  # type: ignore[no-redef]
        def __init__(self, feat_dict=None, col_names_dict=None):
            self.feat_dict = feat_dict or {}
            self.col_names_dict = col_names_dict or {}


from .header import Header


class TableDataAccessor:
    def __init__(self, table_hdu):
        self._table = table_hdu

    def __getitem__(self, key):
        if hasattr(self._table, "_raw_data") and key in self._table._raw_data:
            value = self._table._raw_data[key]
            if isinstance(value, torch.Tensor):
                if value.dim() > 1:
                    return value.squeeze()
                return value
            return value
        if hasattr(self._table, "feat_dict") and key in self._table.feat_dict:
            tensor = self._table.feat_dict[key]
            if tensor.dim() > 1:
                return tensor.squeeze()
            return tensor
        raise KeyError(f"Column '{key}' not found")

    def __contains__(self, key):
        return (hasattr(self._table, "_raw_data") and key in self._table._raw_data) or (
            hasattr(self._table, "feat_dict") and key in self._table.feat_dict
        )

    def keys(self):
        if hasattr(self._table, "_raw_data"):
            return self._table._raw_data.keys()
        if hasattr(self._table, "feat_dict"):
            return self._table.feat_dict.keys()
        return []

    @property
    def columns(self):
        return list(self.keys())

    def __len__(self):
        return self._table.num_rows


class TableHDU(TensorFrame):
    def __init__(
        self,
        tensor_dict: dict,
        col_stats: dict = None,
        header: Optional[Header] = None,
        source_path: Optional[str] = None,
        source_hdu: Optional[int] = None,
    ):
        import numpy as np

        self._raw_data = tensor_dict or {}
        self._source_path = source_path
        self._source_hdu = source_hdu
        feat_dict = {}
        col_names_dict = {}
        string_cols = self._get_string_columns(header)

        for col_name, data in tensor_dict.items():
            try:
                if col_name in string_cols:
                    continue
                if isinstance(data, torch.Tensor):
                    if data.dim() == 1:
                        data = data.unsqueeze(1)
                    elif data.dim() == 0:
                        data = data.unsqueeze(0).unsqueeze(1)
                    feat_dict[col_name] = data
                    col_names_dict[col_name] = [str(i) for i in range(data.shape[1])]
                elif isinstance(data, (list, tuple)):
                    try:
                        if data and isinstance(data[0], str):
                            continue
                        if data and isinstance(data[0], torch.Tensor):
                            continue
                        if data and isinstance(data[0], np.ndarray):
                            continue
                        if data and isinstance(data[0], (int, np.integer)):
                            tensor_data = torch.tensor(data, dtype=torch.long)
                        else:
                            tensor_data = torch.tensor(data, dtype=torch.float32)
                        if tensor_data.dim() == 1:
                            tensor_data = tensor_data.unsqueeze(1)
                        feat_dict[col_name] = tensor_data
                        col_names_dict[col_name] = [
                            str(i) for i in range(tensor_data.shape[1])
                        ]
                    except Exception:
                        continue
                elif isinstance(data, dict):
                    continue
                elif isinstance(data, np.ndarray):
                    if np.iscomplexobj(data) or data.dtype.kind in {"U", "S", "O"}:
                        continue
                    tensor_data = torch.tensor(data)
                    if tensor_data.dim() == 1:
                        tensor_data = tensor_data.unsqueeze(1)
                    feat_dict[col_name] = tensor_data
                    col_names_dict[col_name] = [
                        str(i) for i in range(tensor_data.shape[1])
                    ]
                else:
                    try:
                        if isinstance(data, str):
                            continue
                        tensor_data = torch.tensor(
                            [data] if not hasattr(data, "__len__") else data,
                            dtype=torch.float32,
                        )
                        if tensor_data.dim() == 1:
                            tensor_data = tensor_data.unsqueeze(1)
                        feat_dict[col_name] = tensor_data
                        col_names_dict[col_name] = [
                            str(i) for i in range(tensor_data.shape[1])
                        ]
                    except Exception:
                        continue
            except Exception:
                continue

        if not feat_dict:
            nrows = 1
            if header and "NAXIS2" in header:
                try:
                    nrows = int(header["NAXIS2"])
                except Exception:
                    nrows = 1
            feat_dict = {"dummy": torch.zeros(max(1, nrows), 1)}
            col_names_dict = {"dummy": ["0"]}

        self.header = header or Header()
        if HAS_TORCH_FRAME:
            tf_feat_dict, tf_col_names_dict = self._tensorframe_validation_payload(
                feat_dict
            )
            super().__init__(tf_feat_dict, tf_col_names_dict)
            self.feat_dict = feat_dict
            self.col_names_dict = col_names_dict
        else:
            super().__init__(feat_dict, col_names_dict)

    @staticmethod
    def _tensorframe_validation_payload(
        feat_dict: Dict[str, Tensor],
    ) -> tuple[Dict[Any, Tensor], Dict[Any, List[str]]]:
        num_rows = 1
        num_cols = 0
        names: List[str] = []
        for name, tensor_data in feat_dict.items():
            if not isinstance(tensor_data, torch.Tensor) or tensor_data.dim() < 2:
                continue
            num_rows = int(tensor_data.shape[0])
            width = int(tensor_data.shape[1])
            num_cols += width
            if width == 1:
                names.append(str(name))
            else:
                names.extend(f"{name}_{idx}" for idx in range(width))
        if num_cols <= 0:
            num_cols = 1
            names = ["dummy"]
        st = (
            _torch_frame_stype.numerical
            if _torch_frame_stype is not None
            else "numerical"
        )
        return {st: torch.zeros((num_rows, num_cols), dtype=torch.float32)}, {st: names}

    def _get_string_columns(self, header: Optional[Header]) -> set:
        if not header:
            return set()
        from ..fits_schema import string_column_names

        return set(string_column_names(header))

    # ⚡ Bolt: Cache string column derivation to avoid redundant header parsing and
    # string extraction on repeated access (e.g., during loops or schema validations).
    @functools.cached_property
    def string_columns(self) -> List[str]:
        return sorted(self._get_string_columns(self.header))

    # ⚡ Bolt: Cache schema building to prevent O(N) header traversals
    # for TTYPE*/TFORM* keys on every property access.
    @functools.cached_property
    def schema(self) -> Dict[str, Any]:
        return self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        from ..fits_schema import build_table_schema_dict

        return build_table_schema_dict(self.header)

    def get_vla_column(self, name: str) -> List[Tensor]:
        value = self._raw_data.get(name)
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

    def get_string_column(
        self, name: str, encoding: str = "ascii", strip: bool = True
    ) -> List[str]:
        import numpy as np

        value = self._raw_data.get(name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"Column '{name}' is not a tensor string column")
        if value.dtype != torch.uint8:
            raise TypeError(f"Column '{name}' is not uint8 encoded")
        if value.dim() != 2:
            raise ValueError(f"Column '{name}' must be 2D (rows, width)")

        # Vectorized fixed-width byte decode using numpy
        arr = value.cpu().numpy()
        width = arr.shape[1]
        if width == 0:
            return [""] * arr.shape[0]
        byte_view = arr.view(f"S{width}").ravel()
        decoded = np.char.decode(byte_view, encoding=encoding, errors="ignore")
        if strip:
            decoded = np.char.rstrip(decoded, " \x00")
        return decoded.tolist()

    # ⚡ Bolt: Cache row count extraction to prevent scanning all column tensors
    # and re-evaluating shapes on every length check.
    @functools.cached_property
    def num_rows(self) -> int:
        if hasattr(self, "_raw_data") and self._raw_data:
            import numpy as np

            for value in self._raw_data.values():
                if isinstance(value, torch.Tensor):
                    return value.shape[0] if value.dim() > 0 else 1
                if isinstance(value, np.ndarray):
                    return int(value.shape[0]) if value.ndim > 0 else 1
                if isinstance(value, (list, tuple)):
                    return len(value)
            return 0
        if self.header and "NAXIS2" in self.header:
            try:
                return int(self.header["NAXIS2"])
            except Exception:
                pass
        if hasattr(self, "feat_dict") and self.feat_dict:
            first_tensor = next(iter(self.feat_dict.values()))
            return first_tensor.shape[0] if hasattr(first_tensor, "shape") else 0
        return 0

    @property
    def data(self):
        return TableDataAccessor(self)

    @property
    def columns(self) -> List[str]:
        if hasattr(self, "_raw_data"):
            return [str(k) for k in self._raw_data.keys()]
        if hasattr(self, "feat_dict"):
            return [str(k) for k in self.feat_dict.keys()]
        return []

    @property
    def col_names(self) -> List[str]:
        return self.columns

    @property
    def feat_types(self) -> Dict[str, str]:
        types = {}
        if hasattr(self, "feat_dict"):
            for name, tensor_data in self.feat_dict.items():
                if hasattr(tensor_data, "dtype"):
                    if tensor_data.dtype.is_floating_point:
                        types[str(name)] = "numerical"
                    else:
                        types[str(name)] = "categorical"
                else:
                    types[str(name)] = "categorical"
        return types

    def select(self, cols: List[str]) -> "TableHDU":
        if hasattr(self, "_raw_data"):
            selected_dict = {k: v for k, v in self._raw_data.items() if str(k) in cols}
            return TableHDU(selected_dict, {}, self.header)
        if hasattr(self, "feat_dict"):
            selected_dict = {k: v for k, v in self.feat_dict.items() if str(k) in cols}
            return TableHDU(selected_dict, {}, self.header)
        return self

    def filter(self, condition: str) -> "TableHDU":
        import numpy as np

        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("condition must be a non-empty string")

        data_map = self._raw_data if hasattr(self, "_raw_data") else {}
        if not data_map and hasattr(self, "feat_dict"):
            data_map = self.feat_dict
        if not data_map:
            return self

        num_rows = self.num_rows
        if num_rows <= 0:
            return self

        eval_locals: Dict[str, Any] = {}
        for name, value in data_map.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.shape[0] == num_rows
            ):
                t = value.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                arr = t.numpy()
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                eval_locals[str(name)] = arr
            elif (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == num_rows
            ):
                arr = value
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                eval_locals[str(name)] = arr
            elif isinstance(value, list) and len(value) == num_rows:
                eval_locals[str(name)] = np.asarray(value, dtype=object)

        if not eval_locals:
            raise ValueError("No row-aligned columns available for filtering")

        # Build a minimal Arrow table and delegate to pyarrow.compute predicates.
        import pyarrow as pa

        pa_arrays = {}
        for name, arr in eval_locals.items():
            if isinstance(arr, np.ndarray) and arr.dtype.kind == "O":
                pa_arrays[name] = pa.array(arr.tolist())
            else:
                pa_arrays[name] = pa.array(arr)
        pa_table = pa.table(pa_arrays)

        from .._table.read import _where_mask_for_table

        mask_chunked = _where_mask_for_table(pa_table, condition)
        mask_arr = mask_chunked.to_numpy()
        if mask_arr.ndim == 0:
            mask = np.full(num_rows, bool(mask_arr.item()), dtype=bool)
        else:
            mask = mask_arr.astype(bool, copy=False).reshape(-1)
            if mask.shape[0] != num_rows:
                raise ValueError(
                    f"Filter produced mask of length {mask.shape[0]}, expected {num_rows}"
                )

        filtered: Dict[str, Any] = {}
        for name, value in data_map.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and value.shape[0] == num_rows
            ):
                mask_t = torch.from_numpy(mask)
                if value.device.type != "cpu":
                    mask_t = mask_t.to(value.device)
                filtered[name] = value[mask_t]
            elif (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == num_rows
            ):
                filtered[name] = value[mask]
            elif isinstance(value, list) and len(value) == num_rows:
                filtered[name] = [item for item, keep in zip(value, mask) if keep]
            else:
                filtered[name] = value

        return TableHDU(
            filtered,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def head(self, n: int) -> "TableHDU":
        if hasattr(self, "_raw_data") and self._raw_data:
            new_dict = {}
            for k, v in self._raw_data.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    new_dict[k] = v[:n]
                elif isinstance(v, list):
                    new_dict[k] = v[:n]
                else:
                    new_dict[k] = v
            return TableHDU(new_dict, {}, self.header)
        if hasattr(self, "feat_dict") and self.feat_dict:
            new_dict = {}
            for k, v in self.feat_dict.items():
                if hasattr(v, "shape") and len(v.shape) > 0:
                    new_dict[k] = v[:n]
                else:
                    new_dict[k] = v
            return TableHDU(new_dict, {}, self.header)
        return self

    @staticmethod
    def _value_num_rows(value: Any) -> int:
        import numpy as np

        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.dim() > 0 else 1
        if isinstance(value, np.ndarray):
            return int(value.shape[0]) if value.ndim > 0 else 1
        if isinstance(value, (list, tuple)):
            return len(value)
        raise TypeError(f"Unsupported column type: {type(value)}")

    @staticmethod
    def _append_column_values(name: str, old_value: Any, new_value: Any) -> Any:
        import numpy as np

        if isinstance(old_value, torch.Tensor):
            if not isinstance(new_value, torch.Tensor):
                new_value = torch.as_tensor(
                    np.asarray(new_value), dtype=old_value.dtype
                )
            if old_value.device.type != new_value.device.type:
                new_value = new_value.to(old_value.device)

            if old_value.dim() == 0:
                old_value = old_value.reshape(1)
            if new_value.dim() == 0:
                new_value = new_value.reshape(1)
            if old_value.dim() == 2 and new_value.dim() == 1:
                new_value = new_value.unsqueeze(1)
            if (
                old_value.dim() == 1
                and new_value.dim() == 2
                and new_value.shape[1] == 1
            ):
                new_value = new_value.squeeze(1)
            if old_value.dim() != new_value.dim():
                raise ValueError(
                    f"Column '{name}' rank mismatch: {old_value.dim()} vs {new_value.dim()}"
                )
            if old_value.dim() > 1 and tuple(old_value.shape[1:]) != tuple(
                new_value.shape[1:]
            ):
                raise ValueError(
                    f"Column '{name}' shape mismatch: {tuple(old_value.shape[1:])} vs {tuple(new_value.shape[1:])}"
                )
            return torch.cat([old_value, new_value.to(dtype=old_value.dtype)], dim=0)

        if isinstance(old_value, np.ndarray):
            new_arr = np.asarray(new_value, dtype=old_value.dtype)
            old_arr = old_value
            if old_arr.ndim == 0:
                old_arr = old_arr.reshape(1)
            if new_arr.ndim == 0:
                new_arr = new_arr.reshape(1)
            if old_arr.ndim == 2 and new_arr.ndim == 1:
                new_arr = np.expand_dims(new_arr, axis=1)
            if old_arr.ndim == 1 and new_arr.ndim == 2 and new_arr.shape[1] == 1:
                new_arr = np.squeeze(new_arr, axis=1)
            if old_arr.ndim != new_arr.ndim:
                raise ValueError(
                    f"Column '{name}' rank mismatch: {old_arr.ndim} vs {new_arr.ndim}"
                )
            if old_arr.ndim > 1 and tuple(old_arr.shape[1:]) != tuple(
                new_arr.shape[1:]
            ):
                raise ValueError(
                    f"Column '{name}' shape mismatch: {tuple(old_arr.shape[1:])} vs {tuple(new_arr.shape[1:])}"
                )
            return np.concatenate([old_arr, new_arr], axis=0)

        if isinstance(old_value, list):
            if not isinstance(new_value, (list, tuple)):
                raise ValueError(
                    f"Column '{name}' expects a list/tuple for append; got {type(new_value)}"
                )
            return list(old_value) + list(new_value)

        raise TypeError(
            f"Unsupported column type for append in '{name}': {type(old_value)}"
        )

    def add_column(self, name: str, values: Any, overwrite: bool = False) -> "TableHDU":
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        if name in raw and not overwrite:
            raise KeyError(f"Column '{name}' already exists")

        nrows = self.num_rows
        new_rows = self._value_num_rows(values)
        if nrows > 0 and new_rows != nrows:
            raise ValueError(f"Column '{name}' has {new_rows} rows, expected {nrows}")
        raw[name] = values
        return TableHDU(
            raw,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def drop_columns(self, columns: List[str]) -> "TableHDU":
        if not columns:
            return self
        to_drop = {str(c) for c in columns}
        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        missing = [name for name in to_drop if name not in raw]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        kept = {k: v for k, v in raw.items() if k not in to_drop}
        return TableHDU(
            kept,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def rename_column(self, old_name: str, new_name: str) -> "TableHDU":
        if not isinstance(old_name, str) or not old_name:
            raise ValueError("old_name must be a non-empty string")
        if not isinstance(new_name, str) or not new_name:
            raise ValueError("new_name must be a non-empty string")
        if old_name == new_name:
            return self

        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        if old_name not in raw:
            raise KeyError(f"Column '{old_name}' not found")
        if new_name in raw:
            raise KeyError(f"Column '{new_name}' already exists")

        renamed: Dict[str, Any] = {}
        for key, value in raw.items():
            renamed[new_name if key == old_name else key] = value
        return TableHDU(
            renamed,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def append_rows(self, rows: Dict[str, Any]) -> "TableHDU":
        if not isinstance(rows, dict) or not rows:
            raise ValueError("rows must be a non-empty dictionary")

        raw = dict(self._raw_data) if hasattr(self, "_raw_data") else {}
        if not raw:
            return TableHDU(
                dict(rows),
                {},
                self.header,
                source_path=self._source_path,
                source_hdu=self._source_hdu,
            )

        current_cols = set(raw.keys())
        incoming_cols = set(rows.keys())
        if incoming_cols != current_cols:
            missing = sorted(current_cols - incoming_cols)
            extra = sorted(incoming_cols - current_cols)
            raise ValueError(
                f"append_rows requires exactly matching columns; missing={missing}, extra={extra}"
            )

        appended: Dict[str, Any] = {}
        append_rows_count: Optional[int] = None
        for name in raw.keys():
            new_count = self._value_num_rows(rows[name])
            if append_rows_count is None:
                append_rows_count = new_count
            elif new_count != append_rows_count:
                raise ValueError(
                    f"All appended columns must have same row count; column '{name}' has {new_count}, expected {append_rows_count}"
                )
            appended[name] = self._append_column_values(name, raw[name], rows[name])

        return TableHDU(
            appended,
            {},
            self.header,
            source_path=self._source_path,
            source_hdu=self._source_hdu,
        )

    def __getitem__(self, col_name: str) -> Any:
        if hasattr(self, "_raw_data") and col_name in self._raw_data:
            return self._raw_data[col_name]
        if hasattr(self, "feat_dict") and col_name in self.feat_dict:
            return self.feat_dict[col_name]
        raise KeyError(f"Column '{col_name}' not found")

    def materialize(self) -> "TensorFrame":
        return self

    def to_tensor_dict(self) -> Dict[str, Any]:
        if hasattr(self, "_raw_data"):
            return {
                str(k): v
                for k, v in self._raw_data.items()
                if isinstance(v, torch.Tensor)
            }
        if hasattr(self, "feat_dict"):
            return {str(k): v for k, v in self.feat_dict.items()}
        return {}

    def iter_rows(self, batch_size: int = 1000):
        if hasattr(self, "_raw_data") and self._raw_data:
            total_rows = self.num_rows
            for start in range(0, total_rows, batch_size):
                batch = {}
                for k, v in self._raw_data.items():
                    if isinstance(v, torch.Tensor):
                        batch[str(k)] = v[start : start + batch_size]
                    elif isinstance(v, list):
                        batch[str(k)] = v[start : start + batch_size]
                    else:
                        batch[str(k)] = v
                yield batch
        elif hasattr(self, "feat_dict") and self.feat_dict:
            total_rows = self.num_rows
            for start in range(0, total_rows, batch_size):
                batch = {}
                for k, v in self.feat_dict.items():
                    if hasattr(v, "shape") and len(v.shape) > 0:
                        batch[str(k)] = v[start : start + batch_size]
                    else:
                        batch[str(k)] = v
                yield batch

    @classmethod
    def from_fits(cls, file_path: str, hdu_index: int = 1) -> "TableHDU":
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        if not isinstance(hdu_index, int) or hdu_index < 0:
            raise ValueError("hdu_index must be a non-negative integer")

        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FITS file not found: {file_path}")

        try:
            import torchfits._C as cpp
            from .header import Header as Hdr

            tensor_dict = cpp.read_fits_table(file_path, hdu_index)
            header = Hdr(cpp.read_header_dict(file_path, hdu_index))

            return cls(
                tensor_dict, {}, header, source_path=file_path, source_hdu=hdu_index
            )
        except (IOError, RuntimeError) as e:
            from ..logging import logger

            logger.error(
                f"Failed to read table from {file_path}[{hdu_index}]: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to read table from {file_path}[{hdu_index}]: {e}"
            ) from e
        except Exception as e:
            from ..logging import logger

            logger.critical(
                f"Unexpected error reading {file_path}[{hdu_index}]: {str(e)}"
            )
            raise

    def to_fits(self, file_path: str, overwrite: bool = False):
        import torchfits
        import numpy as np

        payload = (
            dict(self._raw_data)
            if hasattr(self, "_raw_data")
            else self.to_tensor_dict()
        )
        for name in self.string_columns:
            value = payload.get(name)
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.dim() == 2
            ):
                decoded: List[str] = []
                arr = value.detach().cpu().numpy()
                width = arr.shape[1]
                if width == 0:
                    decoded = [""] * arr.shape[0]
                else:
                    byte_view = arr.view(f"S{width}").ravel()
                    decoded_arr = np.char.decode(
                        byte_view, encoding="ascii", errors="ignore"
                    )
                    decoded = np.char.rstrip(decoded_arr, " \x00").tolist()
                payload[name] = decoded

        torchfits.write(
            file_path,
            payload,
            header=self.header if self.header is not None else {},
            overwrite=overwrite,
        )

    def _repr_html_(self):
        import html as pyhtml

        html_parts = [
            '<div tabindex="0" aria-label="FITS Table HDU" style=\'max-height: 400px; overflow: auto; border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 1em;\'>',
            "<table style='border-collapse: collapse; width: 100%; margin: 0;'>",
            "<thead><tr>",
        ]

        name = self.header.get("EXTNAME", "TABLE")
        html_parts.append(
            f"<th colspan='2' style='text-align: left; padding: 8px; position: sticky; top: 0; background-color: var(--theme-ui-colors-background, white); border-bottom: 2px solid rgba(128, 128, 128, 0.3); z-index: 1;'>TableHDU: {pyhtml.escape(str(name))}</th>"
        )
        html_parts.append("</tr></thead><tbody>")

        props = [("Rows", self.num_rows), ("Columns", len(self.columns))]

        for k, v in props:
            html_parts.append("<tr>")
            html_parts.append(
                f"<th scope=\"row\" style='text-align: left; padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2); font-weight: bold;'>{pyhtml.escape(str(k))}</th>"
            )
            html_parts.append(
                f"<td style='padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>{pyhtml.escape(str(v))}</td>"
            )
            html_parts.append("</tr>")

        html_parts.append("</tbody></table></div>")
        return "".join(html_parts)

    def __repr__(self):
        name = self.header.get("EXTNAME", "TABLE")
        return (
            f"TableHDU(name='{name}', rows={self.num_rows}, cols={len(self.columns)})"
        )
