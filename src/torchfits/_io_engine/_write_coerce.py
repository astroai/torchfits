"""Coerce ``torchfits.write`` inputs into the HDU lists the writers take.

``write_api._write_new_file`` performs the writes; the functions here only
turn the caller's ``data`` / ``header`` / ``quantize`` into HDUs and raise
the argument errors (``QuantizeError``, ``TypeError``, ``ValueError``,
``NotImplementedError``) before any byte is written.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch import Tensor

from ..hdu import HDUList, Header, TensorHDU
from ._write_helpers import (
    QuantizeError,
    _TableHDUWriteProxy,
    _apply_image_quantize,
    _coerce_compressed_hdu_item,
    _drop_stale_integer_scale_cards,
    _hdu_with_header,
    _image_hdu_dict_for_fits_write,
    _merge_fits_write_header,
    _merged_write_header,
    _ndarray_to_tensor,
    _require_image_hdu_dict_keys,
    _unsigned_image_storage_for_fits_write,
)

HeaderArg = Optional[Header | Dict[str, Any]]

_MULTI_HDU_QUANTIZE = (
    "quantize= is supported for a single image tensor or dict table, "
    "not multi-HDU sequences"
)
_HDULIST_QUANTIZE = (
    "quantize= is ignored for HDUList writes; quantize each "
    "image tensor before assembling the list"
)


def _with_header_on_first(hdus: List[Any], header: HeaderArg) -> List[Any]:
    """Merge the ``header=`` overlay into the first HDU's header."""
    if header and hdus:
        hdus[0] = _hdu_with_header(
            hdus[0],
            _merged_write_header(getattr(hdus[0], "header", None), header),
        )
    return hdus


def hdulist_hdus_for_write(
    data: HDUList, header: HeaderArg, quantize: Any
) -> List[Any]:
    """The HDUs of an :class:`HDUList` write (compressed or not)."""
    if quantize is not None:
        raise QuantizeError(_HDULIST_QUANTIZE)
    return _with_header_on_first(list(getattr(data, "_hdus", [])), header)


def compressed_hdus_for_write(data: Any, header: HeaderArg, quantize: Any) -> List[Any]:
    """HDUs for a tile-compressed write of a tensor, table, dict or HDU list."""
    import numpy as np

    if isinstance(data, (Tensor, np.ndarray)):
        if isinstance(data, np.ndarray):
            data = _ndarray_to_tensor(data)
        data, header = _apply_image_quantize(data, header, quantize)
        header = _drop_stale_integer_scale_cards(header, data)
        img, img_header = _unsigned_image_storage_for_fits_write(data)
        return [
            TensorHDU(
                data=img,
                header=_merge_fits_write_header(header, img_header),
            )
        ]
    if isinstance(data, HDUList):
        return hdulist_hdus_for_write(data, header, quantize)
    if isinstance(data, dict):
        if "data" in data:
            item: Dict[str, Any] = dict(data)
            _require_image_hdu_dict_keys(item)
            if header:
                item["header"] = _merged_write_header(item.get("header"), header)
            if quantize is not None:
                q_img, q_hdr = _apply_image_quantize(
                    item["data"], item.get("header"), quantize
                )
                item["data"] = q_img
                item["header"] = q_hdr
            return [_coerce_compressed_hdu_item(item)]
        return [_TableHDUWriteProxy(data, header, quantize=quantize)]
    if isinstance(data, (list, tuple)):
        if quantize is not None:
            raise QuantizeError(_MULTI_HDU_QUANTIZE)
        hdus = [_coerce_compressed_hdu_item(item) for item in data]
        return _with_header_on_first(hdus, header)
    raise NotImplementedError(
        "Compressed FITS writing supports tensors, tables, or HDU lists."
    )


def image_hdu_dicts_for_write(
    data: Any, header: HeaderArg, quantize: Any
) -> List[Dict[str, Any]]:
    """Native HDU dicts for an uncompressed image write.

    ``data`` is a tensor, an ndarray, an image dict ``{"data", "header"}``
    or an iterable of those (or of objects with a tensor ``.data``); dict
    tables and :class:`HDUList` take their own writers.
    """
    import numpy as np

    hdus_to_write: List[Dict[str, Any]] = []
    if isinstance(data, Tensor):
        hdus_to_write.append(
            _image_hdu_dict_for_fits_write(data, header, quantize=quantize)
        )

    elif isinstance(data, np.ndarray):
        hdus_to_write.append(
            _image_hdu_dict_for_fits_write(
                _ndarray_to_tensor(data), header, quantize=quantize
            )
        )

    elif isinstance(data, dict):
        # Image dict-HDU form: exactly {"data": ..., "header": ...}.
        _require_image_hdu_dict_keys(data)
        payload = data["data"]
        if isinstance(payload, np.ndarray):
            payload = _ndarray_to_tensor(payload)
        if not isinstance(payload, Tensor):
            raise TypeError("HDU dictionary 'data' must be a torch.Tensor")
        base = (
            _merged_write_header(data.get("header"), header)
            if header
            else data.get("header")
        )
        hdus_to_write.append(
            _image_hdu_dict_for_fits_write(payload, base, quantize=quantize)
        )

    elif hasattr(data, "__iter__") and not isinstance(data, (str, Tensor)):
        if quantize is not None:
            raise QuantizeError(_MULTI_HDU_QUANTIZE)
        for n, item in enumerate(data):
            overlay = header if (n == 0 and header) else None
            if isinstance(item, dict):
                if "data" in item:
                    _require_image_hdu_dict_keys(item)
                    payload = item["data"]
                    if isinstance(payload, np.ndarray):
                        payload = _ndarray_to_tensor(payload)
                    if isinstance(payload, Tensor):
                        base = (
                            _merged_write_header(item.get("header"), overlay)
                            if overlay
                            else item.get("header")
                        )
                        hdus_to_write.append(
                            _image_hdu_dict_for_fits_write(payload, base)
                        )
                    else:
                        raise TypeError("HDU dictionary 'data' must be a torch.Tensor")
                else:
                    raise ValueError("HDU dictionaries must contain a 'data' tensor")
            elif isinstance(item, Tensor):
                hdus_to_write.append(
                    _image_hdu_dict_for_fits_write(
                        item,
                        _merged_write_header(None, overlay) if overlay else None,
                    )
                )
            elif hasattr(item, "data") and isinstance(item.data, Tensor):
                base = (
                    _merged_write_header(getattr(item, "header", None), overlay)
                    if overlay
                    else getattr(item, "header", None)
                )
                hdus_to_write.append(_image_hdu_dict_for_fits_write(item.data, base))
            else:
                raise TypeError(f"Unsupported HDU item type: {type(item).__name__}")
    else:
        raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")

    if not hdus_to_write:
        raise ValueError("At least one writable HDU is required")
    return hdus_to_write
