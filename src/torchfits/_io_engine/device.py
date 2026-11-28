"""Device normalization, validation, and MPS-safe tensor transfer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import Tensor

_MPS_F64_WARNING = (
    "MPS does not support float64; downcasting to float32 (precision loss)"
)
_MPS_C128_WARNING = (
    "MPS does not support complex128; downcasting to complex64 (precision loss)"
)


def validate_device(device: str | torch.device) -> str:
    """Validate and normalize a device identifier.

    Accepts 'cpu', 'cuda', 'cuda:N', 'mps', 'mps:N', or torch.device instances.
    """
    dev_str = str(device)
    if dev_str in ("cpu", "cuda", "mps"):
        return dev_str
    if dev_str.startswith(("cuda:", "mps:")):
        index = dev_str.split(":", 1)[1]
        if index.isascii() and index.isdigit():
            return dev_str
    raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', 'mps' or 'mps:N'")


def to_device(
    tensor: Tensor,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> Tensor:
    """Move a tensor to a device, adapting MPS-unsupported dtypes (float64/complex128)."""
    # Fast path: the overwhelmingly common str-device cases without building
    # torch.device objects or touching dtype tables.
    import torch

    dev_str = device if type(device) is str else str(device)
    if dev_str == "cpu":
        return (
            tensor
            if tensor.device.type == "cpu"
            else tensor.to("cpu", non_blocking=non_blocking)
        )
    if dev_str == "mps" or dev_str.startswith("mps:"):
        if tensor.dtype == torch.float64:
            warnings.warn(_MPS_F64_WARNING, UserWarning, stacklevel=2)
            tensor = tensor.float()
        elif tensor.dtype == torch.complex128:
            warnings.warn(_MPS_C128_WARNING, UserWarning, stacklevel=2)
            tensor = tensor.to(torch.complex64)
    return tensor.to(dev_str, non_blocking=non_blocking)


# Per-tensor size above which batch_to_device transfers tensors one by one
# instead of through one stacked copy.
_STACK_TRANSFER_MAX_BYTES = 1 << 20


def batch_to_device(tensors: list[Tensor], device: str | torch.device) -> list[Tensor]:
    """Move a list of tensors to a device, stacking when shapes match."""
    import torch

    if not tensors:
        return []
    dev_str = str(device)
    if dev_str == "mps" or dev_str.startswith("mps:"):
        # Warn once per call (and thus once per site under the default warning
        # filters) instead of once per tensor.
        warn_f64 = warn_c128 = False
        downcast: list[Tensor] = []
        for t in tensors:
            if t.dtype == torch.float64:
                if not warn_f64:
                    warn_f64 = True
                    warnings.warn(_MPS_F64_WARNING, UserWarning, stacklevel=2)
                downcast.append(t.float())
            elif t.dtype == torch.complex128:
                if not warn_c128:
                    warn_c128 = True
                    warnings.warn(_MPS_C128_WARNING, UserWarning, stacklevel=2)
                downcast.append(t.to(torch.complex64))
            else:
                downcast.append(t)
        tensors = downcast
    if len(tensors) == 1:
        return [tensors[0].to(device, non_blocking=True)]

    first = tensors[0]
    shape = first.shape
    dtype = first.dtype

    if first.numel() * first.element_size() <= _STACK_TRANSFER_MAX_BYTES and all(
        t.shape == shape and t.dtype == dtype for t in tensors
    ):
        # Small same-shape tensors: one transfer beats many per-call
        # latencies, and the host copy / shared device storage is cheap.
        return list(torch.stack(tensors).to(device, non_blocking=True).unbind(0))
    # Large tensors move one by one: stacking first copies the whole batch on
    # the host, and unbind() views keep the whole stacked batch resident on
    # the device for as long as any one of them is alive.
    return [t.to(device, non_blocking=True) for t in tensors]


# Host bytes gathered into one transfer when moving a list of VLA row
# tensors: bounds the temporary host copy while still replacing thousands of
# tiny per-row copies (each a separate host-to-device launch) with a few.
_ROW_LIST_TRANSFER_BYTES = 16 << 20


def _move_row_list(
    items: list[Any],
    device: str | torch.device,
    *,
    non_blocking: bool,
    pin_memory: bool,
) -> list[Any]:
    """Move the tensors of a row-aligned list (VLA rows) to *device*.

    Runs of CPU tensors with one dtype are flattened into a single host
    buffer, moved with one transfer and split back into per-row views on the
    device; anything else (strings, ``None``, mixed dtypes) moves per item.
    """
    import torch

    def _one(t: Tensor) -> Tensor:
        if pin_memory and t.device.type == "cpu":
            t = t.pin_memory()
        return to_device(t, device, non_blocking=non_blocking)

    out: list[Any] = []
    run: list[Tensor] = []
    run_bytes = 0

    def _flush() -> None:
        nonlocal run_bytes
        if len(run) == 1:
            out.append(_one(run[0]))
        elif run:
            flat = _one(torch.cat([t.reshape(-1) for t in run]))
            parts = torch.split(flat, [t.numel() for t in run])
            out.extend(part.view(t.shape) for part, t in zip(parts, run))
        run.clear()
        run_bytes = 0

    for item in items:
        if not isinstance(item, torch.Tensor):
            _flush()
            out.append(item)
            continue
        if item.device.type != "cpu":
            _flush()
            out.append(_one(item))
            continue
        nbytes = item.numel() * item.element_size()
        if run and (
            item.dtype != run[0].dtype or run_bytes + nbytes > _ROW_LIST_TRANSFER_BYTES
        ):
            _flush()
        run.append(item)
        run_bytes += nbytes
    _flush()
    return out


def move_table_columns(
    data: dict[str, Any],
    device: str | torch.device,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> dict[str, Any]:
    """Move a column dict (tensors and row-aligned tensor lists) to *device*.

    The one table device move shared by the read cache, the fallback reader,
    ``read_torch`` / ``TableReaderHandle.read_torch``, ``scan_torch`` and the
    datasets. Tensors inside row-aligned lists (VLA columns) follow their
    sibling columns; other values pass through. A CPU *device* returns
    *data* unchanged.
    """
    import torch

    if str(device) == "cpu":
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if pin_memory and value.device.type == "cpu":
                value = value.pin_memory()
            out[key] = to_device(value, device, non_blocking=non_blocking)
        elif isinstance(value, list):
            out[key] = _move_row_list(
                value, device, non_blocking=non_blocking, pin_memory=pin_memory
            )
        else:
            out[key] = value
    return out
