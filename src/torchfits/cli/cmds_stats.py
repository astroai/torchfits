"""``torchfits stats`` — basic image statistics."""

from __future__ import annotations

import argparse
from typing import Any

import torchfits

from .common import (
    EXIT_OK,
    CliError,
    IoError,
    add_emit_format_args,
    add_file_jobs_arg,
    add_hdu_arg,
    add_jobs_arg,
    configure_torch_jobs,
    emit_records,
    header_extname,
    hdu_type_name,
    resolve_emit_format,
    resolve_file_jobs,
    resolve_paths,
    run_file_jobs,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "stats",
        help="image min/max/mean/std/median of finite pixels",
        description=(
            "Image min/max/mean/std/median over finite pixels, in float64; "
            "NaN/inf/BLANK pixels are excluded and counted in nbad. "
            "The median is the midpoint median (numpy/IRAF). "
            "-j = PyTorch intra-op threads; -J = parallel file workers."
        ),
    )
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_emit_format_args(parser)
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)
    parser.set_defaults(func=run)


def _pixel_stats(tensor: Any) -> dict[str, Any]:
    """min/max/mean/std/median over the finite pixels, in float64.

    Non-finite pixels (NaN, +-inf, and integer ``BLANK`` pixels, which the
    reader returns as NaN) are excluded and counted in ``nbad``; one NaN
    would otherwise poison every statistic. Float64 keeps integers above
    2**24 exact (float32 does not) and avoids float32 accumulation error;
    integer images report exact integer ``min`` / ``max``.
    The median is the conventional midpoint median (numpy / IRAF): the mean
    of the two middle values for an even count, not the lower one.
    """
    import numpy as np
    import torch

    # torch has no reductions for uint16/32/64, and float64 is the decided
    # statistics precision for every dtype anyway.
    flat = tensor.reshape(-1).to(torch.float64)
    finite = torch.isfinite(flat)
    nbad = int(flat.numel() - int(finite.sum()))
    valid = flat if nbad == 0 else flat[finite]
    nan = float("nan")
    # An empty region (or one with no finite pixel) has no defined
    # statistics: every stat is NaN, which -f json/jsonl serializes as null.
    if valid.numel() == 0:
        return {
            "nbad": nbad,
            "min": nan,
            "max": nan,
            "mean": nan,
            "std": nan,
            "median": nan,
        }
    # Integer images report exact integer extrema: float64 cannot represent
    # every int64/uint64 value (e.g. 2**53 + 1). numpy covers the unsigned
    # dtypes torch has no reductions for.
    if tensor.dtype.is_floating_point:
        vmin: Any = float(valid.min())
        vmax: Any = float(valid.max())
    else:
        native = tensor.reshape(-1).numpy()
        vmin, vmax = int(native.min()), int(native.max())
    return {
        "nbad": nbad,
        "min": vmin,
        "max": vmax,
        "mean": float(valid.mean()),
        "std": float(valid.std(unbiased=False)),
        "median": float(np.median(valid.numpy())),
    }


def _stats_one(path: str, hdu: str | None) -> list[dict[str, Any]]:
    import torch

    try:
        with torchfits.open(path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu)
            headers = {index: hdul[index].header for index in indices}
            types = {
                index: hdu_type_name(headers[index], hdul[index]) for index in indices
            }
    except CliError:
        raise
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for index in indices:
        if types[index] != "IMAGE":
            continue
        try:
            tensor = torchfits.read_tensor(path, hdu=index)
        except CliError:
            raise
        except Exception as exc:
            raise IoError(f"{path}:{index} {exc}") from exc
        if not isinstance(tensor, torch.Tensor):
            raise IoError(f"{path}:{index} read_tensor did not return a tensor")
        header = headers[index]
        records.append(
            {
                "file": path,
                "hdu": index,
                "name": header_extname(header, index),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                **_pixel_stats(tensor),
            }
        )
    return records


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(paths))
    configure_torch_jobs(int(args.jobs))
    chunks = run_file_jobs(
        paths,
        lambda path: _stats_one(path, args.hdu),
        file_jobs,
        torch_runtime=True,
    )
    records = [record for chunk in chunks for record in chunk]
    emit_records(records, format=resolve_emit_format(args))
    return EXIT_OK
