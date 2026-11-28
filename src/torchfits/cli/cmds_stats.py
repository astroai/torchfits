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


# Pixels per reduction chunk: bounds the float64 temporaries of a pass at
# 8 MiB however large the image is.
_STATS_CHUNK = 1 << 20


def _pixel_stats(tensor: Any) -> dict[str, Any]:
    """min/max/mean/std/median over the finite pixels, in float64.

    Non-finite pixels (NaN, +-inf, and integer ``BLANK`` pixels, which the
    reader returns as NaN) are excluded and counted in ``nbad``; one NaN
    would otherwise poison every statistic. Float64 keeps integers above
    2**24 exact (float32 does not) and avoids float32 accumulation error;
    integer images report exact integer ``min`` / ``max``.
    The median is the conventional midpoint median (numpy / IRAF): the mean
    of the two middle values for an even count, not the lower one.

    Memory: the sums run over fixed-size float64 chunks, and the median
    partitions one copy of the finite pixels in the image's own dtype, so the
    peak is about twice the image (it was ~9x for a uint8 image: a float64
    copy, a mask, and a float64 copy for the median).
    """
    import numpy as np

    pixels = tensor.reshape(-1).numpy()
    is_float = pixels.dtype.kind in "fc"
    total = int(pixels.size)
    chunks = range(0, total, _STATS_CHUNK)
    nbad = 0
    if is_float:
        nbad = total - sum(
            int(np.count_nonzero(np.isfinite(pixels[i : i + _STATS_CHUNK])))
            for i in chunks
        )
    nan = float("nan")
    # An empty region (or one with no finite pixel) has no defined
    # statistics: every stat is NaN, which -f json/jsonl serializes as null.
    if total - nbad == 0:
        return {
            "nbad": nbad,
            "min": nan,
            "max": nan,
            "mean": nan,
            "std": nan,
            "median": nan,
        }
    # One copy of the finite pixels, partitioned in place for the median.
    valid = pixels[np.isfinite(pixels)] if nbad else pixels.copy()
    count = int(valid.size)
    # Integer images report exact integer extrema: float64 cannot represent
    # every int64/uint64 value (e.g. 2**53 + 1).
    if is_float:
        vmin: Any = float(valid.min())
        vmax: Any = float(valid.max())
    else:
        vmin, vmax = int(valid.min()), int(valid.max())
    steps = range(0, count, _STATS_CHUNK)
    total_sum = 0.0
    for i in steps:
        total_sum += float(valid[i : i + _STATS_CHUNK].astype(np.float64).sum())
    mean = total_sum / count
    sq = 0.0
    for i in steps:
        dev = valid[i : i + _STATS_CHUNK].astype(np.float64)
        dev -= mean
        sq += float(np.dot(dev, dev))
    half = count // 2
    if count % 2:
        valid.partition(half)
        median = float(valid[half])
    else:
        valid.partition((half - 1, half))
        median = (float(valid[half - 1]) + float(valid[half])) / 2.0
    return {
        "nbad": nbad,
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": float(np.sqrt(sq / count)),
        "median": median,
    }


def _stats_one(path: str, hdu: str | None) -> list[dict[str, Any]]:
    import torch

    try:
        with torchfits.open(path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu, path=path)
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
