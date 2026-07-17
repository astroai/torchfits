"""``torchfits stats`` — basic image statistics."""

from __future__ import annotations

import argparse

import torch

import torchfits

from .common import (
    EXIT_OK,
    IoError,
    emit_records,
    header_extname,
    hdu_type_name,
    resolve_paths,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("stats", help="image min/max/mean via read_tensor")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    parser.add_argument("--hdu", help="comma-separated HDU indices (default: all)")
    parser.add_argument("--json", action="store_true", help="emit JSON array")
    parser.add_argument("--jsonl", action="store_true", help="emit JSONL records")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    records: list[dict[str, object]] = []
    for path in paths:
        try:
            with torchfits.open(path) as hdul:
                indices = selected_hdu_indices(len(hdul), args.hdu)
                headers = {index: hdul[index].header for index in indices}
                types = {
                    index: hdu_type_name(headers[index], hdul[index])
                    for index in indices
                }
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
        for index in indices:
            if types[index] != "IMAGE":
                continue
            tensor = torchfits.read_tensor(path, hdu=index)
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
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.float().mean()),
                }
            )
    emit_records(records, json_mode=args.json, jsonl=args.jsonl)
    return EXIT_OK
