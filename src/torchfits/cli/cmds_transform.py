"""``torchfits transform`` — apply a named torchfits.transforms class."""

from __future__ import annotations

import argparse

import torch
import torchfits.transforms as tf_transforms

import torchfits

from .common import EXIT_OK, IoError, UsageError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "transform", help="apply a torchfits.transforms class"
    )
    parser.add_argument("input", help="input FITS image path")
    parser.add_argument("--name", required=True, help="transform class name")
    parser.add_argument("--hdu", type=int, default=0, help="source HDU index")
    parser.add_argument("--out", required=True, help="output FITS path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        cls = getattr(tf_transforms, args.name, None)
        if cls is None or not callable(cls):
            raise UsageError(f"unknown transform: {args.name}")
        transform = cls()
        tensor = torchfits.read_tensor(args.input, hdu=args.hdu)
        if not isinstance(tensor, torch.Tensor):
            raise IoError(
                f"{args.input}:{args.hdu} read_tensor did not return a tensor"
            )
        # Stretches/norms use float ops; integer HDUs must be promoted first.
        if not tensor.is_floating_point():
            tensor = tensor.float()
        header = torchfits.get_header(args.input, args.hdu)
        result = transform(tensor)
        if not isinstance(result, torch.Tensor):
            raise IoError(f"{args.name} did not return a tensor")
        # Do not reuse integer BITPIX headers for float outputs.
        if result.is_floating_point():
            header = None
        torchfits.write_tensor(args.out, result, header=header, overwrite=True)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
