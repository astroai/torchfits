"""``torchfits transform`` — apply a named torchfits.transforms class."""

from __future__ import annotations

import argparse
import inspect
from typing import Any

import torch

import torchfits

from .common import EXIT_OK, IoError, UsageError, add_hdu_arg


def _coerce_transform_kwarg_value(raw: str) -> bool | int | float | str:
    """Coerce a ``key=val`` string value to bool/int/float, else leave as str."""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_transform_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Split ``Name`` or ``Name:key=val,key2=val2`` into (name, kwargs)."""
    name, sep, kwargs_str = spec.partition(":")
    if not sep:
        return name, {}
    kwargs: dict[str, Any] = {}
    for piece in kwargs_str.split(","):
        piece = piece.strip()
        if not piece:
            continue
        key, eq, value = piece.partition("=")
        key = key.strip()
        if not eq or not key:
            raise UsageError(f"invalid --name kwarg (expected key=val): {piece!r}")
        kwargs[key] = _coerce_transform_kwarg_value(value.strip())
    return name, kwargs


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "transform", help="apply a torchfits.transforms class"
    )
    parser.add_argument("input", help="input FITS image path")
    parser.add_argument(
        "--name",
        required=True,
        help=(
            "transform class name; append :key=val,key2=val2 to pass "
            "constructor kwargs, e.g. ArcsinhStretch:a=2.0"
        ),
    )
    add_hdu_arg(parser, type=int, default=0, help="source HDU index")
    parser.add_argument("-o", "--out", required=True, help="output FITS path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import torchfits.transforms as tf_transforms

    try:
        name, kwargs = _parse_transform_spec(args.name)
        public = set(tf_transforms.__all__)
        if name not in public:
            raise UsageError(f"unknown transform: {name}")
        cls = getattr(tf_transforms, name, None)
        if cls is None or not callable(cls):
            raise UsageError(f"unknown transform: {name}")
        if kwargs:
            valid_params = set(inspect.signature(cls).parameters)
            unknown = sorted(set(kwargs) - valid_params)
            if unknown:
                raise UsageError(f"unknown kwarg(s) for {name}: {unknown}")
        transform = cls(**kwargs)
        tensor = torchfits.read_tensor(args.input, hdu=args.hdu)
        if not isinstance(tensor, torch.Tensor):
            raise IoError(
                f"{args.input}:{args.hdu} read_tensor did not return a tensor"
            )
        # Stretches/norms use float ops; integer HDUs must be promoted first.
        if not tensor.is_floating_point():
            tensor = tensor.float()
        header = torchfits.read_header(args.input, args.hdu)
        result = transform(tensor)
        if not isinstance(result, torch.Tensor):
            raise IoError(f"{name} did not return a tensor")
        # Do not reuse integer BITPIX headers for float outputs.
        if result.is_floating_point():
            header = None
        torchfits.write_tensor(args.out, result, header=header, overwrite=True)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
