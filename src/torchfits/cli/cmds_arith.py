"""``torchfits arith`` — image arithmetic by a constant."""

from __future__ import annotations

import argparse
from typing import Callable

import torch

import torchfits

from .common import EXIT_OK, IoError, UsageError, add_hdu_arg

_OPS: dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "add": lambda t, v: t + v,
    "sub": lambda t, v: t - v,
    "mul": lambda t, v: t * v,
    "div": lambda t, v: t / v,
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("arith", help="apply image ±×÷ by a constant")
    parser.add_argument("input", help="input FITS image path")
    parser.add_argument(
        "--op", required=True, choices=tuple(_OPS), help="arithmetic op"
    )
    parser.add_argument("--value", required=True, type=float, help="scalar operand")
    add_hdu_arg(parser, type=int, default=0, help="source HDU index")
    parser.add_argument("-o", "--out", required=True, help="output FITS path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        tensor = torchfits.read_tensor(args.input, hdu=args.hdu)
        if not isinstance(tensor, torch.Tensor):
            raise IoError(
                f"{args.input}:{args.hdu} read_tensor did not return a tensor"
            )
        header = torchfits.read_header(args.input, args.hdu)
        result = _OPS[args.op](tensor, args.value)
        torchfits.write_tensor(args.out, result, header=header, overwrite=True)
    except UsageError:
        raise
    except ZeroDivisionError as exc:
        raise UsageError("division by zero") from exc
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
