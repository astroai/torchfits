"""``torchfits setkey`` — set or rename FITS header keywords (MEF / multi-file)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torchfits
from torchfits.hdu import Header

from .common import EXIT_OK, IoError, UsageError, add_hdu_arg, resolve_paths


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "setkey",
        help="set or rename a FITS header keyword (supports HIERARCH / MEF / files)",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="input FITS path(s); use --stdin for a path list",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read additional paths from stdin (one per line)",
    )
    parser.add_argument(
        "-k",
        "--key",
        help="header keyword to set (supports long / HIERARCH names)",
    )
    parser.add_argument("--value", help="new value when setting --key")
    parser.add_argument(
        "--rename",
        metavar="OLD=NEW",
        help="rename a keyword (e.g. --rename OBJECT=TARGET)",
    )
    add_hdu_arg(
        parser,
        default="0",
        help="HDU index, comma list, or 'all' (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="output path when editing a single input (default: in place)",
    )
    parser.add_argument(
        "--out-dir",
        help="write edited copies into this directory (multi-file)",
    )
    parser.set_defaults(func=run)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _normalize_keyword(raw: str) -> str:
    key = raw.strip()
    if not key or key.upper() in {"HISTORY", "COMMENT"}:
        raise UsageError("keyword must be a normal FITS card (not HISTORY/COMMENT)")
    # Preserve HIERARCH / long keys; uppercase short FITS keywords.
    if key.upper().startswith("HIERARCH ") or " " in key or len(key) > 8:
        if not key.upper().startswith("HIERARCH"):
            return f"HIERARCH {key}"
        return key
    return key.upper()


def _parse_hdus(spec: str, n_hdus: int) -> list[int]:
    text = spec.strip().lower()
    if text in {"all", "*"}:
        return list(range(n_hdus))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_hdus:
            raise UsageError(f"HDU index out of range: {idx} (file has {n_hdus})")
        out.append(idx)
    if not out:
        raise UsageError("--hdu must list indices or 'all'")
    return out


def _apply_edits(
    path: str,
    *,
    hdu_spec: str,
    key: str | None,
    value: str | None,
    rename: str | None,
) -> None:
    with torchfits.open(path) as hdul:
        n_hdus = len(hdul)
    for hdu in _parse_hdus(hdu_spec, n_hdus):
        header = Header(torchfits.read_header(path, hdu))
        if rename:
            if "=" not in rename:
                raise UsageError("--rename must be OLD=NEW")
            old_raw, _, new_raw = rename.partition("=")
            old_key = _normalize_keyword(old_raw)
            new_key = _normalize_keyword(new_raw)
            if old_key not in header:
                raise IoError(f"{path}[{hdu}]: missing keyword {old_key}")
            header[new_key] = header[old_key]
            del header[old_key]
        if key is not None:
            if value is None:
                raise UsageError("--value is required with --key")
            header[_normalize_keyword(key)] = _parse_value(value)
        torchfits.io._write_header_cards_if_supported(path, hdu, header)


def run(args: argparse.Namespace) -> int:
    if not args.key and not args.rename:
        raise UsageError("provide --key/--value and/or --rename OLD=NEW")
    if args.key and args.value is None and not args.rename:
        raise UsageError("--value is required with --key")

    paths = resolve_paths(args.inputs, use_stdin=args.stdin)
    if args.out and len(paths) != 1:
        raise UsageError("--out requires exactly one input path")
    if args.out_dir and args.out:
        raise UsageError("use either --out or --out-dir, not both")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for src in paths:
            if out_dir is not None:
                dest = str(out_dir / Path(src).name)
                with torchfits.open(src) as hdul:
                    hdul.write(dest, overwrite=True)
                target = dest
            elif args.out:
                if args.out != src:
                    with torchfits.open(src) as hdul:
                        hdul.write(args.out, overwrite=True)
                target = args.out
            else:
                target = src
            _apply_edits(
                target,
                hdu_spec=args.hdu,
                key=args.key,
                value=args.value,
                rename=args.rename,
            )
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(str(exc)) from exc
    return EXIT_OK
