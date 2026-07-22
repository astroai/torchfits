"""``torchfits setkey`` — set, rename, or delete FITS header keywords."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torchfits
from torchfits.hdu import Header

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    add_hdu_arg,
    ensure_unique_basenames,
    expand_at_list_paths,
    resolve_file_jobs,
    resolve_paths,
    run_file_jobs,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "setkey",
        help="set, rename, or delete a FITS header keyword (MEF / multi-file)",
        description=(
            "Set (--key/--value), rename (--rename OLD=NEW), or delete (--delete KEY) "
            "header keywords in place via CFITSIO card update/delete (preserves "
            "tile-compressed HDUs). Paths may include @list files. "
            "-J fans out across files; copies use a binary file copy then edit."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="input FITS path(s); @list expands one path per line; --stdin adds more",
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
    parser.add_argument(
        "--delete",
        metavar="KEY",
        action="append",
        default=None,
        help="delete keyword(s); repeatable",
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
    add_file_jobs_arg(parser)
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
    delete_keys: list[str],
) -> None:
    with torchfits.open(path) as hdul:
        n_hdus = len(hdul)
    for hdu in _parse_hdus(hdu_spec, n_hdus):
        header = Header(torchfits.read_header(path, hdu))
        for raw in delete_keys:
            del_key = _normalize_keyword(raw)
            if del_key not in header:
                raise IoError(f"{path}[{hdu}]: missing keyword {del_key}")
            torchfits.io._delete_header_key_if_supported(path, hdu, del_key)
            del header[del_key]
        if rename:
            if "=" not in rename:
                raise UsageError("--rename must be OLD=NEW")
            old_raw, _, new_raw = rename.partition("=")
            old_key = _normalize_keyword(old_raw)
            new_key = _normalize_keyword(new_raw)
            if old_key not in header:
                raise IoError(f"{path}[{hdu}]: missing keyword {old_key}")
            header[new_key] = header[old_key]
            torchfits.io._write_header_cards_if_supported(path, hdu, header)
            torchfits.io._delete_header_key_if_supported(path, hdu, old_key)
            del header[old_key]
        if key is not None:
            if value is None:
                raise UsageError("--value is required with --key")
            header[_normalize_keyword(key)] = _parse_value(value)
            torchfits.io._write_header_cards_if_supported(path, hdu, header)


def _edit_one(job: tuple[str, str, argparse.Namespace, list[str]]) -> None:
    src, dest, args, delete_keys = job
    try:
        if dest != src:
            shutil.copy2(src, dest)
            target = dest
        else:
            target = src
        _apply_edits(
            target,
            hdu_spec=args.hdu,
            key=args.key,
            value=args.value,
            rename=args.rename,
            delete_keys=delete_keys,
        )
    except UsageError:
        raise
    except IoError:
        raise
    except Exception as exc:
        raise IoError(f"{src}: {exc}") from exc


def run(args: argparse.Namespace) -> int:
    delete_keys = list(args.delete or [])
    if not args.key and not args.rename and not delete_keys:
        raise UsageError("provide --key/--value, --rename OLD=NEW, and/or --delete KEY")
    if args.key and args.value is None and not args.rename and not delete_keys:
        raise UsageError("--value is required with --key")

    paths = expand_at_list_paths(resolve_paths(args.inputs, use_stdin=args.stdin))
    if args.out and len(paths) != 1:
        raise UsageError("--out requires exactly one input path")
    if args.out_dir and args.out:
        raise UsageError("use either --out or --out-dir, not both")

    # In-place parallel edits on the same path race CFITSIO writers.
    if not args.out and not args.out_dir:
        seen: dict[str, str] = {}
        for path in paths:
            key = str(Path(path).resolve()) if Path(path).exists() else path
            prior = seen.get(key)
            if prior is not None:
                raise UsageError(
                    f"duplicate in-place path under -J: {path!r} (also {prior})"
                )
            seen[key] = path

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        ensure_unique_basenames(paths)
        out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, argparse.Namespace, list[str]]] = []
    for src in paths:
        if out_dir is not None:
            dest = str(out_dir / Path(src).name)
        elif args.out:
            dest = str(args.out)
        else:
            dest = src
        jobs.append((src, dest, args, delete_keys))

    file_jobs = resolve_file_jobs(int(args.file_jobs), len(jobs))
    run_file_jobs(jobs, _edit_one, file_jobs)
    return EXIT_OK
