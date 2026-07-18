"""Shared CLI helpers: paths, HDU selection, structured output, exit codes."""

from __future__ import annotations

import json
import sys
from typing import Any, Iterable, Iterator, TextIO

EXIT_OK = 0
EXIT_DIFF = 1
EXIT_USAGE = 2
EXIT_IO = 3
EXIT_VERIFY_FAIL = 4

_REMOTE_PREFIXES = ("http://", "https://", "vos://", "vos:")
_EMIT_FORMATS = ("text", "json", "jsonl")


class CliError(Exception):
    """Base CLI error with a stable exit code."""

    exit_code = EXIT_USAGE

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class UsageError(CliError):
    exit_code = EXIT_USAGE


class IoError(CliError):
    exit_code = EXIT_IO


def is_remote_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(_REMOTE_PREFIXES)


def resolve_paths(paths: list[str] | None, *, use_stdin: bool) -> list[str]:
    resolved = [p for p in (paths or []) if p]
    if use_stdin or (not resolved and not sys.stdin.isatty()):
        for line in sys.stdin:
            item = line.strip()
            if item:
                resolved.append(item)
    if not resolved:
        raise UsageError("no input paths (argv or stdin)")
    return resolved


def parse_hdu_list(hdu: str | None) -> list[int] | None:
    if hdu is None:
        return None
    out: list[int] = []
    for part in hdu.split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError as exc:
            raise UsageError(f"invalid HDU index: {piece!r}") from exc
    if not out:
        raise UsageError("--hdu requires at least one HDU index")
    return out


def selected_hdu_indices(num_hdus: int, hdu: str | None) -> list[int]:
    if num_hdus <= 0:
        return []
    wanted = parse_hdu_list(hdu)
    if wanted is None:
        return list(range(num_hdus))
    for idx in wanted:
        if idx < 0 or idx >= num_hdus:
            raise UsageError(f"HDU {idx} out of range (0..{num_hdus - 1})")
    return wanted


def hdu_type_name(header: Any, hdu_obj: Any) -> str:
    from .._hdu.table_hdu_ref import TableHDURef

    if isinstance(hdu_obj, TableHDURef):
        return "TABLE"
    xtension = str(header.get("XTENSION", "")).strip().upper()
    if xtension in {"BINTABLE", "TABLE", "A3DTABLE"}:
        return "TABLE"
    if xtension == "IMAGE":
        return "IMAGE"
    naxis = header.get("NAXIS")
    try:
        if naxis is not None and int(naxis) > 0:
            return "IMAGE"
    except (TypeError, ValueError):
        pass
    return "UNKNOWN"


def header_extname(header: Any, index: int) -> str:
    name = header.get("EXTNAME")
    if name:
        return str(name)
    return "PRIMARY" if index == 0 else f"HDU{index}"


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def add_emit_format_args(parser: Any) -> None:
    """Add ``--format`` / ``--json`` / ``--jsonl`` to an inventory command."""
    parser.add_argument(
        "--format",
        choices=_EMIT_FORMATS,
        default=None,
        help="output format: text (default), json, or jsonl",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON array (alias)")
    parser.add_argument(
        "--jsonl", action="store_true", help="emit JSONL records (alias)"
    )


def resolve_emit_format(args: Any) -> str:
    """Resolve structured-output format from ``--format`` / ``--json`` / ``--jsonl``."""
    fmt = getattr(args, "format", None)
    json_flag = bool(getattr(args, "json", False))
    jsonl_flag = bool(getattr(args, "jsonl", False))
    if json_flag and jsonl_flag:
        raise UsageError("use only one of --json or --jsonl")
    if jsonl_flag:
        if fmt is not None and fmt != "jsonl":
            raise UsageError("conflicting --format and --jsonl")
        return "jsonl"
    if json_flag:
        if fmt is not None and fmt != "json":
            raise UsageError("conflicting --format and --json")
        return "json"
    return fmt or "text"


def emit_records(
    records: Iterable[dict[str, Any]],
    *,
    format: str = "text",
    stream: TextIO | None = None,
) -> None:
    """Emit inventory records as text, JSON, or JSONL."""
    out = stream or sys.stdout
    items = list(records)
    if format == "jsonl":
        for record in items:
            print(json.dumps(record, default=json_default), file=out)
        return
    if format == "json":
        print(json.dumps(items, default=json_default, indent=2), file=out)
        return
    for record in items:
        parts = [f"{key}={record[key]!r}" for key in sorted(record)]
        print(" ".join(parts), file=out)


def iter_file_hdu_pairs(
    paths: Iterable[str], hdu: str | None
) -> Iterator[tuple[str, int, Any]]:
    import torchfits

    for path in paths:
        if is_remote_path(path):
            raise IoError(f"remote paths are not supported: {path}")
        try:
            with torchfits.open(path) as hdul:
                for index in selected_hdu_indices(len(hdul), hdu):
                    yield path, index, hdul[index]
        except CliError:
            raise
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
