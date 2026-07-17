"""``torchfits probe`` — quick file/URL metadata probe."""

from __future__ import annotations

import argparse
import urllib.request
from typing import Any

from ..header_parser import fast_parse_header_cards
from .cmds_info import run as info_run
from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    emit_records,
    is_remote_path,
    resolve_paths,
)

_HEADER_BYTES = 2880 * 2


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("probe", help="probe local files or remote URLs")
    parser.add_argument("paths", nargs="*", help="FITS paths or URLs")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    parser.add_argument("--hdu", help="comma-separated HDU indices (default: all)")
    parser.add_argument("--json", action="store_true", help="emit JSON array")
    parser.add_argument("--jsonl", action="store_true", help="emit JSONL records")
    parser.set_defaults(func=run)


def _is_vos_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("vos://") or lowered.startswith("vos:")


def _probe_vos(path: str) -> dict[str, Any]:
    """Header probe via optional ``vos`` client (CANFAR VOSpace)."""
    try:
        from vos import Client  # type: ignore[import-untyped]
    except ImportError as exc:
        raise UsageError(
            "vos: probe requires the optional 'vos' package "
            "(pip/pixi install vos); HTTP(S) probe needs no extra dep"
        ) from exc
    uri = path if "://" in path else path.replace("vos:", "vos://", 1)
    try:
        client = Client()
        with client.open(uri, mode="rb") as handle:
            chunk = handle.read(_HEADER_BYTES)
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc
    if not chunk or len(chunk) < 2880:
        raise IoError(f"{path}: could not read FITS header from VOSpace")
    cards = _cards_map(chunk[:2880].decode("latin-1", errors="replace"))
    record: dict[str, Any] = {
        "file": path,
        "hdu": 0,
        "simple": cards.get("SIMPLE"),
        "bitpix": cards.get("BITPIX"),
        "naxis": cards.get("NAXIS"),
        "source": "vos",
    }
    try:
        naxis = int(cards.get("NAXIS", 0) or 0)
    except (TypeError, ValueError):
        naxis = 0
    record["type"] = "IMAGE" if naxis > 0 else "UNKNOWN"
    if naxis >= 1:
        record["naxis1"] = cards.get("NAXIS1")
    if naxis >= 2:
        record["naxis2"] = cards.get("NAXIS2")
    return record


def _is_http_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _cards_map(header_text: str) -> dict[str, Any]:
    return {key: value for key, value, _comment in fast_parse_header_cards(header_text)}


def _probe_http(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{_HEADER_BYTES - 1}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            chunk = response.read(_HEADER_BYTES)
    except Exception as exc:
        raise IoError(f"{url}: {exc}") from exc
    if len(chunk) < 2880:
        raise IoError(f"{url}: response too short for FITS header")
    cards = _cards_map(chunk[:2880].decode("latin-1", errors="replace"))
    record: dict[str, Any] = {
        "file": url,
        "hdu": 0,
        "simple": cards.get("SIMPLE"),
        "bitpix": cards.get("BITPIX"),
        "naxis": cards.get("NAXIS"),
    }
    try:
        naxis = int(cards.get("NAXIS", 0) or 0)
    except (TypeError, ValueError):
        naxis = 0
    record["type"] = "IMAGE" if naxis > 0 else "UNKNOWN"
    if naxis >= 1:
        record["naxis1"] = cards.get("NAXIS1")
    if naxis >= 2:
        record["naxis2"] = cards.get("NAXIS2")
    return record


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    remote_records: list[dict[str, Any]] = []
    local_paths: list[str] = []
    for path in paths:
        if not is_remote_path(path):
            local_paths.append(path)
            continue
        if _is_vos_path(path):
            remote_records.append(_probe_vos(path))
            continue
        if _is_http_path(path):
            remote_records.append(_probe_http(path))
            continue
        raise IoError(f"remote paths are not supported: {path}")

    if remote_records and local_paths:
        raise UsageError("mixing local paths and remote URLs is not supported")

    if remote_records:
        emit_records(remote_records, json_mode=args.json, jsonl=args.jsonl)
        if not (args.json or args.jsonl):
            for record in remote_records:
                parts = [f"{key}={record[key]!r}" for key in sorted(record)]
                print(" ".join(parts))
        return EXIT_OK

    args.paths = local_paths
    return info_run(args)
