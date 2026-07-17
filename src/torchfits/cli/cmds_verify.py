"""``torchfits verify`` — checksum verification."""

from __future__ import annotations

import argparse

import torchfits

from .common import (
    EXIT_OK,
    EXIT_VERIFY_FAIL,
    IoError,
    emit_records,
    header_extname,
    resolve_paths,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("verify", help="verify FITS checksum keywords")
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
    all_ok = True
    for path in paths:
        try:
            with torchfits.open(path) as hdul:
                indices = selected_hdu_indices(len(hdul), args.hdu)
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
        for index in indices:
            result = torchfits.verify_checksums(path, hdu=index)
            header = torchfits.get_header(path, index)
            ok = bool(result.get("ok"))
            all_ok = all_ok and ok
            records.append(
                {
                    "file": path,
                    "hdu": index,
                    "name": header_extname(header, index),
                    "ok": ok,
                    "datastatus": result.get("datastatus"),
                    "hdustatus": result.get("hdustatus"),
                }
            )
    if args.json or args.jsonl:
        emit_records(records, json_mode=args.json, jsonl=args.jsonl)
    else:
        for record in records:
            status = "ok" if record["ok"] else "FAIL"
            print(
                f"{record['file']}:{record['hdu']} {record['name']} {status} "
                f"data={record['datastatus']} hdu={record['hdustatus']}"
            )
    return EXIT_OK if all_ok else EXIT_VERIFY_FAIL
