"""``torchfits verify`` — checksum verification."""

from __future__ import annotations

import argparse

import torchfits

from .common import (
    EXIT_OK,
    EXIT_VERIFY_FAIL,
    IoError,
    add_emit_format_args,
    add_hdu_arg,
    emit_records,
    header_extname,
    resolve_emit_format,
    resolve_paths,
    selected_hdu_indices,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("verify", help="verify FITS checksum keywords")
    parser.add_argument("paths", nargs="*", help="FITS file paths")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(parser)
    add_emit_format_args(parser)
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
            header = torchfits.read_header(path, index)
            ok = bool(result.get("ok"))
            status_str = str(result.get("status", "fail"))
            all_ok = all_ok and ok
            records.append(
                {
                    "file": path,
                    "hdu": index,
                    "name": header_extname(header, index),
                    "ok": ok,
                    "status": status_str,
                    "datastatus": result.get("datastatus"),
                    "hdustatus": result.get("hdustatus"),
                }
            )
    fmt = resolve_emit_format(args)
    if fmt == "text":
        for record in records:
            status = record["status"]
            if status == "no_checksums":
                label = "OK (no checksum keywords)"
            elif status == "ok":
                label = "OK"
            else:
                label = "FAIL"
            print(
                f"{record['file']}:{record['hdu']} {record['name']} {label} "
                f"data={record['datastatus']} hdu={record['hdustatus']}"
            )
    else:
        emit_records(records, format=fmt)
    return EXIT_OK if all_ok else EXIT_VERIFY_FAIL
