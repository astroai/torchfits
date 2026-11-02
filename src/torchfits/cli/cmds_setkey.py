"""``torchfits setkey`` — set, rename, or delete FITS header keywords."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
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
    hdu_out_of_range,
    is_compressed_image_header,
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
        nargs="*",
        help="input FITS path(s); @list expands one path per line; --stdin may be the only source",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read paths from stdin (one per line; may be used without positional paths)",
    )
    parser.add_argument(
        "-k",
        "--key",
        help="header keyword to set (supports long / HIERARCH names)",
    )
    parser.add_argument(
        "--value",
        help=(
            "new value when setting --key: true/false -> bool, plain numeric "
            "literals (12, -3, 1.5e3) -> int/float, anything else (e.g. '007', "
            "'1_000') -> string; a non-finite number is a usage error"
        ),
    )
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


# Plain FITS-style numeric literals only: Python's int()/float() also accept
# '1_000', '007', ' 5 ', 'inf' and 'nan', which would silently turn a string
# value (an ID with leading zeros, say) into a number.
_INT_LITERAL = re.compile(r"[+-]?(?:0|[1-9][0-9]*)")
_FLOAT_LITERAL = re.compile(
    r"[+-]?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INT_LITERAL.fullmatch(raw):
        return int(raw)
    if _FLOAT_LITERAL.fullmatch(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise UsageError(f"--value {raw!r} overflows a FITS floating-point value")
        return value
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


# Cards that define the HDU's data layout. CFITSIO maintains them; an edit
# would either be dropped silently or corrupt the file, so they are refused.
_STRUCTURAL_KEY = re.compile(
    r"^(SIMPLE|XTENSION|BITPIX|NAXIS[0-9]*|PCOUNT|GCOUNT|TFIELDS|TFORM[0-9]+"
    r"|TBCOL[0-9]+|EXTEND|THEAP|END)$"
)
# Tile-compression cards: on a compressed HDU they describe how to rebuild
# the image from the tile table.
_COMPRESSION_KEY = re.compile(
    r"^Z(IMAGE|SIMPLE|TENSION|EXTEND|BITPIX|NAXIS[0-9]*|TILE[0-9]+|CMPTYPE"
    r"|NAME[0-9]+|VAL[0-9]+|QUANTIZ|DITHER0|PCOUNT|GCOUNT|BLANK|SCALE|ZERO"
    r"|MASKCMP|HECKSUM|DATASUM|THEAP)$"
)


_CHECKSUM_KEYS = frozenset({"CHECKSUM", "DATASUM"})


def _reject_structural(key: str, *, compressed: bool = False, where: str = "") -> None:
    name = key.upper()
    if _STRUCTURAL_KEY.match(name) or (compressed and _COMPRESSION_KEY.match(name)):
        prefix = f"{where}: " if where else ""
        raise UsageError(
            f"{prefix}{key} is a structural keyword and cannot be edited with setkey"
        )


def _header_key(header: Any, key: str) -> str | None:
    """The spelling under which ``key`` is stored in ``header``, if present.

    ``_normalize_keyword`` spells long keys ``HIERARCH X``, but a parsed
    header stores a HIERARCH card under its bare name ``X``.
    """
    if key in header:
        return key
    if key.upper().startswith("HIERARCH "):
        bare = key[len("HIERARCH ") :].strip()
        if bare in header:
            return bare
    return None


def _parse_hdus(spec: str, n_hdus: int, path: str | None = None) -> list[int]:
    text = spec.strip().lower()
    if text in {"all", "*"}:
        return list(range(n_hdus))
    out: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError as exc:
            raise UsageError(f"invalid HDU index: {part!r}") from exc
        if idx < 0 or idx >= n_hdus:
            raise hdu_out_of_range(path, idx, n_hdus)
        # Same rule as common.parse_hdu_list: a repeated index would plan the
        # same card edits twice. The edits are idempotent today, but the plan
        # length is what other commands turn into an output HDU count.
        if idx in seen:
            raise UsageError(f"duplicate HDU index in --hdu: {idx}")
        seen.add(idx)
        out.append(idx)
    if not out:
        raise UsageError("--hdu must list indices or 'all'")
    return out


def _prepare_edits(
    *,
    key: str | None,
    value: str | None,
    rename: str | None,
    delete_keys: list[str],
) -> tuple[str | None, Any, tuple[str, str] | None, list[str]]:
    """Normalize and validate the file-independent parts of an edit request."""
    del_keys = [_normalize_keyword(raw) for raw in delete_keys]
    rename_pair: tuple[str, str] | None = None
    if rename is not None:
        if "=" not in rename:
            raise UsageError("--rename must be OLD=NEW")
        old_raw, _, new_raw = rename.partition("=")
        old_key = _normalize_keyword(old_raw)
        new_key = _normalize_keyword(new_raw)
        if old_key == new_key:
            raise UsageError(f"--rename {old_raw}={new_raw} names the same keyword")
        rename_pair = (old_key, new_key)
    set_key = _normalize_keyword(key) if key is not None else None
    for edited in (*del_keys, *(rename_pair or ()), *([set_key] if set_key else [])):
        _reject_structural(edited)
    if key is None:
        set_value = None
    elif value is None:  # unreachable: run() validates --key/--value pairing
        raise UsageError("--key requires --value")
    else:
        set_value = _parse_value(value)
    return set_key, set_value, rename_pair, del_keys


_Plan = list[tuple[int, list[tuple[Any, ...]], bool]]


def _plan_edits(
    path: str,
    *,
    hdu_spec: str,
    set_key: str | None,
    set_value: Any,
    rename_pair: tuple[str, str] | None,
    del_keys: list[str],
) -> _Plan:
    """Validate the edit against ``path``'s headers; read-only."""
    with torchfits.open(path) as hdul:
        n_hdus = len(hdul)
    hdus = _parse_hdus(hdu_spec, n_hdus, path)

    # Phase 1: validate every requested edit against the file's current
    # headers and resolve the exact card updates. Nothing is written until the
    # whole batch is known to succeed, so a failed run leaves the file
    # untouched.
    plans: list[list[tuple[Any, ...]]] = []
    refresh: list[bool] = []
    for hdu in hdus:
        header = Header(torchfits.read_header(path, hdu))
        # An HDU that carried CHECKSUM/DATASUM gets them recomputed after the
        # edit (as fthedit / astropy do); a stale CHECKSUM would make verify
        # fail. Edits that name those cards themselves are left as requested.
        touched = {*del_keys, *(rename_pair or ())}
        if set_key is not None:
            touched.add(set_key)
        refresh.append(
            any(k in header for k in _CHECKSUM_KEYS) and not touched & _CHECKSUM_KEYS
        )
        if is_compressed_image_header(header):
            edited_keys = [*del_keys, *(rename_pair or ())]
            if set_key is not None:
                edited_keys.append(set_key)
            for edited in edited_keys:
                _reject_structural(edited, compressed=True, where=f"{path}[{hdu}]")
        ops: list[tuple[Any, ...]] = []
        for del_key in del_keys:
            present = _header_key(header, del_key)
            if present is None:
                raise UsageError(f"{path}[{hdu}]: missing keyword {del_key}")
            ops.append(("delete", present))
            del header[present]
        if rename_pair is not None:
            old_key, new_key = rename_pair
            present = _header_key(header, old_key)
            if present is None:
                raise UsageError(f"{path}[{hdu}]: missing keyword {old_key}")
            if _header_key(header, new_key) is not None:
                raise UsageError(
                    f"{path}[{hdu}]: cannot rename {old_key} to {new_key}: "
                    f"{new_key} already exists"
                )
            card = next(c for c in header.cards if c.key == present)
            ops.append(("set", new_key, card.value, card.comment))
            ops.append(("delete", present))
            header[new_key] = card.value
            del header[present]
        if set_key is not None:
            ops.append(("set", set_key, set_value, ""))
            header[set_key] = set_value
        plans.append(ops)
    return list(zip(hdus, plans, refresh))


def _write_plan(path: str, plan: _Plan) -> None:
    # Phase 2: apply via CFITSIO card update/delete only (never an HDUList
    # rewrite) and only for the edited cards -- replaying a full header would
    # append a second copy of the file's HISTORY/COMMENT cards.
    for hdu, ops, refresh_sums in plan:
        for op in ops:
            if op[0] == "delete":
                torchfits.io._delete_header_key_if_supported(path, hdu, op[1])
            else:
                _, card_key, card_value, comment = op
                write_value = (card_value, comment) if comment else card_value
                torchfits.io._write_header_cards_if_supported(
                    path, hdu, {card_key: write_value}
                )
        if refresh_sums and ops:
            torchfits.write_checksums(path, hdu)
            torchfits.io._invalidate_path_caches(path)


def _edit_one(job: tuple[str, str, str, tuple[Any, ...]]) -> None:
    src, dest, hdu_spec, (set_key, set_value, rename_pair, del_keys) = job
    try:
        # Validate against the source first: a rejected edit must not leave
        # an unedited copy behind at the destination.
        plan = _plan_edits(
            src,
            hdu_spec=hdu_spec,
            set_key=set_key,
            set_value=set_value,
            rename_pair=rename_pair,
            del_keys=del_keys,
        )
        if os.path.realpath(src) == os.path.realpath(dest):
            # dest resolves to src (e.g. --out-dir reaching the input's
            # directory through a symlink): edit in place, as with dest == src.
            _write_plan(src, plan)
            return
        created = not os.path.lexists(dest)
        try:
            shutil.copy2(src, dest)
            _write_plan(dest, plan)
        except BaseException:
            # Never leave a half-edited copy that this run created.
            if created:
                with contextlib.suppress(OSError):
                    os.remove(dest)
            raise
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
    if args.key and args.value is None:
        raise UsageError("--value is required with --key")
    if args.value is not None and not args.key:
        raise UsageError("--value requires --key")
    edits = _prepare_edits(
        key=args.key,
        value=args.value,
        rename=args.rename,
        delete_keys=delete_keys,
    )

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
        ensure_unique_basenames(paths, directory=out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str, tuple[Any, ...]]] = []
    for src in paths:
        if out_dir is not None:
            dest = str(out_dir / Path(src).name)
        elif args.out:
            dest = str(args.out)
        else:
            dest = src
        jobs.append((src, dest, args.hdu, edits))

    file_jobs = resolve_file_jobs(int(args.file_jobs), len(jobs))
    run_file_jobs(jobs, _edit_one, file_jobs)
    return EXIT_OK
