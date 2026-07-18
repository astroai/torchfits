"""``torchfits probe`` — quick file/URL metadata probe."""

from __future__ import annotations

import argparse
import importlib
import ipaddress
import socket
import urllib.parse
import urllib.request
from typing import Any

from ..header_parser import fast_parse_header_cards
from .cmds_info import run as info_run
from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_emit_format_args,
    add_hdu_arg,
    emit_records,
    is_remote_path,
    resolve_emit_format,
    resolve_paths,
)

_DEFAULT_HEADER_BYTES = 2880 * 2
_DEFAULT_TIMEOUT = 30.0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "probe",
        help="probe local files (like info) or remote HTTP(S)/vos header peek",
    )
    parser.add_argument("paths", nargs="*", help="FITS paths or URLs")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(
        parser,
        help="comma-separated HDU indices for local files only (ignored for remote)",
    )
    parser.add_argument(
        "--header-bytes",
        "--bytes",
        type=int,
        default=_DEFAULT_HEADER_BYTES,
        dest="header_bytes",
        help=(
            f"bytes to fetch for remote header peek "
            f"(default: {_DEFAULT_HEADER_BYTES}; --bytes is a deprecated alias)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help=f"remote HTTP timeout seconds (default: {_DEFAULT_TIMEOUT})",
    )
    add_emit_format_args(parser)
    parser.set_defaults(func=run)


def _is_vos_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("vos://") or lowered.startswith("vos:")


def _probe_vos(path: str, *, header_bytes: int) -> dict[str, Any]:
    """Header probe via optional ``vos`` client (CANFAR VOSpace)."""
    # importlib: optional dep; avoids mypy import-not-found vs import-untyped flip-flop
    try:
        vos = importlib.import_module("vos")
    except ImportError as exc:
        raise UsageError(
            "vos: probe requires the optional 'vos' package "
            "(pip/pixi install vos); HTTP(S) probe needs no extra dep"
        ) from exc
    uri = path if "://" in path else path.replace("vos:", "vos://", 1)
    try:
        client = vos.Client()
        with client.open(uri, mode="rb") as handle:
            chunk = handle.read(header_bytes)
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc
    if not chunk or len(chunk) < 2880:
        raise IoError(f"{path}: could not read FITS header from VOSpace")
    cards = _cards_map(chunk[:2880].decode("latin-1", errors="replace"))
    return _remote_record(path, cards, source="vos")


def _is_http_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _cards_map(header_text: str) -> dict[str, Any]:
    return {key: value for key, value, _comment in fast_parse_header_cards(header_text)}


def _remote_record(path: str, cards: dict[str, Any], *, source: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path,
        "hdu": 0,
        "simple": cards.get("SIMPLE"),
        "bitpix": cards.get("BITPIX"),
        "naxis": cards.get("NAXIS"),
        "source": source,
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


def _is_internal_url(url: str) -> bool:
    """True if *url*'s host resolves to any non-public address (or cannot resolve).

    Resolving with :func:`socket.getaddrinfo` and rejecting when *any* returned
    address is private/loopback/link-local/reserved/multicast/unspecified closes
    the DNS-rebinding and multi-record SSRF gaps left by a single
    ``gethostbyname`` lookup. Resolution failure is treated as internal (block).
    """
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:
        return True
    if not hostname:
        return True
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        return True
    for info in infos:
        ip = str(info[4][0]).split("%", 1)[0]
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            return True
    return False


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop so redirects cannot reach internal hosts."""

    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        if _is_internal_url(newurl):
            raise IoError(
                f"{newurl}: redirect to internal or private networks is blocked "
                "for security reasons"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _probe_http(url: str, *, header_bytes: int, timeout: float) -> dict[str, Any]:
    if _is_internal_url(url):
        raise IoError(
            f"{url}: access to internal or private networks is blocked "
            "for security reasons"
        )
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{header_bytes - 1}"},
    )
    opener = urllib.request.build_opener(_ValidatingRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            chunk = response.read(header_bytes)
    except IoError:
        raise
    except Exception as exc:
        raise IoError(f"{url}: {exc}") from exc
    if len(chunk) < 2880:
        raise IoError(f"{url}: response too short for FITS header")
    cards = _cards_map(chunk[:2880].decode("latin-1", errors="replace"))
    return _remote_record(url, cards, source="http")


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    header_bytes = max(2880, int(args.header_bytes))
    timeout = float(args.timeout)
    remote_records: list[dict[str, Any]] = []
    local_paths: list[str] = []
    for path in paths:
        if not is_remote_path(path):
            local_paths.append(path)
            continue
        if _is_vos_path(path):
            remote_records.append(_probe_vos(path, header_bytes=header_bytes))
            continue
        if _is_http_path(path):
            remote_records.append(
                _probe_http(path, header_bytes=header_bytes, timeout=timeout)
            )
            continue
        raise IoError(f"remote paths are not supported: {path}")

    if remote_records and local_paths:
        raise UsageError("mixing local paths and remote URLs is not supported")

    if remote_records:
        emit_records(remote_records, format=resolve_emit_format(args))
        return EXIT_OK

    args.paths = local_paths
    return info_run(args)
