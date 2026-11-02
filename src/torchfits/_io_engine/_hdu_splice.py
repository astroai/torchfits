"""Byte-level HDU splicing for in-place FITS edits.

HDU edits copy every HDU they do not insert or replace byte-for-byte and write
only the new HDU. Decoding an untouched HDU and writing it back changed its
structure: tile-compressed images came back decompressed, scaled storage was
re-derived. The helpers here locate HDUs by their raw 2880-byte header blocks,
assemble the output from byte ranges, and publish it atomically.
"""

from __future__ import annotations

import gzip
from typing import IO, Iterable, List, Sequence, Tuple, Union

from ._write_helpers import _atomic_replace_target, _invalidate_written_target

FITS_BLOCK = 2880
_CARD = 80
_COPY_CHUNK = 1 << 20

# A piece of the assembled file: literal bytes, or ``(path, start, end)``.
Piece = Union[bytes, Tuple[str, int, int]]


def open_fits_bytes(path: str) -> IO[bytes]:
    """Binary stream of the FITS bytes at ``path`` (gunzipped for gzip files)."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")  # type: ignore[return-value]
    return open(path, "rb")


def _header_cards(raw: bytes) -> List[str]:
    """80-character cards of a raw header, up to (excluding) END."""
    text = raw.decode("latin-1")
    cards: List[str] = []
    for i in range(0, len(text), _CARD):
        card = text[i : i + _CARD]
        if card.startswith("END") and not card[3:].strip():
            break
        cards.append(card)
    return cards


def _int_card(cards: Sequence[str], key: str, default: int = 0) -> int:
    for card in cards:
        if card[:8].rstrip() == key and card[8:10] == "= ":
            value = card[10:].split("/", 1)[0].strip()
            try:
                return int(value)
            except ValueError:
                return default
    return default


def _logical_card(cards: Sequence[str], key: str) -> bool:
    for card in cards:
        if card[:8].rstrip() == key and card[8:10] == "= ":
            return card[10:].split("/", 1)[0].strip() == "T"
    return False


def _data_nbytes(cards: Sequence[str]) -> int:
    """Data-unit size in bytes (FITS 4.0 sec. 4.4.1.1), padding excluded."""
    naxis = _int_card(cards, "NAXIS")
    if naxis <= 0:
        return 0
    # Random groups (NAXIS1 = 0, GROUPS = T) leave NAXIS1 out of the product.
    first = (
        2 if _logical_card(cards, "GROUPS") and _int_card(cards, "NAXIS1") == 0 else 1
    )
    count = 1
    for i in range(first, naxis + 1):
        count *= _int_card(cards, f"NAXIS{i}")
    bitpix = abs(_int_card(cards, "BITPIX", 8))
    gcount = _int_card(cards, "GCOUNT", 1)
    pcount = _int_card(cards, "PCOUNT", 0)
    return bitpix // 8 * gcount * (pcount + count)


def _read_header(fh: IO[bytes], pos: int) -> bytes:
    """Raw header blocks starting at ``pos`` (b'' at end of file)."""
    blocks: List[bytes] = []
    while True:
        block = fh.read(FITS_BLOCK)
        if not block and not blocks:
            return b""
        if len(block) < FITS_BLOCK:
            raise ValueError(f"truncated FITS header at byte {pos}")
        pos += FITS_BLOCK
        blocks.append(block)
        if any(
            block[i : i + 3] == b"END" and not block[i + 3 : i + _CARD].strip()
            for i in range(0, FITS_BLOCK, _CARD)
        ):
            return b"".join(blocks)


def scan_hdus(path: str) -> List[Tuple[int, int, int, bytes]]:
    """``(header_start, data_start, end, raw_header)`` of every HDU.

    Offsets are into the FITS byte stream (after gunzip for a gzip file);
    ``end`` is the record-aligned start of the next HDU. One forward pass.
    """
    hdus: List[Tuple[int, int, int, bytes]] = []
    with open_fits_bytes(path) as fh:
        pos = 0
        while True:
            if fh.tell() != pos:
                fh.seek(pos)
            raw = _read_header(fh, pos)
            if not raw:
                break
            data_start = pos + len(raw)
            nbytes = _data_nbytes(_header_cards(raw))
            end = data_start + -(-nbytes // FITS_BLOCK) * FITS_BLOCK
            hdus.append((pos, data_start, end, raw))
            pos = end
    return hdus


def hdu_byte_spans(path: str) -> List[Tuple[int, int, int]]:
    """``(header_start, data_start, end)`` of every HDU, from the raw bytes."""
    return [(start, data, end) for start, data, end, _raw in scan_hdus(path)]


def _copy_range(src: IO[bytes], dst: IO[bytes], start: int, end: int) -> None:
    src.seek(start)
    remaining = end - start
    while remaining > 0:
        chunk = src.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise ValueError(f"FITS file ends before byte {end}")
        dst.write(chunk)
        remaining -= len(chunk)


def publish_spliced(
    path: str, pieces: Iterable[Piece], *, restamp: Iterable[int] = ()
) -> None:
    """Atomically replace ``path`` with the concatenation of ``pieces``.

    ``restamp`` lists the HDU indices (in the new file) whose CHECKSUM/DATASUM
    are recomputed before the file is published.
    """
    with _atomic_replace_target(path) as temp_path:
        with open(temp_path, "wb") as dst:
            for piece in pieces:
                if isinstance(piece, bytes):
                    dst.write(piece)
                    continue
                src_path, start, end = piece
                with open_fits_bytes(src_path) as src:
                    _copy_range(src, dst, start, end)
        for idx in restamp:
            from .checksum_api import write_checksums

            write_checksums(temp_path, hdu=idx)
    _invalidate_written_target(path)


def _fixed_card(key: str, value: str, comment: str = "") -> str:
    card = f"{key:<8}= {value:>20}"
    if comment:
        card += f" / {comment}"
    return card[:_CARD].ljust(_CARD)


def _pad_header(cards: Sequence[str]) -> bytes:
    text = "".join(cards) + "END".ljust(_CARD)
    text += " " * (-len(text) % FITS_BLOCK)
    return text.encode("latin-1")


def _mandatory_axes(cards: Sequence[str], what: str) -> int:
    """Number of leading mandatory cards (first, BITPIX, NAXIS, NAXISn)."""
    naxis = _int_card(cards, "NAXIS")
    expected = ["BITPIX", "NAXIS"] + [f"NAXIS{i}" for i in range(1, naxis + 1)]
    got = [card[:8].rstrip() for card in cards[1 : 1 + len(expected)]]
    if got != expected:
        raise ValueError(f"{what}: mandatory keywords are out of order")
    return 1 + len(expected)


def primary_as_image_extension(raw_header: bytes) -> bytes:
    """Header of a primary HDU rewritten as an IMAGE extension header.

    SIMPLE becomes XTENSION = 'IMAGE', PCOUNT = 0 / GCOUNT = 1 follow the
    axes, EXTEND is dropped; every other card is kept verbatim. The data unit
    is unchanged.
    """
    cards = _header_cards(raw_header)
    if _logical_card(cards, "GROUPS"):
        raise ValueError(
            "The primary HDU holds random groups, which cannot become an image "
            "extension; insert the new HDU at index >= 1."
        )
    head = _mandatory_axes(cards, "primary header")
    out = [_fixed_card("XTENSION", "'IMAGE   '", "IMAGE extension")]
    out += cards[1:head]
    out.append(_fixed_card("PCOUNT", "0", "required keyword; must = 0"))
    out.append(_fixed_card("GCOUNT", "1", "required keyword; must = 1"))
    out += [c for c in cards[head:] if c[:8].rstrip() != "EXTEND"]
    return _pad_header(out)


def image_extension_as_primary(raw_header: bytes) -> bytes:
    """Header of an IMAGE extension rewritten as a primary header.

    XTENSION becomes SIMPLE = T, EXTEND = T follows the axes, PCOUNT/GCOUNT
    are dropped; every other card is kept verbatim. The data unit is
    unchanged.
    """
    cards = _header_cards(raw_header)
    head = _mandatory_axes(cards, "image extension header")
    out = [_fixed_card("SIMPLE", "T", "file does conform to FITS standard")]
    out += cards[1:head]
    out.append(_fixed_card("EXTEND", "T", "FITS dataset may contain extensions"))
    out += [
        c for c in cards[head:] if c[:8].rstrip() not in {"PCOUNT", "GCOUNT", "EXTEND"}
    ]
    return _pad_header(out)


def xtension_of(raw_header: bytes) -> str:
    """XTENSION value of a raw header ('' for a primary header)."""
    for card in _header_cards(raw_header)[:1]:
        if card.startswith("XTENSION= "):
            return card[10:].split("/", 1)[0].strip().strip("'").strip().upper()
    return ""


def header_has_card(raw_header: bytes, *keys: str) -> bool:
    wanted = set(keys)
    return any(card[:8].rstrip() in wanted for card in _header_cards(raw_header))
