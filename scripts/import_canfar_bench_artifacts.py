#!/usr/bin/env python3
"""Extract gzipped CSV payloads embedded in CANFAR session logs."""

from __future__ import annotations

import argparse
import base64
import gzip
import re
from pathlib import Path

_CSV_BLOCK = re.compile(
    r"TORCHFITS_CSV_BEGIN\s+(\S+)\s*(.*?)\s*TORCHFITS_CSV_END\s+\1",
    re.DOTALL,
)


def import_logs(log_text: str, run_id: str, dest: Path) -> list[Path]:
    out_dir = dest / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in _CSV_BLOCK.findall(log_text):
        data = base64.b64decode("".join(payload.split()), validate=False)
        raw = gzip.decompress(data)
        path = out_dir / name
        path.write_bytes(raw)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("--dest", type=Path, default=Path("benchmarks_results"))
    args = parser.parse_args()
    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    paths = import_logs(text, args.run_id, args.dest)
    if not paths:
        raise SystemExit(f"no TORCHFITS_CSV blocks in {args.log_file}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
