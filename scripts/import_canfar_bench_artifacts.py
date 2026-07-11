#!/usr/bin/env python3
"""Extract benchmarks_results tarball embedded in CANFAR session logs."""

from __future__ import annotations

import argparse
import base64
import io
import re
import tarfile
from pathlib import Path

_BEGIN = "TORCHFITS_BENCH_ARTIFACT_BEGIN"
_END = "TORCHFITS_BENCH_ARTIFACT_END"


def extract_artifact(log_text: str, out_dir: Path) -> bool:
    pattern = re.compile(
        rf"{re.escape(_BEGIN)}\s*(.*?)\s*{re.escape(_END)}",
        re.DOTALL,
    )
    match = pattern.search(log_text)
    if not match:
        return False
    payload = "".join(match.group(1).split())
    data = base64.b64decode(payload, validate=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(path=out_dir.parent)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("benchmarks_results"),
        help="Parent directory for extracted run folder",
    )
    args = parser.parse_args()
    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    if not extract_artifact(text, args.dest):
        raise SystemExit(f"no {_BEGIN} .. {_END} block in {args.log_file}")
    print(f"extracted benchmark artifacts under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
