#!/usr/bin/env python3
"""Render ML loader benchmark CSV into docs/benchmarks.md section."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render(csv_path: Path) -> str:
    rows = [r for r in load_csv(csv_path) if r.get("status") == "OK"]
    if not rows:
        return (
            "_No ML loader benchmark CSV yet. Run "
            "`pixi run bench-ml -- --output benchmarks_results/<run-id>/ml_results.csv`._\n"
        )

    by_case: dict[str, list[tuple[str, float]]] = defaultdict(list)
    device = "cpu"
    for row in rows:
        case = row.get("case_id") or row.get("case_label") or "unknown"
        method = row.get("method") or row.get("library") or "?"
        tp = row.get("throughput") or ""
        try:
            rate = float(tp)
        except ValueError:
            continue
        by_case[case].append((method, rate))
        md = row.get("metadata") or ""
        if "cuda" in md.lower():
            device = "cuda"
        elif "mps" in md.lower():
            device = "mps"

    lines = [
        f"Source: `{csv_path}` (device={device}).",
        "",
        "| Case | Method | Median throughput |",
        "|---|---|---:|",
    ]
    for case in sorted(by_case):
        for method, rate in sorted(by_case[case], key=lambda x: x[0]):
            lines.append(f"| {case} | `{method}` | {rate:,.0f} pixels/s |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
