#!/usr/bin/env python3
"""Render quick-bench evidence rows into a markdown table for docs/benchmarks.md.

Reads ``<scope>.json`` files from a directory (default: ``benchmarks_results/quick/``)
where each JSON file holds one quick-bench run summary for a given scope
(``fits`` or ``fitstable``).  The output is a 5-column table suitable for the
"Latest local quick benchmark evidence" section.

JSON schema
-----------
::

    {
        "run_id": "20260625_213448",
        "command": "pixi run python benchmarks/bench_all.py --profile user --fits-only --quick",
        "rows": 27,
        "deficits": 0
    }

If a scope file is missing, the table includes an empty cell for that scope's
run id / command and ``—`` placeholders for ``rows`` and ``deficits``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCOPES: dict[str, str] = {
    "fits": "FITS image I/O",
    "fitstable": "FITS table I/O",
}


def _read_scope(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def render_quick(quick_dir: Path) -> str:
    """Render the full quick-bench table (header + body) as a self-contained block.

    The patch script wraps the result in ``<!-- BENCH_QUICK_BEGIN/END -->``
    markers, so this output is the *entire* table — no external header
    remains.  When no JSON files are present the table stays well-formed
    with placeholder rows so the markdown is never broken.
    """
    rows: list[dict | None] = [
        _read_scope(quick_dir / f"{scope}.json") for scope in SCOPES
    ]

    lines = [
        "| Run ID | Scope | Command | Rows | Deficits |",
        "|---|---|---|---:|---:|",
    ]
    for (_scope, label), data in zip(SCOPES.items(), rows):
        if data is None:
            lines.append(f"| — | {label} | _(no run yet)_ | — | — |")
            continue
        run_id = data.get("run_id", "—")
        command = data.get("command", "—")
        rows_count = data.get("rows", "—")
        deficit_count = data.get("deficits", "—")
        lines.append(
            f"| `{run_id}` | {label} | `{command}` | {rows_count} | {deficit_count} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-dir",
        type=Path,
        default=Path("benchmarks_results/quick"),
        help="Directory containing per-scope quick-bench JSON files.",
    )
    args = parser.parse_args()
    print(render_quick(args.quick_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
