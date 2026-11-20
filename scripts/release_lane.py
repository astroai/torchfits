#!/usr/bin/env python3
"""Render or check torchfits's supported-PyTorch range across the repo.

torchfits's native extension includes and links no libtorch: decoded buffers
reach PyTorch as DLPack capsules through its public Python API, and the
extension targets the CPython stable ABI. One wheel per platform therefore
serves every PyTorch from the tested floor up, with no upper cap.
``scripts/torch_support.json`` holds:

- ``torchfits_version``: the release being prepared (plain ``X.Y.Z``);
- ``torch_min``: the oldest PyTorch minor CI tests (the runtime floor);
- ``torch_tested``: the newest PyTorch minor CI tests.

This script renders or checks:

- ``pyproject.toml``: version and ``torch>=<min>`` (no build-time torch, no
  upper bound);
- ``pixi.toml``: package version and every ``pytorch = ">=<min>"`` pin;
- ``packaging/conda/recipe.yaml``: version and the ``pytorch`` floor;
- ``src/torchfits/__init__.py``: ``__version__``.

It fails when a GitHub workflow hard-codes a torch pin: workflows read the
pins from ``--print-pins``.

Usage::

    release_lane.py                          print the supported range
    release_lane.py --check                  verify committed files match the support map
    release_lane.py --print-pins             print KEY=VALUE pins (CI: >> "$GITHUB_ENV")
    release_lane.py --apply                  render the plain release version
    release_lane.py --apply --prerelease rc1 render a release candidate (1.3.3rc1)

``--check`` is wired into preflight/CI so a pin can never drift from the
support map. A committed pre-release (``1.3.3rc1``) checks as its base.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILE = ROOT / "scripts" / "torch_support.json"
PYPROJECT = ROOT / "pyproject.toml"
PIXI = ROOT / "pixi.toml"
RECIPE = ROOT / "packaging" / "conda" / "recipe.yaml"
INIT = ROOT / "src" / "torchfits" / "__init__.py"
WORKFLOWS = ROOT / ".github" / "workflows"

_VERSION_LINE_RE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)
_INIT_VERSION_RE = re.compile(r'^(__version__ = ")([^"]+)(")$', re.MULTILINE)
_RECIPE_VERSION_RE = re.compile(r'^(  version: ")([^"]+)(")$', re.MULTILINE)
_RECIPE_TORCH_PIN_RE = re.compile(r'^(  torch_pin: ")([^"]+)(")$', re.MULTILINE)
_PYPROJECT_TORCH_RE = re.compile(r'^(\s*")torch[<>=!~][^"]*(",?.*)$', re.MULTILINE)
_PIXI_PYTORCH_RE = re.compile(r'^(pytorch = ")[^"]*(")$', re.MULTILINE)
# A version comparison on torch in a workflow (``torch>=2.14``, ``torch==2.14.0``).
_WORKFLOW_TORCH_RE = re.compile(r"\btorch\s*(?:==|>=|<=|~=|<|>)\s*\d")
_PRERELEASE_RE = re.compile(r"(?:rc|a|b)\d+$")
_MINOR_RE = re.compile(r"\d+\.\d+")


def minor_key(minor: str) -> tuple[int, int]:
    major, rest = minor.split(".")
    return int(major), int(rest)


def load_support() -> dict[str, str]:
    data = json.loads(SUPPORT_FILE.read_text(encoding="utf-8"))
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(data.get("torchfits_version", ""))):
        raise SystemExit("torch_support.json: bad torchfits_version (want X.Y.Z)")
    for key in ("torch_min", "torch_tested"):
        if not _MINOR_RE.fullmatch(str(data.get(key, ""))):
            raise SystemExit(f"torch_support.json: {key} must be a minor such as 2.14")
    if minor_key(data["torch_min"]) > minor_key(data["torch_tested"]):
        raise SystemExit("torch_support.json: torch_min is newer than torch_tested")
    return data


def strip_prerelease(version: str) -> str:
    return _PRERELEASE_RE.sub("", version)


def committed_version() -> str:
    match = _VERSION_LINE_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("pyproject.toml has no version field")
    return match.group(2)


def _replace(
    text: str, pattern: re.Pattern[str], repl: str, what: str, count: int | None = 1
) -> str:
    updated, found = pattern.subn(repl, text)
    if (count is None and found == 0) or (count is not None and found != count):
        want = "at least one" if count is None else str(count)
        raise SystemExit(f"expected {want} match(es) for {what}; found {found}")
    return updated


def render(version: str | None, prerelease: str | None = None) -> dict[Path, str]:
    spec = load_support()
    base = spec["torchfits_version"]
    if prerelease is not None:
        version = f"{base}{prerelease}"
    elif version is None:
        version = base
    elif strip_prerelease(version) != base:
        raise SystemExit(
            f"torch_support.json prepares torchfits {base}, not {version!r}"
        )
    floor = spec["torch_min"]

    rendered: dict[Path, str] = {}
    py = PYPROJECT.read_text(encoding="utf-8")
    py = _replace(py, _VERSION_LINE_RE, rf"\g<1>{version}\g<3>", "pyproject version")
    py = _replace(
        py, _PYPROJECT_TORCH_RE, rf"\g<1>torch>={floor}\g<2>", "runtime torch"
    )
    rendered[PYPROJECT] = py

    pixi = PIXI.read_text(encoding="utf-8")
    pixi = _replace(pixi, _VERSION_LINE_RE, rf"\g<1>{version}\g<3>", "pixi version")
    pixi = _replace(
        pixi, _PIXI_PYTORCH_RE, rf"\g<1>>={floor}\g<2>", "pixi pytorch", count=None
    )
    rendered[PIXI] = pixi

    recipe = RECIPE.read_text(encoding="utf-8")
    recipe = _replace(
        recipe, _RECIPE_VERSION_RE, rf"\g<1>{version}\g<3>", "recipe version"
    )
    recipe = _replace(
        recipe, _RECIPE_TORCH_PIN_RE, rf"\g<1>>={floor}\g<3>", "recipe torch_pin"
    )
    rendered[RECIPE] = recipe

    init = INIT.read_text(encoding="utf-8")
    rendered[INIT] = _replace(
        init, _INIT_VERSION_RE, rf"\g<1>{version}\g<3>", "__init__ version"
    )
    return rendered


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def workflow_problems() -> list[str]:
    problems = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _WORKFLOW_TORCH_RE.search(line):
                problems.append(
                    f"{path.name}:{number}: hard-coded torch pin; "
                    "use release_lane.py --print-pins"
                )
    return problems


def check() -> int:
    try:
        spec = load_support()
        version = committed_version()
        expected = render(version)
    except SystemExit as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1
    print(
        f"[INFO] torchfits {version}, torch >={spec['torch_min']} "
        f"(tested {spec['torch_min']}..{spec['torch_tested']})",
        flush=True,
    )
    failed = False
    for path, want in expected.items():
        if path.read_text(encoding="utf-8") == want:
            print(f"[ OK ] {_display(path)}", flush=True)
        else:
            print(
                f"[FAIL] {_display(path)} drifted from torch_support.json "
                "(re-render with release_lane.py --apply)",
                flush=True,
            )
            failed = True
    for problem in workflow_problems():
        print(f"[FAIL] {problem}", flush=True)
        failed = True
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply", action="store_true", help="rewrite the pinned files in place"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify committed files match the map"
    )
    parser.add_argument(
        "--print-pins", action="store_true", help="print KEY=VALUE pins for CI"
    )
    parser.add_argument("--prerelease", help="PEP 440 suffix such as rc1")
    args = parser.parse_args()
    if args.prerelease is not None and not _PRERELEASE_RE.fullmatch(args.prerelease):
        parser.error(f"--prerelease {args.prerelease!r} is not rcN/aN/bN")

    if args.check:
        return check()
    if args.print_pins:
        # Consumers do `pip install "torch$TORCH_PIN"`; refuse to print pins
        # for a tree whose committed files disagree with the support map.
        with contextlib.redirect_stdout(sys.stderr):
            status = check()
        if status != 0:
            return 1
        spec = load_support()
        # NAME=VALUE lines for $GITHUB_ENV; each VALUE is a full specifier,
        # so `pip install "torch$TORCH_MIN_PIN"` reads torch==2.12.*.
        pins = {
            "VERSION": committed_version(),
            "TORCH_PIN": f">={spec['torch_min']}",
            "TORCH_MIN_PIN": f"=={spec['torch_min']}.*",
            "TORCH_TESTED_PIN": f"=={spec['torch_tested']}.*",
        }
        for name, value in pins.items():
            print(f"{name}={value}")
        return 0
    if not args.apply:
        spec = load_support()
        print(
            f"torchfits {spec['torchfits_version']}: torch >={spec['torch_min']} "
            f"(tested up to {spec['torch_tested']})"
        )
        return 0
    for path, text in render(None, prerelease=args.prerelease).items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {_display(path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
