#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from email.parser import Parser
from pathlib import Path

from archive_safety import zip_safety_problems

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILE = ROOT / "scripts" / "torch_support.json"

# One wheel per (ABI, platform): CPython 3.12+ share the cp312-abi3 wheel (the
# extension links no libtorch and targets the stable ABI); 3.11 is below the
# stable-ABI floor and keeps its own cp311 wheel.
PYTHON_ABIS = (("cp311", "cp311"), ("cp312", "abi3"))
PLATFORM_CELLS = (
    ("manylinux", "x86_64"),
    ("manylinux", "aarch64"),
    ("macosx_14_0", "arm64"),
)
EXPECTED_CELLS = {
    (python, abi, platform, arch)
    for python, abi in PYTHON_ABIS
    for platform, arch in PLATFORM_CELLS
}
WHEEL_RE = re.compile(
    r"^torchfits-(?P<version>[^-]+)-(?P<python>cp\d+)-(?P<abi>cp\d+|abi3)-"
    r"(?P<platform>[^-]+)\.whl$"
)
# Load-command / DT_NEEDED names of libtorch's shared libraries. A native
# member naming one of them is tied to a single PyTorch C++ ABI again.
# (libtorchfits_core is torchfits's own library and must not match.)
LIBTORCH_NAMES = (
    b"libtorch.",
    b"libtorch_cpu",
    b"libtorch_python",
    b"libtorch_cuda",
    b"libc10.",
    b"libc10_cuda",
)


def expected_torch_requirement() -> str:
    """``>=<floor>`` from scripts/torch_support.json (no upper bound)."""
    support = json.loads(SUPPORT_FILE.read_text(encoding="utf-8"))
    return f">={support['torch_min']}"


def _field(message, name: str) -> str | None:
    value = message.get(name)
    return value.strip() if value is not None else None


def _torch_requirement(message) -> str | None:
    for requirement in message.get_all("Requires-Dist", []):
        compact = requirement.replace(" ", "")
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", compact)
        if match is not None:
            name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            if name == "torch" and ";" not in compact:
                return match.group(2)
    return None


def _cell(python: str, abi: str, platform: str) -> tuple[str, str, str, str] | None:
    if platform == "macosx_14_0_arm64":
        return python, abi, "macosx_14_0", "arm64"
    if "manylinux" not in platform:
        return None
    if platform.endswith("_x86_64"):
        arch = "x86_64"
    elif platform.endswith("_aarch64"):
        arch = "aarch64"
    else:
        return None
    platform_tags = set(platform.split("."))
    required = f"manylinux_2_28_{arch}"
    if required not in platform_tags:
        return None
    for tag in platform_tags:
        match = re.fullmatch(rf"manylinux_2_(\d+)_{arch}", tag)
        if match is None or int(match.group(1)) > 28:
            return None
    if not platform_tags:
        return None
    return python, abi, "manylinux", arch


def validate(
    wheel_dir: Path, version: str, torch_requirement: str | None = None
) -> tuple[list[Path], list[str]]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    problems: list[str] = []
    cells: list[tuple[str, str, str, str]] = []
    want_torch = torch_requirement or expected_torch_requirement()
    if len(wheels) != len(EXPECTED_CELLS):
        problems.append(
            f"expected exactly {len(EXPECTED_CELLS)} wheels, found {len(wheels)}"
        )
    for wheel in wheels:
        match = WHEEL_RE.fullmatch(wheel.name)
        if match is None:
            problems.append(f"invalid wheel filename: {wheel.name}")
            continue
        python = match.group("python")
        abi = match.group("abi")
        platform = match.group("platform")
        if match.group("version") != version:
            problems.append(f"{wheel.name}: filename version is not {version}")
        if (python, abi) not in PYTHON_ABIS:
            problems.append(f"{wheel.name}: unexpected Python/ABI tag {python}-{abi}")
        cell = _cell(python, abi, platform)
        if cell is None:
            problems.append(f"{wheel.name}: unexpected platform tag {platform}")
        else:
            cells.append(cell)
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            safety = zip_safety_problems(infos)
            if safety:
                problems.extend(f"{wheel.name}: {problem}" for problem in safety)
                continue
            metadata_names = [
                info.filename
                for info in infos
                if info.filename.endswith(".dist-info/METADATA")
            ]
            wheel_names = [
                info.filename
                for info in infos
                if info.filename.endswith(".dist-info/WHEEL")
            ]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                problems.append(f"{wheel.name}: expected one METADATA and one WHEEL")
                continue
            for info in infos:
                if not info.filename.endswith((".so", ".dylib")):
                    continue
                payload = archive.read(info)
                linked = [name.decode() for name in LIBTORCH_NAMES if name in payload]
                if linked:
                    problems.append(
                        f"{wheel.name}: {info.filename} references libtorch "
                        f"({', '.join(linked)}); the extension must stay torch-free"
                    )
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            wheel_metadata = Parser().parsestr(
                archive.read(wheel_names[0]).decode("utf-8")
            )
        if _field(metadata, "Version") != version:
            problems.append(f"{wheel.name}: METADATA Version is not {version}")
        if _field(metadata, "Requires-Python") != ">=3.11":
            problems.append(f"{wheel.name}: Requires-Python is not >=3.11")
        torch_spec = _torch_requirement(metadata)
        if torch_spec is None or torch_spec.split(",") != [want_torch]:
            problems.append(
                f"{wheel.name}: torch requirement is {torch_spec!r}, not "
                f"{want_torch} with no upper bound"
            )
        expected_tags = {
            f"{python}-{abi}-{platform_tag}" for platform_tag in platform.split(".")
        }
        wheel_tags = set(wheel_metadata.get_all("Tag", []))
        if wheel_tags != expected_tags:
            problems.append(
                f"{wheel.name}: WHEEL tags {sorted(wheel_tags)} != "
                f"filename tags {sorted(expected_tags)}"
            )
    found = set(cells)
    if found != EXPECTED_CELLS:
        missing = sorted(EXPECTED_CELLS - found)
        extra = sorted(found - EXPECTED_CELLS)
        problems.append(f"wheel matrix mismatch; missing={missing}, extra={extra}")
    if len(cells) != len(set(cells)):
        problems.append("wheel matrix contains duplicate cells")
    return wheels, problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(artifacts: list[Path], destination: Path) -> None:
    lines = []
    for artifact in artifacts:
        digest = _sha256(artifact)
        lines.append(f"{digest}  {artifact.stat().st_size}  {artifact.name}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest(artifacts: list[Path], manifest: Path) -> list[str]:
    expected: dict[str, tuple[str, int]] = {}
    problems: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (
            len(fields) != 3
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            or not fields[1].isdigit()
            or Path(fields[2]).name != fields[2]
            or fields[2] in expected
        ):
            problems.append(f"invalid manifest line: {line}")
            continue
        expected[fields[2]] = (fields[0], int(fields[1]))
    if set(expected) != {artifact.name for artifact in artifacts}:
        problems.append("manifest filenames do not match validated artifacts")
        return problems
    for artifact in artifacts:
        digest, size = expected[artifact.name]
        if artifact.stat().st_size != size:
            problems.append(f"{artifact.name}: size does not match manifest")
        if _sha256(artifact) != digest:
            problems.append(f"{artifact.name}: SHA-256 does not match manifest")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    wheels, problems = validate(args.wheel_dir, args.version)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    artifacts = wheels + ([args.sdist] if args.sdist is not None else [])
    if args.manifest is not None:
        write_manifest(artifacts, args.manifest)
    if args.verify_manifest is not None:
        manifest_problems = verify_manifest(artifacts, args.verify_manifest)
        if manifest_problems:
            for problem in manifest_problems:
                print(f"[FAIL] {problem}")
            return 1
    print(f"[ OK ] exact {len(EXPECTED_CELLS)}-wheel release matrix for {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
