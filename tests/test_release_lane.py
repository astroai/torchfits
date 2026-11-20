"""Unit tests for scripts/release_lane.py (no network, no disk writes)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import release_lane as lane  # noqa: E402

WORKFLOWS = ROOT / ".github" / "workflows"


def _rendered_version(rendered: dict[Path, str]) -> str:
    match = re.search(r'^version = "([^"]+)"$', rendered[lane.PYPROJECT], re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_strip_prerelease() -> None:
    assert lane.strip_prerelease("1.4.0") == "1.4.0"
    assert lane.strip_prerelease("1.4.0rc5") == "1.4.0"
    assert lane.strip_prerelease("1.4.0b1") == "1.4.0"
    assert lane.strip_prerelease("1.4.0a2") == "1.4.0"


def test_committed_tree_matches_the_support_map() -> None:
    assert lane.check() == 0
    floor = lane.load_support()["torch_min"]
    pyproject = lane.PYPROJECT.read_text(encoding="utf-8")
    # A floor and no upper cap: the extension links no libtorch.
    assert f'"torch>={floor}",' in pyproject
    assert "torch<" not in pyproject.replace(" ", "")
    build_system = pyproject.split("[build-system]", 1)[1].split("\n[", 1)[0]
    assert "torch" not in build_system
    assert 'wheel.py-api = "cp312"' in pyproject
    recipe = lane.RECIPE.read_text(encoding="utf-8")
    assert f'torch_pin: ">={floor}"' in recipe
    pixi = lane.PIXI.read_text(encoding="utf-8")
    assert re.findall(r'^pytorch = "([^"]+)"$', pixi, re.MULTILINE) == [
        f">={floor}",
        f">={floor}",
    ]


def test_every_version_source_agrees_with_the_support_base() -> None:
    base = lane.load_support()["torchfits_version"]
    version = lane.committed_version()
    assert lane.strip_prerelease(version) == base
    for path in (lane.PYPROJECT, lane.PIXI, lane.INIT, lane.RECIPE):
        assert version in path.read_text(encoding="utf-8"), path


def test_render_prerelease_and_finalize() -> None:
    base = lane.load_support()["torchfits_version"]
    rc = lane.render(None, prerelease="rc3")
    assert _rendered_version(rc) == f"{base}rc3"
    assert f'__version__ = "{base}rc3"' in rc[lane.INIT]
    assert f'version: "{base}rc3"' in rc[lane.RECIPE]
    # A committed rc renders identically to --apply --prerelease.
    assert lane.render(f"{base}rc3") == rc
    assert _rendered_version(lane.render(None)) == base
    with pytest.raises(SystemExit):
        lane.render("9.9.9")


def test_load_support_rejects_an_inverted_range(tmp_path: Path, monkeypatch) -> None:
    support = tmp_path / "torch_support.json"
    support.write_text(
        json.dumps(
            {"torchfits_version": "1.4.0", "torch_min": "2.15", "torch_tested": "2.12"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lane, "SUPPORT_FILE", support)
    with pytest.raises(SystemExit, match="torch_min is newer"):
        lane.load_support()


def test_check_reports_a_drifted_floor(tmp_path: Path, monkeypatch, capsys) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        lane.PYPROJECT.read_text(encoding="utf-8").replace(
            f'"torch>={lane.load_support()["torch_min"]}",', '"torch>=2.10,<2.11",'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lane, "PYPROJECT", pyproject)
    assert lane.check() == 1
    assert "pyproject.toml drifted" in capsys.readouterr().out


def test_workflows_do_not_hard_code_torch_pins() -> None:
    assert lane.workflow_problems() == []
    for name in ("ci.yml", "build_wheels.yml", "docs-drift.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "release_lane.py --print-pins" in text, name
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert 'pip install "torch$TORCH_PIN"' in ci
    assert 'pip install "torch$TORCH_MIN_PIN"' in ci


def test_workflow_pin_scan_flags_a_hard_coded_pin(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "x.yml").write_text(
        'run: pip install "torch>=2.14,<2.15"\n', encoding="utf-8"
    )
    monkeypatch.setattr(lane, "WORKFLOWS", tmp_path)
    assert lane.workflow_problems() == [
        "x.yml:1: hard-coded torch pin; use release_lane.py --print-pins"
    ]


def test_print_pins_are_github_env_lines(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["release_lane.py", "--print-pins"])
    assert lane.main() == 0
    support = lane.load_support()
    pins = dict(
        line.split("=", 1) for line in capsys.readouterr().out.strip().splitlines()
    )
    # $GITHUB_ENV splits at the first "=", so each value is a full specifier.
    assert pins == {
        "VERSION": lane.committed_version(),
        "TORCH_PIN": f">={support['torch_min']}",
        "TORCH_MIN_PIN": f"=={support['torch_min']}.*",
        "TORCH_TESTED_PIN": f"=={support['torch_tested']}.*",
    }


def test_print_pins_fails_loudly_when_the_tree_drifts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    init = tmp_path / "__init__.py"
    init.write_text('__version__ = "0.0.1"\n', encoding="utf-8")
    monkeypatch.setattr(lane, "INIT", init)
    monkeypatch.setattr(sys, "argv", ["release_lane.py", "--print-pins"])
    assert lane.main() == 1
    assert "TORCH_PIN" not in capsys.readouterr().out
