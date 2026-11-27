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
    for name in ("ci.yml", "build_wheels.yml", "docs-drift.yml", "torch-canary.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "release_lane.py --print-pins" in text, name
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert 'pip install "torch$TORCH_TESTED_PIN"' in ci
    assert 'pip install "torch$TORCH_MIN_PIN"' in ci


def test_ci_tests_exactly_torch_tested_and_a_canary_floats() -> None:
    """torch_tested is what CI installs, not a hand-edited claim beside it.

    ci.yml's "newest" lane installed `torch>=<floor>`, i.e. whatever PyTorch
    had last released, while scripts/torch_support.json said torch_tested was
    2.15: nothing tied the two together, and a new PyTorch release silently
    changed what every PR was tested against. The lane is now pinned to
    TORCH_TESTED_PIN and a separate scheduled canary owns the floating newest.
    """
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert '"torch$TORCH_PIN"' not in ci, "a ci.yml job installs floating torch"
    test_job = _job("ci.yml", "test")
    assert "torch: [tested]" in test_job
    assert 'pip install "torch$TORCH_TESTED_PIN"' in test_job
    assert 'pip install "torch$TORCH_MIN_PIN"' in test_job

    canary = (WORKFLOWS / "torch-canary.yml").read_text(encoding="utf-8")
    assert "schedule:" in canary and "workflow_dispatch:" in canary
    assert "push:" not in canary and "pull_request:" not in canary
    assert 'pip install --upgrade "torch$TORCH_PIN"' in canary
    assert "pytest tests/" in canary


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


def _job(workflow: str, name: str) -> str:
    """The text of one top-level job in a workflow file (2-space job keys)."""
    text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
    match = re.search(
        rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{workflow} has no job {name!r}"
    return match.group(1)


def test_release_wheels_are_tested_at_the_floor_and_torch_tested() -> None:
    """The published wheels must be the ones tested at both ends of the range.

    Before 1.3.4 the release workflow ran its floor tests on an editable
    *source* build, and cibuildwheel tested each built wheel only against the
    newest torch for its CPython, so no published wheel ever ran at the floor
    or at the pinned torch_tested minor (TORCH_TESTED_PIN was unused).
    """
    job = _job("build_wheels.yml", "test_wheels")
    assert "needs: build_wheels" in job
    assert "torch: [min, tested]" in job
    for artifact in ("cibw-wheels-ubuntu-latest-x86_64", "cibw-wheels-macos-15-arm64"):
        assert f"artifact: {artifact}" in job
    assert "--no-index --no-deps --find-links wheels torchfits" in job
    assert '"$TORCH_MIN_PIN"' in job and '"$TORCH_TESTED_PIN"' in job
    assert "TORCHFITS_TEST_TORCH_SPEC" in job
    assert "working-directory: ${{ runner.temp }}" in job
    assert 'bash "$GITHUB_WORKSPACE/scripts/cibw_test.sh"' in job
    # Publishing is gated on it.
    for publisher in ("upload_pypi", "github_release"):
        needs = re.search(r"needs: \[([^\]]*)\]", _job("build_wheels.yml", publisher))
        assert needs is not None and "test_wheels" in needs.group(1), publisher

    script = (ROOT / "scripts" / "cibw_test.sh").read_text(encoding="utf-8")
    assert "TORCHFITS_TEST_TORCH_SPEC" in script
    for test in ("test_release_smoke.py", "test_bz2.py", "test_dlpack_boundary.py"):
        assert f"/tests/{test}" in script, test


def test_macos_py315_is_tested_with_torch_2_15_or_newer() -> None:
    """macOS + Python 3.15 is covered, but only with torch >= 2.15.

    The macOS py3.15 CI cell and cibuildwheel's macOS cp315 test were removed
    because torch 2.14's cp315 macOS wheels segfault (pytorch/pytorch#200779).
    torch 2.15 fixed that, so the cells are back: CI runs the macOS py3.15 cell
    on the tested lane only, which is safe while torch_tested >= 2.15, and
    cibw_test.sh adds a >=2.15 bound for that cell.
    """
    test_job = _job("ci.yml", "test")
    assert '"3.15"' in test_job and "macos-15" in test_job
    assert "exclude:" not in test_job
    # The floor lane (torch: min) must not reach macOS py3.15.
    for entry in re.findall(
        r"- os: (\S+)\n\s+python-version: \"([\d.]+)\"\n\s+torch: min", test_job
    ):
        assert entry != ("macos-15", "3.15"), entry
    tested = lane.load_support()["torch_tested"]
    assert lane.minor_key(tested) >= lane.minor_key("2.15"), (
        "torch_tested < 2.15 would put a broken torch on the macOS py3.15 cell"
    )

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    macos = pyproject.split("[tool.cibuildwheel.macos]", 1)[1].split("\n[", 1)[0]
    skip = re.search(r'^skip = "([^"]*)"', macos, re.MULTILINE)
    assert skip is not None and "cp315" not in skip.group(1)
    assert "cp31?t-*" in skip.group(1)
    script = (ROOT / "scripts" / "cibw_test.sh").read_text(encoding="utf-8")
    assert 'TORCH_SPEC="${TORCH_SPEC},>=2.15"' in script
