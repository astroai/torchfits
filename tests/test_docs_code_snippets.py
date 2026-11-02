"""Automated testing for code snippets embedded in documentation markdown files."""

from __future__ import annotations

import ast
import re
import shlex
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch

import torchfits
from torchfits._hdu.header import Header
from torchfits.cli.main import build_parser
from torchfits.hdu import HDUList, TableHDU, TensorHDU

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"


class CodeSnippet(NamedTuple):
    file_path: Path
    line_number: int
    lang: str
    code: str


def _extract_snippets(lang: str) -> list[CodeSnippet]:
    """Extract all fenced code blocks of a given language from docs/*.md."""
    snippets: list[CodeSnippet] = []
    pattern = re.compile(rf"```{lang}[^\n]*\n(.*?)```", re.DOTALL)

    for md_path in sorted(DOCS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            raw_code = match.group(1)
            code = textwrap.dedent(raw_code)
            line_num = text[: match.start()].count("\n") + 1
            snippets.append(
                CodeSnippet(
                    file_path=md_path,
                    line_number=line_num,
                    lang=lang,
                    code=code,
                )
            )
    return snippets


@pytest.fixture(scope="module")
def mock_fits_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory with standard FITS files for doc snippet execution."""
    ws = tmp_path_factory.mktemp("doc_fits_ws")

    img = torch.from_numpy(np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    science = HDUList(
        [
            TensorHDU(
                img,
                Header(
                    {
                        "OBJECT": "M31",
                        "EXPTIME": 300.0,
                        "FILTER": "r",
                        "OBSERVER": "Astronomer",
                        "EXTNAME": "SCI",
                    }
                ),
            ),
            TensorHDU(img, Header({"EXTNAME": "SCI"})),
        ]
    )
    science.write(str(ws / "science.fits"), overwrite=True)

    catalog = {
        "ID": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "RA": np.array([10.1, 10.2, 10.3, 10.4, 10.5], dtype=np.float64),
        "DEC": np.array([-1.0, 0.5, 1.5, -0.2, 2.0], dtype=np.float64),
        "MAG_G": np.array([17.5, 19.2, 21.0, 18.0, 22.5], dtype=np.float32),
        "CLASS_STAR": np.array([0.95, 0.88, 0.12, 0.99, 0.05], dtype=np.float32),
        "FLUX": np.array([10.5, 23.1, 45.0, 12.0, 50.0], dtype=np.float32),
    }
    torchfits.table.write(str(ws / "catalog.fits"), catalog, overwrite=True)

    mosaic = torch.ones(512, 512) * 42.0
    for name in (
        "giant_mosaic.fits",
        "survey_mosaic.fits",
        "mosaic.fits",
        "horsehead.fits",
        "sdss_i.fits",
        "sdss_r.fits",
        "sdss_g.fits",
    ):
        torchfits.write(str(ws / name), mosaic, overwrite=True)

    mef = HDUList(
        [
            TensorHDU(torch.empty(0), Header({"OBJECT": "NGC1234"})),
            TensorHDU(img, Header({"EXTNAME": "SCI"})),
            TableHDU(catalog, header=Header({"EXTNAME": "CATALOG"})),
        ]
    )
    mef.write(str(ws / "observation.fits"), overwrite=True)
    mef.write(str(ws / "mef_survey.fits"), overwrite=True)

    survey_dir = ws / "data" / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        torchfits.write(
            str(survey_dir / f"survey_{i}.fits"),
            img,
            header={
                "CLASS_ID": i % 2,
                "CLASS": "GALAXY" if i % 2 == 0 else "STAR",
            },
            overwrite=True,
        )

    torchfits.table.write(str(ws / "huge_catalog.fits"), catalog, overwrite=True)

    return ws


def test_all_python_snippets_syntax_valid() -> None:
    """Verify that every python code block in docs/*.md is syntactically valid Python."""
    py_snippets = _extract_snippets("python")
    assert len(py_snippets) > 0, "No python snippets found in docs/"

    errors: list[str] = []
    for snippet in py_snippets:
        code = snippet.code.strip()
        try:
            ast.parse(code)
        except SyntaxError as exc:
            # If this is a function call signature like "torchfits.foo(..., *, bar=...)",
            # rewrite as "def foo(..., *, bar=...): pass" to validate function signature syntax
            cleaned = re.sub(r"^torchfits(?:\.\w+)*\.(\w+)\s*\(", r"def \1(", code)
            cleaned = re.sub(r"\)\s*->.*$", "): pass", cleaned, flags=re.DOTALL)
            if not cleaned.endswith(": pass"):
                cleaned += ": pass"
            try:
                ast.parse(cleaned)
            except SyntaxError:
                rel_path = snippet.file_path.relative_to(ROOT)
                errors.append(
                    f"{rel_path}:{snippet.line_number} SyntaxError: {exc}\nCode:\n{code}\n"
                )

    assert not errors, "Syntax errors in python doc snippets:\n" + "\n".join(errors)


def test_all_cli_snippets_valid_commands() -> None:
    """Verify that every `torchfits <subcommand> ...` in bash blocks uses valid subcommands."""
    sh_snippets = _extract_snippets("bash")
    assert len(sh_snippets) > 0, "No bash snippets found in docs/"

    parser = build_parser()
    subcommand_parsers: set[str] = set()
    for action in parser._actions:
        if action.dest == "command" and hasattr(action, "choices"):
            subcommand_parsers = set(action.choices.keys())

    errors: list[str] = []

    for snippet in sh_snippets:
        for line in snippet.code.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("torchfits "):
                try:
                    args = shlex.split(line)[1:]
                except Exception as exc:
                    rel = snippet.file_path.relative_to(ROOT)
                    errors.append(
                        f"{rel}:{snippet.line_number} shlex error on line {line!r}: {exc}"
                    )
                    continue

                if not args:
                    continue

                subcmd = args[0]
                if (
                    subcmd.startswith("$")
                    or subcmd.startswith("<")
                    or subcmd in ("--help", "-h")
                ):
                    continue

                if subcmd not in subcommand_parsers:
                    rel = snippet.file_path.relative_to(ROOT)
                    errors.append(
                        f"{rel}:{snippet.line_number} Unknown subcommand {subcmd!r} in {line!r}"
                    )
                    continue

    assert not errors, "CLI syntax errors in docs:\n" + "\n".join(errors)


def _missing_optional_module(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) is a non-torchfits missing module."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ModuleNotFoundError):
            top = (cur.name or "").partition(".")[0]
            return bool(top) and top != "torchfits"
        if not isinstance(cur, ImportError):
            return False
        cur = cur.__cause__ or cur.__context__
    return False


def test_runnable_workflow_snippets_execute(
    mock_fits_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute runnable workflow snippets from guides, tutorials, and migration pages."""
    monkeypatch.chdir(mock_fits_workspace)

    target_docs = {
        "index.md",
        "quickstart.md",
        "python-workflows.md",
        "examples-transforms.md",
        "compatibility.md",
    }
    py_snippets = [
        s for s in _extract_snippets("python") if s.file_path.name in target_docs
    ]

    executed = 0
    for snippet in py_snippets:
        code = snippet.code.strip()

        # Skip signature definitions or non-executable fragments
        if code.startswith("torchfits.") and (
            "(" in code.splitlines()[0] and "->" in code
        ):
            continue
        if 'device="cuda"' in code and not torch.cuda.is_available():
            # Replace device="cuda" with device="cpu" if running on non-CUDA host
            code = code.replace('device="cuda"', 'device="cpu"')

        # Inject dummy process_batch / handler functions if referenced in loop snippets
        scope: dict = {"process_batch": lambda b: None, "print": lambda *a, **k: None}

        try:
            exec(code, scope)
            executed += 1
        except Exception as exc:
            # Optional third-party package (e.g. polars, astropy) not installed in
            # this environment, raised directly or wrapped by a torchfits
            # converter. A missing torchfits module or name is a stale snippet.
            if _missing_optional_module(exc):
                continue
            rel = snippet.file_path.relative_to(ROOT)
            pytest.fail(
                f"Doc snippet in {rel}:{snippet.line_number} raised {exc.__class__.__name__}: {exc}\nCode:\n{code}"
            )

    assert executed > 0, "No runnable snippets executed"


def _page_snippets(page: str) -> list[CodeSnippet]:
    return [s for s in _extract_snippets("python") if s.file_path.name == page]


def _run_snippet(snippet: CodeSnippet, scope: dict) -> None:
    rel = snippet.file_path.relative_to(ROOT)
    try:
        exec(compile(snippet.code, f"{rel}:{snippet.line_number}", "exec"), scope)
    except Exception as exc:
        # A snippet may document the error it raises, e.g. "# DataStateError: ...".
        if f"# {exc.__class__.__name__}:" in snippet.code:
            return
        pytest.fail(
            f"Doc snippet in {rel}:{snippet.line_number} raised "
            f"{exc.__class__.__name__}: {exc}\nCode:\n{snippet.code}"
        )


def test_api_transforms_snippets_execute_in_page_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every api-transforms.md fence runs, top to bottom, in one shared scope.

    Fences build on each other (``pipeline`` is defined once and reused), so
    they run in page order; the scope is seeded only with the sample arrays the
    page names as inputs (``image``, ``counts``, ``payload``, ...).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    torchfits.write(str(tmp_path / "images" / "a.fits"), torch.ones(8, 8))

    gen = torch.Generator().manual_seed(0)
    image = 100.0 + torch.randn(64, 64, generator=gen)
    ivar = torch.full((64, 64), 4.0)
    scope: dict = {
        "torch": torch,
        "image": image,
        "counts": image.clone(),
        "finite_mask": torch.isfinite(image),
        "ivar": ivar,
        "payload": {
            "flux": image.clone(),
            "ivar": ivar,
            "dq": torch.zeros(64, 64, dtype=torch.int32),
        },
    }
    snippets = _page_snippets("api-transforms.md")
    assert len(snippets) >= 10
    for snippet in snippets:
        _run_snippet(snippet, scope)


def test_examples_ml_pipeline_snippet_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Galaxy Zoo preprocessing fence builds its pipeline and loader.

    ``file_paths`` / ``target_labels`` come from the download step the page
    runs first; local stand-ins replace them here. The pipeline is applied to
    one item so a child Compose cannot call is exercised, not just built.
    """
    monkeypatch.chdir(tmp_path)
    paths = []
    for i in range(2):
        path = tmp_path / f"cutout_{i}.fits"
        stamp = torch.randn(3, 16, 16, generator=torch.Generator().manual_seed(i))
        stamp[0, 0, 0] = float("nan")
        torchfits.write(str(path), stamp)
        paths.append(str(path))

    snippet = next(s for s in _page_snippets("examples-ml.md") if "Compose(" in s.code)
    scope: dict = {"file_paths": paths, "target_labels": [0, 1]}
    _run_snippet(snippet, scope)
    image, label = scope["dataset"][0]
    assert image.shape[-2:] == (16, 16)
    assert int(label) == 0
