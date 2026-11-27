from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchfits"
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _requirement_name(requirement: str) -> str:
    """The distribution name of a PEP 508 requirement string.

    Splitting on the specifier/version characters keeps ``torchvision`` from
    being read as torch: a naive ``startswith("torch")`` would count every
    torch-family requirement as a second torch pin.
    """
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0]


def _torch_floor() -> str:
    support = json.loads(
        (REPO_ROOT / "scripts" / "torch_support.json").read_text(encoding="utf-8")
    )
    return support["torch_min"]


def test_torch_support_range_is_consistent() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    pixi = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))
    floor = _torch_floor()

    # The build needs no torch at all: the extension links no libtorch.
    assert not [
        r
        for r in pyproject["build-system"]["requires"]
        if _requirement_name(r) in {"torch", "numpy"}
    ]
    # Runtime: the supported floor, deliberately uncapped, and the only torch
    # requirement (no per-flavor extras pinning one minor).
    torch_reqs = [
        r
        for group in (
            pyproject["project"]["dependencies"],
            *pyproject["project"]["optional-dependencies"].values(),
        )
        for r in group
        if _requirement_name(r) == "torch"
    ]
    assert torch_reqs == [f"torch>={floor}"]
    assert pyproject["tool"]["scikit-build"]["wheel"]["py-api"] == "cp312"
    # Pixi: PyTorch is a run dependency only, with the same floor.
    for section in ("build-dependencies", "host-dependencies"):
        assert "pytorch" not in pixi["package"][section]
    assert pixi["package"]["run-dependencies"]["pytorch"] == f">={floor}"
    assert pixi["dependencies"]["pytorch"] == f">={floor}"

    workflow_paths = (
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "build_wheels.yml",
    )
    for path in workflow_paths:
        workflow = path.read_text(encoding="utf-8")
        # CI resolves the range from scripts/torch_support.json at runtime.
        assert "release_lane.py --print-pins" in workflow
        assert "pip install torch " not in workflow
    ci = workflow_paths[0].read_text(encoding="utf-8")
    # CI pins both ends of the tested range; only the nightly canary floats.
    assert 'pip install "torch$TORCH_TESTED_PIN"' in ci
    assert 'pip install "torch$TORCH_MIN_PIN"' in ci
    assert '"torch$TORCH_PIN"' not in ci

    wheel_workflow = (
        REPO_ROOT / ".github" / "workflows" / "build_wheels.yml"
    ).read_text(encoding="utf-8")
    cibw = pyproject.get("tool", {}).get("cibuildwheel", {})
    frontend = cibw.get("build-frontend")
    assert frontend == {"name": "pip", "args": ["--no-build-isolation"]} or (
        isinstance(frontend, str) and "no-build-isolation" in frontend
    )
    assert "cp311-*" in str(cibw.get("build", ""))
    assert "cp312-*" in str(cibw.get("build", ""))
    assert "cp31?t-*" in str(cibw.get("skip", ""))
    # Nothing to exclude any more: the wheel carries no libtorch reference.
    assert "--exclude" not in cibw["linux"]["repair-wheel-command"]
    assert "ubuntu-24.04-arm" in wheel_workflow
    cibw_pin = re.search(r"pypa/cibuildwheel@([0-9a-f]{40})", wheel_workflow)
    assert cibw_pin is not None, "cibuildwheel action pin not found"
    action_refs = re.findall(
        r"^\s*-?\s*uses:\s*[^@\s]+@([^\s]+)", wheel_workflow, re.MULTILINE
    )
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
    # PyPI must not get an sdist — unmatched CPython/arch would compile.
    assert "pattern: cibw-wheels-*" in wheel_workflow
    # The fork receives the same tag and has no PyPI trusted publisher.
    assert "github.repository == 'astroai/torchfits'" in wheel_workflow
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish" in wheel_workflow
    )
    assert "needs.release_gate.outputs.release_kind == 'stable'" in wheel_workflow
    assert "needs.release_gate.outputs.release_kind == 'rc'" in wheel_workflow
    assert "name: validated-release-artifacts" in wheel_workflow
    assert "python scripts/check_wheel_contents.py dist/wheels/*.whl" in wheel_workflow
    assert "python scripts/validate_release_wheels.py" in wheel_workflow
    assert wheel_workflow.count("--verify-manifest dist/RELEASE-MANIFEST.sha256") == 2
    assert wheel_workflow.count("python scripts/validate_release_sdist.py") == 4
    assert "--sdist dist/sdist/*.tar.gz" in wheel_workflow
    assert (
        "python scripts/check_macos_wheel_floor.py wheelhouse/*.whl" in wheel_workflow
    )
    assert "_PYTHON_HOST_PLATFORM" in str(cibw["macos"]["environment"])
    assert cibw["macos"]["environment"]["_PYTHON_HOST_PLATFORM"] == (
        "macosx-14.0-arm64"
    )
    cmake_args = " ".join(pyproject["tool"]["scikit-build"]["cmake"]["args"])
    assert "CONDA_PREFIX" not in cmake_args
    assert "-DUSE_CUDA=OFF" in cmake_args
    assert not (REPO_ROOT / "constraints-wheel.txt").exists()

    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert "find_package(Torch" not in cmake
    assert "TORCHFITS_TORCH_ABI" not in cmake
    assert "STABLE_ABI" in cmake
    assert "SKBUILD_SABI_COMPONENT" in cmake
    assert "matching_abi" not in bindings
    assert "pip should not compile" in cmake
    assert (REPO_ROOT / "scripts" / "cibw_before_build.sh").is_file()
    assert (REPO_ROOT / "scripts" / "cibw_test.sh").is_file()
    assert (REPO_ROOT / "scripts" / "cibuildwheel.sh").is_file()
    assert pyproject["tool"]["scikit-build"]["sdist"]["force-include"] == {
        "extern/cfitsio": "extern/cfitsio"
    }
    assert (
        "bash extern/vendor.sh --cfitsio-version extern/VERSIONS.txt" in wheel_workflow
    )

    # [dev] is the library toolchain. Comparators live in torchfits-harness.
    dev = set(pyproject["project"]["optional-dependencies"]["dev"])
    assert "ipykernel" not in dev
    assert any(x.startswith("pytest") for x in dev)
    assert any(x.startswith("matplotlib") for x in dev)
    extras = pyproject["project"]["optional-dependencies"]
    assert "bench" not in extras
    for group in ("dev", "test"):
        for req in extras[group]:
            name = req.split(">=")[0].split("[")[0]
            assert name not in {"astropy", "fitsio"}, req


def test_every_tracked_file_is_re_addable() -> None:
    """No tracked file may sit under a .gitignore rule that can never re-include it.

    `git check-ignore` skips tracked paths, so an ignore rule covering a tracked
    file is invisible in normal use: the file keeps showing up in `git status`
    as tracked, and the breakage only appears on the delete/restore cycle. Git
    cannot re-include a file whose *parent directory* is excluded, so a rule like
    `benchmarks/replays/` plus `!benchmarks/replays/<manifest>` leaves the
    negation unreachable -- `git add` of that manifest then fails outright. Here
    the manifest was tracked anyway, which is why the dead `!` line went unseen.
    `--no-index` is what makes the shadowing visible, and the assertion is over
    the whole tracked set so the next one is caught where it is introduced.
    """
    probe = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )
    if probe.returncode != 0:
        pytest.skip(f"not a git checkout: {probe.stderr.strip()}")
    tracked = probe.stdout.splitlines()
    assert tracked, "git ls-files returned nothing; the check would be vacuous"
    shadowed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(tracked) + "\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert not shadowed.stdout.strip(), (
        "these tracked files are shadowed by an unreachable .gitignore rule, so "
        "git add would refuse them after a delete/restore cycle:\n"
        f"{shadowed.stdout}"
    )


def test_sdist_readme_names_the_runtime_requirement_not_a_torch_version() -> None:
    """The packager-facing build note must not restate a torch version.

    SDIST-README.txt ships inside the sdist tarball, which is exactly the
    audience a wrong version claim reaches: downstream packagers reading it
    have no other statement of the PyTorch requirement. It points at
    `project.dependencies` as the authority instead of restating a version, so
    it cannot drift when the floor moves, and says the build itself needs no
    PyTorch (the extension links no libtorch).
    """
    readme = (REPO_ROOT / "SDIST-README.txt").read_text(encoding="utf-8")
    assert "project.dependencies" in readme
    assert "needs no PyTorch" in readme
    # No torch version claim of any shape.
    assert not re.search(r"[Pp]y[Tt]orch\s*[\(<>=]*\s*\d", readme), (
        "SDIST-README.txt must not state a torch version; point at the pin instead"
    )


def _native_artifacts() -> list[Path]:
    import torchfits._C as native
    import torchfits._core as core

    paths = {Path(native.__file__), Path(core.__file__)}
    lib_dir = Path(native.__file__).parent
    paths.update(lib_dir.glob("libtorchfits_core.*"))
    return sorted(paths)


def test_native_extensions_link_no_libtorch() -> None:
    """No native artifact names a libtorch library in its load commands.

    A libtorch link ties the extension to one PyTorch minor's C++ ABI (there is
    no stable one), which is what the per-minor release lanes existed to manage.
    """
    forbidden = (
        b"libtorch.",
        b"libtorch_cpu",
        b"libtorch_python",
        b"libc10.",
        b"THPVariable_Wrap",
    )
    artifacts = _native_artifacts()
    assert any(p.name.startswith("_C") for p in artifacts)
    for path in artifacts:
        payload = path.read_bytes()
        hits = [name.decode() for name in forbidden if name in payload]
        assert not hits, f"{path.name} references {hits}"


def test_native_extension_imports_without_torch() -> None:
    """Importing the extension (and reading metadata) never loads torch."""
    script = """
import sys
import torchfits._C as C
assert C.__core_build_id__
assert not hasattr(C, "_check_torch_abi")
assert "torch" not in sys.modules, "importing torchfits._C loaded torch"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_native_extension_accepts_any_reported_torch_version(tmp_path) -> None:
    """No import-time ABI gate: results reach torch through DLPack only."""
    path = tmp_path / "x.fits"
    script = f"""
import torch
torch.__version__ = "9.99.0"
import torchfits
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
torchfits.write({str(path)!r}, x, overwrite=True)
y = torchfits.read({str(path)!r})
assert isinstance(y, torch.Tensor) and torch.equal(x, y), y
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_metadata_works_with_neither_torch_nor_numpy(tmp_path) -> None:
    """The torch-free core must not need numpy either.

    The clean-install case is the one that matters: a machine that only wants to
    look at a header has no reason to have torch, and torchfits declares numpy
    as a hard dependency for the tensor/Arrow paths, not for metadata. Reading
    a header, a shape and a column list in a process where importing either
    raises is the strongest available statement that the split is real.

    The fixture is written by this test's parent process (which does have the
    full stack) and only read by the child.
    """
    import numpy as np

    import torchfits

    image = tmp_path / "image.fits"
    torchfits.write(
        str(image), np.arange(64, dtype=np.float32).reshape(8, 8), overwrite=True
    )
    table = tmp_path / "table.fits"
    torchfits.table.write(
        str(table),
        {"ID": np.arange(3, dtype=np.int32), "MAG": np.linspace(18.0, 20.0, 3)},
        overwrite=True,
        extname="SCI",
    )

    script = """
import importlib.abc, json, sys


class _Block(importlib.abc.MetaPathFinder):
    BLOCKED = ("torch", "numpy")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.BLOCKED:
            raise ImportError(root + " is blocked: " + fullname)
        return None


sys.meta_path.insert(0, _Block())

import torchfits
import torchfits._core as core

image, table = sys.argv[1], sys.argv[2]
print(json.dumps({
    "core_library_build_id": core.core_library_build_id(),
    "module_build_id": core.__build_id__,
    "num_hdus": torchfits.read_num_hdus(table),
    "hdu_type": torchfits.read_hdu_type(table, 1),
    "nrows": torchfits.read_nrows(table, 1),
    "colnames": torchfits.read_colnames(table, 1),
    "shape": torchfits.read_shape(image, 0),
    "keys": torchfits.read_keys(table, ["EXTNAME", "NAXIS2"], hdu=1),
    "header_size": len(torchfits.read_header(table, 1)),
    "table_info": sorted(torchfits.read_table_info(table, 1)),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(image), str(table)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["module_build_id"] == payload["core_library_build_id"]
    assert payload["num_hdus"] == 2
    assert payload["hdu_type"] == "BINARY_TABLE"
    assert payload["nrows"] == 3
    assert payload["colnames"] == ["ID", "MAG"]
    assert payload["shape"][0] == -32
    assert list(payload["shape"][1]) == [8, 8]
    assert payload["keys"] == {"EXTNAME": "SCI", "NAXIS2": 3}
    assert payload["header_size"] > 0
    assert payload["table_info"] == ["colnames", "nrows", "tforms"]


def test_torchfits_source_does_not_reference_torchsky() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.suffix not in {".py", ".cpp", ".h"} and path.name != "CMakeLists.txt":
            continue
        if "torchsky" in path.read_text(encoding="utf-8", errors="ignore").lower():
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == []


def test_torchfits_contains_only_fits_native_sources() -> None:
    native_root = PACKAGE_ROOT / "cpp_src"
    # Positive half first. "Contains only FITS native sources" is a
    # containment claim, so the container itself has to be asserted: with the
    # four absence checks alone the test is satisfied by an empty package, and
    # deleting or renaming cpp_src/ wholesale left it green. The positive
    # anchor is the house style here -- test_check_duplicate_cpp.py carries
    # `assert files, "the real cpp_src tree must not be empty"` for the
    # same tree.
    assert native_root.is_dir(), f"native source root is missing: {native_root}"
    sources = sorted(
        p.name for p in native_root.rglob("*") if p.suffix in {".cpp", ".h"}
    )
    assert sources, f"cpp_src must not be empty: {native_root}"
    assert "fits_file.cpp" in sources, (
        "the FITS native sources are what this package is allowed to contain; "
        f"found {sources}"
    )
    assert not (native_root / "wcs.cpp").exists()
    assert not (native_root / "healpix.cpp").exists()
    assert not (PACKAGE_ROOT / "wcs").exists()
    assert not (PACKAGE_ROOT / "sphere").exists()


def test_torchfits_python_sources_never_import_astropy_or_fitsio() -> None:
    """Runtime I/O must use vendored CFITSIO + _C, not Python astropy/fitsio."""
    forbidden = (
        "import astropy",
        "from astropy",
        "import fitsio",
        "from fitsio",
    )
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                if pattern in stripped:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {stripped}")
    assert not offenders, "\n".join(offenders)


def test_root_import_stays_runtime_light() -> None:
    script = """
import sys
import torchfits
for name in ('torch', 'numpy', 'pyarrow', 'torchfits._C'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_import_sets_kmp_duplicate_lib_ok() -> None:
    if sys.platform != "darwin":
        pytest.skip("KMP_DUPLICATE_LIB_OK is only set by __init__ on Darwin")
    script = """
import os
os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
import torchfits
assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE", os.environ.get("KMP_DUPLICATE_LIB_OK")
"""
    env = {**os.environ}
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_duplicate_libomp_survives_after_import() -> None:
    """Homebrew libomp + PyTorch libomp abort with OMP Error #15 unless the
    import guard ran first. Skip when Homebrew's dylib is not installed."""
    if sys.platform != "darwin":
        pytest.skip("Homebrew libomp is a macOS-only scenario")
    brew_omp = next(
        (
            path
            for path in (
                Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
                Path("/usr/local/opt/libomp/lib/libomp.dylib"),
            )
            if path.is_file()
        ),
        None,
    )
    if brew_omp is None:
        pytest.skip("Homebrew libomp dylib not installed")
    script = f"""
import ctypes, os
os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
import torchfits
assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
lib = ctypes.CDLL({str(brew_omp)!r}, mode=ctypes.RTLD_GLOBAL)
lib.omp_get_max_threads.restype = ctypes.c_int
lib.omp_get_max_threads()
import torch
print(torch.__version__, flush=True)
"""
    env = {**os.environ}
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_removed_native_cache_environment_is_ignored() -> None:
    """The removed TORCHFITS_CFITSIO_CACHE_* knobs must not break imports."""
    env = {
        **os.environ,
        "TORCHFITS_CFITSIO_CACHE_MB": "256",
        "TORCHFITS_CFITSIO_CACHE_FILES": "32",
    }
    result = subprocess.run(
        [sys.executable, "-c", "import torchfits; torchfits.read"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mmap", [True, False])
def test_table_read_where_does_not_import_torch(tmp_path, mmap: bool) -> None:
    """An Arrow ``table.read(where=...)`` must not pull torch into the process."""
    import numpy as np

    import torchfits

    path = tmp_path / "where.fits"
    torchfits.table.write(
        str(path),
        {"ID": np.arange(10, dtype=np.int32), "MAG": np.linspace(18.0, 20.0, 10)},
        overwrite=True,
    )
    script = f"""
import sys
import torchfits
table = torchfits.table.read(sys.argv[1], where="MAG > 19", mmap={mmap})
assert table.num_rows == 5, table.num_rows
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_documented_torch_free_entry_points_never_import_torch(tmp_path) -> None:
    """Every entry point the architecture page lists as torch-free runs with
    ``import torch`` blocked. ``delete_hdu`` used to load torch through the
    tensor write module that re-exported it."""
    import numpy as np
    import torch

    import torchfits

    image = tmp_path / "image.fits"
    torchfits.write(
        str(image), np.arange(16, dtype=np.float32).reshape(4, 4), overwrite=True
    )
    table = tmp_path / "table.fits"
    torchfits.table.write(
        str(table),
        {"ID": np.arange(5, dtype=np.int32), "MAG": np.linspace(18.0, 20.0, 5)},
        overwrite=True,
        extname="SCI",
    )
    mef = tmp_path / "mef.fits"
    torchfits.write(
        str(mef),
        torchfits.HDUList(
            [
                torchfits.TensorHDU(torch.zeros(0)),
                torchfits.TensorHDU(torch.ones(2, 2), {"EXTNAME": "A"}),
                torchfits.TensorHDU(torch.ones(3), {"EXTNAME": "B"}),
            ]
        ),
        overwrite=True,
    )

    script = """
import contextlib, importlib.abc, io, sys


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "torch":
            raise ImportError("torch is blocked: " + fullname)
        return None


sys.meta_path.insert(0, _Block())

import torchfits
from torchfits.cli.main import main as cli_main

image, table, mef = sys.argv[1:4]
with contextlib.redirect_stdout(io.StringIO()):
    with torchfits.open(table) as hdul:
        hdul.info()
        assert hdul[1].header["EXTNAME"] == "SCI"
        assert hdul["SCI"].header["NAXIS2"] == 5
    torchfits.write_checksums(image)
    assert torchfits.verify_checksums(image)
    for argv in (["info", table], ["header", table], ["verify", image],
                 ["setkey", image, "--key", "OBSERVER", "--value", "x"]):
        assert cli_main(argv) in (0, None), argv
    assert sum(b.num_rows for b in torchfits.table.scan(table, batch_size=2)) == 5
    assert "MAG" in str(torchfits.table.schema(table))
    torchfits.delete_hdu(mef, "A")
assert torchfits.read_num_hdus(mef) == 2
assert "torch" not in sys.modules
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(image), str(table), str(mef)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
