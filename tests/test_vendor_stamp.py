"""A vendored extern/cfitsio must match the checked-in inputs it is built from.

Measured on 2026-11-27: a worktree whose extern/cfitsio was vendored before the
2026-11-03 cfitsio-4.7.0-http-*.patch security patches built and installed with
`pixi run dev` without a word -- CMake only vendored when extern/cfitsio was
absent -- and seven tests failed against the unpatched CFITSIO. Nothing
compared the tree with extern/VERSIONS.txt or extern/patches.

extern/vendor.sh now stamps the tree with the sha256 of each input
(extern/cfitsio/.torchfits-vendor-stamp) and CMake recomputes the same lines
(extern/vendor_stamp.cmake), re-vendoring a missing/stale tree in a git
checkout and refusing it elsewhere. The two definitions live in two languages,
so these tests run both and require byte-identical output, including the patch
order (byte order, whatever the locale) and which tag's patches count.

Hermetic: `curl` is stubbed with a script that copies a prebuilt tarball.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERN = ROOT / "extern"
STAMP = ".torchfits-vendor-stamp"
TAG = "cfitsio-9.9.9"

pytestmark = [
    pytest.mark.skipif(
        shutil.which("bash") is None or shutil.which("tar") is None,
        reason="extern/vendor.sh needs bash and tar",
    ),
    pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake"),
]

_STUB_CURL = """#!/usr/bin/env bash
set -eu
out=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "-o" ]]; then out="$arg"; fi
  prev="$arg"
done
cp "$TF_FAKE_ARCHIVE" "$out"
"""

# Applies to the fake archive's imcompress.c.
_PATCH = """--- a/imcompress.c
+++ b/imcompress.c
@@ -1 +1,2 @@
 /* stand-in for imcompress.c */
+/* patched */
"""


def _fake_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in {
            "imcompress.c": b"/* stand-in for imcompress.c */\n",
            "CMakeLists.txt": b"# stand-in\n",
        }.items():
            info = tarfile.TarInfo(f"cfitsio-{TAG}/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A throwaway checkout: vendor.sh, the CMake helper, a pin and patches."""
    extern = tmp_path / "extern"
    (extern / "patches").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    shutil.copy(EXTERN / "vendor.sh", extern / "vendor.sh")
    shutil.copy(EXTERN / "vendor_stamp.cmake", extern / "vendor_stamp.cmake")
    archive = tmp_path / "fake.tar.gz"
    archive.write_bytes(_fake_archive())
    (extern / "VERSIONS.txt").write_text(
        "cfitsio_repo=HEASARC/cfitsio\n"
        f"cfitsio_tag={TAG}\n"
        f"cfitsio_sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    # Upper-case sorts first in byte order but not in most locales' collation:
    # both implementations must agree on the order anyway.
    (extern / "patches" / f"{TAG}-a-mark.patch").write_text(_PATCH, encoding="utf-8")
    (extern / "patches" / f"{TAG}-B-empty.patch").write_text("", encoding="utf-8")
    # Another tag's patch is neither applied nor stamped.
    (extern / "patches" / "cfitsio-0.0.1-other.patch").write_text(
        "not applied\n", encoding="utf-8"
    )
    curl = tmp_path / "bin" / "curl"
    curl.write_text(_STUB_CURL, encoding="utf-8")
    curl.chmod(0o755)
    return tmp_path


def _env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{root / 'bin'}{os.pathsep}{env['PATH']}"
    env["TF_FAKE_ARCHIVE"] = str(root / "fake.tar.gz")
    env["LC_ALL"] = "en_US.UTF-8"
    return env


def _vendor(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "extern" / "vendor.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
        env=_env(root),
    )


def _bash_stamp(root: Path) -> str:
    result = _vendor(root, "--print-stamp")
    assert result.returncode == 0, result.stderr
    return result.stdout


def _cmake_stamp(root: Path, out: Path) -> tuple[str, str]:
    result = subprocess.run(
        [
            "cmake",
            f"-DTORCHFITS_REPO_ROOT={root}",
            f"-DOUT={out}",
            "-P",
            str(root / "extern" / "vendor_stamp.cmake"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    status = Path(f"{out}.status").read_text(encoding="utf-8").strip()
    return out.read_text(encoding="utf-8"), status


def test_bash_and_cmake_compute_the_same_stamp_for_the_repo(tmp_path: Path) -> None:
    bash_stamp = _bash_stamp(ROOT)
    cmake_stamp, _ = _cmake_stamp(ROOT, tmp_path / "stamp")
    assert bash_stamp == cmake_stamp
    labels = [line.split("  ", 1)[1] for line in bash_stamp.splitlines()]
    tag = next(
        line.split("=", 1)[1]
        for line in (EXTERN / "VERSIONS.txt").read_text().splitlines()
        if line.startswith("cfitsio_tag=")
    )
    expected_patches = sorted(
        f"patches/{p.name}" for p in (EXTERN / "patches").glob(f"{tag}-*.patch")
    )
    assert labels == ["VERSIONS.txt", "vendor.sh", *expected_patches]


def test_bash_and_cmake_agree_on_patch_order_and_tag_selection(
    tree: Path, tmp_path: Path
) -> None:
    bash_stamp = _bash_stamp(tree)
    cmake_stamp, status = _cmake_stamp(tree, tmp_path / "stamp")
    assert bash_stamp == cmake_stamp
    assert [line.split("  ", 1)[1] for line in bash_stamp.splitlines()] == [
        "VERSIONS.txt",
        "vendor.sh",
        f"patches/{TAG}-B-empty.patch",
        f"patches/{TAG}-a-mark.patch",
    ]
    assert status == "missing"  # nothing vendored yet


def test_vendor_run_stamps_tree_and_any_input_change_makes_it_stale(
    tree: Path, tmp_path: Path
) -> None:
    result = _vendor(tree, "--cfitsio-version", "extern/VERSIONS.txt")
    assert result.returncode == 0, result.stderr
    vendored = tree / "extern" / "cfitsio"
    assert "/* patched */" in (vendored / "imcompress.c").read_text()
    stamp = (vendored / STAMP).read_text(encoding="utf-8")
    assert stamp == _bash_stamp(tree)
    assert _vendor(tree, "--check-stamp").returncode == 0
    assert _cmake_stamp(tree, tmp_path / "s0")[1] == "fresh"

    # Editing an applied patch, adding one, or touching the pin each stale it.
    patch = tree / "extern" / "patches" / f"{TAG}-a-mark.patch"
    patch.write_text(_PATCH + "\n", encoding="utf-8")
    check = _vendor(tree, "--check-stamp")
    assert check.returncode == 1
    assert f"{TAG}-a-mark.patch" in check.stderr
    assert _cmake_stamp(tree, tmp_path / "s1")[1] == "stale"
    patch.write_text(_PATCH, encoding="utf-8")
    assert _cmake_stamp(tree, tmp_path / "s2")[1] == "fresh"

    added = tree / "extern" / "patches" / f"{TAG}-new.patch"
    added.write_text("", encoding="utf-8")
    assert _vendor(tree, "--check-stamp").returncode == 1
    assert _cmake_stamp(tree, tmp_path / "s3")[1] == "stale"
    added.unlink()

    # A patch for another tag is not an input.
    (tree / "extern" / "patches" / "cfitsio-0.0.1-other.patch").write_text("x\n")
    assert _cmake_stamp(tree, tmp_path / "s4")[1] == "fresh"

    (vendored / STAMP).unlink()
    assert _vendor(tree, "--check-stamp").returncode == 1
    assert _cmake_stamp(tree, tmp_path / "s5")[1] == "missing"


def test_bare_tag_preview_leaves_no_stamp(tree: Path) -> None:
    """A preview of another tag is not the pinned tree: the build must not
    accept it as current."""
    result = subprocess.run(
        ["bash", str(tree / "extern" / "vendor.sh"), "--cfitsio-version", TAG],
        capture_output=True,
        text=True,
        check=False,
        cwd=tree,
        env={**_env(tree), "TORCHFITS_VENDOR_ALLOW_UNPINNED": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert (tree / "extern" / "cfitsio" / "imcompress.c").is_file()
    assert not (tree / "extern" / "cfitsio" / STAMP).exists()
    assert _vendor(tree, "--check-stamp").returncode == 1


def test_cmake_build_checks_the_stamp_before_using_the_tree() -> None:
    cmake = (ROOT / "src" / "torchfits" / "cpp_src" / "CMakeLists.txt").read_text()
    assert 'include("${TORCHFITS_REPO_ROOT}/extern/vendor_stamp.cmake")' in cmake
    check = cmake.index("torchfits_vendor_stamp_check(")
    assert check < cmake.index('add_subdirectory("${CFITSIO_SOURCE_DIR}"')
    assert "CMAKE_CONFIGURE_DEPENDS" in cmake[check:]
