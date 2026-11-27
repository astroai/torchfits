#!/usr/bin/env bash
# Installed-wheel test: CPU torch from the supported range + release smoke.
# cibuildwheel runs it as test-command with {project} (the mounted source tree)
# as $1; build_wheels.yml's wheel-test job runs it again on the built wheel with
# TORCHFITS_TEST_TORCH_SPEC set to the floor and to the tested pin. Run it from
# outside the source tree so the installed wheel, not src/, is imported.
set -euo pipefail

PROJECT="${1:?cibuildwheel test-command must pass project path}"
# Default: the supported floor, uncapped (scripts/torch_support.json), so pip
# resolves the newest PyTorch that has a wheel for this CPython.
FLOOR="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["torch_min"])' \
  "${PROJECT}/scripts/torch_support.json")"
test -n "${FLOOR}"
TORCH_SPEC="${TORCHFITS_TEST_TORCH_SPEC:->=${FLOOR}}"

python -m pip install "torch${TORCH_SPEC}" --extra-index-url https://download.pytorch.org/whl/cpu
# The spec must have held: pip leaves an already-installed torch alone only if
# it satisfies it, so this names the torch the tests below actually ran.
python - "${TORCH_SPEC}" <<'PY'
import sys
from importlib.metadata import version

spec = sys.argv[1]
installed = version("torch").split("+")[0]
print(f"torch {installed} (requested torch{spec})")
if spec.startswith("==") and spec.endswith(".*"):
    want = spec[2:-2]
    assert installed == want or installed.startswith(want + "."), (
        f"torch {installed} does not satisfy torch{spec}"
    )
PY

# The extension links no libtorch: importing it must neither need torch's
# library directory on the loader path nor import the torch module.
python -c 'import sys, torchfits._C as C; assert "torch" not in sys.modules, "importing _C loaded torch"; assert C.HAS_BZIP2, "wheel must link libbz2 (HAS_BZIP2)"'
# The three native artifacts ship together and share one build id. If the wheel
# lost libtorchfits_core, or shipped a _core from a different build than the
# library it dlopens, the failure is a metadata ImportError (or a cross-version
# call into C) that no functional test above would reach.
python -c 'import torchfits._core as c; assert c.TORCH_FREE, "the metadata core must be torch-free"; assert c.core_library_build_id() == c.__build_id__, "build id mismatch"; import torchfits._C as C; assert C.__core_build_id__ == c.__build_id__, "extension and library are from different builds"'
python -m pytest -c /dev/null --noconftest -p no:cacheprovider \
  "${PROJECT}/tests/test_release_smoke.py" \
  "${PROJECT}/tests/test_bz2.py" \
  "${PROJECT}/tests/test_dlpack_boundary.py" -q
