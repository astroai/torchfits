#!/usr/bin/env bash
# cibuildwheel test-command: CPU torch from the supported range + release smoke.
# {project} is passed as $1 (the mounted source tree).
set -euo pipefail

PROJECT="${1:?cibuildwheel test-command must pass project path}"
# The supported floor, uncapped (scripts/torch_support.json): pip resolves the
# newest PyTorch that has a wheel for this CPython.
FLOOR="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["torch_min"])' \
  "${PROJECT}/scripts/torch_support.json")"
test -n "${FLOOR}"

python -m pip install "torch>=${FLOOR}" --extra-index-url https://download.pytorch.org/whl/cpu

# The extension links no libtorch: importing it must neither need torch's
# library directory on the loader path nor import the torch module.
python -c 'import sys, torchfits._C as C; assert "torch" not in sys.modules, "importing _C loaded torch"; assert C.HAS_BZIP2, "wheel must link libbz2 (HAS_BZIP2)"'
# The three native artifacts ship together and share one build id. If the wheel
# lost libtorchfits_core, or shipped a _core from a different build than the
# library it dlopens, the failure is a metadata ImportError (or a cross-version
# call into C) that no functional test above would reach.
python -c 'import torchfits._core as c; assert c.TORCH_FREE, "the metadata core must be torch-free"; assert c.core_library_build_id() == c.__build_id__, "build id mismatch"; import torchfits._C as C; assert C.__core_build_id__ == c.__build_id__, "extension and library are from different builds"'
python -m pytest -c /dev/null --noconftest \
  "${PROJECT}/tests/test_release_smoke.py" \
  "${PROJECT}/tests/test_bz2.py" -q
