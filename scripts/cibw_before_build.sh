#!/usr/bin/env bash
# Install the build frontend into the cibuildwheel build venv. The extension
# links no libtorch (DLPack boundary), so the build needs no PyTorch.
set -euo pipefail

python -m pip install -U pip
python -m pip install \
  "scikit-build-core>=1.0.3" \
  "nanobind>=3.0.1" \
  "cmake>=3.21" \
  "ninja>=1.10"
