#!/usr/bin/env bash
# Bootstrap the CUDA-enabled PyTorch wheel for the pixi bench-gpu env.
# macOS is intentionally skipped - conda-forge pytorch (CPU) is the default there.
set -euo pipefail

case "$(uname)" in
    Darwin*)
        echo "skipping CUDA bootstrap on macOS"
        exit 0
        ;;
esac

INDEX="${TORCHFITS_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# Avoid writing into ~/.local on CANFAR (/arc/home) — concurrent sessions
# corrupt shared user-site packages mid-uninstall.
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMPDIR:-/tmp}/torchfits-pip-cache}"
mkdir -p "${PIP_CACHE_DIR}"

# Match package pin torch>=2.10,<2.11 (cu128 index otherwise installs latest 2.11).
TORCH_SPEC="${TORCHFITS_TORCH_SPEC:-torch>=2.10,<2.11}"

python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    --no-user \
    "${TORCH_SPEC}" \
    --index-url "${INDEX}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
