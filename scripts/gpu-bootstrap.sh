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

python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    torch \
    --index-url "${INDEX}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
