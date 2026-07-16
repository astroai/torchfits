#!/usr/bin/env bash
# Upload torchfits CANFAR bench artifacts to VOSpace via vcp.
set -euo pipefail

bench_out="${1:?usage: $0 <benchmarks_results/run-id-dir>}"
vos_dest="${2:?usage: $0 <bench-out> <vos:user/path/run-id>}"

# Install vos into a private prefix (no root, no --user / usersite).
_vos_root="${HOME}/.local/torchfits-vos"
_vos_bin="${_vos_root}/bin"
mkdir -p "${_vos_bin}" "${_vos_root}/lib/python"
if ! command -v vcp >/dev/null; then
  PYTHONNOUSERSITE=1 python3 -m pip install -q --target "${_vos_root}/lib/python" vos
fi
export PYTHONPATH="${_vos_root}/lib/python${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${_vos_bin}:${PATH}"
# vcp entrypoint may live under the target bin or Scripts layout.
if ! command -v vcp >/dev/null; then
  if [[ -x "${_vos_root}/lib/python/bin/vcp" ]]; then
    ln -sfn "${_vos_root}/lib/python/bin/vcp" "${_vos_bin}/vcp"
  elif [[ -f "${_vos_root}/lib/python/vos/commands/vcp.py" ]]; then
    printf '%s\n' '#!/usr/bin/env bash' \
      "export PYTHONPATH=\"${_vos_root}/lib/python\${PYTHONPATH:+:\${PYTHONPATH}}\"" \
      "exec python3 -m vos.commands.vcp \"\$@\"" >"${_vos_bin}/vcp"
    chmod +x "${_vos_bin}/vcp"
  fi
fi

if ! command -v vcp >/dev/null && ! python3 -c 'import vos' 2>/dev/null; then
  echo "vcp/vos not available after private-prefix install" >&2
  exit 1
fi

if command -v vmkdir >/dev/null; then
  vmkdir -p "${vos_dest}" 2>/dev/null || true
elif python3 -c 'import vos' 2>/dev/null; then
  python3 - <<PY || true
from vos import Client
Client().mkdir("${vos_dest}")
PY
fi

# ponytail: copy directory *contents* (not the directory name) into vos_dest
if command -v vcp >/dev/null; then
  vcp "${bench_out}/." "${vos_dest}/"
else
  python3 - <<PY
from pathlib import Path
from vos import Client
c = Client()
src = Path("${bench_out}")
dest = "${vos_dest}".rstrip("/") + "/"
for p in src.rglob("*"):
    if p.is_file():
        rel = p.relative_to(src).as_posix()
        c.copy(str(p), dest + rel)
PY
fi
echo "TORCHFITS_VOS_URI=${vos_dest}"
