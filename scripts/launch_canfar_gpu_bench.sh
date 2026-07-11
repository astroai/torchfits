#!/usr/bin/env bash
# Launch a headless GPU session on CANFAR staging, clone torchfits, run pixi bench/tests,
# and capture platform logs locally under benchmarks_results/canfar_<run-id>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GIT_REF="${TORCHFITS_GIT_REF:-main}"
RUN_ID="${TORCHFITS_BENCH_RUN_ID:-exhaustive_cuda_0.7.0_$(date -u +%Y%m%d_%H%M%S)}"
MODE="${TORCHFITS_BENCH_MODE:-exhaustive}"
IMAGE="${TORCHFITS_CANFAR_IMAGE:-astroai/base:latest}"
GPU="${TORCHFITS_CANFAR_GPU:-1}"
# ponytail: CANFAR session names allow [A-Za-z0-9-] only (no dots/underscores)
SAFE_TAG="$(printf '%s' "${RUN_ID}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
NAME="${TORCHFITS_CANFAR_NAME:-torchfits-gpu-${SAFE_TAG}}"
REPO_URL="${TORCHFITS_GIT_URL:-https://github.com/astroai/torchfits.git}"
LOCAL_OUT="${ROOT_DIR}/benchmarks_results/canfar_${RUN_ID}"
POLL_SECS="${TORCHFITS_CANFAR_POLL_SECS:-30}"
CLONE_DIR="/scratch/torchfits"
VOS_DEST="${TORCHFITS_VOS_DEST:-vos:sfabbro/torchfits-gpu-bench/${RUN_ID}}"

mkdir -p "$LOCAL_OUT"

if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found (install from canfar-portal or CANFAR image venv)" >&2
  exit 1
fi

echo "=== CANFAR GPU bench launcher ===" | tee "${LOCAL_OUT}/launcher.log"
echo "server: $(canfar auth show 2>&1 | rg 'Server' || true)" | tee -a "${LOCAL_OUT}/launcher.log"
echo "image=${IMAGE} gpu=${GPU} ref=${GIT_REF} mode=${MODE} run_id=${RUN_ID}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "vos_dest=${VOS_DEST}" | tee -a "${LOCAL_OUT}/launcher.log"

# ponytail: skaha splits cmd args on spaces; tabs keep bash -c script as one token (no $ or &)
REMOTE_PLAIN="git clone --depth 1 --branch ${GIT_REF} ${REPO_URL} ${CLONE_DIR}; cd ${CLONE_DIR}; bash scripts/canfar_gpu_bench_incontainer.sh"
REMOTE_CMD="$(printf '%s' "${REMOTE_PLAIN}" | tr ' ' '\t')"

CREATE_LOG="${LOCAL_OUT}/create.log"
set +o pipefail
canfar create headless "${IMAGE}" \
  --name "${NAME}" \
  --gpu "${GPU}" \
  --env "TORCHFITS_BENCH_RUN_ID=${RUN_ID}" \
  --env "TORCHFITS_BENCH_MODE=${MODE}" \
  --env "TORCHFITS_GIT_REF=${GIT_REF}" \
  --env "TORCHFITS_BENCH_LOG_REDIRECTED=1" \
  --env "PIXI_HOME=/scratch/torchfits-pixi-home" \
  --env "PIXI_CACHE_DIR=/scratch/torchfits-pixi-cache" \
  --env "TORCHFITS_VOS_DEST=${VOS_DEST}" \
  -- bash -c "${REMOTE_CMD}" 2>&1 | tee "${CREATE_LOG}"
CREATE_RC=${PIPESTATUS[0]}
set -o pipefail
if [[ "${CREATE_RC}" -ne 0 ]]; then
  if rg -q 'No authentication provided for unknown or private image' "${CREATE_LOG}" 2>/dev/null; then
    cat >&2 <<EOF
canfar create failed: private registry image (${IMAGE}).

${IMAGE} is not in \`canfar image ls\`; x509 alone only pulls contributed/public
images. For astroai/base configure Harbor once:

  canfar config set registry.url https://images.canfar.net
  canfar config set registry.username <harbor-user>
  canfar config set registry.secret <harbor-token>

Or use a listed image, e.g.:
  TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest pixi run bench-canfar-gpu
EOF
  fi
  echo "canfar create failed (rc=${CREATE_RC}); see ${CREATE_LOG}" >&2
  exit 1
fi

SESSION_ID="$(
  python3 - "${CREATE_LOG}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
m = re.search(r"ID:\s*([^)]+)\)", text)
print(m.group(1).strip() if m else "")
PY
)"
if [[ -z "${SESSION_ID}" ]]; then
  echo "could not parse session ID from ${CREATE_LOG}" >&2
  exit 1
fi
echo "${SESSION_ID}" > "${LOCAL_OUT}/session_id.txt"
echo "session_id=${SESSION_ID}" | tee -a "${LOCAL_OUT}/launcher.log"

terminal_status() {
  canfar info "${SESSION_ID}" 2>/dev/null | python3 -c '
import re, sys

text = sys.stdin.read()
m = re.search(r"^\s*Status\s+(\S+)", text, re.MULTILINE)
print(m.group(1) if m else "")
'
}

STATUS=""
while true; do
  STATUS="$(terminal_status || true)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${STATUS:-unknown}" | tee -a "${LOCAL_OUT}/launcher.log"
  case "${STATUS}" in
    Succeeded|Completed|Failed|Error|Terminating)
      break
      ;;
  esac
  sleep "${POLL_SECS}"
done

canfar logs "${SESSION_ID}" > "${LOCAL_OUT}/canfar_logs.txt" 2>&1 || true
canfar events "${SESSION_ID}" > "${LOCAL_OUT}/canfar_events.txt" 2>&1 || true

if [[ "${STATUS}" == "Succeeded" || "${STATUS}" == "Completed" ]]; then
  if command -v vcp >/dev/null && bash scripts/fetch_canfar_bench_vos.sh "${RUN_ID}"; then
    echo "fetched benchmarks_results/${RUN_ID} from ${VOS_DEST}" | tee -a "${LOCAL_OUT}/launcher.log"
  elif python3 scripts/import_canfar_bench_artifacts.py "${LOCAL_OUT}/canfar_logs.txt" "${RUN_ID}" --dest "${ROOT_DIR}/benchmarks_results" 2>/dev/null; then
    echo "imported benchmarks_results/${RUN_ID} from session logs (vcp fallback)" | tee -a "${LOCAL_OUT}/launcher.log"
  else
    echo "fetch results: bash scripts/fetch_canfar_bench_vos.sh ${RUN_ID}" | tee -a "${LOCAL_OUT}/launcher.log"
    echo "  (vos: ${VOS_DEST})" | tee -a "${LOCAL_OUT}/launcher.log"
  fi
fi

echo "finished status=${STATUS}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "local artifacts: ${LOCAL_OUT}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "vos artifacts: ${VOS_DEST}" | tee -a "${LOCAL_OUT}/launcher.log"

case "${STATUS}" in
  Succeeded|Completed) exit 0 ;;
  *) exit 1 ;;
esac
