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

mkdir -p "$LOCAL_OUT"

if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found (install from canfar-portal or CANFAR image venv)" >&2
  exit 1
fi

echo "=== CANFAR GPU bench launcher ===" | tee "${LOCAL_OUT}/launcher.log"
echo "server: $(canfar auth show 2>&1 | rg 'Server' || true)" | tee -a "${LOCAL_OUT}/launcher.log"
echo "image=${IMAGE} gpu=${GPU} ref=${GIT_REF} mode=${MODE} run_id=${RUN_ID}" | tee -a "${LOCAL_OUT}/launcher.log"

REMOTE_CMD=$(
  cat <<EOF
set -euo pipefail
SCRATCH="\${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="\${SCRATCH}/torchfits-gpu-bench/${RUN_ID}"
mkdir -p "\$RUN_DIR"
exec > >(tee "\$RUN_DIR/stdout.log") 2> >(tee "\$RUN_DIR/stderr.log" >&2)
SRC="\${TMP_SRC_DIR:-/tmp/src}"
mkdir -p "\$SRC"
cd "\$SRC"
if [[ -d torchfits/.git ]]; then
  cd torchfits
  git fetch origin --tags
  git checkout ${GIT_REF}
  git pull --ff-only origin ${GIT_REF} 2>/dev/null || true
else
  if git clone --depth 1 --branch ${GIT_REF} ${REPO_URL} torchfits 2>/dev/null; then
    cd torchfits
  else
    git clone ${REPO_URL} torchfits
    cd torchfits
    git checkout ${GIT_REF}
  fi
fi
export TORCHFITS_BENCH_RUN_ID=${RUN_ID}
export TORCHFITS_BENCH_MODE=${MODE}
export TORCHFITS_BENCH_LOG_REDIRECTED=1
bash scripts/canfar_gpu_bench_incontainer.sh
EOF
)

CREATE_LOG="${LOCAL_OUT}/create.log"
set +o pipefail
canfar create headless "${IMAGE}" \
  --name "${NAME}" \
  --gpu "${GPU}" \
  --env "TORCHFITS_BENCH_RUN_ID=${RUN_ID}" \
  --env "TORCHFITS_BENCH_MODE=${MODE}" \
  --env "TORCHFITS_GIT_REF=${GIT_REF}" \
  -- bash -lc "${REMOTE_CMD}" 2>&1 | tee "${CREATE_LOG}"
CREATE_RC=${PIPESTATUS[0]}
set -o pipefail
if [[ "${CREATE_RC}" -ne 0 ]]; then
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
  canfar ps --json | python3 - "${SESSION_ID}" <<'PY'
import json, sys
sid = sys.argv[1]
for row in json.load(sys.stdin):
    if row.get("id") == sid:
        print(row.get("status", ""))
        raise SystemExit(0)
print("")
PY
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

echo "finished status=${STATUS}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "local artifacts: ${LOCAL_OUT}" | tee -a "${LOCAL_OUT}/launcher.log"
echo "scratch path (ephemeral): \${TMP_SCRATCH_DIR:-/scratch}/torchfits-gpu-bench/${RUN_ID}" | tee -a "${LOCAL_OUT}/launcher.log"

case "${STATUS}" in
  Succeeded|Completed) exit 0 ;;
  *) exit 1 ;;
esac
