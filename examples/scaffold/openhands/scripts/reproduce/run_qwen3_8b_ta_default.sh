#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B + vLLM + ThunderAgent(default router) + OpenHands SWE-bench Lite.
# Runs 50 instances with 8 workers and stores service logs/profiles under outputs.
#
# Usage from repo root:
#   bash examples/scaffold/openhands/scripts/reproduce/run_qwen3_8b_ta_default_50_workers8.sh
#
# Useful overrides:
#   OUTPUT_NAME=my_run VLLM_PORT=4748 TA_PORT=9001 bash ...
#   START_VLLM=0 START_THUNDERAGENT=0 bash ...   # reuse already running services
#   KEEP_SERVICES=1 bash ...                     # keep vLLM/ThunderAgent alive after eval
#   VLLM_BIN=/path/to/vllm PYTHON_BIN=/path/to/python bash ...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SCAFFOLD_DIR="${REPO_ROOT}/examples/scaffold"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-4747}"
TA_PORT="${TA_PORT:-9000}"
ROUTER_MODE="${ROUTER_MODE:-default}"

SWEBENCH_DATASET="${SWEBENCH_DATASET:-princeton-nlp/SWE-bench_Lite}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
SWEBENCH_LIMIT="${SWEBENCH_LIMIT:-50}"
SWEBENCH_WORKERS="${SWEBENCH_WORKERS:-4}"
SWEBENCH_MAX_ITERATIONS="${SWEBENCH_MAX_ITERATIONS:-30}"

OUTPUT_NAME="${OUTPUT_NAME:-qwen3_8b_ta_default_30_workers4}"
OUTPUT_DIR="${SCAFFOLD_DIR}/outputs/${OUTPUT_NAME}"
LOG_DIR="${OUTPUT_DIR}/runtime_logs"
PROFILE_DIR="${OUTPUT_DIR}/thunderagent_profiles"
RUN_CONFIG="${OUTPUT_DIR}/openhands_config.toml"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
SWEBENCH_LOG="${LOG_DIR}/openhands_swebench.log"
PID_DIR="${LOG_DIR}/pids"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

VLLM_ENV_DIR="${VLLM_ENV_DIR:-/raid0/liyi/vllm}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-/raid0/liyi/ThunderAgent}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/.venv/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/.venv/bin/python}"

VLLM_PID=""
TA_PID=""

log() {
  echo "[$(date '+%F %T')] $*"
}

die() {
  echo "[$(date '+%F %T')] ERROR: $*" >&2
  exit 1
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local start now out
  start="$(date +%s)"
  while true; do
    out=""
    if out="$(curl -fsS --max-time 3 "${url}" 2>&1)"; then
      log "${name} is ready: ${url}"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= HEALTH_TIMEOUT_S )); then
      log "${name} did not become ready within ${HEALTH_TIMEOUT_S}s"
      log "Last curl output: ${out}"
      return 1
    fi
    sleep 3
  done
}

cleanup() {
  if [[ "${KEEP_SERVICES}" == "1" ]]; then
    log "KEEP_SERVICES=1; leaving services running."
    return
  fi
  set +e
  if [[ -n "${TA_PID}" ]] && kill -0 "${TA_PID}" 2>/dev/null; then
    log "Stopping ThunderAgent pid=${TA_PID}"
    kill "${TA_PID}" 2>/dev/null
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "Stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

check_port_free() {
  local port="$1"
  local name="$2"
  local port_var="$3"
  local start_var="$4"
  if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${port} )" | grep -q ":${port}"; then
    die "${name} port ${port} is already in use. Stop the old service, set ${port_var}, or set ${start_var}=0."
  fi
}

write_run_config() {
  cat > "${RUN_CONFIG}" <<EOF
[llm.vllm_local]
model = "${MODEL_REPO}"
custom_llm_provider = "openai"
base_url = "http://127.0.0.1:${TA_PORT}/v1"
api_key = "EMPTY"
temperature = 0.0
native_tool_calling = true
timeout = 240
num_retries = 3
EOF
}

start_vllm() {
  if [[ "${START_VLLM}" != "1" ]]; then
    log "START_VLLM=${START_VLLM}; reusing vLLM at http://127.0.0.1:${VLLM_PORT}"
    wait_for_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"
    return
  fi

  check_port_free "${VLLM_PORT}" "vLLM" "VLLM_PORT" "START_VLLM"
  log "Starting vLLM on port ${VLLM_PORT}; log: ${VLLM_LOG}"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  nohup "${VLLM_BIN}" serve "${MODEL_REPO}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}' \
    --max-model-len 128K \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-mfu-metrics \
    --kv-cache-metrics \
    > "${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"
  echo "${VLLM_PID}" > "${PID_DIR}/vllm.pid"
  wait_for_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"
}

start_thunderagent() {
  if [[ "${START_THUNDERAGENT}" != "1" ]]; then
    log "START_THUNDERAGENT=${START_THUNDERAGENT}; reusing ThunderAgent at http://127.0.0.1:${TA_PORT}"
    wait_for_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"
    return
  fi

  check_port_free "${TA_PORT}" "ThunderAgent" "TA_PORT" "START_THUNDERAGENT"
  log "Starting ThunderAgent on port ${TA_PORT}; log: ${TA_LOG}"
  nohup "${PYTHON_BIN}" -m ThunderAgent \
    --backend-type vllm \
    --backends "http://127.0.0.1:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --router "${ROUTER_MODE}" \
    --metrics \
    --profile \
    --profile-dir "${PROFILE_DIR}" \
    > "${TA_LOG}" 2>&1 &
  TA_PID="$!"
  echo "${TA_PID}" > "${PID_DIR}/thunderagent.pid"
  wait_for_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"
}

run_swebench() {
  log "Running SWE-bench Lite: limit=${SWEBENCH_LIMIT}, workers=${SWEBENCH_WORKERS}, max_iter=${SWEBENCH_MAX_ITERATIONS}"
  log "OpenHands output dir: ${OUTPUT_DIR}"
  cd "${SCAFFOLD_DIR}/openhands"
  export PYTHONPATH="${THUNDERAGENT_ENV_DIR}:${PYTHONPATH:-}"
  export EVAL_DOCKER_IMAGE_SOURCE=epoch 
  "${PYTHON_BIN}" -m evaluation.benchmarks.swe_bench.run_infer \
    --config-file "${RUN_CONFIG}" \
    --llm-config vllm_local \
    --agent-cls CodeActAgent \
    --dataset "${SWEBENCH_DATASET}" \
    --split "${SWEBENCH_SPLIT}" \
    --max-iterations "${SWEBENCH_MAX_ITERATIONS}" \
    --eval-num-workers "${SWEBENCH_WORKERS}" \
    --eval-n-limit "${SWEBENCH_LIMIT}" \
    --eval-output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${SWEBENCH_LOG}"
}

main() {
  [[ -d "${SCAFFOLD_DIR}/openhands" ]] || die "OpenHands directory not found: ${SCAFFOLD_DIR}/openhands"
  [[ -x "${VLLM_BIN}" ]] || die "vLLM binary is not executable: ${VLLM_BIN}"
  [[ -x "${PYTHON_BIN}" ]] || die "Python binary is not executable: ${PYTHON_BIN}"
  command -v curl >/dev/null 2>&1 || die "Missing required command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing required command: tee"

  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${PID_DIR}"
  write_run_config

  log "Run config: ${RUN_CONFIG}"
  log "vLLM environment: ${VLLM_ENV_DIR}"
  log "vLLM binary: ${VLLM_BIN}"
  log "ThunderAgent/OpenHands environment: ${THUNDERAGENT_ENV_DIR}"
  log "ThunderAgent/OpenHands Python: ${PYTHON_BIN}"
  log "Logs: ${LOG_DIR}"
  log "ThunderAgent profiles: ${PROFILE_DIR}"

  start_vllm
  start_thunderagent
  run_swebench

  log "Done."
  log "vLLM log: ${VLLM_LOG}"
  log "ThunderAgent log: ${TA_LOG}"
  log "OpenHands log: ${SWEBENCH_LOG}"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
}

main "$@"
