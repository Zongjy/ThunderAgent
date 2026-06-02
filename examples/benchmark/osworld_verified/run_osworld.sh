#!/usr/bin/env bash
set -euo pipefail

# Run ThunderAgent and OSWorld against an already-running vLLM service.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCAFFOLD_DIR="${REPO_ROOT}/examples/scaffold/osworld"
OSWORLD_ENV_DIR="${OSWORLD_ENV_DIR:-${REPO_ROOT}/.venv-osworld}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${OSWORLD_ENV_DIR}}"

if [[ -z "${OSWORLD_SOURCE_ROOT:-}" ]]; then
  if [[ -d "${SCAFFOLD_DIR}/OSWorld" ]]; then
    OSWORLD_SOURCE_ROOT="${SCAFFOLD_DIR}/OSWorld"
  elif [[ -d "/raid0/liyi/OSWorld" ]]; then
    OSWORLD_SOURCE_ROOT="/raid0/liyi/OSWorld"
  elif [[ -d "${HOME}/OSWorld" ]]; then
    OSWORLD_SOURCE_ROOT="${HOME}/OSWorld"
  else
    OSWORLD_SOURCE_ROOT="${SCAFFOLD_DIR}/OSWorld"
  fi
fi

OSWORLD_AGENT_KIND="${OSWORLD_AGENT_KIND:-qwen35}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
if [[ -z "${SERVED_MODEL_NAME:-}" ]]; then
  if [[ "${MODEL}" == "Qwen/Qwen3.5-27B" ]]; then
    SERVED_MODEL_NAME="qwen3.5-27B"
  else
    SERVED_MODEL_NAME="${MODEL}"
  fi
fi
AGENT_LLM="${AGENT_LLM:-${SERVED_MODEL_NAME}}"

VLLM_PORT="${VLLM_PORT:-4747}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-${VLLM_BASE_URL}/health}"
VLLM_METRICS_URL="${VLLM_METRICS_URL:-${VLLM_BASE_URL}/metrics}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OSWORLD_REWRITE_HF_URLS="${OSWORLD_REWRITE_HF_URLS:-1}"
TA_PORT="${TA_PORT:-9000}"
ROUTER_MODE="${ROUTER_MODE:-default}"
THUNDERAGENT_BACKEND_TYPE="${THUNDERAGENT_BACKEND_TYPE:-vllm}"
THUNDERAGENT_BACKENDS="${THUNDERAGENT_BACKENDS:-${VLLM_BASE_URL}}"
THUNDERAGENT_KV_CAPACITY_TOKENS="${THUNDERAGENT_KV_CAPACITY_TOKENS:-217725}"
THUNDERAGENT_SCHEDULER_INTERVAL="${THUNDERAGENT_SCHEDULER_INTERVAL:-5}"
THUNDERAGENT_ACTING_TOKEN_WEIGHT="${THUNDERAGENT_ACTING_TOKEN_WEIGHT:-0.5}"
THUNDERAGENT_USE_ACTING_TOKEN_DECAY="${THUNDERAGENT_USE_ACTING_TOKEN_DECAY:-1}"

OSWORLD_DOMAIN="${OSWORLD_DOMAIN:-all}"
OSWORLD_TASK_IDS="${OSWORLD_TASK_IDS:-}"
OSWORLD_TASK_START="${OSWORLD_TASK_START:-0}"
OSWORLD_NUM_TASKS="${OSWORLD_NUM_TASKS:-1}" # 0 means all selected tasks
OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-100}"
OSWORLD_MAX_CONCURRENCY="${OSWORLD_MAX_CONCURRENCY:-16}"
OSWORLD_ACTION_SPACE="${OSWORLD_ACTION_SPACE:-pyautogui}"
OSWORLD_OBSERVATION_TYPE="${OSWORLD_OBSERVATION_TYPE:-screenshot}"
OSWORLD_SLEEP_AFTER_EXECUTION="${OSWORLD_SLEEP_AFTER_EXECUTION:-3.0}"
OSWORLD_HEADLESS="${OSWORLD_HEADLESS:-1}"
OSWORLD_PROGRESS="${OSWORLD_PROGRESS:-1}"
OSWORLD_PROGRESS_INTERVAL="${OSWORLD_PROGRESS_INTERVAL:-10}"
OSWORLD_PROVIDER_NAME="${OSWORLD_PROVIDER_NAME:-docker}"
OSWORLD_REGION="${OSWORLD_REGION:-us-east-1}"
OSWORLD_PATH_TO_VM="${OSWORLD_PATH_TO_VM:-}"
OSWORLD_CLIENT_PASSWORD="${OSWORLD_CLIENT_PASSWORD:-}"
OSWORLD_SCREEN_WIDTH="${OSWORLD_SCREEN_WIDTH:-1920}"
OSWORLD_SCREEN_HEIGHT="${OSWORLD_SCREEN_HEIGHT:-1080}"
OSWORLD_TEST_CONFIG_BASE_DIR="${OSWORLD_TEST_CONFIG_BASE_DIR:-${OSWORLD_SOURCE_ROOT}/evaluation_examples}"
OSWORLD_TEST_ALL_META_PATH="${OSWORLD_TEST_ALL_META_PATH:-${OSWORLD_SOURCE_ROOT}/evaluation_examples/test_nogdrive.json}"

OSWORLD_TEMPERATURE="${OSWORLD_TEMPERATURE:-0}"
OSWORLD_TOP_P="${OSWORLD_TOP_P:-0.9}"
OSWORLD_MAX_OUTPUT_TOKENS="${OSWORLD_MAX_OUTPUT_TOKENS:-32768}"
OSWORLD_EXTRA_BODY="${OSWORLD_EXTRA_BODY:-}"

QWEN3VL_MODEL="${QWEN3VL_MODEL:-${AGENT_LLM}}"
QWEN3VL_BASE_URL="${QWEN3VL_BASE_URL:-http://127.0.0.1:${TA_PORT}/v1}"
QWEN3VL_API_KEY="${QWEN3VL_API_KEY:-EMPTY}"
QWEN3VL_MAX_TOKENS="${QWEN3VL_MAX_TOKENS:-${OSWORLD_MAX_OUTPUT_TOKENS}}"
QWEN3VL_COORDINATE_TYPE="${QWEN3VL_COORDINATE_TYPE:-relative}"
QWEN3VL_HISTORY_N="${QWEN3VL_HISTORY_N:-4}"
QWEN3VL_REQUEST_TIMEOUT="${QWEN3VL_REQUEST_TIMEOUT:-500}"
QWEN3VL_MAX_RETRIES="${QWEN3VL_MAX_RETRIES:-5}"
QWEN3VL_ENABLE_THINKING="${QWEN3VL_ENABLE_THINKING:-0}"
QWEN3VL_THINKING_BUDGET="${QWEN3VL_THINKING_BUDGET:-32768}"
QWEN3VL_USE_VLLM_TOOL_CALLS="${QWEN3VL_USE_VLLM_TOOL_CALLS:-1}"
QWEN3VL_TOOL_CHOICE="${QWEN3VL_TOOL_CHOICE:-named}"

CHECK_VLLM="${CHECK_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/bin/python}"
RUN_NAME="${RUN_NAME:-osworld_verified_${OSWORLD_AGENT_KIND}_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/osworld_verified/runs}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
MANIFEST="${RUN_DIR}/manifest.env"
RESULTS_DIR="${RUN_DIR}/osworld_outputs"
SUMMARY_DIR="${RUN_DIR}/_analysis"

TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
OSWORLD_LOG="${LOG_DIR}/osworld_verified.log"

TA_PID=""
SAMPLER_PID=""

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

wait_url() {
  local url="$1" name="$2" start now_ts
  start="$(date +%s)"
  until curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
    now_ts="$(date +%s)"
    (( now_ts - start < HEALTH_TIMEOUT_S )) || die "${name} health timeout: ${url}"
    sleep 3
  done
  log "${name} ready: ${url}"
}

ensure_port_free() {
  local port="$1" name="$2"
  if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${port} )" | grep -q ":${port}"; then
    die "${name} port ${port} is in use; disable START_${name} or use another port"
  fi
}

cleanup() {
  [[ "${KEEP_SERVICES}" == "1" ]] && { log "KEEP_SERVICES=1; ThunderAgent left running."; return; }
  set +e
  [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null
  [[ -n "${TA_PID}" ]] && kill "${TA_PID}" 2>/dev/null
}
trap cleanup EXIT

write_manifest() {
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${RESULTS_DIR}" "${SUMMARY_DIR}"
  cat > "${MANIFEST}" <<EOF
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
AGENT_LLM=${AGENT_LLM}
VLLM_PORT=${VLLM_PORT}
VLLM_BASE_URL=${VLLM_BASE_URL}
VLLM_HEALTH_URL=${VLLM_HEALTH_URL}
VLLM_METRICS_URL=${VLLM_METRICS_URL}
HF_ENDPOINT=${HF_ENDPOINT}
OSWORLD_REWRITE_HF_URLS=${OSWORLD_REWRITE_HF_URLS}
TA_PORT=${TA_PORT}
ROUTER_MODE=${ROUTER_MODE}
THUNDERAGENT_BACKEND_TYPE=${THUNDERAGENT_BACKEND_TYPE}
THUNDERAGENT_BACKENDS=${THUNDERAGENT_BACKENDS}
THUNDERAGENT_KV_CAPACITY_TOKENS=${THUNDERAGENT_KV_CAPACITY_TOKENS}
THUNDERAGENT_SCHEDULER_INTERVAL=${THUNDERAGENT_SCHEDULER_INTERVAL}
THUNDERAGENT_ACTING_TOKEN_WEIGHT=${THUNDERAGENT_ACTING_TOKEN_WEIGHT}
THUNDERAGENT_USE_ACTING_TOKEN_DECAY=${THUNDERAGENT_USE_ACTING_TOKEN_DECAY}
OSWORLD_SOURCE_ROOT=${OSWORLD_SOURCE_ROOT}
OSWORLD_DOMAIN=${OSWORLD_DOMAIN}
OSWORLD_TASK_IDS=${OSWORLD_TASK_IDS}
OSWORLD_TASK_START=${OSWORLD_TASK_START}
OSWORLD_NUM_TASKS=${OSWORLD_NUM_TASKS}
OSWORLD_MAX_STEPS=${OSWORLD_MAX_STEPS}
OSWORLD_MAX_CONCURRENCY=${OSWORLD_MAX_CONCURRENCY}
OSWORLD_ACTION_SPACE=${OSWORLD_ACTION_SPACE}
OSWORLD_OBSERVATION_TYPE=${OSWORLD_OBSERVATION_TYPE}
OSWORLD_SLEEP_AFTER_EXECUTION=${OSWORLD_SLEEP_AFTER_EXECUTION}
OSWORLD_HEADLESS=${OSWORLD_HEADLESS}
OSWORLD_PROGRESS=${OSWORLD_PROGRESS}
OSWORLD_PROGRESS_INTERVAL=${OSWORLD_PROGRESS_INTERVAL}
OSWORLD_PROVIDER_NAME=${OSWORLD_PROVIDER_NAME}
OSWORLD_REGION=${OSWORLD_REGION}
OSWORLD_PATH_TO_VM=${OSWORLD_PATH_TO_VM}
OSWORLD_CLIENT_PASSWORD=${OSWORLD_CLIENT_PASSWORD}
OSWORLD_SCREEN_WIDTH=${OSWORLD_SCREEN_WIDTH}
OSWORLD_SCREEN_HEIGHT=${OSWORLD_SCREEN_HEIGHT}
OSWORLD_TEST_CONFIG_BASE_DIR=${OSWORLD_TEST_CONFIG_BASE_DIR}
OSWORLD_TEST_ALL_META_PATH=${OSWORLD_TEST_ALL_META_PATH}
OSWORLD_TEMPERATURE=${OSWORLD_TEMPERATURE}
OSWORLD_TOP_P=${OSWORLD_TOP_P}
OSWORLD_MAX_OUTPUT_TOKENS=${OSWORLD_MAX_OUTPUT_TOKENS}
OSWORLD_EXTRA_BODY=${OSWORLD_EXTRA_BODY}
QWEN3VL_MODEL=${QWEN3VL_MODEL}
QWEN3VL_BASE_URL=${QWEN3VL_BASE_URL}
QWEN3VL_API_KEY=${QWEN3VL_API_KEY}
QWEN3VL_MAX_TOKENS=${QWEN3VL_MAX_TOKENS}
QWEN3VL_COORDINATE_TYPE=${QWEN3VL_COORDINATE_TYPE}
QWEN3VL_HISTORY_N=${QWEN3VL_HISTORY_N}
QWEN3VL_REQUEST_TIMEOUT=${QWEN3VL_REQUEST_TIMEOUT}
QWEN3VL_MAX_RETRIES=${QWEN3VL_MAX_RETRIES}
QWEN3VL_ENABLE_THINKING=${QWEN3VL_ENABLE_THINKING}
QWEN3VL_THINKING_BUDGET=${QWEN3VL_THINKING_BUDGET}
QWEN3VL_USE_VLLM_TOOL_CALLS=${QWEN3VL_USE_VLLM_TOOL_CALLS}
QWEN3VL_TOOL_CHOICE=${QWEN3VL_TOOL_CHOICE}
CHECK_VLLM=${CHECK_VLLM}
START_THUNDERAGENT=${START_THUNDERAGENT}
KEEP_SERVICES=${KEEP_SERVICES}
SAMPLE_METRICS=${SAMPLE_METRICS}
PYTHON_BIN=${PYTHON_BIN}
THUNDERAGENT_ENV_DIR=${THUNDERAGENT_ENV_DIR}
OSWORLD_ENV_DIR=${OSWORLD_ENV_DIR}
RUN_NAME=${RUN_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_DIR=${RUN_DIR}
EOF
}

check_vllm() {
  [[ "${CHECK_VLLM}" == "1" ]] || { log "CHECK_VLLM=0; skip vLLM health check."; return; }
  wait_url "${VLLM_HEALTH_URL}" "vLLM"
}

start_thunderagent() {
  [[ "${START_THUNDERAGENT}" == "1" ]] || { log "Reuse ThunderAgent: ${TA_PORT}"; wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"; return; }
  ensure_port_free "${TA_PORT}" "THUNDERAGENT"

  local cmd=(
    "${PYTHON_BIN}" -m ThunderAgent
    --backend-type "${THUNDERAGENT_BACKEND_TYPE}"
    --backends "${THUNDERAGENT_BACKENDS}"
    --port "${TA_PORT}"
    --router "${ROUTER_MODE}"
    --metrics
    --metrics-interval "${METRICS_INTERVAL_S}"
    --profile
    --profile-dir "${PROFILE_DIR}"
    --scheduler-interval "${THUNDERAGENT_SCHEDULER_INTERVAL}"
    --acting-token-weight "${THUNDERAGENT_ACTING_TOKEN_WEIGHT}"
  )
  [[ "${THUNDERAGENT_USE_ACTING_TOKEN_DECAY}" == "1" ]] && cmd+=(--use-acting-token-decay)

  log "Start ThunderAgent router=${ROUTER_MODE}: ${TA_LOG}"
  nohup env \
    VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" \
    PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" \
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    THUNDERAGENT_KV_CAPACITY_TOKENS="${THUNDERAGENT_KV_CAPACITY_TOKENS}" \
    "${cmd[@]}" > "${TA_LOG}" 2>&1 &
  TA_PID="$!"
  wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"
}

start_sampler() {
  [[ "${SAMPLE_METRICS}" == "1" ]] || return 0
  (
    while true; do
      printf '{"sample_ts":%s,"payload":%s}\n' "$(date +%s)" "$(curl -fsS --max-time 5 "http://127.0.0.1:${TA_PORT}/health" 2>/dev/null || echo '{}')" >> "${LOG_DIR}/thunderagent_health.jsonl"
      printf '{"sample_ts":%s,"payload":%s}\n' "$(date +%s)" "$(curl -fsS --max-time 5 "http://127.0.0.1:${TA_PORT}/metrics" 2>/dev/null || echo '{}')" >> "${LOG_DIR}/thunderagent_metrics.jsonl"
      { printf '# sample_ts=%s\n' "$(date +%s)"; curl -fsS --max-time 5 "${VLLM_METRICS_URL}" 2>/dev/null || true; printf '\n'; } >> "${LOG_DIR}/vllm_metrics.prom"
      sleep "${METRICS_INTERVAL_S}"
    done
  ) &
  SAMPLER_PID="$!"
}

run_osworld() {
  local args=(
    "${SCAFFOLD_DIR}/run_osworld_verified.py"
    --osworld-root "${OSWORLD_SOURCE_ROOT}"
    --test-config-base-dir "${OSWORLD_TEST_CONFIG_BASE_DIR}"
    --test-all-meta-path "${OSWORLD_TEST_ALL_META_PATH}"
    --base-url "http://127.0.0.1:${TA_PORT}/v1"
    --model "${AGENT_LLM}"
    --domain "${OSWORLD_DOMAIN}"
    --task-start "${OSWORLD_TASK_START}"
    --num-tasks "${OSWORLD_NUM_TASKS}"
    --max-steps "${OSWORLD_MAX_STEPS}"
    --max-concurrency "${OSWORLD_MAX_CONCURRENCY}"
    --action-space "${OSWORLD_ACTION_SPACE}"
    --observation-type "${OSWORLD_OBSERVATION_TYPE}"
    --sleep-after-execution "${OSWORLD_SLEEP_AFTER_EXECUTION}"
    --progress-interval "${OSWORLD_PROGRESS_INTERVAL}"
    --provider-name "${OSWORLD_PROVIDER_NAME}"
    --region "${OSWORLD_REGION}"
    --screen-width "${OSWORLD_SCREEN_WIDTH}"
    --screen-height "${OSWORLD_SCREEN_HEIGHT}"
    --client-password "${OSWORLD_CLIENT_PASSWORD}"
    --temperature "${OSWORLD_TEMPERATURE}"
    --top-p "${OSWORLD_TOP_P}"
    --max-output-tokens "${OSWORLD_MAX_OUTPUT_TOKENS}"
    --agent-kind "${OSWORLD_AGENT_KIND}"
    --qwen3vl-model "${QWEN3VL_MODEL}"
    --qwen3vl-base-url "${QWEN3VL_BASE_URL}"
    --qwen3vl-api-key "${QWEN3VL_API_KEY}"
    --qwen3vl-max-tokens "${QWEN3VL_MAX_TOKENS}"
    --qwen3vl-coordinate-type "${QWEN3VL_COORDINATE_TYPE}"
    --qwen3vl-history-n "${QWEN3VL_HISTORY_N}"
    --qwen3vl-request-timeout "${QWEN3VL_REQUEST_TIMEOUT}"
    --qwen3vl-max-retries "${QWEN3VL_MAX_RETRIES}"
    --qwen3vl-thinking-budget "${QWEN3VL_THINKING_BUDGET}"
    --qwen3vl-tool-choice "${QWEN3VL_TOOL_CHOICE}"
    --output-dir "${RESULTS_DIR}"
  )
  [[ -n "${OSWORLD_TASK_IDS}" ]] && args+=(--task-ids "${OSWORLD_TASK_IDS}")
  [[ -n "${OSWORLD_PATH_TO_VM}" ]] && args+=(--path-to-vm "${OSWORLD_PATH_TO_VM}")
  [[ -n "${OSWORLD_EXTRA_BODY}" ]] && args+=(--extra-body "${OSWORLD_EXTRA_BODY}")
  [[ "${OSWORLD_HEADLESS}" == "1" ]] && args+=(--headless) || args+=(--no-headless)
  [[ "${OSWORLD_PROGRESS}" == "1" ]] && args+=(--progress) || args+=(--no-progress)
  [[ "${QWEN3VL_ENABLE_THINKING}" == "1" ]] && args+=(--qwen3vl-enable-thinking) || args+=(--qwen3vl-disable-thinking)
  [[ "${QWEN3VL_USE_VLLM_TOOL_CALLS}" == "1" ]] && args+=(--qwen3vl-use-vllm-tool-calls) || args+=(--no-qwen3vl-use-vllm-tool-calls)

  log "Run OSWorld-Verified: domain=${OSWORLD_DOMAIN}, tasks=${OSWORLD_TASK_IDS:-slice ${OSWORLD_TASK_START}+${OSWORLD_NUM_TASKS}}, max_steps=${OSWORLD_MAX_STEPS}, concurrency=${OSWORLD_MAX_CONCURRENCY}"
  PYTHONPATH="${REPO_ROOT}:${OSWORLD_SOURCE_ROOT}:${PYTHONPATH:-}" VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" HF_ENDPOINT="${HF_ENDPOINT}" OSWORLD_REWRITE_HF_URLS="${OSWORLD_REWRITE_HF_URLS}" "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${OSWORLD_LOG}"
}

main() {
  [[ -f "${SCAFFOLD_DIR}/run_osworld_verified.py" ]] || die "OSWorld scaffold runner not found: ${SCAFFOLD_DIR}/run_osworld_verified.py"
  [[ -d "${OSWORLD_SOURCE_ROOT}" ]] || die "OSWORLD_SOURCE_ROOT not found: ${OSWORLD_SOURCE_ROOT}. Clone https://github.com/xlang-ai/OSWorld and set OSWORLD_SOURCE_ROOT."
  [[ -f "${OSWORLD_TEST_ALL_META_PATH}" ]] || die "OSWorld test meta not found: ${OSWORLD_TEST_ALL_META_PATH}"
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find Python: ${PYTHON_BIN}. Run: OSWORLD_SOURCE_ROOT=${OSWORLD_SOURCE_ROOT} bash examples/scripts/setup_benchmark_env.sh osworld"

  write_manifest
  log "Run dir: ${RUN_DIR}"
  check_vllm
  start_thunderagent
  start_sampler
  local osworld_status=0
  run_osworld || osworld_status=$?
  local results_json="${RESULTS_DIR}/summary/results.json"
  log "Done. Results: ${results_json}"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
  return "${osworld_status}"
}

main "$@"
