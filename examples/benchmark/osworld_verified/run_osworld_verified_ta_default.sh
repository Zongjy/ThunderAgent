#!/usr/bin/env bash
set -euo pipefail

# OSWorld-Verified + OpenCUA/Agent-S scaffold + ThunderAgent + vLLM.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCAFFOLD_DIR="${REPO_ROOT}/examples/scaffold/osworld"
OSWORLD_ENV_DIR="${OSWORLD_ENV_DIR:-${REPO_ROOT}/.venv-osworld}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${OSWORLD_ENV_DIR}}"
VLLM_ROOT="${VLLM_ROOT:-${HOME}/vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${VLLM_ROOT}/.venv}"
VLLM_WORKDIR="${VLLM_WORKDIR:-${VLLM_ROOT}}"

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

OSWORLD_AGENT_KIND="${OSWORLD_AGENT_KIND:-opencua}"
if [[ -z "${MODEL:-}" ]]; then
  if [[ "${OSWORLD_AGENT_KIND}" == "opencua" ]]; then
    MODEL="xlangai/OpenCUA-7B"
  else
    MODEL="ByteDance-Seed/UI-TARS-1.5-7B"
  fi
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
if [[ "${OSWORLD_AGENT_KIND}" == "opencua" && "${SERVED_MODEL_NAME}" == "${MODEL}" ]]; then
  SERVED_MODEL_NAME="opencua-7b"
fi
AGENT_LLM="${AGENT_LLM:-${SERVED_MODEL_NAME}}"
VLLM_PORT="${VLLM_PORT:-4747}"
TA_PORT="${TA_PORT:-9000}"
ROUTER_MODE="${ROUTER_MODE:-default}"
THUNDERAGENT_BACKEND_TYPE="${THUNDERAGENT_BACKEND_TYPE:-vllm}"
THUNDERAGENT_BACKENDS="${THUNDERAGENT_BACKENDS:-http://127.0.0.1:${VLLM_PORT}}"
THUNDERAGENT_KV_CAPACITY_TOKENS="${THUNDERAGENT_KV_CAPACITY_TOKENS:-}"
THUNDERAGENT_SCHEDULER_INTERVAL="${THUNDERAGENT_SCHEDULER_INTERVAL:-5}"
THUNDERAGENT_ACTING_TOKEN_WEIGHT="${THUNDERAGENT_ACTING_TOKEN_WEIGHT:-1.0}"
THUNDERAGENT_USE_ACTING_TOKEN_DECAY="${THUNDERAGENT_USE_ACTING_TOKEN_DECAY:-0}"

OSWORLD_DOMAIN="${OSWORLD_DOMAIN:-all}"
OSWORLD_TASK_IDS="${OSWORLD_TASK_IDS:-}"
OSWORLD_TASK_START="${OSWORLD_TASK_START:-0}"
OSWORLD_NUM_TASKS="${OSWORLD_NUM_TASKS:-1}" # 0 means all selected tasks
OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-100}"
OSWORLD_MAX_CONCURRENCY="${OSWORLD_MAX_CONCURRENCY:-1}"
OSWORLD_ACTION_SPACE="${OSWORLD_ACTION_SPACE:-pyautogui}"
OSWORLD_OBSERVATION_TYPE="${OSWORLD_OBSERVATION_TYPE:-screenshot}"
OSWORLD_SLEEP_AFTER_EXECUTION="${OSWORLD_SLEEP_AFTER_EXECUTION:-3.0}"
OSWORLD_RESET_SLEEP="${OSWORLD_RESET_SLEEP:-60.0}"
OSWORLD_SETTLE_SLEEP="${OSWORLD_SETTLE_SLEEP:-20.0}"
OSWORLD_HEADLESS="${OSWORLD_HEADLESS:-1}"
OSWORLD_RECORDING="${OSWORLD_RECORDING:-1}"
OSWORLD_ENABLE_PROXY="${OSWORLD_ENABLE_PROXY:-1}"
OSWORLD_PROVIDER_NAME="${OSWORLD_PROVIDER_NAME:-aws}"
OSWORLD_REGION="${OSWORLD_REGION:-us-east-1}"
OSWORLD_PATH_TO_VM="${OSWORLD_PATH_TO_VM:-}"
OSWORLD_SNAPSHOT_NAME="${OSWORLD_SNAPSHOT_NAME:-}"
OSWORLD_OS_TYPE="${OSWORLD_OS_TYPE:-Ubuntu}"
OSWORLD_CLIENT_PASSWORD="${OSWORLD_CLIENT_PASSWORD:-}"
OSWORLD_PASSWORD="${OSWORLD_PASSWORD:-osworld-public-evaluation}"
OSWORLD_SCREEN_WIDTH="${OSWORLD_SCREEN_WIDTH:-1920}"
OSWORLD_SCREEN_HEIGHT="${OSWORLD_SCREEN_HEIGHT:-1080}"
OSWORLD_TEST_CONFIG_BASE_DIR="${OSWORLD_TEST_CONFIG_BASE_DIR:-${OSWORLD_SOURCE_ROOT}/evaluation_examples}"
OSWORLD_TEST_ALL_META_PATH="${OSWORLD_TEST_ALL_META_PATH:-${OSWORLD_SOURCE_ROOT}/evaluation_examples/test_nogdrive.json}"

OSWORLD_TEMPERATURE="${OSWORLD_TEMPERATURE:-0}"
OSWORLD_TOP_P="${OSWORLD_TOP_P:-0.9}"
if [[ -z "${OSWORLD_MAX_OUTPUT_TOKENS:-}" ]]; then
  if [[ "${OSWORLD_AGENT_KIND}" == "opencua" ]]; then
    OSWORLD_MAX_OUTPUT_TOKENS="2048"
  else
    OSWORLD_MAX_OUTPUT_TOKENS="1000"
  fi
fi
OSWORLD_EXTRA_BODY="${OSWORLD_EXTRA_BODY:-}"

OPENCUA_MODEL="${OPENCUA_MODEL:-${AGENT_LLM}}"
OPENCUA_BASE_URL="${OPENCUA_BASE_URL:-http://127.0.0.1:${TA_PORT}/v1}"
OPENCUA_API_KEY="${OPENCUA_API_KEY:-EMPTY}"
OPENCUA_EXTRA_BODY="${OPENCUA_EXTRA_BODY:-}"
OPENCUA_HISTORY_TYPE="${OPENCUA_HISTORY_TYPE:-action_history}"
OPENCUA_COORDINATE_TYPE="${OPENCUA_COORDINATE_TYPE:-qwen25}"
OPENCUA_COT_LEVEL="${OPENCUA_COT_LEVEL:-l2}"
OPENCUA_MAX_IMAGE_HISTORY_LENGTH="${OPENCUA_MAX_IMAGE_HISTORY_LENGTH:-3}"
OPENCUA_USE_OLD_SYS_PROMPT="${OPENCUA_USE_OLD_SYS_PROMPT:-1}"
OPENCUA_REQUEST_TIMEOUT="${OPENCUA_REQUEST_TIMEOUT:-500}"
OPENCUA_MAX_RETRIES="${OPENCUA_MAX_RETRIES:-20}"

AGENT_S_VERSION="${AGENT_S_VERSION:-0.3.2}"
AGENT_S_ENGINE_TYPE="${AGENT_S_ENGINE_TYPE:-openai}"
AGENT_S_API_KEY="${AGENT_S_API_KEY:-EMPTY}"
AGENT_S_BASE_URL="${AGENT_S_BASE_URL:-http://127.0.0.1:${TA_PORT}/v1}"
AGENT_S_MAIN_ENGINE_TYPE="${AGENT_S_MAIN_ENGINE_TYPE:-${AGENT_S_ENGINE_TYPE}}"
AGENT_S_MAIN_MODEL="${AGENT_S_MAIN_MODEL:-${AGENT_LLM}}"
AGENT_S_MAIN_BASE_URL="${AGENT_S_MAIN_BASE_URL:-${AGENT_S_BASE_URL}}"
AGENT_S_MAIN_API_KEY="${AGENT_S_MAIN_API_KEY:-${AGENT_S_API_KEY}}"
AGENT_S_MAIN_EXTRA_BODY="${AGENT_S_MAIN_EXTRA_BODY:-}"
AGENT_S_GROUNDING_ENGINE_TYPE="${AGENT_S_GROUNDING_ENGINE_TYPE:-${AGENT_S_ENGINE_TYPE}}"
AGENT_S_GROUNDING_MODEL="${AGENT_S_GROUNDING_MODEL:-${AGENT_LLM}}"
AGENT_S_GROUNDING_BASE_URL="${AGENT_S_GROUNDING_BASE_URL:-${AGENT_S_BASE_URL}}"
AGENT_S_GROUNDING_API_KEY="${AGENT_S_GROUNDING_API_KEY:-${AGENT_S_API_KEY}}"
AGENT_S_GROUNDING_EXTRA_BODY="${AGENT_S_GROUNDING_EXTRA_BODY:-}"
AGENT_S_GROUNDING_WIDTH="${AGENT_S_GROUNDING_WIDTH:-${OSWORLD_SCREEN_WIDTH}}"
AGENT_S_GROUNDING_HEIGHT="${AGENT_S_GROUNDING_HEIGHT:-${OSWORLD_SCREEN_HEIGHT}}"
AGENT_S_CODE_ENGINE_TYPE="${AGENT_S_CODE_ENGINE_TYPE:-${AGENT_S_MAIN_ENGINE_TYPE}}"
AGENT_S_CODE_MODEL="${AGENT_S_CODE_MODEL:-${AGENT_S_MAIN_MODEL}}"
AGENT_S_CODE_BASE_URL="${AGENT_S_CODE_BASE_URL:-${AGENT_S_MAIN_BASE_URL}}"
AGENT_S_CODE_API_KEY="${AGENT_S_CODE_API_KEY:-${AGENT_S_MAIN_API_KEY}}"
AGENT_S_CODE_EXTRA_BODY="${AGENT_S_CODE_EXTRA_BODY:-}"
AGENT_S_ENABLE_CODE_AGENT="${AGENT_S_ENABLE_CODE_AGENT:-0}"
AGENT_S_CODE_AGENT_BUDGET="${AGENT_S_CODE_AGENT_BUDGET:-20}"
AGENT_S_MAX_TRAJECTORY_LENGTH="${AGENT_S_MAX_TRAJECTORY_LENGTH:-8}"
AGENT_S_ENABLE_REFLECTION="${AGENT_S_ENABLE_REFLECTION:-1}"
AGENT_S_STRICT_VERSION="${AGENT_S_STRICT_VERSION:-1}"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_VLLM_USAGE_FLAGS="${ENABLE_VLLM_USAGE_FLAGS:-1}"
ENABLE_VLLM_TOOL_CALLING="${ENABLE_VLLM_TOOL_CALLING:-0}"
ENABLE_VLLM_KV_METRICS="${ENABLE_VLLM_KV_METRICS:-1}"
ENABLE_VLLM_LANGUAGE_MODEL_ONLY="${ENABLE_VLLM_LANGUAGE_MODEL_ONLY:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code}"

RUN_NAME="${RUN_NAME:-osworld_verified_${OSWORLD_AGENT_KIND}_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/osworld_verified/runs}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
MANIFEST="${RUN_DIR}/manifest.env"
RESULTS_DIR="${RUN_DIR}/osworld_outputs"
SUMMARY_DIR="${RUN_DIR}/_analysis"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
OSWORLD_LOG="${LOG_DIR}/osworld_verified.log"

VLLM_PID=""
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
  [[ "${KEEP_SERVICES}" == "1" ]] && { log "KEEP_SERVICES=1; services left running."; return; }
  set +e
  [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null
  [[ -n "${TA_PID}" ]] && kill "${TA_PID}" 2>/dev/null
  [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" 2>/dev/null
}
trap cleanup EXIT

write_manifest() {
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${RESULTS_DIR}" "${SUMMARY_DIR}"
  cat > "${MANIFEST}" <<EOF
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
AGENT_LLM=${AGENT_LLM}
VLLM_PORT=${VLLM_PORT}
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
OSWORLD_RESET_SLEEP=${OSWORLD_RESET_SLEEP}
OSWORLD_SETTLE_SLEEP=${OSWORLD_SETTLE_SLEEP}
OSWORLD_HEADLESS=${OSWORLD_HEADLESS}
OSWORLD_RECORDING=${OSWORLD_RECORDING}
OSWORLD_ENABLE_PROXY=${OSWORLD_ENABLE_PROXY}
OSWORLD_PROVIDER_NAME=${OSWORLD_PROVIDER_NAME}
OSWORLD_REGION=${OSWORLD_REGION}
OSWORLD_PATH_TO_VM=${OSWORLD_PATH_TO_VM}
OSWORLD_SNAPSHOT_NAME=${OSWORLD_SNAPSHOT_NAME}
OSWORLD_OS_TYPE=${OSWORLD_OS_TYPE}
OSWORLD_CLIENT_PASSWORD=${OSWORLD_CLIENT_PASSWORD}
OSWORLD_PASSWORD=${OSWORLD_PASSWORD}
OSWORLD_SCREEN_WIDTH=${OSWORLD_SCREEN_WIDTH}
OSWORLD_SCREEN_HEIGHT=${OSWORLD_SCREEN_HEIGHT}
OSWORLD_TEST_CONFIG_BASE_DIR=${OSWORLD_TEST_CONFIG_BASE_DIR}
OSWORLD_TEST_ALL_META_PATH=${OSWORLD_TEST_ALL_META_PATH}
OSWORLD_TEMPERATURE=${OSWORLD_TEMPERATURE}
OSWORLD_TOP_P=${OSWORLD_TOP_P}
OSWORLD_MAX_OUTPUT_TOKENS=${OSWORLD_MAX_OUTPUT_TOKENS}
OSWORLD_EXTRA_BODY=${OSWORLD_EXTRA_BODY}
AGENT_S_VERSION=${AGENT_S_VERSION}
AGENT_S_ENGINE_TYPE=${AGENT_S_ENGINE_TYPE}
AGENT_S_API_KEY=${AGENT_S_API_KEY}
AGENT_S_BASE_URL=${AGENT_S_BASE_URL}
AGENT_S_MAIN_ENGINE_TYPE=${AGENT_S_MAIN_ENGINE_TYPE}
AGENT_S_MAIN_MODEL=${AGENT_S_MAIN_MODEL}
AGENT_S_MAIN_BASE_URL=${AGENT_S_MAIN_BASE_URL}
AGENT_S_MAIN_API_KEY=${AGENT_S_MAIN_API_KEY}
AGENT_S_MAIN_EXTRA_BODY=${AGENT_S_MAIN_EXTRA_BODY}
AGENT_S_GROUNDING_ENGINE_TYPE=${AGENT_S_GROUNDING_ENGINE_TYPE}
AGENT_S_GROUNDING_MODEL=${AGENT_S_GROUNDING_MODEL}
AGENT_S_GROUNDING_BASE_URL=${AGENT_S_GROUNDING_BASE_URL}
AGENT_S_GROUNDING_API_KEY=${AGENT_S_GROUNDING_API_KEY}
AGENT_S_GROUNDING_EXTRA_BODY=${AGENT_S_GROUNDING_EXTRA_BODY}
AGENT_S_GROUNDING_WIDTH=${AGENT_S_GROUNDING_WIDTH}
AGENT_S_GROUNDING_HEIGHT=${AGENT_S_GROUNDING_HEIGHT}
AGENT_S_CODE_ENGINE_TYPE=${AGENT_S_CODE_ENGINE_TYPE}
AGENT_S_CODE_MODEL=${AGENT_S_CODE_MODEL}
AGENT_S_CODE_BASE_URL=${AGENT_S_CODE_BASE_URL}
AGENT_S_CODE_API_KEY=${AGENT_S_CODE_API_KEY}
AGENT_S_CODE_EXTRA_BODY=${AGENT_S_CODE_EXTRA_BODY}
AGENT_S_ENABLE_CODE_AGENT=${AGENT_S_ENABLE_CODE_AGENT}
AGENT_S_CODE_AGENT_BUDGET=${AGENT_S_CODE_AGENT_BUDGET}
AGENT_S_MAX_TRAJECTORY_LENGTH=${AGENT_S_MAX_TRAJECTORY_LENGTH}
AGENT_S_ENABLE_REFLECTION=${AGENT_S_ENABLE_REFLECTION}
AGENT_S_STRICT_VERSION=${AGENT_S_STRICT_VERSION}
START_VLLM=${START_VLLM}
START_THUNDERAGENT=${START_THUNDERAGENT}
KEEP_SERVICES=${KEEP_SERVICES}
SAMPLE_METRICS=${SAMPLE_METRICS}
PYTHON_BIN=${PYTHON_BIN}
THUNDERAGENT_ENV_DIR=${THUNDERAGENT_ENV_DIR}
OSWORLD_ENV_DIR=${OSWORLD_ENV_DIR}
VLLM_ROOT=${VLLM_ROOT}
VLLM_ENV_DIR=${VLLM_ENV_DIR}
VLLM_WORKDIR=${VLLM_WORKDIR}
VLLM_BIN=${VLLM_BIN}
HF_ENDPOINT=${HF_ENDPOINT}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}
ENABLE_VLLM_USAGE_FLAGS=${ENABLE_VLLM_USAGE_FLAGS}
ENABLE_VLLM_TOOL_CALLING=${ENABLE_VLLM_TOOL_CALLING}
ENABLE_VLLM_KV_METRICS=${ENABLE_VLLM_KV_METRICS}
ENABLE_VLLM_LANGUAGE_MODEL_ONLY=${ENABLE_VLLM_LANGUAGE_MODEL_ONLY}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
REASONING_PARSER=${REASONING_PARSER}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}
RUN_NAME=${RUN_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_DIR=${RUN_DIR}
EOF
}

start_vllm() {
  [[ "${START_VLLM}" == "1" ]] || { log "Reuse vLLM: ${VLLM_PORT}"; wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"; return; }
  ensure_port_free "${VLLM_PORT}" "VLLM"

  local cmd=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
  )
  [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && cmd+=(--enable-prefix-caching)
  [[ -n "${REASONING_PARSER}" ]] && cmd+=(--reasoning-parser "${REASONING_PARSER}")
  [[ "${ENABLE_VLLM_LANGUAGE_MODEL_ONLY}" == "1" ]] && cmd+=(--language-model-only)
  [[ "${ENABLE_VLLM_USAGE_FLAGS}" == "1" ]] && cmd+=(--enable-prompt-tokens-details --enable-force-include-usage)
  [[ "${ENABLE_VLLM_TOOL_CALLING}" == "1" ]] && cmd+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
  [[ "${ENABLE_VLLM_KV_METRICS}" == "1" ]] && cmd+=(--enable-mfu-metrics --kv-cache-metrics)
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
    cmd+=("${extra_args[@]}")
  fi

  log "Start vLLM from ${VLLM_WORKDIR}: ${VLLM_LOG}"
  pushd "${VLLM_WORKDIR}" >/dev/null
  nohup env VIRTUAL_ENV="${VLLM_ENV_DIR}" PATH="${VLLM_ENV_DIR}/bin:${PATH}" HF_ENDPOINT="${HF_ENDPOINT}" "${cmd[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"
  popd >/dev/null
  wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"
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
      { printf '# sample_ts=%s\n' "$(date +%s)"; curl -fsS --max-time 5 "http://127.0.0.1:${VLLM_PORT}/metrics" 2>/dev/null || true; printf '\n'; } >> "${LOG_DIR}/vllm_metrics.prom"
      sleep "${METRICS_INTERVAL_S}"
    done
  ) &
  SAMPLER_PID="$!"
}

check_agent_s_version() {
  [[ "${AGENT_S_STRICT_VERSION}" == "1" ]] || return 0
  local installed
  installed="$("${PYTHON_BIN}" -c 'from importlib.metadata import version; print(version("gui-agents"))')" \
    || die "Cannot import pinned Agent-S package gui-agents==${AGENT_S_VERSION}. Run: OSWORLD_SOURCE_ROOT=${OSWORLD_SOURCE_ROOT} bash examples/scripts/setup_benchmark_env.sh osworld"
  [[ "${installed}" == "${AGENT_S_VERSION}" ]] \
    || die "Agent-S version mismatch: expected gui-agents==${AGENT_S_VERSION}, found ${installed}"
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
    --reset-sleep "${OSWORLD_RESET_SLEEP}"
    --settle-sleep "${OSWORLD_SETTLE_SLEEP}"
    --provider-name "${OSWORLD_PROVIDER_NAME}"
    --region "${OSWORLD_REGION}"
    --screen-width "${OSWORLD_SCREEN_WIDTH}"
    --screen-height "${OSWORLD_SCREEN_HEIGHT}"
    --os-type "${OSWORLD_OS_TYPE}"
    --client-password "${OSWORLD_CLIENT_PASSWORD}"
    --password "${OSWORLD_PASSWORD}"
    --temperature "${OSWORLD_TEMPERATURE}"
    --top-p "${OSWORLD_TOP_P}"
    --max-output-tokens "${OSWORLD_MAX_OUTPUT_TOKENS}"
    --agent-kind "agent-s"
    --agent-s-engine-type "${AGENT_S_ENGINE_TYPE}"
    --agent-s-main-engine-type "${AGENT_S_MAIN_ENGINE_TYPE}"
    --agent-s-main-model "${AGENT_S_MAIN_MODEL}"
    --agent-s-main-base-url "${AGENT_S_MAIN_BASE_URL}"
    --agent-s-main-api-key "${AGENT_S_MAIN_API_KEY}"
    --agent-s-grounding-engine-type "${AGENT_S_GROUNDING_ENGINE_TYPE}"
    --agent-s-grounding-model "${AGENT_S_GROUNDING_MODEL}"
    --agent-s-grounding-base-url "${AGENT_S_GROUNDING_BASE_URL}"
    --agent-s-grounding-api-key "${AGENT_S_GROUNDING_API_KEY}"
    --agent-s-grounding-width "${AGENT_S_GROUNDING_WIDTH}"
    --agent-s-grounding-height "${AGENT_S_GROUNDING_HEIGHT}"
    --agent-s-code-engine-type "${AGENT_S_CODE_ENGINE_TYPE}"
    --agent-s-code-model "${AGENT_S_CODE_MODEL}"
    --agent-s-code-base-url "${AGENT_S_CODE_BASE_URL}"
    --agent-s-code-api-key "${AGENT_S_CODE_API_KEY}"
    --agent-s-code-agent-budget "${AGENT_S_CODE_AGENT_BUDGET}"
    --agent-s-max-trajectory-length "${AGENT_S_MAX_TRAJECTORY_LENGTH}"
    --output-dir "${RESULTS_DIR}"
  )
  [[ -n "${OSWORLD_TASK_IDS}" ]] && args+=(--task-ids "${OSWORLD_TASK_IDS}")
  [[ -n "${OSWORLD_PATH_TO_VM}" ]] && args+=(--path-to-vm "${OSWORLD_PATH_TO_VM}")
  [[ -n "${OSWORLD_SNAPSHOT_NAME}" ]] && args+=(--snapshot-name "${OSWORLD_SNAPSHOT_NAME}")
  [[ -n "${OSWORLD_EXTRA_BODY}" ]] && args+=(--extra-body "${OSWORLD_EXTRA_BODY}")
  [[ -n "${AGENT_S_MAIN_EXTRA_BODY}" ]] && args+=(--agent-s-main-extra-body "${AGENT_S_MAIN_EXTRA_BODY}")
  [[ -n "${AGENT_S_GROUNDING_EXTRA_BODY}" ]] && args+=(--agent-s-grounding-extra-body "${AGENT_S_GROUNDING_EXTRA_BODY}")
  [[ -n "${AGENT_S_CODE_EXTRA_BODY}" ]] && args+=(--agent-s-code-extra-body "${AGENT_S_CODE_EXTRA_BODY}")
  [[ "${OSWORLD_HEADLESS}" == "1" ]] && args+=(--headless) || args+=(--no-headless)
  [[ "${OSWORLD_ENABLE_PROXY}" == "1" ]] && args+=(--enable-proxy) || args+=(--no-enable-proxy)
  [[ "${AGENT_S_ENABLE_CODE_AGENT}" == "1" ]] && args+=(--agent-s-enable-code-agent) || args+=(--no-agent-s-enable-code-agent)
  [[ "${AGENT_S_ENABLE_REFLECTION}" == "1" ]] && args+=(--agent-s-enable-reflection) || args+=(--no-agent-s-enable-reflection)
  [[ "${AGENT_S_STRICT_VERSION}" == "1" ]] && args+=(--agent-s-strict-version) || args+=(--no-agent-s-strict-version)
  [[ "${OSWORLD_RECORDING}" != "1" ]] && args+=(--no-recording)

  log "Run OSWorld-Verified: domain=${OSWORLD_DOMAIN}, tasks=${OSWORLD_TASK_IDS:-slice ${OSWORLD_TASK_START}+${OSWORLD_NUM_TASKS}}, max_steps=${OSWORLD_MAX_STEPS}, concurrency=${OSWORLD_MAX_CONCURRENCY}"
  PYTHONPATH="${REPO_ROOT}:${OSWORLD_SOURCE_ROOT}:${PYTHONPATH:-}" VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${OSWORLD_LOG}"
}

main() {
  [[ -f "${SCAFFOLD_DIR}/run_osworld_verified.py" ]] || die "OSWorld scaffold runner not found: ${SCAFFOLD_DIR}/run_osworld_verified.py"
  [[ -d "${OSWORLD_SOURCE_ROOT}" ]] || die "OSWORLD_SOURCE_ROOT not found: ${OSWORLD_SOURCE_ROOT}. Clone https://github.com/xlang-ai/OSWorld and set OSWORLD_SOURCE_ROOT."
  [[ -f "${OSWORLD_TEST_ALL_META_PATH}" ]] || die "OSWorld test meta not found: ${OSWORLD_TEST_ALL_META_PATH}"
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find Python: ${PYTHON_BIN}. Run: OSWORLD_SOURCE_ROOT=${OSWORLD_SOURCE_ROOT} bash examples/scripts/setup_benchmark_env.sh osworld"
  [[ "${START_VLLM}" != "1" ]] || [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"

  check_agent_s_version
  write_manifest
  log "Run dir: ${RUN_DIR}"
  start_vllm
  start_thunderagent
  start_sampler
  local osworld_status=0
  run_osworld || osworld_status=$?
  if [[ -f "${RESULTS_DIR}/results.json" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/benchmark/osworld_verified/analyze_osworld_outputs.py" --root "${RESULTS_DIR}" --output-dir "${SUMMARY_DIR}"
  else
    log "Skip analysis: ${RESULTS_DIR}/results.json was not written"
  fi
  log "Done. Results: ${RESULTS_DIR}/results.json"
  log "Summary: ${SUMMARY_DIR}/osworld_verified_analysis.json"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
  return "${osworld_status}"
}

main "$@"
