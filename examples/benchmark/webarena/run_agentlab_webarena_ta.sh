#!/usr/bin/env bash
set -euo pipefail

# AgentLab + WebArena + ThunderAgent + vLLM.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${REPO_ROOT}/examples/scaffold/agentlab_webarena}"
WEBARENA_ENV_DIR="${WEBARENA_ENV_DIR:-${REPO_ROOT}/.venv-webarena}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${WEBARENA_ENV_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/bin/python}"
NLTK_DATA="${NLTK_DATA:-${WEBARENA_ENV_DIR}/nltk_data}"
WEBARENA_ENV_FILE="${WEBARENA_ENV_FILE:-${REPO_ROOT}/examples/benchmark/webarena/webarena.env}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
AGENT_LLM="${AGENT_LLM:-${SERVED_MODEL_NAME}}"
VLLM_ROOT="${VLLM_ROOT:-${HOME}/vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${VLLM_ROOT}/.venv}"
VLLM_WORKDIR="${VLLM_WORKDIR:-${VLLM_ROOT}}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-4747}"
TA_PORT="${TA_PORT:-9000}"

ROUTER_MODE="${ROUTER_MODE:-default}"
THUNDERAGENT_BACKEND_TYPE="${THUNDERAGENT_BACKEND_TYPE:-vllm}"
THUNDERAGENT_BACKENDS="${THUNDERAGENT_BACKENDS:-http://127.0.0.1:${VLLM_PORT}}"
THUNDERAGENT_KV_CAPACITY_TOKENS="${THUNDERAGENT_KV_CAPACITY_TOKENS:-}"
THUNDERAGENT_SCHEDULER_INTERVAL="${THUNDERAGENT_SCHEDULER_INTERVAL:-5}"
THUNDERAGENT_ACTING_TOKEN_WEIGHT="${THUNDERAGENT_ACTING_TOKEN_WEIGHT:-1.0}"
THUNDERAGENT_USE_ACTING_TOKEN_DECAY="${THUNDERAGENT_USE_ACTING_TOKEN_DECAY:-0}"

WEBARENA_TASK_IDS="${WEBARENA_TASK_IDS:-}"
WEBARENA_TASK_START="${WEBARENA_TASK_START:-0}"
WEBARENA_NUM_TASKS="${WEBARENA_NUM_TASKS:-0}"
WEBARENA_SITE_FILTER="${WEBARENA_SITE_FILTER:-shopping}"
WEBARENA_BENCHMARK="${WEBARENA_BENCHMARK:-webarena}"
WEBARENA_MAX_STEPS="${WEBARENA_MAX_STEPS:-100}"
WEBARENA_MAX_CONCURRENCY="${WEBARENA_MAX_CONCURRENCY:-1}"
WEBARENA_SEED="${WEBARENA_SEED:-0}"
WEBARENA_HEADLESS="${WEBARENA_HEADLESS:-1}"
WEBARENA_RECORD_VIDEO="${WEBARENA_RECORD_VIDEO:-0}"
WEBARENA_SKIP_BACKEND_MASSAGE="${WEBARENA_SKIP_BACKEND_MASSAGE:-1}"

AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0}"
AGENT_MAX_TOTAL_TOKENS="${AGENT_MAX_TOTAL_TOKENS:-128000}"
AGENT_MAX_INPUT_TOKENS="${AGENT_MAX_INPUT_TOKENS:-128000}"
AGENT_MAX_OUTPUT_TOKENS="${AGENT_MAX_OUTPUT_TOKENS:-2048}"
AGENT_MAX_PROMPT_TOKENS="${AGENT_MAX_PROMPT_TOKENS:-40000}"
AGENT_LLM_MAX_RETRY="${AGENT_LLM_MAX_RETRY:-4}"
AGENT_LLM_RETRY_WAIT="${AGENT_LLM_RETRY_WAIT:-5}"
AGENT_ACTION_PARSE_RETRY="${AGENT_ACTION_PARSE_RETRY:-4}"
AGENTLAB_PARALLEL_BACKEND="${AGENTLAB_PARALLEL_BACKEND:-}"
AGENTLAB_N_RELAUNCH="${AGENTLAB_N_RELAUNCH:-1}"
AGENTLAB_IGNORE_DEPENDENCIES="${AGENTLAB_IGNORE_DEPENDENCIES:-0}"
AGENTLAB_STUDY_SUFFIX="${AGENTLAB_STUDY_SUFFIX:-thunderagent-shopping}"
AGENTLAB_PROGRESS_INTERVAL="${AGENTLAB_PROGRESS_INTERVAL:-30}"

WEBARENA_EVALUATOR_PROVIDER="${WEBARENA_EVALUATOR_PROVIDER:-none}"
WEBARENA_EVALUATOR_BASE_URL="${WEBARENA_EVALUATOR_BASE_URL:-}"
WEBARENA_EVALUATOR_MODEL="${WEBARENA_EVALUATOR_MODEL:-}"
WEBARENA_EVALUATOR_API_KEY="${WEBARENA_EVALUATOR_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_VLLM_USAGE_FLAGS="${ENABLE_VLLM_USAGE_FLAGS:-1}"
ENABLE_VLLM_TOOL_CALLING="${ENABLE_VLLM_TOOL_CALLING:-1}"
ENABLE_VLLM_KV_METRICS="${ENABLE_VLLM_KV_METRICS:-1}"
ENABLE_VLLM_LANGUAGE_MODEL_ONLY="${ENABLE_VLLM_LANGUAGE_MODEL_ONLY:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

RUN_NAME="${RUN_NAME:-agentlab_${WEBARENA_BENCHMARK}_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/webarena/runs}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
MANIFEST="${RUN_DIR}/manifest.env"
RESULTS_DIR="${RUN_DIR}/agentlab_outputs"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
WEBARENA_LOG="${LOG_DIR}/agentlab_webarena.log"

VLLM_PID=""
TA_PID=""
SAMPLER_PID=""

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

write_env_manifest() {
  local path="$1"
  shift
  : > "${path}"
  local name
  for name in "$@"; do
    printf '%s=%q\n' "${name}" "${!name}" >> "${path}"
  done
}

load_webarena_env_file() {
  if [[ -f "${WEBARENA_ENV_FILE}" ]]; then
    log "Load WebArena env file: ${WEBARENA_ENV_FILE}"
    set -a
    # shellcheck disable=SC1090
    source "${WEBARENA_ENV_FILE}"
    set +a
  fi
}

check_webarena_env() {
  if [[ -z "${WA_HOMEPAGE:-}" && -n "${WA_SHOPPING:-}" ]]; then
    export WA_HOMEPAGE="${WA_SHOPPING}"
  fi
  if [[ "${WEBARENA_SITE_FILTER}" == "shopping" ]]; then
    for key in WA_SHOPPING_ADMIN WA_REDDIT WA_GITLAB WA_WIKIPEDIA WA_MAP; do
      if [[ -z "${!key:-}" ]]; then
        export "${key}=todo"
      fi
    done
  fi
  [[ -n "${WA_SHOPPING:-}" ]] || die "WA_SHOPPING is not set. Edit ${WEBARENA_ENV_FILE}."
  [[ -n "${WA_HOMEPAGE:-}" ]] || die "WA_HOMEPAGE is not set. Edit ${WEBARENA_ENV_FILE}."
}

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
  if is_enabled "${KEEP_SERVICES}"; then
    log "KEEP_SERVICES=1; services left running."
    return
  fi
  set +e
  [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null
  [[ -n "${TA_PID}" ]] && kill "${TA_PID}" 2>/dev/null
  [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" 2>/dev/null
}
trap cleanup EXIT

prepare_run_dirs() {
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${RESULTS_DIR}"
}

write_manifest() {
  write_env_manifest "${MANIFEST}" \
    REPO_ROOT SCAFFOLD_DIR WEBARENA_ENV_DIR THUNDERAGENT_ENV_DIR PYTHON_BIN NLTK_DATA WEBARENA_ENV_FILE \
    MODEL SERVED_MODEL_NAME AGENT_LLM VLLM_ROOT VLLM_ENV_DIR VLLM_WORKDIR VLLM_BIN VLLM_PORT TA_PORT \
    ROUTER_MODE THUNDERAGENT_BACKEND_TYPE THUNDERAGENT_BACKENDS THUNDERAGENT_KV_CAPACITY_TOKENS \
    THUNDERAGENT_SCHEDULER_INTERVAL THUNDERAGENT_ACTING_TOKEN_WEIGHT THUNDERAGENT_USE_ACTING_TOKEN_DECAY \
    WEBARENA_TASK_IDS WEBARENA_TASK_START WEBARENA_NUM_TASKS WEBARENA_SITE_FILTER WEBARENA_BENCHMARK \
    WEBARENA_MAX_STEPS WEBARENA_MAX_CONCURRENCY WEBARENA_SEED WEBARENA_HEADLESS WEBARENA_RECORD_VIDEO \
    WEBARENA_SKIP_BACKEND_MASSAGE \
    AGENT_TEMPERATURE AGENT_MAX_TOTAL_TOKENS AGENT_MAX_INPUT_TOKENS AGENT_MAX_OUTPUT_TOKENS \
    AGENT_MAX_PROMPT_TOKENS AGENT_LLM_MAX_RETRY AGENT_LLM_RETRY_WAIT AGENT_ACTION_PARSE_RETRY \
    AGENTLAB_PARALLEL_BACKEND AGENTLAB_N_RELAUNCH AGENTLAB_IGNORE_DEPENDENCIES AGENTLAB_STUDY_SUFFIX \
    AGENTLAB_PROGRESS_INTERVAL \
    WEBARENA_EVALUATOR_PROVIDER WEBARENA_EVALUATOR_BASE_URL WEBARENA_EVALUATOR_MODEL WEBARENA_EVALUATOR_API_KEY \
    START_VLLM START_THUNDERAGENT KEEP_SERVICES SAMPLE_METRICS METRICS_INTERVAL_S HEALTH_TIMEOUT_S \
    HF_ENDPOINT GPU_MEMORY_UTILIZATION MAX_MODEL_LEN ENABLE_PREFIX_CACHING ENABLE_VLLM_USAGE_FLAGS \
    ENABLE_VLLM_TOOL_CALLING ENABLE_VLLM_KV_METRICS ENABLE_VLLM_LANGUAGE_MODEL_ONLY TOOL_CALL_PARSER \
    REASONING_PARSER VLLM_EXTRA_ARGS RUN_NAME OUTPUT_ROOT RUN_DIR LOG_DIR PROFILE_DIR RESULTS_DIR
}

start_vllm() {
  if ! is_enabled "${START_VLLM}"; then
    log "Reuse vLLM: ${VLLM_PORT}"
    wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"
    return
  fi
  ensure_port_free "${VLLM_PORT}" "VLLM"

  local cmd=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
  )
  is_enabled "${ENABLE_PREFIX_CACHING}" && cmd+=(--enable-prefix-caching)
  [[ -n "${REASONING_PARSER}" ]] && cmd+=(--reasoning-parser "${REASONING_PARSER}")
  is_enabled "${ENABLE_VLLM_LANGUAGE_MODEL_ONLY}" && cmd+=(--language-model-only)
  is_enabled "${ENABLE_VLLM_USAGE_FLAGS}" && cmd+=(--enable-prompt-tokens-details --enable-force-include-usage)
  is_enabled "${ENABLE_VLLM_TOOL_CALLING}" && cmd+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
  is_enabled "${ENABLE_VLLM_KV_METRICS}" && cmd+=(--enable-mfu-metrics --kv-cache-metrics)
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    local extra_args=()
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
  if ! is_enabled "${START_THUNDERAGENT}"; then
    log "Reuse ThunderAgent: ${TA_PORT}"
    wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"
    return
  fi
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
  is_enabled "${THUNDERAGENT_USE_ACTING_TOKEN_DECAY}" && cmd+=(--use-acting-token-decay)

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
  is_enabled "${SAMPLE_METRICS}" || return 0
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

run_webarena() {
  local args=(
    "${SCAFFOLD_DIR}/run_agentlab_webarena.py"
    --env-file "${WEBARENA_ENV_FILE}"
    --base-url "http://127.0.0.1:${TA_PORT}/v1"
    --api-key "EMPTY"
    --model "${AGENT_LLM}"
    --benchmark "${WEBARENA_BENCHMARK}"
    --task-ids "${WEBARENA_TASK_IDS}"
    --site-filter "${WEBARENA_SITE_FILTER}"
    --task-start "${WEBARENA_TASK_START}"
    --num-tasks "${WEBARENA_NUM_TASKS}"
    --seed "${WEBARENA_SEED}"
    --max-steps "${WEBARENA_MAX_STEPS}"
    --n-jobs "${WEBARENA_MAX_CONCURRENCY}"
    --output-dir "${RESULTS_DIR}"
    --temperature "${AGENT_TEMPERATURE}"
    --max-total-tokens "${AGENT_MAX_TOTAL_TOKENS}"
    --max-input-tokens "${AGENT_MAX_INPUT_TOKENS}"
    --max-output-tokens "${AGENT_MAX_OUTPUT_TOKENS}"
    --max-prompt-tokens "${AGENT_MAX_PROMPT_TOKENS}"
    --llm-max-retry "${AGENT_LLM_MAX_RETRY}"
    --llm-retry-wait "${AGENT_LLM_RETRY_WAIT}"
    --action-parse-retry "${AGENT_ACTION_PARSE_RETRY}"
    --n-relaunch "${AGENTLAB_N_RELAUNCH}"
    --study-suffix "${AGENTLAB_STUDY_SUFFIX}"
    --progress-interval "${AGENTLAB_PROGRESS_INTERVAL}"
    --evaluator-provider "${WEBARENA_EVALUATOR_PROVIDER}"
    --evaluator-api-key "${WEBARENA_EVALUATOR_API_KEY}"
  )
  is_enabled "${WEBARENA_HEADLESS}" && args+=(--headless) || args+=(--no-headless)
  is_enabled "${WEBARENA_RECORD_VIDEO}" && args+=(--record-video)
  is_enabled "${WEBARENA_SKIP_BACKEND_MASSAGE}" && args+=(--skip-backend-massage) || args+=(--no-skip-backend-massage)
  is_enabled "${AGENTLAB_IGNORE_DEPENDENCIES}" && args+=(--ignore-dependencies)
  [[ -n "${AGENTLAB_PARALLEL_BACKEND}" ]] && args+=(--parallel-backend "${AGENTLAB_PARALLEL_BACKEND}")
  [[ -n "${WEBARENA_EVALUATOR_BASE_URL}" ]] && args+=(--evaluator-base-url "${WEBARENA_EVALUATOR_BASE_URL}")
  [[ -n "${WEBARENA_EVALUATOR_MODEL}" ]] && args+=(--evaluator-model "${WEBARENA_EVALUATOR_MODEL}")

  local task_summary
  if [[ -n "${WEBARENA_TASK_IDS}" ]]; then
    task_summary="ids=${WEBARENA_TASK_IDS}"
  elif [[ "${WEBARENA_NUM_TASKS}" == "0" ]]; then
    task_summary="site=${WEBARENA_SITE_FILTER},start=${WEBARENA_TASK_START},num=all"
  else
    task_summary="site=${WEBARENA_SITE_FILTER},start=${WEBARENA_TASK_START},num=${WEBARENA_NUM_TASKS}"
  fi

  log "Run AgentLab ${WEBARENA_BENCHMARK}: tasks=${task_summary}, max_steps=${WEBARENA_MAX_STEPS}, jobs=${WEBARENA_MAX_CONCURRENCY}, progress_interval=${AGENTLAB_PROGRESS_INTERVAL}s"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" \
    PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" \
    NLTK_DATA="${NLTK_DATA}" \
    AGENTLAB_EXP_ROOT="${RESULTS_DIR}" \
    "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${WEBARENA_LOG}"
}

main() {
  [[ -f "${SCAFFOLD_DIR}/run_agentlab_webarena.py" ]] || die "AgentLab WebArena runner not found: ${SCAFFOLD_DIR}/run_agentlab_webarena.py"
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find Python: ${PYTHON_BIN}. Run: bash examples/scripts/setup_benchmark_env.sh webarena"
  [[ "${START_VLLM}" != "1" ]] || [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"
  [[ "${START_VLLM}" != "1" ]] || [[ -x "${VLLM_BIN}" ]] || command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "Cannot find vLLM: ${VLLM_BIN}"

  load_webarena_env_file
  check_webarena_env
  prepare_run_dirs
  write_manifest
  log "Run dir: ${RUN_DIR}"
  start_vllm
  start_thunderagent
  start_sampler
  local webarena_status=0
  run_webarena || webarena_status=$?
  log "Done. AgentLab output root: ${RESULTS_DIR}"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
  return "${webarena_status}"
}

main "$@"
