#!/usr/bin/env bash
set -euo pipefail

# SWE-bench Verified + OpenHands + ThunderAgent(default router) + vLLM.
# Run from repo root:
#   bash examples/benchmark/swebench/run_openhands_ta_default.sh

# ---- Configurable parameters -------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENHANDS_DIR="${REPO_ROOT}/examples/scaffold/openhands"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${REPO_ROOT}/.venv}"
VLLM_ROOT="${VLLM_ROOT:-${HOME}/vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${VLLM_ROOT}/.venv}"
VLLM_WORKDIR="${VLLM_WORKDIR:-${VLLM_ROOT}}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
VLLM_PORT="${VLLM_PORT:-4747}"
TA_PORT="${TA_PORT:-9000}"
ROUTER_MODE="${ROUTER_MODE:-default}"

SWEBENCH_DATASET="${SWEBENCH_DATASET:-princeton-nlp/SWE-bench_Verified}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
SWEBENCH_LIMIT="${SWEBENCH_LIMIT:-100}"          # 0 means all remaining tasks
SWEBENCH_WORKERS="${SWEBENCH_WORKERS:-4}"
SWEBENCH_MAX_ITERATIONS="${SWEBENCH_MAX_ITERATIONS:-100}"
SWEBENCH_MODE="${SWEBENCH_MODE:-swe}"

OPENHANDS_AGENT_CLS="${OPENHANDS_AGENT_CLS:-CodeActAgent}"
OPENHANDS_AGENT_CONFIG="${OPENHANDS_AGENT_CONFIG:-swe_lego_no_plan}"
OPENHANDS_ENABLE_PLAN_MODE="${OPENHANDS_ENABLE_PLAN_MODE:-false}"
OPENHANDS_NATIVE_TOOL_CALLING="${OPENHANDS_NATIVE_TOOL_CALLING:-true}"
OPENHANDS_LLM_TIMEOUT="${OPENHANDS_LLM_TIMEOUT:-240}"
OPENHANDS_EVAL_NOTE="${OPENHANDS_EVAL_NOTE:-no-hint-no-plan-no-icl}"
USE_HINT_TEXT="${USE_HINT_TEXT:-false}"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_VLLM_USAGE_FLAGS="${ENABLE_VLLM_USAGE_FLAGS:-1}"
ENABLE_VLLM_TOOL_CALLING="${ENABLE_VLLM_TOOL_CALLING:-1}"
ENABLE_VLLM_KV_METRICS="${ENABLE_VLLM_KV_METRICS:-1}"
ENABLE_VLLM_LANGUAGE_MODEL_ONLY="${ENABLE_VLLM_LANGUAGE_MODEL_ONLY:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
EVAL_DOCKER_IMAGE_SOURCE="${EVAL_DOCKER_IMAGE_SOURCE:-epoch}"
EPOCH_DOCKER_IMAGE_PREFIX="${EPOCH_DOCKER_IMAGE_PREFIX:-ghcr.io/epoch-research}"

RUN_NAME="${RUN_NAME:-openhands_swebench_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/swebench/runs}"

# ---- Derived paths -----------------------------------------------------------
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
RUN_CONFIG="${RUN_DIR}/openhands_config.toml"
MANIFEST="${RUN_DIR}/manifest.env"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
SWEBENCH_LOG="${LOG_DIR}/openhands_swebench.log"

VLLM_PID=""
TA_PID=""
SAMPLER_PID=""

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

toml_bool() {
  case "${1,,}" in
    1|true|yes|on) echo "true" ;;
    0|false|no|off) echo "false" ;;
    *) die "Invalid boolean value: $1" ;;
  esac
}

wait_url() {
  local url="$1" name="$2" start now
  start="$(date +%s)"
  until curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
    now="$(date +%s)"
    (( now - start < HEALTH_TIMEOUT_S )) || die "${name} health timeout: ${url}"
    sleep 3
  done
  log "${name} ready: ${url}"
}

ensure_port_free() {
  local port="$1" name="$2"
  if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${port} )" | grep -q ":${port}"; then
    die "${name} port ${port} is in use; set START_${name}=0 or use another port"
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

write_run_files() {
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
  local native_tool_calling enable_plan_mode
  native_tool_calling="$(toml_bool "${OPENHANDS_NATIVE_TOOL_CALLING}")"
  enable_plan_mode="$(toml_bool "${OPENHANDS_ENABLE_PLAN_MODE}")"

  cat > "${RUN_CONFIG}" <<EOF
[agent.${OPENHANDS_AGENT_CONFIG}]
enable_plan_mode = ${enable_plan_mode}
enable_prompt_extensions = false

[llm.vllm_local]
model = "${SERVED_MODEL_NAME}"
custom_llm_provider = "openai"
base_url = "http://127.0.0.1:${TA_PORT}/v1"
api_key = "EMPTY"
temperature = 0.6
top_p = 0.95
top_k = 20
max_output_tokens = 32768
native_tool_calling = ${native_tool_calling}
timeout = ${OPENHANDS_LLM_TIMEOUT}
num_retries = 3

[llm.vllm_local.completion_kwargs]
min_p = 0.0
presence_penalty = 1.0
EOF

  cat > "${MANIFEST}" <<EOF
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
VLLM_PORT=${VLLM_PORT}
TA_PORT=${TA_PORT}
ROUTER_MODE=${ROUTER_MODE}
SWEBENCH_DATASET=${SWEBENCH_DATASET}
SWEBENCH_SPLIT=${SWEBENCH_SPLIT}
SWEBENCH_LIMIT=${SWEBENCH_LIMIT}
SWEBENCH_WORKERS=${SWEBENCH_WORKERS}
SWEBENCH_MAX_ITERATIONS=${SWEBENCH_MAX_ITERATIONS}
SWEBENCH_MODE=${SWEBENCH_MODE}
OPENHANDS_AGENT_CLS=${OPENHANDS_AGENT_CLS}
OPENHANDS_AGENT_CONFIG=${OPENHANDS_AGENT_CONFIG}
OPENHANDS_ENABLE_PLAN_MODE=${OPENHANDS_ENABLE_PLAN_MODE}
OPENHANDS_NATIVE_TOOL_CALLING=${OPENHANDS_NATIVE_TOOL_CALLING}
OPENHANDS_LLM_TIMEOUT=${OPENHANDS_LLM_TIMEOUT}
OPENHANDS_EVAL_NOTE=${OPENHANDS_EVAL_NOTE}
USE_HINT_TEXT=${USE_HINT_TEXT}
START_VLLM=${START_VLLM}
START_THUNDERAGENT=${START_THUNDERAGENT}
KEEP_SERVICES=${KEEP_SERVICES}
SAMPLE_METRICS=${SAMPLE_METRICS}
METRICS_INTERVAL_S=${METRICS_INTERVAL_S}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S}
PYTHON_BIN=${PYTHON_BIN}
THUNDERAGENT_ENV_DIR=${THUNDERAGENT_ENV_DIR}
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
EVAL_DOCKER_IMAGE_SOURCE=${EVAL_DOCKER_IMAGE_SOURCE}
EPOCH_DOCKER_IMAGE_PREFIX=${EPOCH_DOCKER_IMAGE_PREFIX}
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

  log "Start vLLM from ${VLLM_WORKDIR} with ${VLLM_BIN}: ${VLLM_LOG}"
  pushd "${VLLM_WORKDIR}" >/dev/null
  nohup env VIRTUAL_ENV="${VLLM_ENV_DIR}" PATH="${VLLM_ENV_DIR}/bin:${PATH}" HF_ENDPOINT="${HF_ENDPOINT}" "${cmd[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"
  popd >/dev/null
  wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"
}

start_thunderagent() {
  [[ "${START_THUNDERAGENT}" == "1" ]] || { log "Reuse ThunderAgent: ${TA_PORT}"; wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"; return; }
  ensure_port_free "${TA_PORT}" "THUNDERAGENT"

  log "Start ThunderAgent router=${ROUTER_MODE} with ${PYTHON_BIN}: ${TA_LOG}"
  nohup env VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m ThunderAgent \
    --backend-type vllm \
    --backends "http://127.0.0.1:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --router "${ROUTER_MODE}" \
    --metrics \
    --profile \
    --profile-dir "${PROFILE_DIR}" \
    > "${TA_LOG}" 2>&1 &
  TA_PID="$!"
  wait_url "http://127.0.0.1:${TA_PORT}/health" "ThunderAgent"
}

start_sampler() {
  [[ "${SAMPLE_METRICS}" == "1" ]] || return
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

run_swebench() {
  cd "${OPENHANDS_DIR}"
  local args=(
    -m evaluation.benchmarks.swe_bench.run_infer
    --config-file "${RUN_CONFIG}"
    --llm-config vllm_local
    --agent-cls "${OPENHANDS_AGENT_CLS}"
    --agent-config "${OPENHANDS_AGENT_CONFIG}"
    --dataset "${SWEBENCH_DATASET}"
    --split "${SWEBENCH_SPLIT}"
    --mode "${SWEBENCH_MODE}"
    --max-iterations "${SWEBENCH_MAX_ITERATIONS}"
    --eval-num-workers "${SWEBENCH_WORKERS}"
    --eval-note "${OPENHANDS_EVAL_NOTE}"
    --eval-output-dir "${RUN_DIR}/openhands_outputs"
  )
  [[ "${SWEBENCH_LIMIT}" == "0" ]] || args+=(--eval-n-limit "${SWEBENCH_LIMIT}")

  log "Run OpenHands: agent=${OPENHANDS_AGENT_CLS}, limit=${SWEBENCH_LIMIT}, workers=${SWEBENCH_WORKERS}, max_iter=${SWEBENCH_MAX_ITERATIONS}"
  VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" USE_HINT_TEXT="${USE_HINT_TEXT}" EVAL_DOCKER_IMAGE_SOURCE="${EVAL_DOCKER_IMAGE_SOURCE}" EPOCH_DOCKER_IMAGE_PREFIX="${EPOCH_DOCKER_IMAGE_PREFIX}" "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${SWEBENCH_LOG}"
}

main() {
  [[ -d "${OPENHANDS_DIR}" ]] || die "OpenHands not found: ${OPENHANDS_DIR}"
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find ThunderAgent/OpenHands python: ${PYTHON_BIN}"
  [[ "${START_VLLM}" != "1" ]] || [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"
  [[ "${START_VLLM}" != "1" ]] || [[ -x "${VLLM_BIN}" ]] || command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "Cannot find vLLM: ${VLLM_BIN}"

  write_run_files
  log "Run dir: ${RUN_DIR}"
  log "OpenHands config: ${RUN_CONFIG}"
  start_vllm
  start_thunderagent
  start_sampler
  run_swebench
  log "Done. Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
}

main "$@"
