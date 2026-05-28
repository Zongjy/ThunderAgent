#!/usr/bin/env bash
set -euo pipefail

# SWE-bench + mini-swe-agent + ThunderAgent + vLLM.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINISWEAGENT_ENV_DIR="${MINISWEAGENT_ENV_DIR:-${REPO_ROOT}/.venv-minisweagent}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${MINISWEAGENT_ENV_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-${MINISWEAGENT_ENV_DIR}/bin/python}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
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
THUNDERAGENT_VLLM_LOG_PATHS="${THUNDERAGENT_VLLM_LOG_PATHS:-}"
THUNDERAGENT_SCHEDULER_INTERVAL="${THUNDERAGENT_SCHEDULER_INTERVAL:-5}"
THUNDERAGENT_ACTING_TOKEN_WEIGHT="${THUNDERAGENT_ACTING_TOKEN_WEIGHT:-1.0}"
THUNDERAGENT_USE_ACTING_TOKEN_DECAY="${THUNDERAGENT_USE_ACTING_TOKEN_DECAY:-0}"

SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-verified}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
SWEBENCH_SLICE="${SWEBENCH_SLICE:-0:50}"
SWEBENCH_FILTER="${SWEBENCH_FILTER:-}"
SWEBENCH_SHUFFLE="${SWEBENCH_SHUFFLE:-0}"
SWEBENCH_WORKERS="${SWEBENCH_WORKERS:-4}"
SWEBENCH_ROLLOUTS_PER_INSTANCE="${SWEBENCH_ROLLOUTS_PER_INSTANCE:-1}"
SWEBENCH_CHECKPOINT_INTERVAL_STEPS="${SWEBENCH_CHECKPOINT_INTERVAL_STEPS:-0}"
SWEBENCH_REDO_EXISTING="${SWEBENCH_REDO_EXISTING:-0}"
SWEBENCH_CLEANUP_IMAGES="${SWEBENCH_CLEANUP_IMAGES:-0}"
EVAL_DOCKER_IMAGE_SOURCE="${EVAL_DOCKER_IMAGE_SOURCE:-epoch}"
EPOCH_DOCKER_IMAGE_PREFIX="${EPOCH_DOCKER_IMAGE_PREFIX:-ghcr.io/epoch-research}"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_VLLM_USAGE_FLAGS="${ENABLE_VLLM_USAGE_FLAGS:-1}"
ENABLE_VLLM_TOOL_CALLING="${ENABLE_VLLM_TOOL_CALLING:-1}"
ENABLE_VLLM_KV_METRICS="${ENABLE_VLLM_KV_METRICS:-1}"
ENABLE_VLLM_LANGUAGE_MODEL_ONLY="${ENABLE_VLLM_LANGUAGE_MODEL_ONLY:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

MINISWEAGENT_CONFIG="${MINISWEAGENT_CONFIG:-${REPO_ROOT}/examples/scaffold/mini_swe_agent/config/swebench_thunderagent.yaml}"
MINISWEAGENT_MODEL_CLASS="${MINISWEAGENT_MODEL_CLASS:-examples.scaffold.mini_swe_agent.thunderagent_vllm_model.ThunderAgentVLLMModel}"

RUN_NAME="${RUN_NAME:-minisweagent_swebench_ta_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/swebench/runs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
MANIFEST="${RUN_DIR}/manifest.env"
MINISWEAGENT_OUTPUT_DIR="${RUN_DIR}/minisweagent_outputs"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
SWEBENCH_LOG="${LOG_DIR}/minisweagent_swebench.log"

if [[ -z "${THUNDERAGENT_VLLM_LOG_PATHS}" ]]; then
  THUNDERAGENT_VLLM_LOG_PATHS="${VLLM_LOG}"
fi

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
    die "${name} port ${port} is in use; set START_${name}=0 to reuse it or choose another ${name}_PORT"
  fi
}

extract_vllm_kv_capacity() {
  [[ -f "${VLLM_LOG}" ]] || return 1
  local line value
  line="$(grep -E 'GPU KV cache size: [0-9,]+ tokens' "${VLLM_LOG}" | tail -n 1 || true)"
  [[ -n "${line}" ]] || return 1
  value="$(sed -E 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/' <<< "${line}" | tr -d ',')"
  [[ -n "${value}" ]] || return 1
  printf '%s\n' "${value}"
}

wait_vllm_kv_capacity() {
  [[ "${THUNDERAGENT_BACKEND_TYPE}" == "vllm" ]] || return 0
  is_enabled "${START_VLLM}" || return 0
  [[ -z "${THUNDERAGENT_KV_CAPACITY_TOKENS}" ]] || return 0

  local start now_ts capacity
  start="$(date +%s)"
  while true; do
    capacity="$(extract_vllm_kv_capacity || true)"
    if [[ -n "${capacity}" ]]; then
      THUNDERAGENT_KV_CAPACITY_TOKENS="${capacity}"
      log "Detected vLLM KV cache capacity: ${THUNDERAGENT_KV_CAPACITY_TOKENS} tokens"
      return 0
    fi
    now_ts="$(date +%s)"
    (( now_ts - start < HEALTH_TIMEOUT_S )) || die "vLLM KV capacity not found in log: ${VLLM_LOG}"
    sleep 1
  done
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

write_run_files() {
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${MINISWEAGENT_OUTPUT_DIR}"
  write_env_manifest "${MANIFEST}" \
    REPO_ROOT MINISWEAGENT_ENV_DIR THUNDERAGENT_ENV_DIR PYTHON_BIN \
    MODEL SERVED_MODEL_NAME VLLM_ROOT VLLM_ENV_DIR VLLM_WORKDIR VLLM_BIN VLLM_PORT TA_PORT \
    ROUTER_MODE THUNDERAGENT_BACKEND_TYPE THUNDERAGENT_BACKENDS THUNDERAGENT_KV_CAPACITY_TOKENS THUNDERAGENT_VLLM_LOG_PATHS \
    THUNDERAGENT_SCHEDULER_INTERVAL THUNDERAGENT_ACTING_TOKEN_WEIGHT THUNDERAGENT_USE_ACTING_TOKEN_DECAY \
    SWEBENCH_SUBSET SWEBENCH_SPLIT SWEBENCH_SLICE SWEBENCH_FILTER SWEBENCH_SHUFFLE SWEBENCH_WORKERS \
    SWEBENCH_ROLLOUTS_PER_INSTANCE SWEBENCH_CHECKPOINT_INTERVAL_STEPS SWEBENCH_REDO_EXISTING SWEBENCH_CLEANUP_IMAGES \
    EVAL_DOCKER_IMAGE_SOURCE EPOCH_DOCKER_IMAGE_PREFIX \
    START_VLLM START_THUNDERAGENT KEEP_SERVICES SAMPLE_METRICS METRICS_INTERVAL_S HEALTH_TIMEOUT_S \
    HF_ENDPOINT GPU_MEMORY_UTILIZATION MAX_MODEL_LEN MAX_NUM_SEQS ENABLE_PREFIX_CACHING ENABLE_VLLM_USAGE_FLAGS \
    ENABLE_VLLM_TOOL_CALLING ENABLE_VLLM_KV_METRICS ENABLE_VLLM_LANGUAGE_MODEL_ONLY TOOL_CALL_PARSER \
    REASONING_PARSER VLLM_EXTRA_ARGS MINISWEAGENT_CONFIG MINISWEAGENT_MODEL_CLASS \
    RUN_NAME OUTPUT_ROOT RUN_DIR LOG_DIR PROFILE_DIR MINISWEAGENT_OUTPUT_DIR
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
    --max-num-seqs "${MAX_NUM_SEQS}"
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
  [[ -n "${THUNDERAGENT_VLLM_LOG_PATHS}" ]] && cmd+=(--vllm-log-paths "${THUNDERAGENT_VLLM_LOG_PATHS}")
  [[ -n "${THUNDERAGENT_KV_CAPACITY_TOKENS}" ]] && cmd+=(--kv-capacity-tokens "${THUNDERAGENT_KV_CAPACITY_TOKENS}")

  log "Start ThunderAgent router=${ROUTER_MODE}: ${TA_LOG}"
  nohup env \
    VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" \
    PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" \
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    THUNDERAGENT_KV_CAPACITY_TOKENS="${THUNDERAGENT_KV_CAPACITY_TOKENS}" \
    THUNDERAGENT_VLLM_LOG_PATHS="${THUNDERAGENT_VLLM_LOG_PATHS}" \
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

run_swebench() {
  local args=(
    "${REPO_ROOT}/examples/scaffold/mini_swe_agent/run_swebench_ta.py"
    --subset "${SWEBENCH_SUBSET}"
    --split "${SWEBENCH_SPLIT}"
    --workers "${SWEBENCH_WORKERS}"
    --output "${MINISWEAGENT_OUTPUT_DIR}"
    --config "${MINISWEAGENT_CONFIG}"
    --model "${SERVED_MODEL_NAME}"
    --model-class "${MINISWEAGENT_MODEL_CLASS}"
    --base-url "http://127.0.0.1:${TA_PORT}/v1"
    --rollouts-per-instance "${SWEBENCH_ROLLOUTS_PER_INSTANCE}"
    --checkpoint-interval-steps "${SWEBENCH_CHECKPOINT_INTERVAL_STEPS}"
  )
  [[ -z "${SWEBENCH_SLICE}" ]] || args+=(--slice "${SWEBENCH_SLICE}")
  [[ -z "${SWEBENCH_FILTER}" ]] || args+=(--filter "${SWEBENCH_FILTER}")
  is_enabled "${SWEBENCH_SHUFFLE}" && args+=(--shuffle)
  is_enabled "${SWEBENCH_REDO_EXISTING}" && args+=(--redo-existing)
  is_enabled "${SWEBENCH_CLEANUP_IMAGES}" && args+=(--cleanup-images)

  log "Run mini-swe-agent SWE-bench: subset=${SWEBENCH_SUBSET}, split=${SWEBENCH_SPLIT}, slice=${SWEBENCH_SLICE}, workers=${SWEBENCH_WORKERS}"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    VIRTUAL_ENV="${MINISWEAGENT_ENV_DIR}" \
    PATH="${MINISWEAGENT_ENV_DIR}/bin:${PATH}" \
    EVAL_DOCKER_IMAGE_SOURCE="${EVAL_DOCKER_IMAGE_SOURCE}" \
    EPOCH_DOCKER_IMAGE_PREFIX="${EPOCH_DOCKER_IMAGE_PREFIX}" \
    "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${SWEBENCH_LOG}"
}

main() {
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find Python: ${PYTHON_BIN}. Run: bash examples/scripts/setup_benchmark_env.sh mini-swe-agent"
  [[ "${START_VLLM}" != "1" ]] || [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"
  [[ "${START_VLLM}" != "1" ]] || [[ -x "${VLLM_BIN}" ]] || command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "Cannot find vLLM: ${VLLM_BIN}"
  [[ -f "${MINISWEAGENT_CONFIG}" ]] || die "mini-swe-agent config not found: ${MINISWEAGENT_CONFIG}"

  write_run_files
  log "Run dir: ${RUN_DIR}"
  start_vllm
  wait_vllm_kv_capacity
  write_run_files
  start_thunderagent
  start_sampler
  run_swebench
  log "Done. mini-swe-agent outputs: ${MINISWEAGENT_OUTPUT_DIR}"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
}

main "$@"
