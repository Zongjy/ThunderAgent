#!/usr/bin/env bash
set -euo pipefail

# Official tau2/τ³ benchmark + ThunderAgent + vLLM.
# Run from repo root:
#   bash examples/benchmark/tau3/run_tau3_ta_default.sh

# ---- Configurable parameters -------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TAU3_DIR="${REPO_ROOT}/examples/scaffold/tau3"
TAU3_ENV_DIR="${TAU3_ENV_DIR:-${REPO_ROOT}/.venv-tau3}"
THUNDERAGENT_ENV_DIR="${THUNDERAGENT_ENV_DIR:-${TAU3_ENV_DIR}}"
VLLM_ROOT="${VLLM_ROOT:-${HOME}/vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${VLLM_ROOT}/.venv}"
VLLM_WORKDIR="${VLLM_WORKDIR:-${VLLM_ROOT}}"
TAU2_SOURCE_ROOT="${TAU2_SOURCE_ROOT:-/raid0/liyi/tau2-bench}"
if [[ ! -d "${TAU2_SOURCE_ROOT}/src/tau2" ]]; then
  TAU2_SOURCE_ROOT=""
fi
if [[ -z "${TAU2_DATA_DIR:-}" ]]; then
  if [[ -n "${TAU2_SOURCE_ROOT}" && -d "${TAU2_SOURCE_ROOT}/data/tau2/domains" ]]; then
    TAU2_DATA_DIR="${TAU2_SOURCE_ROOT}/data"
  else
    TAU2_DATA_DIR="${REPO_ROOT}/.tau2_data"
  fi
fi

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
AGENT_LLM="${AGENT_LLM:-}"
USER_LLM="${USER_LLM:-}"
VLLM_PORT="${VLLM_PORT:-4747}"
TA_PORT="${TA_PORT:-9000}"
ROUTER_MODE="${ROUTER_MODE:-default}"

TAU3_DOMAIN="${TAU3_DOMAIN:-airline}"
TAU3_TASK_SET_NAME="${TAU3_TASK_SET_NAME:-}"
TAU3_TASK_SPLIT_NAME="${TAU3_TASK_SPLIT_NAME:-base}"
TAU3_TASK_IDS="${TAU3_TASK_IDS:-}"
TAU3_NUM_TASKS="${TAU3_NUM_TASKS:-${TAU3_LIMIT:-0}}" # 0 means all tasks in the split
TAU3_NUM_TRIALS="${TAU3_NUM_TRIALS:-1}"
TAU3_MAX_STEPS="${TAU3_MAX_STEPS:-100}"
TAU3_MAX_ERRORS="${TAU3_MAX_ERRORS:-10}"
TAU3_MAX_CONCURRENCY="${TAU3_MAX_CONCURRENCY:-4}"
TAU3_SEED="${TAU3_SEED:-300}"
TAU3_TEMPERATURE="${TAU3_TEMPERATURE:-0.6}"
TAU3_MAX_TOKENS="${TAU3_MAX_TOKENS:-32768}"
TAU3_RETRIEVAL_CONFIG="${TAU3_RETRIEVAL_CONFIG:-}"
TAU3_RETRIEVAL_CONFIG_KWARGS="${TAU3_RETRIEVAL_CONFIG_KWARGS:-}"
TAU3_VERBOSE_LOGS="${TAU3_VERBOSE_LOGS:-1}"
TAU3_AUTO_RESUME="${TAU3_AUTO_RESUME:-1}"

START_VLLM="${START_VLLM:-1}"
START_THUNDERAGENT="${START_THUNDERAGENT:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
SAMPLE_METRICS="${SAMPLE_METRICS:-1}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

PYTHON_BIN="${PYTHON_BIN:-${THUNDERAGENT_ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
HF_BIN="${HF_BIN:-${THUNDERAGENT_ENV_DIR}/bin/hf}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TAU2_DATASET_REPO="${TAU2_DATASET_REPO:-HuggingFaceH4/tau2-bench-data}"
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

RUN_NAME="${RUN_NAME:-tau3_${TAU3_DOMAIN}_${ROUTER_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/benchmark/tau3/runs}"

# ---- Derived paths -----------------------------------------------------------
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
PROFILE_DIR="${RUN_DIR}/thunderagent_profiles"
MANIFEST="${RUN_DIR}/manifest.env"
OFFICIAL_DIR="${RUN_DIR}/official"
RESULTS_JSON="${OFFICIAL_DIR}/results.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"

VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}_${ROUTER_MODE}.log"
TAU3_LOG="${LOG_DIR}/tau3_official.log"
TAU2_DOMAIN_DATA_DIR="${TAU2_DATA_DIR}/tau2/domains/${TAU3_DOMAIN}"

VLLM_PID=""
TA_PID=""
SAMPLER_PID=""

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

repo_pythonpath() {
  local value="${REPO_ROOT}"
  if [[ -n "${TAU2_SOURCE_ROOT}" ]]; then
    value="${value}:${TAU2_SOURCE_ROOT}/src"
  fi
  if [[ -n "${PYTHONPATH:-}" ]]; then
    value="${value}:${PYTHONPATH}"
  fi
  printf '%s' "${value}"
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

discover_vllm_model() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/v1/models"
with urllib.request.urlopen(url, timeout=10) as response:
    payload = json.load(response)
for item in payload.get("data", []):
    model_id = item.get("id")
    if model_id:
        print(model_id)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

resolve_model_names() {
  if [[ -z "${SERVED_MODEL_NAME}" ]]; then
    log "Discovering served model name from local vLLM"
    SERVED_MODEL_NAME="$(discover_vllm_model "http://127.0.0.1:${VLLM_PORT}")" || die "Cannot discover vLLM served model name; set SERVED_MODEL_NAME explicitly"
  fi
  AGENT_LLM="${AGENT_LLM:-openai/${SERVED_MODEL_NAME}}"
  USER_LLM="${USER_LLM:-openai/${SERVED_MODEL_NAME}}"
  log "Using model names: served=${SERVED_MODEL_NAME}, agent=${AGENT_LLM}, user=${USER_LLM}"
}

have_tau2_domain_data() {
  case "${TAU3_DOMAIN}" in
    airline|retail)
      [[ -f "${TAU2_DOMAIN_DATA_DIR}/db.json" && -f "${TAU2_DOMAIN_DATA_DIR}/policy.md" && -f "${TAU2_DOMAIN_DATA_DIR}/tasks.json" ]]
      ;;
    telecom)
      [[ -f "${TAU2_DOMAIN_DATA_DIR}/db.toml" && -f "${TAU2_DOMAIN_DATA_DIR}/user_db.toml" && -f "${TAU2_DOMAIN_DATA_DIR}/main_policy.md" && -f "${TAU2_DOMAIN_DATA_DIR}/tasks.json" ]]
      ;;
    banking_knowledge)
      [[ -f "${TAU2_DOMAIN_DATA_DIR}/db.json" && -d "${TAU2_DOMAIN_DATA_DIR}/tasks" && -d "${TAU2_DOMAIN_DATA_DIR}/documents" ]]
      ;;
    *)
      [[ -d "${TAU2_DOMAIN_DATA_DIR}" ]]
      ;;
  esac
}

ensure_tau2_data() {
  if have_tau2_domain_data; then
    log "Using tau2 data: ${TAU2_DOMAIN_DATA_DIR}"
    return
  fi

  [[ -x "${HF_BIN}" ]] || command -v "${HF_BIN}" >/dev/null 2>&1 || die "Cannot find Hugging Face CLI: ${HF_BIN}"
  mkdir -p "${TAU2_DATA_DIR}/tau2"
  log "Downloading official tau2 data for domain=${TAU3_DOMAIN} from ${TAU2_DATASET_REPO} via HF_ENDPOINT=${HF_ENDPOINT}"
  env HF_ENDPOINT="${HF_ENDPOINT}" "${HF_BIN}" download "${TAU2_DATASET_REPO}" \
    --repo-type dataset \
    --include "domains/${TAU3_DOMAIN}/**" \
    --local-dir "${TAU2_DATA_DIR}/tau2" \
    --max-workers 8

  have_tau2_domain_data || die "tau2 data download did not produce required files under ${TAU2_DOMAIN_DATA_DIR}"

  if [[ "${TAU3_DOMAIN}" == "telecom" && "${TAU3_TASK_SPLIT_NAME}" == "base" && ! -f "${TAU2_DOMAIN_DATA_DIR}/split_tasks.json" ]]; then
    log "No telecom split_tasks.json found in downloaded data; using all telecom tasks"
    TAU3_TASK_SPLIT_NAME=""
  fi
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
  mkdir -p "${LOG_DIR}" "${PROFILE_DIR}" "${RUN_DIR}"

  cat > "${MANIFEST}" <<EOF
MODEL=${MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
AGENT_LLM=${AGENT_LLM}
USER_LLM=${USER_LLM}
VLLM_PORT=${VLLM_PORT}
TA_PORT=${TA_PORT}
ROUTER_MODE=${ROUTER_MODE}
TAU3_DOMAIN=${TAU3_DOMAIN}
TAU3_TASK_SET_NAME=${TAU3_TASK_SET_NAME}
TAU3_TASK_SPLIT_NAME=${TAU3_TASK_SPLIT_NAME}
TAU3_TASK_IDS=${TAU3_TASK_IDS}
TAU3_NUM_TASKS=${TAU3_NUM_TASKS}
TAU3_NUM_TRIALS=${TAU3_NUM_TRIALS}
TAU3_MAX_STEPS=${TAU3_MAX_STEPS}
TAU3_MAX_ERRORS=${TAU3_MAX_ERRORS}
TAU3_MAX_CONCURRENCY=${TAU3_MAX_CONCURRENCY}
TAU3_TEMPERATURE=${TAU3_TEMPERATURE}
TAU3_MAX_TOKENS=${TAU3_MAX_TOKENS}
TAU3_RETRIEVAL_CONFIG=${TAU3_RETRIEVAL_CONFIG}
TAU3_RETRIEVAL_CONFIG_KWARGS=${TAU3_RETRIEVAL_CONFIG_KWARGS}
TAU2_DATA_DIR=${TAU2_DATA_DIR}
TAU2_SOURCE_ROOT=${TAU2_SOURCE_ROOT}
TAU2_DATASET_REPO=${TAU2_DATASET_REPO}
START_VLLM=${START_VLLM}
START_THUNDERAGENT=${START_THUNDERAGENT}
KEEP_SERVICES=${KEEP_SERVICES}
SAMPLE_METRICS=${SAMPLE_METRICS}
PYTHON_BIN=${PYTHON_BIN}
THUNDERAGENT_ENV_DIR=${THUNDERAGENT_ENV_DIR}
TAU3_ENV_DIR=${TAU3_ENV_DIR}
VLLM_ROOT=${VLLM_ROOT}
VLLM_ENV_DIR=${VLLM_ENV_DIR}
VLLM_WORKDIR=${VLLM_WORKDIR}
VLLM_BIN=${VLLM_BIN}
HF_ENDPOINT=${HF_ENDPOINT}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}
ENABLE_VLLM_TOOL_CALLING=${ENABLE_VLLM_TOOL_CALLING}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
REASONING_PARSER=${REASONING_PARSER}
RUN_NAME=${RUN_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_DIR=${RUN_DIR}
EOF
}

start_vllm() {
  [[ "${START_VLLM}" == "1" ]] || { log "Reuse vLLM: ${VLLM_PORT}"; wait_url "http://127.0.0.1:${VLLM_PORT}/health" "vLLM"; return; }
  ensure_port_free "${VLLM_PORT}" "VLLM"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"

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

  log "Start ThunderAgent router=${ROUTER_MODE}: ${TA_LOG}"
  nohup env VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" PYTHONPATH="$(repo_pythonpath)" "${PYTHON_BIN}" -m ThunderAgent \
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

run_tau3_official() {
  local cmd=(
    "${PYTHON_BIN}" "${TAU3_DIR}/run_official.py"
    --domain "${TAU3_DOMAIN}"
    --num-trials "${TAU3_NUM_TRIALS}"
    --max-steps "${TAU3_MAX_STEPS}"
    --max-errors "${TAU3_MAX_ERRORS}"
    --max-concurrency "${TAU3_MAX_CONCURRENCY}"
    --seed "${TAU3_SEED}"
    --log-level INFO
    --temperature "${TAU3_TEMPERATURE}"
    --max-tokens "${TAU3_MAX_TOKENS}"
    --agent-llm "${AGENT_LLM}"
    --agent-base-url "http://127.0.0.1:${TA_PORT}/v1"
    --user-llm "${USER_LLM}"
    --user-base-url "http://127.0.0.1:${VLLM_PORT}/v1"
    --save-to "${RUN_NAME}"
    --output-dir "${OFFICIAL_DIR}"
  )
  [[ -n "${TAU3_TASK_SPLIT_NAME}" ]] && cmd+=(--task-split-name "${TAU3_TASK_SPLIT_NAME}")
  [[ -n "${TAU3_TASK_SET_NAME}" ]] && cmd+=(--task-set-name "${TAU3_TASK_SET_NAME}")
  [[ -n "${TAU3_TASK_IDS}" ]] && cmd+=(--task-ids "${TAU3_TASK_IDS}")
  [[ -n "${TAU3_NUM_TASKS}" && "${TAU3_NUM_TASKS}" != "0" ]] && cmd+=(--num-tasks "${TAU3_NUM_TASKS}")
  [[ -n "${TAU3_RETRIEVAL_CONFIG}" ]] && cmd+=(--retrieval-config "${TAU3_RETRIEVAL_CONFIG}")
  [[ -n "${TAU3_RETRIEVAL_CONFIG_KWARGS}" ]] && cmd+=(--retrieval-config-kwargs "${TAU3_RETRIEVAL_CONFIG_KWARGS}")
  [[ "${TAU3_VERBOSE_LOGS}" == "1" ]] && cmd+=(--verbose-logs)
  [[ "${TAU3_AUTO_RESUME}" == "1" ]] && cmd+=(--auto-resume)

  log "Running official tau2/τ³ domain=${TAU3_DOMAIN}: ${TAU3_LOG}"
  VIRTUAL_ENV="${THUNDERAGENT_ENV_DIR}" PATH="${THUNDERAGENT_ENV_DIR}/bin:${PATH}" PYTHONPATH="$(repo_pythonpath)" TAU2_DATA_DIR="${TAU2_DATA_DIR}" HF_ENDPOINT="${HF_ENDPOINT}" "${cmd[@]}" 2>&1 | tee -a "${TAU3_LOG}"
}

main() {
  [[ -d "${TAU3_DIR}" ]] || die "tau^3 scaffold not found: ${TAU3_DIR}"
  command -v curl >/dev/null 2>&1 || die "Missing command: curl"
  command -v tee >/dev/null 2>&1 || die "Missing command: tee"
  [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Cannot find Python: ${PYTHON_BIN}. Run: bash examples/scripts/setup_benchmark_env.sh tau3"
  [[ "${START_VLLM}" != "1" ]] || [[ -d "${VLLM_WORKDIR}" ]] || die "vLLM workdir not found: ${VLLM_WORKDIR}"

  export TAU2_DATA_DIR HF_ENDPOINT
  ensure_tau2_data
  write_manifest
  log "Run dir: ${RUN_DIR}"
  start_vllm
  resolve_model_names
  write_manifest
  start_thunderagent
  start_sampler
  run_tau3_official
  "${PYTHON_BIN}" "${REPO_ROOT}/examples/benchmark/tau3/analyze_tau3_outputs.py" --results "${RESULTS_JSON}" --output "${SUMMARY_JSON}"
  log "Done. Official results: ${RESULTS_JSON}"
  log "Summary: ${SUMMARY_JSON}"
  log "Profile CSV: ${PROFILE_DIR}/step_profiles.csv"
}

main "$@"
